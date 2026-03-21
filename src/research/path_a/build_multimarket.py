"""
Multi-market synthetic data generator for Leviathan regime research.
======================================================================
Generates N independent synthetic housing-market panels with controlled
heterogeneity so that Leviathan's regime-only overlay can be evaluated
across many distinct regime realizations rather than a single panel.

Design
------
Each market has its own:
  - real-rate path  (amplitude, phase, frequency, level offset)
  - DTI path        (trend start/end, regime boost, noise level)
  - price process   (base growth rate, regime boost, noise)
  - crash params    (GFC depth, joint-trigger probability, shock magnitude)

Invariants across all markets (for comparability):
  - Same calendar span (default 1992-Q1 to 2024-Q4)
  - Same regime definition: regime = (real_rate < 0).astype(int)
  - Same GFC calendar anchor: 2007-Q1 to 2009-Q1 (global event)
  - Same rolling-pct window: 20 quarters
  - Same joint-trigger threshold: 0.65

Reproducibility:
  - meta_seed controls parameter sampling (one draw per market, consumed
    sequentially so market order is stable)
  - base_seed + market_id seeds the per-market data RNG

Path A convention: no imports from src.evaluation.* — all computations
(rolling pct rank, regime) are inline.

Usage
-----
    from src.research.path_a.build_multimarket import build_all_markets
    df = build_all_markets(n_markets=12)
    # df: market_id, date, dti, real_rate, regime, real_price_index
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (shared with build_dataset.py)
# ---------------------------------------------------------------------------

_ROLL_W        = 20      # rolling percentile window (quarters)
_JOINT_THRESH  = 0.65    # dti_pct_roll threshold for joint-trigger eligibility
_JOINT_DUR_MIN = 3
_JOINT_DUR_MAX = 4
_BG_DUR_MIN    = 2
_BG_DUR_MAX    = 3
_GFC_START_DATE = "2007-03-31"
_GFC_END_DATE   = "2009-03-31"

N_MARKETS_DEFAULT = 12
_META_SEED        = 999
_BASE_SEED        = 1000


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def _sample_market_params(meta_rng: np.random.Generator) -> dict:
    """Draw market-specific parameters from the meta RNG.

    Ranges are chosen to produce plausible synthetic housing markets:
    positive long-run price growth, regime switches roughly every 3–6 years,
    and crash episodes of realistic magnitude (10–40% peak-to-trough).
    """
    return {
        # real-rate path
        "rate_amplitude":      float(meta_rng.uniform(1.0,   2.5)),
        "rate_phase":          float(meta_rng.uniform(0.0,   2 * np.pi)),
        "rate_freq_mult":      float(meta_rng.uniform(0.6,   1.5)),
        "rate_level":          float(meta_rng.uniform(-0.5,  0.5)),
        "rate_noise_std":      float(meta_rng.uniform(0.4,   0.9)),
        # DTI path
        "dti_start":           float(meta_rng.uniform(75.0,  100.0)),
        "dti_end":             float(meta_rng.uniform(115.0, 155.0)),
        "dti_regime_boost":    float(meta_rng.uniform(3.0,   9.0)),
        "dti_noise_std":       float(meta_rng.uniform(1.5,   3.0)),
        # price process
        "growth_base":         float(meta_rng.uniform(0.004, 0.013)),
        "growth_regime_boost": float(meta_rng.uniform(0.001, 0.007)),
        "price_noise_std":     float(meta_rng.uniform(0.007, 0.016)),
        # GFC crash
        "gfc_shock":           float(meta_rng.uniform(0.045, 0.085)),
        # joint-trigger crash
        "joint_prob":          float(meta_rng.uniform(0.25,  0.60)),
        "joint_shock_lo":      float(meta_rng.uniform(0.050, 0.080)),
        "joint_shock_hi":      float(meta_rng.uniform(0.080, 0.120)),
        # background noise crashes
        "bg_prob":             float(meta_rng.uniform(0.02,  0.05)),
    }


# ---------------------------------------------------------------------------
# Single-market builder
# ---------------------------------------------------------------------------

def build_market(
    market_id: int,
    params: dict,
    *,
    start: str = "1992-01-01",
    end:   str = "2024-12-31",
    seed:  int,
) -> pd.DataFrame:
    """Build one synthetic market panel with the given parameters.

    Returns a DataFrame with columns:
        market_id, date, dti, real_rate, regime, real_price_index

    Regime is computed inline: regime = (real_rate < 0).astype(int)
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="QE")
    n     = len(dates)

    # ------------------------------------------------------------------
    # Real rate and regime
    # ------------------------------------------------------------------
    t_vec     = np.linspace(0, 8 * np.pi * params["rate_freq_mult"], n)
    cycle     = (params["rate_amplitude"] * np.sin(t_vec + params["rate_phase"])
                 + params["rate_level"])
    real_rate = cycle + rng.normal(0, params["rate_noise_std"], size=n)
    regime    = (real_rate < 0).astype(int)

    # ------------------------------------------------------------------
    # DTI
    # ------------------------------------------------------------------
    trend = np.linspace(params["dti_start"], params["dti_end"], n)
    dti   = (trend
             + regime * params["dti_regime_boost"]
             + rng.normal(0, params["dti_noise_std"], size=n))
    dti   = np.clip(dti, 60, 180)

    # ------------------------------------------------------------------
    # Base returns
    # ------------------------------------------------------------------
    base_growth = params["growth_base"] + regime * params["growth_regime_boost"]
    noise       = rng.normal(0, params["price_noise_std"], size=n)

    # ------------------------------------------------------------------
    # Rolling percentile rank of DTI (inline — Path A convention)
    # ------------------------------------------------------------------
    dti_pct_roll = np.empty(n)
    for t_idx in range(n):
        w_start              = max(0, t_idx - _ROLL_W + 1)
        seg                  = dti[w_start : t_idx + 1]
        dti_pct_roll[t_idx]  = np.sum(seg <= dti[t_idx]) / len(seg)

    # ------------------------------------------------------------------
    # Crash mechanism (mirrors build_dataset.py structure)
    # ------------------------------------------------------------------
    crash = np.zeros(n)

    gfc_s = int(np.searchsorted(dates, pd.Timestamp(_GFC_START_DATE), side="left"))
    gfc_e = int(np.searchsorted(dates, pd.Timestamp(_GFC_END_DATE),   side="left"))
    crash[gfc_s:gfc_e] = -params["gfc_shock"]
    gfc_suppress = set(range(max(0, gfc_s - 2), min(n, gfc_e + 3)))

    joint_eligible = (regime == 0) & (dti_pct_roll > _JOINT_THRESH)
    active_until   = -1

    for t_idx in range(n):
        if t_idx <= active_until or t_idx in gfc_suppress:
            continue

        if joint_eligible[t_idx] and rng.random() < params["joint_prob"]:
            dur   = int(rng.integers(_JOINT_DUR_MIN, _JOINT_DUR_MAX + 1))
            shock = float(rng.uniform(params["joint_shock_lo"], params["joint_shock_hi"]))
            for q in range(dur):
                if t_idx + q < n:
                    crash[t_idx + q] -= shock
            active_until = t_idx + dur - 1
            continue

        if rng.random() < params["bg_prob"]:
            dur   = int(rng.integers(_BG_DUR_MIN, _BG_DUR_MAX + 1))
            shock = float(rng.uniform(0.020, 0.040))
            for q in range(dur):
                if t_idx + q < n:
                    crash[t_idx + q] -= shock
            active_until = t_idx + dur - 1

    # ------------------------------------------------------------------
    # Price index
    # ------------------------------------------------------------------
    returns          = base_growth + noise + crash
    real_price_index = 100 * np.exp(np.cumsum(returns))
    real_price_index = real_price_index / real_price_index[0] * 100

    return pd.DataFrame({
        "market_id":        market_id,
        "date":             dates,
        "dti":              dti,
        "real_rate":        real_rate,
        "regime":           regime,
        "real_price_index": real_price_index,
    })


# ---------------------------------------------------------------------------
# Multi-market builder
# ---------------------------------------------------------------------------

def build_all_markets(
    n_markets:  int  = N_MARKETS_DEFAULT,
    *,
    start:      str  = "1992-01-01",
    end:        str  = "2024-12-31",
    base_seed:  int  = _BASE_SEED,
    meta_seed:  int  = _META_SEED,
    verbose:    bool = False,
) -> pd.DataFrame:
    """Build N synthetic market panels.

    Parameters
    ----------
    n_markets : int
        Number of independent synthetic markets (default 12).
    start, end : str
        Panel date range (inclusive).
    base_seed : int
        Per-market seed = base_seed + market_id.
    meta_seed : int
        Seed for parameter sampling across markets.
    verbose : bool
        If True, print per-market regime diagnostics.

    Returns
    -------
    pd.DataFrame
        Combined panel with columns:
        market_id, date, dti, real_rate, regime, real_price_index
    """
    meta_rng = np.random.default_rng(meta_seed)
    frames   = []

    for m in range(n_markets):
        params = _sample_market_params(meta_rng)
        df     = build_market(m, params, start=start, end=end, seed=base_seed + m)
        frames.append(df)

        if verbose:
            n_sw  = int((pd.Series(df["regime"]).diff().abs() > 0).sum())
            p_adv = float((df["regime"] == 0).mean())
            print(f"  market {m:2d}: "
                  f"regime_switches={n_sw:2d}  "
                  f"pct_adverse={p_adv:.2f}  "
                  f"rate_amp={params['rate_amplitude']:.2f}  "
                  f"rate_phase={params['rate_phase']:.2f}  "
                  f"freq={params['rate_freq_mult']:.2f}")

    return pd.concat(frames, ignore_index=True)
