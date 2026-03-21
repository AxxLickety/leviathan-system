"""
Adversarial multi-market synthetic data generator — Leviathan stress test.
===========================================================================
Generates N synthetic housing-market panels that are deliberately harder for
the regime-only overlay than the base multi-market world in two ways:

  1. De-synchronized crashes: each market has one large "idiosyncratic" crash
     at a market-specific calendar date (drawn uniformly across the panel,
     training or test, any regime state). No synchronized GFC.

  2. Reduced regime dominance: crashes are regime-conditional (adverse regime
     raises probability) but regime is imperfect — non-trivial crash
     probability exists in regime==1, and many regime-0 quarters have no
     crash. The joint-trigger pct_roll requirement is dropped entirely.

Target world properties (vs base multi-market):
  - Fraction of crash quarters in regime==0: ~60–70%  (was ~85–90%)
  - Fraction of regime-0 quarters actively crashing: ~30–40%
  - Fraction of regime-1 quarters actively crashing: ~12–20%
  - Major crash can fall in either regime state (50/50 by construction)

Invariants (for comparability with base multi-market world):
  - Same calendar span (default 1992-Q1 to 2024-Q4)
  - Same regime definition: (real_rate < 0).astype(int)
  - Same rolling-pct window, same rate/DTI/price parameter ranges
  - Same train/test split (≤ 2007-12-31 / ≥ 2008-03-31)

Returns
-------
build_all_markets_adversarial() -> (pd.DataFrame, list[dict])
    Combined panel + list of per-market crash diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROLL_W        = 20
_BG_DUR_MIN    = 2
_BG_DUR_MAX    = 3

N_MARKETS_DEFAULT = 12
_META_SEED        = 7777     # different from base world so params differ
_BASE_SEED        = 2000     # different from base world

# Major crash onset window: [onset_lo_frac, onset_hi_frac] of panel length
# Allows crash to fall in train or test, in either regime state.
_ONSET_LO = 0.20
_ONSET_HI = 0.75


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def _sample_adversarial_params(meta_rng: np.random.Generator) -> dict:
    """Draw market parameters for one adversarial market.

    Rate/DTI/price ranges match the base multi-market world exactly so that
    differences in results are attributable to crash design, not to different
    macro environments.
    """
    return {
        # real-rate path (identical ranges to base world)
        "rate_amplitude":      float(meta_rng.uniform(1.0,   2.5)),
        "rate_phase":          float(meta_rng.uniform(0.0,   2 * np.pi)),
        "rate_freq_mult":      float(meta_rng.uniform(0.6,   1.5)),
        "rate_level":          float(meta_rng.uniform(-0.5,  0.5)),
        "rate_noise_std":      float(meta_rng.uniform(0.4,   0.9)),
        # DTI path (identical ranges to base world)
        "dti_start":           float(meta_rng.uniform(75.0,  100.0)),
        "dti_end":             float(meta_rng.uniform(115.0, 155.0)),
        "dti_regime_boost":    float(meta_rng.uniform(3.0,   9.0)),
        "dti_noise_std":       float(meta_rng.uniform(1.5,   3.0)),
        # price process (identical ranges to base world)
        "growth_base":         float(meta_rng.uniform(0.004, 0.013)),
        "growth_regime_boost": float(meta_rng.uniform(0.001, 0.007)),
        "price_noise_std":     float(meta_rng.uniform(0.007, 0.016)),
        # -------------------------------------------------------------------
        # Adversarial crash design
        # -------------------------------------------------------------------
        # Major idiosyncratic crash (replaces synchronized GFC)
        "major_onset_frac":    float(meta_rng.uniform(_ONSET_LO, _ONSET_HI)),
        "major_depth":         float(meta_rng.uniform(0.040,  0.070)),
        "major_dur":           int(meta_rng.integers(5, 10)),    # 5–9 quarters
        # Regime-conditional probabilistic crashes
        # p_r0 > p_r1 so regime is informative but imperfect
        "p_crash_r0":          float(meta_rng.uniform(0.08,   0.18)),
        "p_crash_r1":          float(meta_rng.uniform(0.03,   0.08)),
        "crash_dur_min":       int(meta_rng.integers(2, 4)),
        "crash_dur_max":       int(meta_rng.integers(4, 7)),
        "crash_shock_lo":      float(meta_rng.uniform(0.030,  0.055)),
        "crash_shock_hi":      float(meta_rng.uniform(0.055,  0.090)),
        # Background noise (slightly elevated vs base world)
        "bg_prob":             float(meta_rng.uniform(0.03,   0.07)),
        "bg_shock_lo":         float(meta_rng.uniform(0.015,  0.025)),
        "bg_shock_hi":         float(meta_rng.uniform(0.025,  0.045)),
    }


# ---------------------------------------------------------------------------
# Single-market builder
# ---------------------------------------------------------------------------

def build_market_adversarial(
    market_id: int,
    params: dict,
    *,
    start: str = "1992-01-01",
    end:   str = "2024-12-31",
    seed:  int,
) -> tuple[pd.DataFrame, dict]:
    """Build one adversarial synthetic market.

    Returns
    -------
    (df, diag)
        df   — market panel (market_id, date, dti, real_rate, regime, real_price_index)
        diag — crash diagnostic dict for Phase 4 reporting
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="QE")
    n     = len(dates)

    # ------------------------------------------------------------------
    # Real rate and regime (inline — Path A convention)
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
    # Crash mechanism — adversarial design
    # ------------------------------------------------------------------
    crash = np.zeros(n)
    crash_events: list[tuple[int, str]] = []   # (start_idx, regime_at_start)

    # --- Major idiosyncratic crash (market-specific calendar) -----------
    major_onset = int(np.clip(params["major_onset_frac"] * n, 0, n - params["major_dur"] - 1))
    major_dur   = params["major_dur"]
    for q in range(major_dur):
        if major_onset + q < n:
            crash[major_onset + q] -= params["major_depth"]
    major_regime = "r0" if regime[major_onset] == 0 else "r1"
    crash_events.append((major_onset, major_regime))

    # Suppress zone around major crash
    major_suppress = set(range(max(0, major_onset - 2),
                               min(n, major_onset + major_dur + 3)))

    # --- Regime-conditional + background probabilistic crashes ----------
    active_until = major_onset + major_dur - 1   # suppress overlap with major crash

    for t_idx in range(n):
        if t_idx <= active_until or t_idx in major_suppress:
            continue

        p = params["p_crash_r0"] if regime[t_idx] == 0 else params["p_crash_r1"]

        if rng.random() < p:
            dur   = int(rng.integers(params["crash_dur_min"],
                                     params["crash_dur_max"] + 1))
            shock = float(rng.uniform(params["crash_shock_lo"],
                                      params["crash_shock_hi"]))
            for q in range(dur):
                if t_idx + q < n:
                    crash[t_idx + q] -= shock
            active_until = t_idx + dur - 1
            r_label = "r0" if regime[t_idx] == 0 else "r1"
            crash_events.append((t_idx, r_label))
            continue

        if rng.random() < params["bg_prob"]:
            dur   = int(rng.integers(_BG_DUR_MIN, _BG_DUR_MAX + 1))
            shock = float(rng.uniform(params["bg_shock_lo"], params["bg_shock_hi"]))
            for q in range(dur):
                if t_idx + q < n:
                    crash[t_idx + q] -= shock
            active_until = t_idx + dur - 1
            r_label = "r0" if regime[t_idx] == 0 else "r1"
            crash_events.append((t_idx, "bg_" + r_label))

    # ------------------------------------------------------------------
    # Price index
    # ------------------------------------------------------------------
    returns          = base_growth + noise + crash
    real_price_index = 100 * np.exp(np.cumsum(returns))
    real_price_index = real_price_index / real_price_index[0] * 100

    # ------------------------------------------------------------------
    # Phase 4 diagnostics
    # ------------------------------------------------------------------
    n_r0 = int((regime == 0).sum())
    n_r1 = n - n_r0

    # Crash-active quarters per regime (any quarter where crash[t] < 0)
    crash_active    = crash < 0
    r0_crash_q      = int(((regime == 0) & crash_active).sum())
    r1_crash_q      = int(((regime == 1) & crash_active).sum())
    total_crash_q   = r0_crash_q + r1_crash_q

    # Crash event counts (by type)
    n_major      = 1
    n_prob_r0    = sum(1 for (_, r) in crash_events if r == "r0")
    n_prob_r1    = sum(1 for (_, r) in crash_events if r == "r1")
    n_bg_r0      = sum(1 for (_, r) in crash_events if r == "bg_r0")
    n_bg_r1      = sum(1 for (_, r) in crash_events if r == "bg_r1")
    n_all_events = len(crash_events)

    frac_crash_q_in_r0 = (r0_crash_q / total_crash_q) if total_crash_q > 0 else float("nan")
    frac_r0_in_crash   = (r0_crash_q / n_r0)           if n_r0 > 0          else float("nan")
    frac_r1_in_crash   = (r1_crash_q / n_r1)           if n_r1 > 0          else float("nan")

    diag = {
        "market_id":            market_id,
        "major_onset_q":        major_onset,
        "major_onset_date":     str(dates[major_onset].date()),
        "major_regime":         major_regime,
        "n_crash_events":       n_all_events,
        "n_prob_r0":            n_prob_r0,
        "n_prob_r1":            n_prob_r1,
        "n_bg_r0":              n_bg_r0,
        "n_bg_r1":              n_bg_r1,
        "crash_q_in_r0":        r0_crash_q,
        "crash_q_in_r1":        r1_crash_q,
        "total_crash_q":        total_crash_q,
        "frac_crash_q_in_r0":   round(frac_crash_q_in_r0, 3),
        "frac_r0_in_crash":     round(frac_r0_in_crash,   3),
        "frac_r1_in_crash":     round(frac_r1_in_crash,   3),
        "regime_switches":      int((pd.Series(regime).diff().abs() > 0).sum()),
    }

    df = pd.DataFrame({
        "market_id":        market_id,
        "date":             dates,
        "dti":              dti,
        "real_rate":        real_rate,
        "regime":           regime,
        "real_price_index": real_price_index,
    })

    return df, diag


# ---------------------------------------------------------------------------
# Multi-market builder
# ---------------------------------------------------------------------------

def build_all_markets_adversarial(
    n_markets:  int  = N_MARKETS_DEFAULT,
    *,
    start:      str  = "1992-01-01",
    end:        str  = "2024-12-31",
    base_seed:  int  = _BASE_SEED,
    meta_seed:  int  = _META_SEED,
    verbose:    bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """Build N adversarial synthetic market panels.

    Returns
    -------
    (df_combined, list_of_diagnostics)
    """
    meta_rng  = np.random.default_rng(meta_seed)
    frames    = []
    diag_list = []

    for m in range(n_markets):
        params     = _sample_adversarial_params(meta_rng)
        df, diag   = build_market_adversarial(
            m, params, start=start, end=end, seed=base_seed + m
        )
        frames.append(df)
        diag_list.append(diag)

        if verbose:
            print(f"  market {m:2d}: "
                  f"switches={diag['regime_switches']:2d}  "
                  f"major={diag['major_onset_date']}[{diag['major_regime']}]  "
                  f"crash_q_r0={diag['crash_q_in_r0']:2d}  "
                  f"crash_q_r1={diag['crash_q_in_r1']:2d}  "
                  f"frac_in_r0={diag['frac_crash_q_in_r0']:.2f}")

    return pd.concat(frames, ignore_index=True), diag_list
