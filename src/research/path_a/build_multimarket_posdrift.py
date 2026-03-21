"""
Positive-drift adversarial multi-market generator — Leviathan attribution test.
================================================================================
This world is designed to isolate whether Leviathan's regime-only overlay has
SIGNAL VALUE (regime predicts bad outcomes) vs PARTICIPATION-REDUCTION VALUE
(being out 50% of the time helps when always_in has negative expected return).

How it differs from the adversarial world:
  - Higher base drift: always_in expected return is positive in most markets.
    Being out of the market now has a real opportunity cost.
  - Lighter, shorter crashes: positive drift dominates in most quarters,
    making the test period broadly favorable for staying invested.
  - Crash mechanism is otherwise structurally identical to the adversarial world:
    market-specific major crash (no synchronized GFC), regime-conditional
    probabilistic crashes (p_r0 > p_r1 but not overwhelmingly), background noise.

Target world properties:
  - always_in mean return > 0 in most markets (clearly positive Sharpe common)
  - Fraction of crash-quarters in regime-0: ~60–75%  (regime is informative, not dominant)
  - Fraction of regime-0 quarters crashing: ~20–35%
  - Fraction of regime-1 quarters crashing: ~8–18%
  - Major crash: market-specific calendar, any regime state

The crash-regime ratio (p_r0 / p_r1 ≈ 2.5×) is stronger than the adversarial world
(1.3×) but weaker than the base world (near-infinite). Regime is more informative here
than in the adversarial world, but the positive drift creates a real opportunity cost
for the regime filter — the harder attribution question.

Invariants:
  - Same calendar span (1992-Q1 to 2024-Q4)
  - Same regime definition: (real_rate < 0).astype(int)
  - Same rate/DTI parameter ranges as base and adversarial worlds
  - Same train/test split (≤ 2007-12-31 / ≥ 2008-03-31)

Returns
-------
build_all_markets_posdrift() -> (pd.DataFrame, list[dict])
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
_META_SEED        = 5555     # distinct from base (999) and adversarial (7777)
_BASE_SEED        = 3000     # distinct from base (1000) and adversarial (2000)

_ONSET_LO = 0.20
_ONSET_HI = 0.75


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def _sample_posdrift_params(meta_rng: np.random.Generator) -> dict:
    """Draw parameters for one positive-drift adversarial market.

    Key differences vs adversarial world:
      - growth_base / growth_regime_boost: substantially higher
        → positive always_in expected return; opportunity cost of exiting
      - p_crash_r0, p_crash_r1: reduced; crashes are less frequent
        → drift can dominate in most quarters
      - major_depth / major_dur: lighter and shorter
        → single large crash does not overwhelm the positive drift

    Rate/DTI ranges are identical to base and adversarial worlds so that
    macro environment differences don't confound the comparison.
    """
    return {
        # real-rate path (identical ranges to all prior worlds)
        "rate_amplitude":      float(meta_rng.uniform(1.0,   2.5)),
        "rate_phase":          float(meta_rng.uniform(0.0,   2 * np.pi)),
        "rate_freq_mult":      float(meta_rng.uniform(0.6,   1.5)),
        "rate_level":          float(meta_rng.uniform(-0.5,  0.5)),
        "rate_noise_std":      float(meta_rng.uniform(0.4,   0.9)),
        # DTI path (identical ranges to all prior worlds)
        "dti_start":           float(meta_rng.uniform(75.0,  100.0)),
        "dti_end":             float(meta_rng.uniform(115.0, 155.0)),
        "dti_regime_boost":    float(meta_rng.uniform(3.0,   9.0)),
        "dti_noise_std":       float(meta_rng.uniform(1.5,   3.0)),
        # ---------------------------------------------------------------
        # Price process — elevated drift (KEY CHANGE)
        # growth_base: 0.012–0.022/q  ≈  4.8–8.8% annual real appreciation
        # vs adversarial: 0.004–0.013/q ≈ 1.6–5.2%
        # ---------------------------------------------------------------
        "growth_base":         float(meta_rng.uniform(0.012, 0.022)),
        "growth_regime_boost": float(meta_rng.uniform(0.004, 0.010)),
        "price_noise_std":     float(meta_rng.uniform(0.007, 0.016)),
        # ---------------------------------------------------------------
        # Crash design — structurally identical to adversarial world,
        # but lighter and less frequent so positive drift can dominate.
        # ---------------------------------------------------------------
        # Major idiosyncratic crash (market-specific calendar)
        "major_onset_frac":    float(meta_rng.uniform(_ONSET_LO, _ONSET_HI)),
        "major_depth":         float(meta_rng.uniform(0.025,  0.050)),  # lighter
        "major_dur":           int(meta_rng.integers(3, 7)),            # shorter: 3–6q
        # Regime-conditional probabilistic crashes
        # p_r0 / p_r1 ≈ 2.5× — informative but imperfect
        # Both are reduced vs adversarial so drift can dominate
        "p_crash_r0":          float(meta_rng.uniform(0.05,   0.12)),
        "p_crash_r1":          float(meta_rng.uniform(0.02,   0.05)),
        "crash_dur_min":       int(meta_rng.integers(2, 4)),
        "crash_dur_max":       int(meta_rng.integers(4, 6)),
        "crash_shock_lo":      float(meta_rng.uniform(0.025,  0.045)),
        "crash_shock_hi":      float(meta_rng.uniform(0.045,  0.070)),
        # Background noise (unchanged from adversarial)
        "bg_prob":             float(meta_rng.uniform(0.03,   0.07)),
        "bg_shock_lo":         float(meta_rng.uniform(0.010,  0.020)),
        "bg_shock_hi":         float(meta_rng.uniform(0.020,  0.035)),
    }


# ---------------------------------------------------------------------------
# Single-market builder (structure identical to adversarial)
# ---------------------------------------------------------------------------

def build_market_posdrift(
    market_id: int,
    params: dict,
    *,
    start: str = "1992-01-01",
    end:   str = "2024-12-31",
    seed:  int,
) -> tuple[pd.DataFrame, dict]:
    """Build one positive-drift adversarial market panel.

    Returns (df, diag) — identical interface to build_market_adversarial.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="QE")
    n     = len(dates)

    # Real rate and regime
    t_vec     = np.linspace(0, 8 * np.pi * params["rate_freq_mult"], n)
    cycle     = (params["rate_amplitude"] * np.sin(t_vec + params["rate_phase"])
                 + params["rate_level"])
    real_rate = cycle + rng.normal(0, params["rate_noise_std"], size=n)
    regime    = (real_rate < 0).astype(int)

    # DTI
    trend = np.linspace(params["dti_start"], params["dti_end"], n)
    dti   = (trend
             + regime * params["dti_regime_boost"]
             + rng.normal(0, params["dti_noise_std"], size=n))
    dti   = np.clip(dti, 60, 180)

    # Base returns — elevated drift is the only substantive difference
    base_growth = params["growth_base"] + regime * params["growth_regime_boost"]
    noise       = rng.normal(0, params["price_noise_std"], size=n)

    # Crash mechanism (identical structure to adversarial)
    crash        = np.zeros(n)
    crash_events: list[tuple[int, str]] = []

    major_onset  = int(np.clip(params["major_onset_frac"] * n,
                               0, n - params["major_dur"] - 1))
    major_dur    = params["major_dur"]
    for q in range(major_dur):
        if major_onset + q < n:
            crash[major_onset + q] -= params["major_depth"]
    major_regime = "r0" if regime[major_onset] == 0 else "r1"
    crash_events.append((major_onset, major_regime))

    major_suppress = set(range(max(0, major_onset - 2),
                               min(n, major_onset + major_dur + 3)))
    active_until   = major_onset + major_dur - 1

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
            crash_events.append((t_idx, "r0" if regime[t_idx] == 0 else "r1"))
            continue

        if rng.random() < params["bg_prob"]:
            dur   = int(rng.integers(_BG_DUR_MIN, _BG_DUR_MAX + 1))
            shock = float(rng.uniform(params["bg_shock_lo"], params["bg_shock_hi"]))
            for q in range(dur):
                if t_idx + q < n:
                    crash[t_idx + q] -= shock
            active_until = t_idx + dur - 1
            crash_events.append((t_idx, "bg_" + ("r0" if regime[t_idx] == 0 else "r1")))

    # Price index
    returns          = base_growth + noise + crash
    real_price_index = 100 * np.exp(np.cumsum(returns))
    real_price_index = real_price_index / real_price_index[0] * 100

    # Diagnostics
    crash_active        = crash < 0
    r0_crash_q          = int(((regime == 0) & crash_active).sum())
    r1_crash_q          = int(((regime == 1) & crash_active).sum())
    total_crash_q       = r0_crash_q + r1_crash_q
    n_r0                = int((regime == 0).sum())
    n_r1                = n - n_r0

    frac_crash_q_in_r0  = (r0_crash_q / total_crash_q) if total_crash_q > 0 else float("nan")
    frac_r0_in_crash    = (r0_crash_q / n_r0)           if n_r0 > 0          else float("nan")
    frac_r1_in_crash    = (r1_crash_q / n_r1)           if n_r1 > 0          else float("nan")

    diag = {
        "market_id":           market_id,
        "major_onset_date":    str(dates[major_onset].date()),
        "major_regime":        major_regime,
        "n_crash_events":      len(crash_events),
        "crash_q_in_r0":       r0_crash_q,
        "crash_q_in_r1":       r1_crash_q,
        "total_crash_q":       total_crash_q,
        "frac_crash_q_in_r0":  round(frac_crash_q_in_r0, 3),
        "frac_r0_in_crash":    round(frac_r0_in_crash,   3),
        "frac_r1_in_crash":    round(frac_r1_in_crash,   3),
        "regime_switches":     int((pd.Series(regime).diff().abs() > 0).sum()),
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

def build_all_markets_posdrift(
    n_markets:  int  = N_MARKETS_DEFAULT,
    *,
    start:      str  = "1992-01-01",
    end:        str  = "2024-12-31",
    base_seed:  int  = _BASE_SEED,
    meta_seed:  int  = _META_SEED,
    verbose:    bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """Build N positive-drift adversarial markets.

    Returns (df_combined, list_of_diagnostics).
    """
    meta_rng  = np.random.default_rng(meta_seed)
    frames    = []
    diag_list = []

    for m in range(n_markets):
        params   = _sample_posdrift_params(meta_rng)
        df, diag = build_market_posdrift(
            m, params, start=start, end=end, seed=base_seed + m
        )
        frames.append(df)
        diag_list.append(diag)

        if verbose:
            print(f"  market {m:2d}: "
                  f"growth_base={params['growth_base']:.4f}  "
                  f"switches={diag['regime_switches']:2d}  "
                  f"major={diag['major_onset_date']}[{diag['major_regime']}]  "
                  f"crash_q_r0={diag['crash_q_in_r0']:2d}  "
                  f"crash_q_r1={diag['crash_q_in_r1']:2d}  "
                  f"frac_in_r0={diag['frac_crash_q_in_r0']:.2f}")

    return pd.concat(frames, ignore_index=True), diag_list
