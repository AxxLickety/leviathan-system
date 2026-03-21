"""
OOS Synthetic Sensitivity Sweep — Leviathan Phase OOS
======================================================
Runs a 3×3×3 = 27-configuration sweep over synthetic crash design choices:

  crash_shock     : per-quarter shock magnitude  in {0.05, 0.075, 0.10}
  frag_prob       : fragile-row trigger probability in {0.10, 0.20, 0.30}
  dti_percentile  : DTI threshold for fragility eligibility (top X%) in {70, 75, 80}

For each configuration:
  - Build synthetic panel (same seed, same structural crash placement)
  - Compute train/test label distributions
  - Fit interaction logit on training window
  - Run walk-forward quantile search
  - Apply overlay to test period
  - Report key metrics

Does NOT touch production scripts (oos_train.py / oos_eval.py) or frozen outputs.

Run:
    python scripts/oos_sensitivity_sweep.py

Outputs:
    outputs/oos/robustness/sweep_results.csv
"""
from __future__ import annotations

import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.research.path_a.label_correction import add_correction_label
from src.evaluation.backtest import compute_forward_return
from src.evaluation.regime import assign_fragility_regime
from src.backtests.evaluation import summarize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_END  = "2007-12-31"
TEST_START = "2008-03-31"
OUT_DIR    = Path("outputs/oos/robustness")

SEED = 42

# Sweep grid
_CRASH_SHOCKS     = [0.050, 0.075, 0.100]
_FRAG_PROBS       = [0.10,  0.20,  0.30]
_DTI_PERCENTILES  = [70.0,  75.0,  80.0]    # fragility threshold = top (100-p)%

# Walk-forward parameters (match oos_train.py exactly)
_WF_WINDOW_Q    = 20
_WF_QGRID       = np.arange(0.60, 0.96, 0.05)
_WF_FALLBACK_Q  = 0.80
_WF_MIN_REGIME0 = 10

# Logit threshold probability level
_LOGIT_PROB = 0.10


# ---------------------------------------------------------------------------
# Parameterized panel builder
# (self-contained; does not call build_master_df to avoid print noise)
# ---------------------------------------------------------------------------

def _build_sweep_df(
    crash_shock: float,
    frag_prob: float,
    dti_percentile: float,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Build the synthetic panel with configurable crash parameters.

    Replicates build_master_df logic exactly, varying three knobs:
      crash_shock    : per-quarter magnitude for all deterministic crashes
      frag_prob      : trigger probability for fragility-eligible rows (Layer 4)
      dti_percentile : DTI percentile above which a row is "high DTI" for Layer 4

    Structural crash placement (Layers 1–3) is unchanged so that the
    training crash at row 24 is always present; only its depth varies.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(start="1999-01-01", end="2024-12-31", freq="QE")
    n     = len(dates)

    # Real rate and regime (unchanged)
    cycle     = 1.5 * np.sin(np.linspace(0, 8 * np.pi, n))
    shocks    = rng.normal(0, 0.6, size=n)
    real_rate = cycle + shocks
    regime    = (real_rate < 0).astype(int)

    # DTI (unchanged)
    trend = np.linspace(85, 135, n)
    dti   = trend + regime * 6 + rng.normal(0, 2.0, size=n)
    dti   = np.clip(dti, 60, 180)

    # Base returns (unchanged)
    base_growth = 0.008 + regime * 0.004
    noise       = rng.normal(0, 0.01, size=n)

    # --- Training boundary ---
    _LABEL_HORIZON = 20
    train_end_idx  = int(
        np.searchsorted(dates, pd.Timestamp(TRAIN_END), side="right") - 1
    )
    _last_labeled = train_end_idx - _LABEL_HORIZON   # = 15

    crash = np.zeros(n)

    # Layer 1: Deterministic training crash (depth = crash_shock)
    _TC_START = _last_labeled + 9    # = 24
    _TC_DUR   = 4
    for q in range(_TC_DUR):
        crash[_TC_START + q] -= crash_shock

    # Layer 2: GFC-proxy crash (depth = crash_shock, same placement)
    _GFC_START = int(n * 0.30)
    _GFC_END   = int(n * 0.38)
    crash[_GFC_START:_GFC_END] = -crash_shock

    # Layer 3: Post-GFC test crash (depth = crash_shock)
    _PCT_START = 68
    _PCT_DUR   = 4
    for q in range(_PCT_DUR):
        if _PCT_START + q < n:
            crash[_PCT_START + q] -= crash_shock

    # Layer 4: Probabilistic fragility-driven crashes
    dti_thr = float(np.percentile(dti, dti_percentile))
    fragile  = (regime == 0) & (dti > dti_thr)

    _det_suppress: set[int] = set()
    for start_row, dur in [(_TC_START, _TC_DUR), (_GFC_START, _GFC_END - _GFC_START), (_PCT_START, _PCT_DUR)]:
        for q in range(dur + 2):
            _det_suppress.add(start_row - 1 + q)
            _det_suppress.add(start_row + q)

    active_until = -1
    for t in range(n):
        if t <= active_until or t in _det_suppress:
            continue
        if fragile[t] and rng.random() < frag_prob:
            dur_p   = int(rng.integers(3, 5))
            shock_p = float(rng.uniform(0.065, 0.085))
            for q in range(dur_p):
                if t + q < n:
                    crash[t + q] -= shock_p
            active_until = t + dur_p - 1

    returns         = base_growth + noise + crash
    real_price_index = 100 * np.exp(np.cumsum(returns))
    real_price_index /= real_price_index[0] / 100.0

    return pd.DataFrame({
        "date":             dates,
        "dti":              dti,
        "real_rate":        real_rate,
        "regime":           regime,
        "real_price_index": real_price_index,
    })


# ---------------------------------------------------------------------------
# Single-config evaluation
# ---------------------------------------------------------------------------

def _eval_config(crash_shock: float, frag_prob: float, dti_pct: float) -> dict:
    """Build, train, evaluate one sweep configuration. Returns metrics dict."""

    # Build panel
    df = _build_sweep_df(crash_shock, frag_prob, dti_pct)
    df = compute_forward_return(df)
    df = assign_fragility_regime(df)

    # Train/test split and labeling (suppress all warnings)
    df_train_full = df[df["date"] <= pd.Timestamp(TRAIN_END)].copy()
    df_test_full  = df[df["date"] >= pd.Timestamp(TEST_START)].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_train = add_correction_label(df_train_full)
        df_test  = add_correction_label(df_test_full)

    n_train    = len(df_train)
    train_y1   = int(df_train["y"].sum())
    train_y0   = n_train - train_y1
    train_prev = train_y1 / n_train if n_train > 0 else 0.0

    n_test    = len(df_test)
    test_y1   = int(df_test["y"].sum())
    test_prev = test_y1 / n_test if n_test > 0 else 0.0

    # always_in baseline
    ai = summarize(df_test["fwd_return"], name="always_in")

    # Logit (interaction, matching oos_train.py)
    converged   = False
    pseudo_r2   = np.nan
    logit_cutoff = np.nan

    if train_y1 > 0 and train_y0 > 0:   # skip if perfectly separated or all-one-class
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_tr2 = df_train.copy()
                df_tr2["dti_x_regime"] = df_tr2["dti"] * df_tr2["regime"]
                X = sm.add_constant(df_tr2[["dti", "regime", "dti_x_regime"]])
                y = df_tr2["y"]
                model   = sm.Logit(y, X)
                res     = model.fit(disp=False)
                converged = bool(res.mle_retvals.get("converged", False))
                try:
                    pseudo_r2 = float(res.prsquared)
                    if np.isinf(pseudo_r2):
                        pseudo_r2 = np.nan
                except Exception:
                    pseudo_r2 = np.nan
                # Analytical threshold at regime=0, p=0.10
                b0 = float(res.params["const"])
                b1 = float(res.params["dti"])
                b2 = float(res.params["regime"])
                b3 = float(res.params["dti_x_regime"])
                logit_p = np.log(_LOGIT_PROB / (1 - _LOGIT_PROB))
                denom = b1 + b3 * 0
                if abs(denom) > 1e-12:
                    logit_cutoff = (logit_p - b0 - b2 * 0) / denom
        except Exception:
            pass   # degenerate fit — leave defaults

    # Walk-forward cutoff
    df_wf = (
        df_train_full
        .dropna(subset=["fwd_return", "dti", "regime"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    wf_cutoff: float | None = None
    for t in range(_WF_WINDOW_Q, len(df_wf)):
        train_wf = df_wf.iloc[:t]
        r0 = train_wf.loc[train_wf["regime"] == 0, "dti"].dropna()
        if len(r0) < _WF_MIN_REGIME0:
            thr = float(train_wf["dti"].quantile(_WF_FALLBACK_Q))
        else:
            best_thr = float(r0.quantile(_WF_FALLBACK_Q))
            best_val = -np.inf
            for q in _WF_QGRID:
                cand = float(r0.quantile(q))
                hold = ~((train_wf["regime"] == 0) & (train_wf["dti"] > cand))
                val  = float((train_wf["fwd_return"] * hold.astype(float)).quantile(0.05))
                if val > best_val:
                    best_val, best_thr = val, cand
            thr = best_thr
        wf_cutoff = thr

    # OOS overlay with walk-forward cutoff
    if wf_cutoff is not None:
        hold = ~((df_test["regime"] == 0) & (df_test["dti"] > wf_cutoff))
        ov_ret = df_test["fwd_return"] * hold.astype(float)
        ov = summarize(ov_ret, name="overlay")
    else:
        ov = {k: np.nan for k in ["p05", "sharpe", "maxdd", "mean"]}

    return {
        "crash_shock":    crash_shock,
        "frag_prob":      frag_prob,
        "dti_pct":        dti_pct,
        "train_y1":       train_y1,
        "train_y0":       train_y0,
        "train_prev":     round(train_prev, 3),
        "test_y1":        test_y1,
        "test_prev":      round(test_prev, 3),
        "logit_converged": converged,
        "pseudo_r2":      round(pseudo_r2, 4) if not np.isnan(pseudo_r2) else np.nan,
        "logit_cutoff":   round(logit_cutoff, 4) if not np.isnan(logit_cutoff) else np.nan,
        "wf_cutoff":      round(wf_cutoff, 4) if wf_cutoff is not None else np.nan,
        "ov_p05":         round(float(ov["p05"]),    6),
        "ov_sharpe":      round(float(ov["sharpe"]), 4) if not np.isnan(ov["sharpe"]) else np.nan,
        "ov_maxdd":       round(float(ov["maxdd"]),  4),
        "ov_mean":        round(float(ov["mean"]),   6),
        "ai_p05":         round(float(ai["p05"]),    6),
        "ai_sharpe":      round(float(ai["sharpe"]), 4),
        "ai_maxdd":       round(float(ai["maxdd"]),  4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = list(product(_CRASH_SHOCKS, _FRAG_PROBS, _DTI_PERCENTILES))
    n_total = len(configs)
    print(f"[SWEEP] Running {n_total} configurations ...")

    results = []
    for i, (cs, fp, dp) in enumerate(configs, 1):
        print(f"  [{i:2d}/{n_total}]  crash_shock={cs:.3f}  frag_prob={fp:.2f}  dti_pct={dp:.0f}", end="  ")
        row = _eval_config(cs, fp, dp)
        print(f"train_y1={row['train_y1']}  test_y1={row['test_y1']}  "
              f"wf_cut={row['wf_cutoff']}  conv={row['logit_converged']}  "
              f"pR2={row['pseudo_r2']}  ov_p05={row['ov_p05']:.4f}")
        results.append(row)

    df_out = pd.DataFrame(results)
    out_path = OUT_DIR / "sweep_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n[SWEEP] Wrote {out_path}")

    # Summary of extremes
    print("\n[SWEEP] Configurations with zero training y=1:")
    zero = df_out[df_out["train_y1"] == 0]
    if len(zero):
        print(zero[["crash_shock","frag_prob","dti_pct","train_y1","test_y1"]].to_string(index=False))
    else:
        print("  None")

    print("\n[SWEEP] Configurations with all training y=1 (n_y1=16):")
    all_pos = df_out[df_out["train_y1"] == 16]
    if len(all_pos):
        print(all_pos[["crash_shock","frag_prob","dti_pct","train_y1","logit_converged"]].to_string(index=False))
    else:
        print("  None")

    print("\n[SWEEP] Top 5 by overlay Sharpe:")
    print(df_out.nlargest(5, "ov_sharpe")[
        ["crash_shock","frag_prob","dti_pct","train_y1","pseudo_r2","wf_cutoff","ov_sharpe","ov_p05","ov_maxdd"]
    ].to_string(index=False))

    print("\n[SWEEP] Bottom 5 by overlay Sharpe (excluding NaN):")
    valid = df_out.dropna(subset=["ov_sharpe"])
    print(valid.nsmallest(5, "ov_sharpe")[
        ["crash_shock","frag_prob","dti_pct","train_y1","pseudo_r2","wf_cutoff","ov_sharpe","ov_p05","ov_maxdd"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
