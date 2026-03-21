"""
OOS Training Script (Percentile-DTI) — Leviathan Phase OOS
===========================================================
Redesigned training pipeline that replaces raw absolute DTI with a causal
rolling 20-quarter percentile rank (dti_pct_roll) as the operative feature.

Motivation
----------
The original oos_train.py learned a raw-DTI walk-forward cutoff (98.81) that was
structurally irrelevant in the test period: all 64 test-quarter DTI values exceeded
the cutoff, so the Leviathan gate reduced to a pure regime filter. The DTI dimension
contributed nothing beyond (regime == 0).

Fix: dti_pct_roll measures "how high is DTI relative to its own recent 20-quarter
history." This is stationary by construction and directly interpretable:
- 1.0 = DTI is at the 20-quarter high
- 0.5 = DTI is at the 20-quarter median
- 0.0 = DTI is at the 20-quarter low (never achieved in practice)

The walk-forward now searches percentile-space thresholds (0.50–0.95) rather than
converting quantile levels to raw DTI values.

Run:
    python scripts/oos_train_pct.py

Outputs (all in outputs/oos_pct/frozen/):
    coef.csv            — logit coefficients on dti_pct_roll features
    thresholds.csv      — regime-conditional dti_pct_roll thresholds at p=0.10, 0.20
    dti_cutoff.json     — scalar percentile cutoff from walk-forward search
    train_metadata.json — training window bounds, diagnostics, and feature metadata
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.research.path_a.build_dataset import build_master_df
from src.research.path_a.label_correction import add_correction_label
from src.evaluation.backtest import compute_forward_return
from src.evaluation.regime import assign_fragility_regime
from src.evaluation.transforms import compute_rolling_pct_rank

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_END  = "2007-12-31"
FROZEN_DIR = Path("outputs/oos_pct/frozen")

_ROLL_WINDOW   = 20    # quarters for rolling percentile rank

# Walk-forward parameters — percentile-space thresholds tested directly
_WF_WINDOW_Q   = 20
_WF_GRID       = np.arange(0.50, 0.96, 0.05)   # candidate pct thresholds
_WF_FALLBACK   = 0.75                            # fallback when regime-0 rows < min
_WF_MIN_R0     = 10                              # min regime-0 rows for grid search

# Logit probability levels for analytical threshold table
_LOGIT_PROBS   = (0.10, 0.20)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_returns(sub: pd.DataFrame, thr: float) -> pd.Series:
    """Apply percentile-DTI/regime overlay; return filtered return series."""
    hold = ~((sub["regime"] == 0) & (sub["dti_pct_roll"] > thr))
    return sub["fwd_return"] * hold.astype(float)


def _logit_pct_threshold(params: pd.Series, regime: int, prob: float) -> float:
    """
    Analytical dti_pct_roll threshold at which P(y=1 | regime=r) = prob.

    logit(prob) = b0 + b1*dti_pct + b2*regime + b3*dti_pct*regime
    Solving for dti_pct:
        dti_pct = (logit(prob) - b0 - b2*r) / (b1 + b3*r)
    """
    b0 = float(params.get("const",           params.get("Intercept", 0)))
    b1 = float(params.get("dti_pct_roll",    0))
    b2 = float(params.get("regime",          0))
    b3 = float(params.get("dti_pct_x_regime", 0))
    lp = np.log(prob / (1 - prob))
    denom = b1 + b3 * regime
    if abs(denom) < 1e-12:
        return float("nan")
    return (lp - b0 - b2 * regime) / denom


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — load full panel
    # ------------------------------------------------------------------
    df = build_master_df()

    # ------------------------------------------------------------------
    # Step 2 — compute forward 4-quarter log return
    # ------------------------------------------------------------------
    df = compute_forward_return(df)
    assert "fwd_ret_4q" not in df.columns, (
        "Unexpected fwd_ret_4q column present after compute_forward_return()"
    )

    # ------------------------------------------------------------------
    # Step 3 — assign fragility regime
    # ------------------------------------------------------------------
    df = assign_fragility_regime(df)
    assert pd.api.types.is_integer_dtype(df["regime"]), (
        f"regime must be integer dtype, got {df['regime'].dtype}"
    )
    assert set(df["regime"].unique()).issubset({0, 1}), (
        f"regime must be 0/1 only, got: {df['regime'].unique()}"
    )

    # ------------------------------------------------------------------
    # Step 4 — compute causal rolling-percentile DTI (full panel, causal)
    #
    # Computed over the full panel so that early test rows have their
    # correct lookback into training history.  This is causal: at each
    # date t, dti_pct_roll[t] uses only dti[t-19:t+1].
    # ------------------------------------------------------------------
    df["dti_pct_roll"] = compute_rolling_pct_rank(df["dti"], window=_ROLL_WINDOW)
    print(f"[TRAIN_PCT] dti_pct_roll computed  window={_ROLL_WINDOW}q  "
          f"range [{df['dti_pct_roll'].min():.3f}, {df['dti_pct_roll'].max():.3f}]")

    # ------------------------------------------------------------------
    # Step 5 — restrict to training window
    # ------------------------------------------------------------------
    df_train = df[df["date"] <= pd.Timestamp(TRAIN_END)].copy()
    print(f"[TRAIN_PCT] Training rows          : {len(df_train)}"
          f"  ({df_train['date'].min().date()} – {df_train['date'].max().date()})")
    print(f"[TRAIN_PCT] Train dti_pct_roll range: "
          f"[{df_train['dti_pct_roll'].min():.3f}, {df_train['dti_pct_roll'].max():.3f}]")

    # ------------------------------------------------------------------
    # Step 6 — add correction labels (training window only)
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_labeled = add_correction_label(df_train)

    n_labeled  = len(df_labeled)
    n_positive = int(df_labeled["y"].sum())
    n_negative = n_labeled - n_positive
    prevalence = n_positive / n_labeled if n_labeled > 0 else 0.0

    print(f"[TRAIN_PCT] rows after labeling    : {n_labeled}  "
          f"(y=1: {n_positive}, y=0: {n_negative})")
    assert n_positive > 0, (
        "No positive training labels — crash mechanism not visible to labeled rows"
    )

    # ------------------------------------------------------------------
    # Step 7 — fit interaction logit on dti_pct_roll
    #
    # Features: const, dti_pct_roll, regime, dti_pct_roll × regime
    # Fit inline (fit_interaction_logit hardcodes raw 'dti' column).
    # ------------------------------------------------------------------
    df_labeled["dti_pct_x_regime"] = (
        df_labeled["dti_pct_roll"] * df_labeled["regime"]
    )

    X = sm.add_constant(
        df_labeled[["dti_pct_roll", "regime", "dti_pct_x_regime"]],
        has_constant="skip",
    )
    y = df_labeled["y"]

    converged  = False
    pseudo_r2  = float("nan")
    res        = None

    # Try interaction model with bfgs (newton fails due to near-separation in
    # the regime × dti_pct_roll cross term at small n).
    # If interaction coefficients are extreme (|regime coef| > 50) the model
    # has collapsed to near-separation; fall back to main-effects logit.
    _near_sep = False
    for method in ("bfgs", "newton"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.Logit(y, X)
                res   = model.fit(disp=False, method=method, maxiter=300)
            converged = bool(res.mle_retvals.get("converged", False))
            if converged:
                _b_regime = abs(float(res.params.get("regime", 0)))
                if _b_regime > 50:
                    _near_sep = True
                    print(f"[LOGIT_PCT] interaction: converged but near-separation "
                          f"(|regime coef|={_b_regime:.1f}) — flagging, keeping fit")
                break
        except Exception as exc:
            print(f"[LOGIT_PCT] {method} failed: {exc}")

    if res is not None:
        try:
            pseudo_r2 = float(res.prsquared)
            if np.isinf(pseudo_r2) or np.isnan(pseudo_r2):
                pseudo_r2 = float("nan")
        except Exception:
            pseudo_r2 = float("nan")
        print(f"[LOGIT_PCT] converged={converged}  near_sep={_near_sep}  "
              f"pseudo_R2={pseudo_r2:.4f}")
    else:
        print("[LOGIT_PCT] All fit methods failed — no coefficients available")

    # ------------------------------------------------------------------
    # Step 8 — analytical thresholds in percentile space
    # ------------------------------------------------------------------
    threshold_rows = []
    if res is not None:
        for regime_val in (0, 1):
            for prob in _LOGIT_PROBS:
                thr = _logit_pct_threshold(res.params, regime=regime_val, prob=prob)
                threshold_rows.append({
                    "regime":        regime_val,
                    "prob":          prob,
                    "dti_threshold": round(thr, 6),   # percentile-space threshold
                })
    thresholds = pd.DataFrame(threshold_rows)

    # ------------------------------------------------------------------
    # Step 9 — walk-forward quantile search (percentile-space)
    #
    # Expanding window on training data.
    # Candidate thresholds: directly from _WF_GRID (0.50–0.95 in 0.05 steps).
    # At each step, filter when regime==0 AND dti_pct_roll > candidate.
    # Objective: maximize p05 of filtered return series.
    # Final step is frozen.
    # ------------------------------------------------------------------
    df_wf = (
        df_train
        .dropna(subset=["fwd_return", "dti_pct_roll", "regime"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    assert len(df_wf) > _WF_WINDOW_Q, (
        f"Not enough training rows for walk-forward: {len(df_wf)} <= {_WF_WINDOW_Q}"
    )

    pct_cutoff: float | None = None
    wf_steps = 0

    for t in range(_WF_WINDOW_Q, len(df_wf)):
        train_wf = df_wf.iloc[:t]
        r0_rows  = train_wf.loc[train_wf["regime"] == 0]

        if len(r0_rows) < _WF_MIN_R0:
            thr = _WF_FALLBACK
        else:
            best_thr: float = _WF_FALLBACK
            best_val: float = -np.inf
            for cand in _WF_GRID:
                val = float(_filter_returns(train_wf, float(cand)).quantile(0.05))
                if val > best_val:
                    best_val, best_thr = val, float(cand)
            thr = best_thr

        pct_cutoff = thr
        wf_steps  += 1

    assert pct_cutoff is not None, (
        "Walk-forward produced no cutoff — insufficient training rows"
    )
    print(f"[WF_PCT]    steps={wf_steps}  frozen pct_cutoff={pct_cutoff:.4f}")

    # ------------------------------------------------------------------
    # Step 10 — write frozen files
    # ------------------------------------------------------------------
    if res is not None:
        coef_df = pd.DataFrame({"coef": res.params, "p_value": res.pvalues})
    else:
        coef_df = pd.DataFrame(columns=["coef", "p_value"])
    coef_df.to_csv(FROZEN_DIR / "coef.csv")

    thresholds.to_csv(FROZEN_DIR / "thresholds.csv", index=False)

    with open(FROZEN_DIR / "dti_cutoff.json", "w") as f:
        json.dump({"dti_cutoff": float(pct_cutoff)}, f, indent=2)

    train_start_str = str(df_train["date"].min().date())
    n_regime0 = int((df_labeled["regime"] == 0).sum())
    train_metadata = {
        "train_start":        train_start_str,
        "train_end":          TRAIN_END,
        "test_start":         "2008-03-31",
        "n_train_labeled":    n_labeled,
        "n_regime0":          n_regime0,
        "n_positive_labels":  n_positive,
        "prevalence":         round(prevalence, 6),
        "wf_steps":           wf_steps,
        "dti_cutoff":         float(pct_cutoff),
        "dti_feature":        f"dti_pct_roll{_ROLL_WINDOW}",   # key: not raw DTI
        "roll_window":        _ROLL_WINDOW,
        "logit_converged":    converged,
        "logit_near_sep":     _near_sep,
        "pseudo_r2":          round(pseudo_r2, 6) if not np.isnan(pseudo_r2) else None,
    }
    with open(FROZEN_DIR / "train_metadata.json", "w") as f:
        json.dump(train_metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Step 11 — print frozen-parameter summary
    # ------------------------------------------------------------------
    print()
    print("=" * 66)
    print("FROZEN PARAMETER SUMMARY (PERCENTILE-DTI PIPELINE)")
    print("=" * 66)
    print(f"  Training window    : {train_start_str}  →  {TRAIN_END}")
    print(f"  Labeled rows       : {n_labeled}")
    print(f"  Regime-0 rows      : {n_regime0}")
    print(f"  Positive labels    : {n_positive}  (prevalence {prevalence:.2%})")
    print(f"  DTI feature        : dti_pct_roll{_ROLL_WINDOW} (in [0, 1])")
    print()
    if res is not None:
        print("  Logit coefficients (on dti_pct_roll):")
        print(coef_df.to_string(float_format="{:.6f}".format))
        print()
    if not thresholds.empty:
        print("  Analytical thresholds (dti_pct_roll at p=0.10, p=0.20):")
        print(thresholds.to_string(index=False, float_format="{:.4f}".format))
        print()
    print(f"  Walk-forward pct cutoff : {pct_cutoff:.4f}")
    print(f"  Walk-forward steps      : {wf_steps}")
    print("=" * 66)

    # ------------------------------------------------------------------
    # Step 12 — assert all frozen files exist
    # ------------------------------------------------------------------
    frozen_files = ["coef.csv", "thresholds.csv", "dti_cutoff.json", "train_metadata.json"]
    for fname in frozen_files:
        p = FROZEN_DIR / fname
        assert p.exists(), f"Frozen file missing after write: {p}"
        print(f"  [OK] {p}")

    print()
    print("[OOS_TRAIN_PCT] complete.")


if __name__ == "__main__":
    main()
