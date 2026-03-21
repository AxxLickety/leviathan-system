"""
OOS Training Script — Leviathan Phase OOS
==========================================
Loads the full panel, fits all models on the training window only
(date <= TRAIN_END), and writes frozen parameters to outputs/oos/frozen/.

Run:
    python scripts/oos_train.py

Outputs (all in outputs/oos/frozen/):
    coef.csv            — logit coefficients and p-values (index = param name)
    thresholds.csv      — regime-conditional DTI thresholds at p=0.10 and p=0.20
    dti_cutoff.json     — scalar DTI cutoff from walk-forward quantile search
    train_metadata.json — training window bounds and diagnostics
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.path_a.build_dataset import build_master_df
from src.research.path_a.label_correction import add_correction_label
from src.research.path_a.fit_logit import fit_interaction_logit
from src.research.path_a.thresholds import compute_dti_thresholds
from src.evaluation.backtest import compute_forward_return
from src.evaluation.regime import assign_fragility_regime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_END = "2007-12-31"

FROZEN_DIR = Path("outputs/oos/frozen")

# Walk-forward quantile search parameters — kept inline per design
_WF_WINDOW_Q   = 20                         # minimum quarters before walk-forward starts
_WF_QGRID      = np.arange(0.60, 0.96, 0.05)  # quantile grid for DTI threshold candidates
_WF_FALLBACK_Q = 0.80                        # fallback quantile when regime-0 rows < threshold
_WF_MIN_REGIME0 = 10                         # minimum regime-0 rows to use grid search


# ---------------------------------------------------------------------------
# Helpers (inline — not shared with eval)
# ---------------------------------------------------------------------------

def _filter_returns(sub: pd.DataFrame, thr: float) -> pd.Series:
    """Apply DTI/regime overlay to sub; return filtered return series."""
    hold = ~((sub["regime"] == 0) & (sub["dti"] > thr))
    return sub["fwd_return"] * hold.astype(float)


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
    df = compute_forward_return(df)                      # adds "fwd_return"
    assert "fwd_ret_4q" not in df.columns, (
        "Unexpected fwd_ret_4q column present after compute_forward_return()"
    )

    # ------------------------------------------------------------------
    # Step 3 — assign fragility regime (overwrites inline regime from build_master_df)
    # ------------------------------------------------------------------
    df = assign_fragility_regime(df)
    assert pd.api.types.is_integer_dtype(df["regime"]), (
        f"regime must be integer dtype, got {df['regime'].dtype}"
    )
    assert set(df["regime"].unique()).issubset({0, 1}), (
        f"regime must be 0/1 only, got values: {df['regime'].unique()}"
    )

    # ------------------------------------------------------------------
    # Step 4 — filter to training window
    # ------------------------------------------------------------------
    df_train = df[df["date"] <= pd.Timestamp(TRAIN_END)].copy()
    assert len(df_train) > 0, "Training window is empty"
    assert df_train["date"].max() <= pd.Timestamp(TRAIN_END), (
        "Post-training rows leaked into df_train"
    )
    print(f"[TRAIN] rows before labeling: {len(df_train)} "
          f"({df_train['date'].min().date()} – {df_train['date'].max().date()})")

    # ------------------------------------------------------------------
    # Step 5 — add correction label (drops last horizon_max_q rows)
    # ------------------------------------------------------------------
    df_labeled = add_correction_label(df_train)
    assert "y" in df_labeled.columns, "add_correction_label() did not produce 'y'"
    assert df_labeled["y"].dtype == int or pd.api.types.is_integer_dtype(df_labeled["y"]), (
        f"y must be integer, got {df_labeled['y'].dtype}"
    )
    print(f"[TRAIN] rows after labeling : {len(df_labeled)}  "
          f"(y=1: {df_labeled['y'].sum()}, y=0: {(df_labeled['y']==0).sum()})")

    # ------------------------------------------------------------------
    # Step 6 — fit interaction logit
    # ------------------------------------------------------------------
    res, df_pred = fit_interaction_logit(df_labeled)
    print(f"[LOGIT] converged={res.mle_retvals['converged']}  "
          f"pseudo_R2={res.prsquared:.4f}")

    # ------------------------------------------------------------------
    # Step 7 — compute regime-conditional DTI thresholds
    # ------------------------------------------------------------------
    thresholds = compute_dti_thresholds(res.params)

    # ------------------------------------------------------------------
    # Step 8 — walk-forward quantile search (regime==0 DTI, training rows only)
    #
    # Objective: maximize p05 of filtered returns on expanding training window.
    # The final step's threshold is frozen as dti_cutoff.
    # ------------------------------------------------------------------
    df_wf = (
        df_train
        .dropna(subset=["fwd_return", "dti", "regime"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    assert len(df_wf) > _WF_WINDOW_Q, (
        f"Not enough training rows for walk-forward: {len(df_wf)} <= {_WF_WINDOW_Q}"
    )

    dti_cutoff: float | None = None
    wf_steps = 0

    for t in range(_WF_WINDOW_Q, len(df_wf)):
        train_wf = df_wf.iloc[:t]
        r0_dti = train_wf.loc[train_wf["regime"] == 0, "dti"].dropna()

        if len(r0_dti) < _WF_MIN_REGIME0:
            thr = float(train_wf["dti"].quantile(_WF_FALLBACK_Q))
        else:
            best_thr: float = float(r0_dti.quantile(_WF_FALLBACK_Q))
            best_val: float = -np.inf
            for q in _WF_QGRID:
                candidate = float(r0_dti.quantile(q))
                val = float(_filter_returns(train_wf, candidate).quantile(0.05))
                if val > best_val:
                    best_val, best_thr = val, candidate
            thr = best_thr

        dti_cutoff = thr   # overwrite each step; final iteration is frozen
        wf_steps += 1

    assert dti_cutoff is not None, (
        "Walk-forward produced no cutoff — insufficient training rows"
    )
    print(f"[WF]    steps={wf_steps}  frozen dti_cutoff={dti_cutoff:.4f}")

    # ------------------------------------------------------------------
    # Step 9 — training diagnostics
    # ------------------------------------------------------------------
    n_labeled  = int(len(df_labeled))
    n_regime0  = int((df_labeled["regime"] == 0).sum())
    n_positive = int(df_labeled["y"].sum())
    prevalence = round(n_positive / n_labeled, 6)
    train_start_str = str(df_train["date"].min().date())

    # ------------------------------------------------------------------
    # Step 10 — write frozen files
    # ------------------------------------------------------------------
    # coef.csv — index preserved so parameter names survive round-trip
    coef_df = pd.DataFrame({"coef": res.params, "p_value": res.pvalues})
    coef_df.to_csv(FROZEN_DIR / "coef.csv")   # index=True (default)

    # thresholds.csv
    thresholds.to_csv(FROZEN_DIR / "thresholds.csv", index=False)

    # dti_cutoff.json
    with open(FROZEN_DIR / "dti_cutoff.json", "w") as f:
        json.dump({"dti_cutoff": float(dti_cutoff)}, f, indent=2)

    # train_metadata.json
    train_metadata = {
        "train_start":        train_start_str,
        "train_end":          TRAIN_END,
        "test_start":         "2008-03-31",
        "n_train_labeled":    n_labeled,
        "n_regime0":          n_regime0,
        "n_positive_labels":  n_positive,
        "prevalence":         prevalence,
        "wf_steps":           wf_steps,
        "dti_cutoff":         float(dti_cutoff),
    }
    with open(FROZEN_DIR / "train_metadata.json", "w") as f:
        json.dump(train_metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Step 11 — print frozen-parameter summary
    # ------------------------------------------------------------------
    print()
    print("=" * 62)
    print("FROZEN PARAMETER SUMMARY")
    print("=" * 62)
    print(f"  Training window   : {train_start_str}  →  {TRAIN_END}")
    print(f"  Labeled rows      : {n_labeled}")
    print(f"  Regime-0 rows     : {n_regime0}")
    print(f"  Positive labels   : {n_positive}  (prevalence {prevalence:.2%})")
    print()
    print("  Logit coefficients:")
    print(coef_df.to_string(float_format="{:.6f}".format))
    print()
    print("  DTI thresholds (regime-conditional):")
    print(thresholds.to_string(index=False, float_format="{:.4f}".format))
    print()
    print(f"  Walk-forward DTI cutoff : {dti_cutoff:.4f}")
    print(f"  Walk-forward steps      : {wf_steps}")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Step 12 — assert all four frozen files exist
    # ------------------------------------------------------------------
    frozen_files = ["coef.csv", "thresholds.csv", "dti_cutoff.json", "train_metadata.json"]
    for fname in frozen_files:
        p = FROZEN_DIR / fname
        assert p.exists(), f"Frozen file missing after write: {p}"
        print(f"  [OK] {p}")
    print()
    print("[OOS_TRAIN] complete.")


if __name__ == "__main__":
    main()
