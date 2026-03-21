"""
OOS Evaluation Script — Leviathan Phase OOS
============================================
Loads frozen parameters from outputs/oos/frozen/, applies them to the
test period (date >= TEST_START), and writes all evaluation artifacts.

FIREWALL: This script must not recompute any training-derived parameter.
All coefficients, thresholds, and the DTI cutoff are read from disk only.

Run:
    python scripts/oos_eval.py

Outputs:
    outputs/oos/tables/regime_supply_counts.csv
    outputs/oos/tables/crash_frequency_ci.csv
    outputs/oos/tables/overlay_performance.csv
    outputs/oos/tables/sensitivity_analysis.csv
    outputs/oos/plots/equity_curve.png
    outputs/oos/verdict.txt
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.path_a.build_dataset import build_master_df
from src.research.path_a.label_correction import add_correction_label
from src.evaluation.backtest import compute_forward_return
from src.evaluation.regime import assign_fragility_regime
from src.backtests.evaluation import summarize
from src.evaluation.oos_helpers import (
    load_frozen_params,
    regime_supply_count_table,
    crash_frequency_table,
    apply_overlay_filter,
    plot_equity_curves,
    write_verdict,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_START  = "2008-03-31"
_TRAIN_END  = "2007-12-31"
FROZEN_DIR  = "outputs/oos/frozen/"
OUT_DIR     = Path("outputs/oos")

_TABLE_DIR  = OUT_DIR / "tables"
_PLOT_DIR   = OUT_DIR / "plots"

_SENSITIVITY_MULTIPLIERS = [0.80, 0.90, 1.10, 1.20]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _TABLE_DIR.mkdir(parents=True, exist_ok=True)
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — load frozen params (firewall: first substantive operation)
    # ------------------------------------------------------------------
    frozen = load_frozen_params(FROZEN_DIR)
    dti_cutoff = frozen["dti_cutoff"]
    meta = frozen["train_metadata"]

    print(f"[EVAL] Frozen params loaded from {FROZEN_DIR}")
    print(f"[EVAL] Training window : {meta['train_start']}  →  {meta['train_end']}")
    print(f"[EVAL] dti_cutoff      : {dti_cutoff:.4f}")

    # Detect degenerate training — drives caveat labeling throughout
    is_smoke_test = (meta.get("n_positive_labels", -1) == 0)
    if is_smoke_test:
        print()
        print("  *** SMOKE TEST RUN — training window had zero positive labels ***")
        print("  *** Pipeline mechanics are being validated, not research findings. ***")
        print()

    # ------------------------------------------------------------------
    # Step 2 — load full panel (same source as oos_train.py)
    # ------------------------------------------------------------------
    df = build_master_df()

    # ------------------------------------------------------------------
    # Step 3 — compute fwd_return and regime (mirrors oos_train.py exactly)
    # ------------------------------------------------------------------
    df = compute_forward_return(df)
    assert "fwd_ret_4q" not in df.columns, (
        "Unexpected fwd_ret_4q present after compute_forward_return()"
    )
    df = assign_fragility_regime(df)
    assert pd.api.types.is_integer_dtype(df["regime"]), (
        f"regime must be integer dtype, got {df['regime'].dtype}"
    )
    assert set(df["regime"].unique()).issubset({0, 1}), (
        f"regime must be 0/1 only, got: {df['regime'].unique()}"
    )

    # ------------------------------------------------------------------
    # Step 4 — firewall assertions before any test-row evaluation
    # ------------------------------------------------------------------
    assert meta["train_end"]   == _TRAIN_END, (
        f"Frozen train_end '{meta['train_end']}' != expected '{_TRAIN_END}'"
    )
    assert meta["test_start"]  == TEST_START, (
        f"Frozen test_start '{meta['test_start']}' != expected '{TEST_START}'"
    )
    assert dti_cutoff is not None, "Frozen dti_cutoff is None — re-run oos_train.py"
    assert "fwd_return" in df.columns,    "'fwd_return' missing from panel"
    assert "fwd_ret_4q" not in df.columns, "'fwd_ret_4q' must not be present"

    # ------------------------------------------------------------------
    # Step 5 — split into test and train context frames
    # ------------------------------------------------------------------
    df_test  = df[df["date"] >= pd.Timestamp(TEST_START)].copy()
    df_train = df[df["date"] <= pd.Timestamp(_TRAIN_END)].copy()

    assert len(df_test) > 0,  "Test period is empty"
    assert len(df_train) > 0, "Train context frame is empty"
    print(f"[EVAL] Test rows   : {len(df_test)}  "
          f"({df_test['date'].min().date()} – {df_test['date'].max().date()})")
    print(f"[EVAL] Train (ctx) : {len(df_train)} rows")

    # ------------------------------------------------------------------
    # Step 6 — add correction label to df_test only
    # ------------------------------------------------------------------
    # Suppress convergence warnings; add_correction_label does no fitting,
    # but statsmodels may warn if called elsewhere in the import chain.
    df_test_labeled = add_correction_label(df_test)
    n_test_positive = int(df_test_labeled["y"].sum())
    print(f"[EVAL] Test labeled rows : {len(df_test_labeled)}  "
          f"(y=1: {n_test_positive}, y=0: {(df_test_labeled['y']==0).sum()})")

    # df_train context for count tables only (no label needed for regime counts)
    df_train_labeled = add_correction_label(df_train)

    # ------------------------------------------------------------------
    # Step 7 — construct strategies on labeled test rows
    # ------------------------------------------------------------------
    df_strat = df_test_labeled.copy()

    # always_in: hold every period
    df_strat["always_in"] = df_strat["fwd_return"]

    # overlay: hold unless regime==0 and dti > cutoff
    df_strat["overlay"] = apply_overlay_filter(
        df_strat,
        dti_cutoff=dti_cutoff,
        regime_col="regime",
        dti_col="dti",
        return_col="fwd_return",
    )

    # always_out: flat every period
    df_strat["always_out"] = 0.0

    # naive_benchmark: regime-1 only (hold when accommodative)
    df_strat["naive_benchmark"] = df_strat["fwd_return"].where(
        df_strat["regime"] == 1, other=0.0
    )

    # ------------------------------------------------------------------
    # Step 8 — regime_supply_counts.csv
    # ------------------------------------------------------------------
    counts_df = regime_supply_count_table(
        df_train=df_train_labeled,
        df_test=df_test_labeled,
        regime_col="regime",
        supply_col=None,        # no supply column in synthetic panel
        label_col="y",
    )
    counts_path = _TABLE_DIR / "regime_supply_counts.csv"
    counts_df.to_csv(counts_path, index=False)
    print(f"[EVAL] Wrote {counts_path}")

    # ------------------------------------------------------------------
    # Step 9 — crash_frequency_ci.csv
    # ------------------------------------------------------------------
    crash_df = crash_frequency_table(
        df=df_test_labeled,
        regime_col="regime",
        supply_col=None,
        label_col="y",
    )
    crash_path = _TABLE_DIR / "crash_frequency_ci.csv"
    crash_df.to_csv(crash_path, index=False)
    print(f"[EVAL] Wrote {crash_path}")

    # ------------------------------------------------------------------
    # Step 10 — overlay_performance.csv
    # ------------------------------------------------------------------
    strat_cols = ["always_in", "overlay", "always_out", "naive_benchmark"]
    perf_rows = [summarize(df_strat[col], name=col) for col in strat_cols]
    perf_df = pd.DataFrame(perf_rows)
    perf_path = _TABLE_DIR / "overlay_performance.csv"
    perf_df.to_csv(perf_path, index=False)
    print(f"[EVAL] Wrote {perf_path}")

    # ------------------------------------------------------------------
    # Step 11 — sensitivity_analysis.csv (cutoff multipliers)
    # ------------------------------------------------------------------
    sens_rows = []
    for mult in _SENSITIVITY_MULTIPLIERS:
        adj_cutoff = dti_cutoff * mult
        adj_ret = apply_overlay_filter(
            df_strat,
            dti_cutoff=adj_cutoff,
            regime_col="regime",
            dti_col="dti",
            return_col="fwd_return",
        )
        s = summarize(adj_ret, name=f"overlay_x{mult:.2f}")
        s["cutoff_multiplier"] = mult
        s["adj_cutoff"] = adj_cutoff
        sens_rows.append(s)
    sens_df = pd.DataFrame(sens_rows)
    sens_path = _TABLE_DIR / "sensitivity_analysis.csv"
    sens_df.to_csv(sens_path, index=False)
    print(f"[EVAL] Wrote {sens_path}")

    # ------------------------------------------------------------------
    # Step 12 — equity_curve.png
    # ------------------------------------------------------------------
    plot_cols = ["always_in", "overlay", "always_out", "naive_benchmark"]
    markers = {
        "always_in":       "Always In",
        "overlay":         f"Overlay (cutoff={dti_cutoff:.1f})",
        "always_out":      "Always Out",
        "naive_benchmark": "Naive (regime==1 only)",
    }
    equity_path = _PLOT_DIR / "equity_curve.png"
    plot_equity_curves(
        df=df_strat,
        strategy_cols=plot_cols,
        markers=markers,
        outpath=equity_path,
    )
    print(f"[EVAL] Wrote {equity_path}")

    # ------------------------------------------------------------------
    # Step 13 — verdict.txt
    # ------------------------------------------------------------------
    primary = {
        "always_in":  summarize(df_strat["always_in"],  name="always_in"),
        "overlay":    summarize(df_strat["overlay"],     name="overlay"),
        "always_out": summarize(df_strat["always_out"],  name="always_out"),
    }
    secondary = {s["name"]: s for s in sens_rows}

    verdict_label = (
        "SMOKE_TEST_DEGENERATE_TRAINING"
        if is_smoke_test
        else "PIPELINE_COMPLETE"
    )

    verdict_path = OUT_DIR / "verdict.txt"
    write_verdict(
        verdict=verdict_label,
        primary_summaries=primary,
        secondary_summaries=secondary,
        outpath=verdict_path,
    )
    print(f"[EVAL] Wrote {verdict_path}")

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    print()
    print("=" * 62)
    print("OOS EVALUATION SUMMARY")
    if is_smoke_test:
        print()
        print("  *** SMOKE TEST ONLY — training window had zero positive  ***")
        print("  *** labels. Metrics below validate pipeline only.        ***")
    print("=" * 62)
    print(perf_df[["name", "n", "mean", "sharpe", "p05", "maxdd"]].to_string(index=False))
    print()
    print(f"  Verdict : {verdict_label}")
    print("=" * 62)
    print()
    print("[OOS_EVAL] complete.")


if __name__ == "__main__":
    main()
