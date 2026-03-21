"""
OOS Evaluation Script (Percentile-DTI) — Leviathan Phase OOS
=============================================================
Loads frozen percentile-space parameters from outputs/oos_pct/frozen/,
applies them to the test period (date >= TEST_START), and writes all
evaluation artifacts.

FIREWALL: This script must not recompute any training-derived parameter.
All coefficients, thresholds, and the DTI percentile cutoff are read from
disk only.

The DTI feature used here is dti_pct_roll (causal rolling 20q percentile
rank), computed identically to oos_train_pct.py. The frozen cutoff is a
value in [0, 1] — it is a percentile level, not a raw DTI value.

Run:
    python scripts/oos_eval_pct.py

Outputs:
    outputs/oos_pct/tables/regime_supply_counts.csv
    outputs/oos_pct/tables/crash_frequency_ci.csv
    outputs/oos_pct/tables/overlay_performance.csv
    outputs/oos_pct/tables/sensitivity_analysis.csv
    outputs/oos_pct/plots/equity_curve.png
    outputs/oos_pct/verdict.txt
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
from src.evaluation.transforms import compute_rolling_pct_rank
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
FROZEN_DIR  = "outputs/oos_pct/frozen/"
OUT_DIR     = Path("outputs/oos_pct")

_TABLE_DIR  = OUT_DIR / "tables"
_PLOT_DIR   = OUT_DIR / "plots"

_ROLL_WINDOW = 20

# Sensitivity: test the cutoff at fixed additive offsets in [0, 1] space
_SENSITIVITY_OFFSETS = [-0.15, -0.10, 0.10, 0.15]

_EXPECTED_DTI_FEATURE = f"dti_pct_roll{_ROLL_WINDOW}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _TABLE_DIR.mkdir(parents=True, exist_ok=True)
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — load frozen params (firewall: first substantive operation)
    # ------------------------------------------------------------------
    frozen     = load_frozen_params(FROZEN_DIR)
    pct_cutoff = frozen["dti_cutoff"]
    meta       = frozen["train_metadata"]

    print(f"[EVAL_PCT] Frozen params loaded from {FROZEN_DIR}")
    print(f"[EVAL_PCT] Training window : {meta['train_start']}  →  {meta['train_end']}")
    print(f"[EVAL_PCT] DTI feature     : {meta.get('dti_feature', 'unknown')}")
    print(f"[EVAL_PCT] pct_cutoff      : {pct_cutoff:.4f}")

    # ------------------------------------------------------------------
    # Step 2 — load full panel
    # ------------------------------------------------------------------
    df = build_master_df()

    # ------------------------------------------------------------------
    # Step 3 — compute fwd_return and regime (mirrors oos_train_pct.py)
    # ------------------------------------------------------------------
    df = compute_forward_return(df)
    assert "fwd_ret_4q" not in df.columns, (
        "Unexpected fwd_ret_4q present after compute_forward_return()"
    )
    df = assign_fragility_regime(df)
    assert pd.api.types.is_integer_dtype(df["regime"]), (
        f"regime must be integer dtype, got {df['regime'].dtype}"
    )

    # ------------------------------------------------------------------
    # Step 4 — compute dti_pct_roll (same transform as oos_train_pct.py)
    #
    # Computed on the full panel so early test rows have their correct
    # 20-quarter lookback into training history. Causal by construction.
    # ------------------------------------------------------------------
    df["dti_pct_roll"] = compute_rolling_pct_rank(df["dti"], window=_ROLL_WINDOW)

    # ------------------------------------------------------------------
    # Step 5 — firewall assertions
    # ------------------------------------------------------------------
    assert meta["train_end"]  == _TRAIN_END, (
        f"Frozen train_end '{meta['train_end']}' != expected '{_TRAIN_END}'"
    )
    assert meta["test_start"] == TEST_START, (
        f"Frozen test_start '{meta['test_start']}' != expected '{TEST_START}'"
    )
    assert pct_cutoff is not None, (
        "Frozen pct_cutoff is None — re-run oos_train_pct.py"
    )
    assert 0.0 < pct_cutoff < 1.0, (
        f"pct_cutoff={pct_cutoff:.4f} outside (0, 1) — wrong frozen dir? "
        "This script expects a percentile-space cutoff from oos_train_pct.py."
    )
    assert meta.get("dti_feature") == _EXPECTED_DTI_FEATURE, (
        f"Frozen dti_feature='{meta.get('dti_feature')}' does not match "
        f"expected '{_EXPECTED_DTI_FEATURE}'. Re-run oos_train_pct.py."
    )
    assert "fwd_return"    in df.columns, "'fwd_return' missing from panel"
    assert "dti_pct_roll"  in df.columns, "'dti_pct_roll' missing from panel"

    # ------------------------------------------------------------------
    # Step 6 — split into test and training context frames
    # ------------------------------------------------------------------
    df_test  = df[df["date"] >= pd.Timestamp(TEST_START)].copy()
    df_train = df[df["date"] <= pd.Timestamp(_TRAIN_END)].copy()

    print(f"[EVAL_PCT] Test rows   : {len(df_test)}  "
          f"({df_test['date'].min().date()} – {df_test['date'].max().date()})")
    print(f"[EVAL_PCT] Train (ctx) : {len(df_train)} rows")

    # Diagnostic: dti_pct_roll distribution in test vs train
    tr_pct = df_train["dti_pct_roll"]
    te_pct = df_test.dropna(subset=["fwd_return"])["dti_pct_roll"]
    print(f"[EVAL_PCT] Train dti_pct_roll: "
          f"mean={tr_pct.mean():.3f}  p25={tr_pct.quantile(.25):.3f}  "
          f"p75={tr_pct.quantile(.75):.3f}")
    print(f"[EVAL_PCT] Test  dti_pct_roll: "
          f"mean={te_pct.mean():.3f}  p25={te_pct.quantile(.25):.3f}  "
          f"p75={te_pct.quantile(.75):.3f}")

    gate_active = (df_test["regime"] == 0) & (df_test["dti_pct_roll"] > pct_cutoff)
    print(f"[EVAL_PCT] Leviathan gate fires: {gate_active.sum()}/{len(df_test)} "
          f"test rows  ({gate_active.mean():.1%})")
    print(f"           → regime==0 rows:               "
          f"{(df_test['regime']==0).sum()}")
    print(f"           → regime==0 AND pct_roll>{pct_cutoff:.2f}: "
          f"{gate_active.sum()}")

    # ------------------------------------------------------------------
    # Step 7 — add correction labels
    # ------------------------------------------------------------------
    df_test_labeled  = add_correction_label(df_test)
    df_train_labeled = add_correction_label(df_train)

    n_test_pos = int(df_test_labeled["y"].sum())
    print(f"[EVAL_PCT] Test labeled rows : {len(df_test_labeled)}  "
          f"(y=1: {n_test_pos}, y=0: {(df_test_labeled['y']==0).sum()})")

    # ------------------------------------------------------------------
    # Step 8 — construct strategies on labeled test rows
    # ------------------------------------------------------------------
    df_strat = df_test_labeled.copy()

    df_strat["always_in"] = df_strat["fwd_return"]

    # overlay: hold unless regime==0 AND dti_pct_roll > frozen pct_cutoff
    df_strat["overlay"] = apply_overlay_filter(
        df_strat,
        dti_cutoff=pct_cutoff,
        regime_col="regime",
        dti_col="dti_pct_roll",
        return_col="fwd_return",
    )

    # regime_only: hold when regime==1 only — isolates the regime component
    df_strat["regime_only"] = df_strat["fwd_return"].where(
        df_strat["regime"] == 1, other=0.0
    )

    # always_out
    df_strat["always_out"] = 0.0

    # ------------------------------------------------------------------
    # Step 9 — regime_supply_counts.csv
    # ------------------------------------------------------------------
    counts_df = regime_supply_count_table(
        df_train=df_train_labeled,
        df_test=df_test_labeled,
        regime_col="regime",
        supply_col=None,
        label_col="y",
    )
    counts_path = _TABLE_DIR / "regime_supply_counts.csv"
    counts_df.to_csv(counts_path, index=False)
    print(f"[EVAL_PCT] Wrote {counts_path}")

    # ------------------------------------------------------------------
    # Step 10 — crash_frequency_ci.csv
    # ------------------------------------------------------------------
    crash_df = crash_frequency_table(
        df=df_test_labeled,
        regime_col="regime",
        supply_col=None,
        label_col="y",
    )
    crash_path = _TABLE_DIR / "crash_frequency_ci.csv"
    crash_df.to_csv(crash_path, index=False)
    print(f"[EVAL_PCT] Wrote {crash_path}")

    # ------------------------------------------------------------------
    # Step 11 — overlay_performance.csv
    # ------------------------------------------------------------------
    strat_cols = ["always_in", "overlay", "regime_only", "always_out"]
    perf_rows  = [summarize(df_strat[col], name=col) for col in strat_cols]
    perf_df    = pd.DataFrame(perf_rows)

    # Annotate participation rate and gate comparison
    for row in perf_rows:
        col = row["name"]
        if col in df_strat.columns:
            row["pct_invested"] = round(
                (df_strat[col] != 0).mean(), 4
            )

    perf_df["pct_invested"] = [(df_strat[c] != 0).mean()
                               for c in strat_cols]
    perf_path = _TABLE_DIR / "overlay_performance.csv"
    perf_df.to_csv(perf_path, index=False)
    print(f"[EVAL_PCT] Wrote {perf_path}")

    # ------------------------------------------------------------------
    # Step 12 — sensitivity_analysis.csv (additive offsets in pct space)
    # ------------------------------------------------------------------
    sens_rows = []
    for offset in _SENSITIVITY_OFFSETS:
        adj_cutoff = float(np.clip(pct_cutoff + offset, 0.01, 0.99))
        adj_ret = apply_overlay_filter(
            df_strat,
            dti_cutoff=adj_cutoff,
            regime_col="regime",
            dti_col="dti_pct_roll",
            return_col="fwd_return",
        )
        s = summarize(adj_ret, name=f"overlay_{offset:+.2f}")
        s["cutoff_offset"] = offset
        s["adj_cutoff"]    = round(adj_cutoff, 4)
        s["pct_invested"]  = round((adj_ret != 0).mean(), 4)
        sens_rows.append(s)
    sens_df = pd.DataFrame(sens_rows)
    sens_path = _TABLE_DIR / "sensitivity_analysis.csv"
    sens_df.to_csv(sens_path, index=False)
    print(f"[EVAL_PCT] Wrote {sens_path}")

    # ------------------------------------------------------------------
    # Step 13 — equity_curve.png
    # ------------------------------------------------------------------
    plot_cols = ["always_in", "overlay", "regime_only", "always_out"]
    markers = {
        "always_in":   "Always In",
        "overlay":     f"Overlay (pct_cutoff={pct_cutoff:.2f})",
        "regime_only": "Regime Only (regime==1)",
        "always_out":  "Always Out",
    }
    equity_path = _PLOT_DIR / "equity_curve.png"
    plot_equity_curves(
        df=df_strat,
        strategy_cols=plot_cols,
        markers=markers,
        outpath=equity_path,
    )
    print(f"[EVAL_PCT] Wrote {equity_path}")

    # ------------------------------------------------------------------
    # Step 14 — verdict.txt
    # ------------------------------------------------------------------
    primary = {
        "always_in":  summarize(df_strat["always_in"],  name="always_in"),
        "overlay":    summarize(df_strat["overlay"],     name="overlay"),
        "always_out": summarize(df_strat["always_out"],  name="always_out"),
    }
    secondary = {s["name"]: s for s in sens_rows}

    verdict_path = OUT_DIR / "verdict.txt"
    write_verdict(
        verdict="PERCENTILE_PIPELINE_COMPLETE",
        primary_summaries=primary,
        secondary_summaries=secondary,
        outpath=verdict_path,
    )
    print(f"[EVAL_PCT] Wrote {verdict_path}")

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    overlay_s    = summarize(df_strat["overlay"],    name="overlay")
    always_in_s  = summarize(df_strat["always_in"],  name="always_in")
    regime_only_s = summarize(df_strat["regime_only"], name="regime_only")

    print()
    print("=" * 66)
    print("OOS EVALUATION SUMMARY — PERCENTILE-DTI PIPELINE")
    print("=" * 66)
    print(perf_df[["name", "n", "mean", "sharpe", "p05", "maxdd",
                   "pct_invested"]].to_string(index=False))
    print()

    # DTI contribution analysis: overlay vs regime_only
    delta_sharpe = float(overlay_s["sharpe"]) - float(regime_only_s["sharpe"])
    delta_p05    = float(overlay_s["p05"])    - float(regime_only_s["p05"])
    delta_maxdd  = float(overlay_s["maxdd"])  - float(regime_only_s["maxdd"])
    print("  Overlay vs regime-only (DTI contribution):")
    print(f"    Δsharpe = {delta_sharpe:+.4f}")
    print(f"    Δp05    = {delta_p05:+.4f}")
    print(f"    Δmaxdd  = {delta_maxdd:+.4f}")
    if abs(delta_sharpe) < 0.02 and abs(delta_p05) < 0.01 and abs(delta_maxdd) < 0.02:
        dti_verdict = "DTI adds NEGLIGIBLE margin over regime-only filter"
    elif delta_p05 > 0.01 or delta_maxdd > 0.02:
        dti_verdict = "DTI adds MEANINGFUL downside improvement over regime-only"
    else:
        dti_verdict = "DTI adds MIXED signal over regime-only filter"
    print(f"    → {dti_verdict}")
    print()

    print(f"  Pct cutoff (frozen) : {pct_cutoff:.4f}")
    print(f"  Gate fires          : {gate_active.sum()}/{len(df_test)} "
          f"({gate_active.mean():.1%} of test quarters)")
    print("=" * 66)
    print()
    print("[OOS_EVAL_PCT] complete.")


if __name__ == "__main__":
    main()
