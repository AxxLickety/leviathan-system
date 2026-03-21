"""
OOS Joint-Trigger Evaluation Script — Leviathan
================================================
Evaluates whether a percentile-DTI filter adds value beyond a regime-only
filter in the joint-trigger world (where crashes are triggered by
regime==0 AND dti_pct_roll > 0.65).

Run:
    python scripts/oos_joint_trigger_eval.py

Outputs (all in outputs/oos_joint_trigger/):
    comparison.csv       — all 4 strategies, full metrics
    lift.csv             — lift of overlays vs always_in, plus joint_overlay vs regime_only
    comparison_print.txt — text summary
    summary.csv          — one-row summary with key fields
"""
from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_END   = "2007-12-31"
TEST_START  = "2008-03-31"
OUT_DIR     = Path("outputs/oos_joint_trigger")

_ROLL_WINDOW   = 20
_JOINT_THRESH  = 0.65    # must match build_dataset._JOINT_THRESH

# Walk-forward parameters
_WF_WINDOW_Q   = 20
_WF_GRID       = np.arange(0.50, 0.96, 0.05)
_WF_FALLBACK   = 0.75
_WF_MIN_R0     = 10
_WF_MIN_ROWS   = 20


# ---------------------------------------------------------------------------
# Walk-forward helpers
# ---------------------------------------------------------------------------

def _filter_returns(sub: pd.DataFrame, thr: float) -> pd.Series:
    """Apply percentile-DTI/regime overlay; return filtered return series."""
    hold = ~((sub["regime"] == 0) & (sub["dti_pct_roll"] > thr))
    return sub["fwd_return"] * hold.astype(float)


def _walk_forward_pct_cutoff(df_train: pd.DataFrame) -> float:
    """
    Walk-forward search for optimal pct_cutoff on training data.

    Expanding window. Grid: np.arange(0.50, 0.96, 0.05).
    Objective: maximize p05 of filtered return series.
    Returns frozen cutoff from final step.
    """
    df_wf = (
        df_train
        .dropna(subset=["fwd_return", "dti_pct_roll", "regime"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    assert len(df_wf) > _WF_WINDOW_Q, (
        f"Not enough training rows for walk-forward: {len(df_wf)} <= {_WF_WINDOW_Q}"
    )

    pct_cutoff: float = _WF_FALLBACK
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

    print(f"[WF_PCT]    steps={wf_steps}  frozen pct_cutoff={pct_cutoff:.4f}")
    return pct_cutoff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Build panel using joint-trigger dataset
    # ------------------------------------------------------------------
    df = build_master_df()

    # ------------------------------------------------------------------
    # Step 2 — Compute forward 4-quarter log return
    # ------------------------------------------------------------------
    df = compute_forward_return(df)

    # ------------------------------------------------------------------
    # Step 3 — Assign fragility regime (canonical: overwrites inline regime)
    # ------------------------------------------------------------------
    df = assign_fragility_regime(df)
    assert pd.api.types.is_integer_dtype(df["regime"]), (
        f"regime must be integer dtype, got {df['regime'].dtype}"
    )
    assert set(df["regime"].unique()).issubset({0, 1}), (
        f"regime must be 0/1 only, got: {df['regime'].unique()}"
    )

    # ------------------------------------------------------------------
    # Step 4 — Compute causal rolling-percentile DTI (full panel)
    # ------------------------------------------------------------------
    df["dti_pct_roll"] = compute_rolling_pct_rank(df["dti"], window=_ROLL_WINDOW)

    # ------------------------------------------------------------------
    # Step 5 — Split train / test
    # ------------------------------------------------------------------
    df_train_full = df[df["date"] <= pd.Timestamp(TRAIN_END)].copy()
    df_test_raw   = df[df["date"] >= pd.Timestamp(TEST_START)].copy()
    df_test       = df_test_raw.dropna(subset=["fwd_return"]).copy()

    n_train_rows = len(df_train_full)
    n_test_rows  = len(df_test)

    # ------------------------------------------------------------------
    # Step 6 — Walk-forward pct_cutoff (training only)
    # ------------------------------------------------------------------
    pct_cutoff = _walk_forward_pct_cutoff(df_train_full)

    # ------------------------------------------------------------------
    # Step 7 — Label training and test windows
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_train_labeled = add_correction_label(df_train_full)
        df_test_labeled  = add_correction_label(df_test_raw)

    n_train_labeled  = len(df_train_labeled)
    n_train_y1       = int(df_train_labeled["y"].sum())
    n_train_y0       = n_train_labeled - n_train_y1

    n_test_labeled   = len(df_test_labeled)
    n_test_y1        = int(df_test_labeled["y"].sum())
    n_test_y0        = n_test_labeled - n_test_y1

    # ------------------------------------------------------------------
    # Step 8 — Compute joint-trigger eligibility diagnostics
    # ------------------------------------------------------------------
    joint_elig_train = (
        (df_train_full["regime"] == 0) & (df_train_full["dti_pct_roll"] > _JOINT_THRESH)
    ).sum()
    joint_elig_test = (
        (df_test["regime"] == 0) & (df_test["dti_pct_roll"] > _JOINT_THRESH)
    ).sum()

    # ------------------------------------------------------------------
    # Step 9 — Print startup diagnostic
    # ------------------------------------------------------------------
    print()
    print("[JT_EVAL] Joint trigger world:")
    print(f"  Train eligible (regime==0 AND pct>{_JOINT_THRESH}): "
          f"{joint_elig_train} / {n_train_rows}")
    print(f"  Test  eligible (regime==0 AND pct>{_JOINT_THRESH}): "
          f"{joint_elig_test} / {n_test_rows}")
    print(f"  Walk-forward pct_cutoff: {pct_cutoff:.2f} (from training only)")
    print(f"  Train labeled rows: {n_train_labeled}  "
          f"(y=1: {n_train_y1}, y=0: {n_train_y0})")
    print(f"  Test  labeled rows: {n_test_labeled}  "
          f"(y=1: {n_test_y1}, y=0: {n_test_y0})")

    # ------------------------------------------------------------------
    # Step 10 — Compute trailing 4q return for trend_regime filter
    # ------------------------------------------------------------------
    df_test = df_test.copy()
    df_test["trailing_4q"] = (
        np.log(df_test["real_price_index"])
        - np.log(df_test["real_price_index"].shift(4))
    )

    # ------------------------------------------------------------------
    # Step 11 — Define four test strategies
    # ------------------------------------------------------------------
    # always_in: always hold
    always_in = df_test["fwd_return"].copy()

    # regime_only: hold when regime==1, else 0
    regime_only = df_test["fwd_return"].where(df_test["regime"] == 1, other=0.0)

    # joint_overlay: hold unless regime==0 AND dti_pct_roll > pct_cutoff
    joint_overlay_mask = ~((df_test["regime"] == 0) & (df_test["dti_pct_roll"] > pct_cutoff))
    joint_overlay = df_test["fwd_return"] * joint_overlay_mask.astype(float)

    # trend_regime: hold when regime==1 AND trailing_4q > 0
    trend_regime_mask = (df_test["regime"] == 1) & (df_test["trailing_4q"] > 0)
    trend_regime = df_test["fwd_return"].where(trend_regime_mask, other=0.0)

    # ------------------------------------------------------------------
    # Step 12 — Compute pct_invested for each strategy
    # ------------------------------------------------------------------
    n_test = len(df_test)

    def pct_invested(s: pd.Series) -> float:
        return float((s != 0).sum()) / n_test if n_test > 0 else float("nan")

    # ------------------------------------------------------------------
    # Step 13 — Compute metrics via summarize()
    # ------------------------------------------------------------------
    strat_map = {
        "always_in":    always_in,
        "regime_only":  regime_only,
        "joint_overlay": joint_overlay,
        "trend_regime": trend_regime,
    }

    rows = []
    for name, series in strat_map.items():
        row = summarize(series, name=name)
        row["pct_invested"] = pct_invested(series)
        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Step 14 — Compute lift table
    # ------------------------------------------------------------------
    ai_row = comparison_df.loc[comparison_df["name"] == "always_in"].iloc[0]
    ro_row = comparison_df.loc[comparison_df["name"] == "regime_only"].iloc[0]
    jo_row = comparison_df.loc[comparison_df["name"] == "joint_overlay"].iloc[0]

    lift_rows = []
    for name, series in strat_map.items():
        row = comparison_df.loc[comparison_df["name"] == name].iloc[0]
        lift_row = {
            "name":            name,
            "vs_always_in_sharpe": row["sharpe"] - ai_row["sharpe"],
            "vs_always_in_p05":    row["p05"]    - ai_row["p05"],
            "vs_always_in_maxdd":  row["maxdd"]  - ai_row["maxdd"],
        }
        if name == "joint_overlay":
            lift_row["vs_regime_only_sharpe"] = jo_row["sharpe"] - ro_row["sharpe"]
            lift_row["vs_regime_only_p05"]    = jo_row["p05"]    - ro_row["p05"]
            lift_row["vs_regime_only_maxdd"]  = jo_row["maxdd"]  - ro_row["maxdd"]
        else:
            lift_row["vs_regime_only_sharpe"] = float("nan")
            lift_row["vs_regime_only_p05"]    = float("nan")
            lift_row["vs_regime_only_maxdd"]  = float("nan")
        lift_rows.append(lift_row)

    lift_df = pd.DataFrame(lift_rows)

    # ------------------------------------------------------------------
    # Step 15 — DTI contribution (joint_overlay vs regime_only)
    # ------------------------------------------------------------------
    dti_delta_sharpe = float(jo_row["sharpe"] - ro_row["sharpe"])
    dti_delta_p05    = float(jo_row["p05"]    - ro_row["p05"])
    dti_delta_maxdd  = float(jo_row["maxdd"]  - ro_row["maxdd"])

    # Verdict logic
    if dti_delta_p05 > 0.01 or dti_delta_maxdd > 0.02:
        verdict = "DTI_ADDS_VALUE"
    elif abs(dti_delta_p05) < 0.01 and abs(dti_delta_maxdd) < 0.01:
        verdict = "DTI_NEUTRAL"
    elif dti_delta_p05 < -0.01 and dti_delta_maxdd < -0.01:
        verdict = "DTI_HURTS"
    else:
        verdict = "DTI_NEUTRAL"

    # ------------------------------------------------------------------
    # Step 16 — Print comparison table
    # ------------------------------------------------------------------
    print()
    print("=== TEST PERIOD STRATEGY COMPARISON ===")
    print(comparison_df.to_string(
        index=False,
        float_format="{:.4f}".format,
        columns=["name", "n", "mean", "vol", "sharpe", "p05", "p50", "maxdd", "pct_invested"],
    ))

    print()
    print("=== CORE QUESTION: Does DTI add value beyond regime? ===")
    print("  joint_overlay vs regime_only:")
    print(f"    Δsharpe = {dti_delta_sharpe:.4f}")
    print(f"    Δp05    = {dti_delta_p05:.4f}")
    print(f"    Δmaxdd  = {dti_delta_maxdd:.4f}")
    print(f"    → {verdict}")

    # ------------------------------------------------------------------
    # Step 17 — Write output files
    # ------------------------------------------------------------------
    comparison_df.to_csv(OUT_DIR / "comparison.csv", index=False)
    lift_df.to_csv(OUT_DIR / "lift.csv", index=False)

    # Text summary
    text_lines = [
        "=== OOS Joint-Trigger Evaluation ===",
        "",
        f"Train window  : up to {TRAIN_END}",
        f"Test window   : {TEST_START} onward",
        f"pct_cutoff    : {pct_cutoff:.4f}  (walk-forward, training only)",
        "",
        "Train labels:",
        f"  n_labeled   : {n_train_labeled}",
        f"  y=1         : {n_train_y1}",
        f"  y=0         : {n_train_y0}",
        "",
        "Test labels:",
        f"  n_labeled   : {n_test_labeled}",
        f"  y=1         : {n_test_y1}",
        f"  y=0         : {n_test_y0}",
        "",
        "=== Strategy Comparison (test period) ===",
        comparison_df.to_string(
            index=False,
            float_format="{:.4f}".format,
            columns=["name", "n", "mean", "vol", "sharpe", "p05", "p50", "maxdd", "pct_invested"],
        ),
        "",
        "=== DTI Contribution (joint_overlay vs regime_only) ===",
        f"  Δsharpe = {dti_delta_sharpe:.4f}",
        f"  Δp05    = {dti_delta_p05:.4f}",
        f"  Δmaxdd  = {dti_delta_maxdd:.4f}",
        f"  verdict = {verdict}",
    ]
    (OUT_DIR / "comparison_print.txt").write_text("\n".join(text_lines))

    # One-row summary
    summary_row = {
        "pct_cutoff":           pct_cutoff,
        "n_train_labeled":      n_train_labeled,
        "n_train_y1":           n_train_y1,
        "n_test_labeled":       n_test_labeled,
        "n_test_y1":            n_test_y1,
        "regime_only_sharpe":   float(ro_row["sharpe"]),
        "regime_only_p05":      float(ro_row["p05"]),
        "regime_only_maxdd":    float(ro_row["maxdd"]),
        "joint_overlay_sharpe": float(jo_row["sharpe"]),
        "joint_overlay_p05":    float(jo_row["p05"]),
        "joint_overlay_maxdd":  float(jo_row["maxdd"]),
        "dti_delta_sharpe":     dti_delta_sharpe,
        "dti_delta_p05":        dti_delta_p05,
        "dti_delta_maxdd":      dti_delta_maxdd,
        "verdict":              verdict,
    }
    pd.DataFrame([summary_row]).to_csv(OUT_DIR / "summary.csv", index=False)

    print()
    print(f"[JT_EVAL] Outputs written to {OUT_DIR}/")
    for fname in ["comparison.csv", "lift.csv", "comparison_print.txt", "summary.csv"]:
        p = OUT_DIR / fname
        assert p.exists(), f"Output file missing: {p}"
        print(f"  [OK] {p}")

    print()
    print("[JT_EVAL] complete.")


if __name__ == "__main__":
    main()
