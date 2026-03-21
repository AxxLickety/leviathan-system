"""
OOS Regime Multi-Market Evaluation — Leviathan
===============================================
Tests whether the regime-only overlay generalizes across N independent
synthetic housing-market worlds with heterogeneous rate paths, DTI levels,
housing cycles, and crash parameters.

Core question:
    Is the regime-only result a single-world artifact, or a structural
    relationship that holds across many distinct regime realizations?

Model (parameter-free):
    hold[t] = 1  if  regime[t] == 1  (real_rate < 0, accommodative)
    hold[t] = 0  if  regime[t] == 0  (real_rate >= 0, restrictive)

Strategies per market:
    always_in    — fully invested every quarter
    regime_only  — hold when regime == 1
    trend_regime — hold when regime == 1 AND trailing 4q log return > 0

OOS split:
    Train: ≤ 2007-12-31  (unused — model has no learned parameters)
    Test:  ≥ 2008-03-31

Phase 4 diagnostics (regime episode metadata):
    - regime switches per market (full panel and test period)
    - adverse quarters in train and test

Outputs:
    outputs/oos_regime_multimarket/market_level_results.csv
    outputs/oos_regime_multimarket/summary.csv
    outputs/oos_regime_multimarket/summary_print.txt

Run:
    python scripts/oos_regime_multimarket.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.research.path_a.build_multimarket import build_all_markets
from src.evaluation.backtest import compute_forward_return
from src.evaluation.regime import assign_fragility_regime
from src.backtests.evaluation import summarize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OOS_START = "2008-03-31"
_TRAIN_END  = "2007-12-31"
N_MARKETS   = 12

OUT_DIR     = Path("outputs/oos_regime_multimarket")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regime_switches(regime: pd.Series) -> int:
    return int((regime.diff().abs() > 0).sum())


def _pct_invested(hold: pd.Series) -> float:
    return float(hold.astype(float).mean())


def _turnover(hold: pd.Series) -> int:
    return int((hold.astype(int).diff().abs() > 0).sum())


# ---------------------------------------------------------------------------
# Per-market evaluation
# ---------------------------------------------------------------------------

def eval_market(df_mkt: pd.DataFrame, market_id: int) -> dict | None:
    """Evaluate always_in, regime_only, and trend_regime for one market."""
    df = compute_forward_return(df_mkt.copy())
    df = assign_fragility_regime(df)

    # Trailing 4q log return for trend signal (causal)
    log_px         = np.log(df["real_price_index"])
    df["trail_4q"] = log_px - log_px.shift(4)

    train = df[df["date"] <= pd.Timestamp(_TRAIN_END)].copy()
    test  = (
        df[df["date"] >= pd.Timestamp(_OOS_START)]
        .dropna(subset=["fwd_return"])
        .copy()
        .reset_index(drop=True)
    )

    if len(test) < 10:
        return None   # skip degenerate panels

    ret      = test["fwd_return"]
    hold_ai  = pd.Series(True,  index=test.index)
    hold_ro  = test["regime"] == 1
    hold_tr  = hold_ro & (test["trail_4q"] > 0)

    ai_s  = summarize(ret * hold_ai.astype(float), name="always_in")
    ro_s  = summarize(ret * hold_ro.astype(float), name="regime_only")
    tr_s  = summarize(ret * hold_tr.astype(float), name="trend_regime")

    # Regime episode metadata
    full_sw     = _regime_switches(df["regime"])
    test_sw     = _regime_switches(test["regime"])
    adv_train   = int((train["regime"] == 0).sum())
    adv_test    = int((test["regime"] == 0).sum())

    delta_sharpe = round(float(ro_s["sharpe"]) - float(ai_s["sharpe"]), 3)
    delta_p05    = round(float(ro_s["p05"])    - float(ai_s["p05"]),    4)
    delta_maxdd  = round(float(ro_s["maxdd"])  - float(ai_s["maxdd"]),  4)

    return {
        "market_id":             market_id,
        # always_in
        "ai_mean":               round(float(ai_s["mean"]),   4),
        "ai_sharpe":             round(float(ai_s["sharpe"]), 3),
        "ai_p05":                round(float(ai_s["p05"]),    4),
        "ai_maxdd":              round(float(ai_s["maxdd"]),  4),
        # regime_only
        "ro_mean":               round(float(ro_s["mean"]),   4),
        "ro_sharpe":             round(float(ro_s["sharpe"]), 3),
        "ro_p05":                round(float(ro_s["p05"]),    4),
        "ro_maxdd":              round(float(ro_s["maxdd"]),  4),
        "ro_pct_invested":       round(_pct_invested(hold_ro), 3),
        "ro_turnover":           _turnover(hold_ro),
        # trend+regime
        "tr_sharpe":             round(float(tr_s["sharpe"]), 3),
        "tr_p05":                round(float(tr_s["p05"]),    4),
        "tr_maxdd":              round(float(tr_s["maxdd"]),  4),
        "tr_pct_invested":       round(_pct_invested(hold_tr), 3),
        # deltas: regime_only vs always_in
        "d_sharpe":              delta_sharpe,
        "d_p05":                 delta_p05,
        "d_maxdd":               delta_maxdd,
        # Phase 4 — regime episode metadata
        "regime_switches_total": full_sw,
        "regime_switches_test":  test_sw,
        "adverse_q_train":       adv_train,
        "adverse_q_test":        adv_test,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[MULTIMARKET] Building {N_MARKETS} synthetic markets ...")
    df_all = build_all_markets(n_markets=N_MARKETS, verbose=True)

    market_rows: list[dict] = []
    for mid, grp in df_all.groupby("market_id"):
        row = eval_market(grp.reset_index(drop=True), int(mid))
        if row is not None:
            market_rows.append(row)

    df_markets = pd.DataFrame(market_rows).sort_values("market_id").reset_index(drop=True)
    n = len(df_markets)

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    n_p05_improved    = int((df_markets["d_p05"]    > 0).sum())
    n_maxdd_improved  = int((df_markets["d_maxdd"]  > 0).sum())
    n_sharpe_improved = int((df_markets["d_sharpe"] > 0).sum())

    mean_d_sharpe = float(df_markets["d_sharpe"].mean())
    mean_d_p05    = float(df_markets["d_p05"].mean())
    mean_d_maxdd  = float(df_markets["d_maxdd"].mean())

    med_d_sharpe  = float(df_markets["d_sharpe"].median())
    med_d_p05     = float(df_markets["d_p05"].median())
    med_d_maxdd   = float(df_markets["d_maxdd"].median())

    best_idx  = int(df_markets["d_sharpe"].idxmax())
    worst_idx = int(df_markets["d_sharpe"].idxmin())

    avg_switches = float(df_markets["regime_switches_total"].mean())
    avg_adv_test = float(df_markets["adverse_q_test"].mean())

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print_str = (
        "OOS Regime Multi-Market Evaluation\n"
        "====================================\n"
        f"N markets        : {n}\n"
        f"OOS period       : {_OOS_START}  onward\n"
        f"Model            : regime_only (exit when real_rate >= 0)\n"
        f"Avg regime switches (full panel): {avg_switches:.1f}\n"
        f"Avg adverse (regime-0) quarters in test: {avg_adv_test:.1f}\n\n"
    )

    print_str += "=== Market-Level Results ===\n"
    cols = [
        "market_id",
        "ai_sharpe", "ai_p05", "ai_maxdd",
        "ro_sharpe", "ro_p05", "ro_maxdd",
        "d_sharpe",  "d_p05",  "d_maxdd",
        "ro_pct_invested", "regime_switches_total",
    ]
    print_str += df_markets[cols].to_string(index=False) + "\n\n"

    print_str += "=== Aggregate: regime_only vs always_in ===\n"
    print_str += (
        f"  p05 improved   : {n_p05_improved}/{n} markets\n"
        f"  maxdd improved : {n_maxdd_improved}/{n} markets\n"
        f"  Sharpe improved: {n_sharpe_improved}/{n} markets\n\n"
        f"  Mean  Δsharpe  : {mean_d_sharpe:+.3f}\n"
        f"  Mean  Δp05     : {mean_d_p05:+.4f}\n"
        f"  Mean  Δmaxdd   : {mean_d_maxdd:+.4f}\n\n"
        f"  Median Δsharpe : {med_d_sharpe:+.3f}\n"
        f"  Median Δp05    : {med_d_p05:+.4f}\n"
        f"  Median Δmaxdd  : {med_d_maxdd:+.4f}\n\n"
    )

    best  = df_markets.iloc[best_idx]
    worst = df_markets.iloc[worst_idx]
    print_str += (
        f"  Best market  (Δsharpe={best['d_sharpe']:+.3f}): "
        f"market_id={int(best['market_id'])}  "
        f"ro_sharpe={best['ro_sharpe']:.3f}  "
        f"ai_sharpe={best['ai_sharpe']:.3f}  "
        f"regime_switches={int(best['regime_switches_total'])}\n"
        f"  Worst market (Δsharpe={worst['d_sharpe']:+.3f}): "
        f"market_id={int(worst['market_id'])}  "
        f"ro_sharpe={worst['ro_sharpe']:.3f}  "
        f"ai_sharpe={worst['ai_sharpe']:.3f}  "
        f"regime_switches={int(worst['regime_switches_total'])}\n\n"
    )

    # Distribution of Δsharpe
    q10 = float(df_markets["d_sharpe"].quantile(0.10))
    q25 = float(df_markets["d_sharpe"].quantile(0.25))
    q75 = float(df_markets["d_sharpe"].quantile(0.75))
    q90 = float(df_markets["d_sharpe"].quantile(0.90))
    print_str += (
        f"  Δsharpe distribution:\n"
        f"    p10={q10:+.3f}  p25={q25:+.3f}  "
        f"median={med_d_sharpe:+.3f}  "
        f"p75={q75:+.3f}  p90={q90:+.3f}\n\n"
    )

    # Decision
    if n_p05_improved >= int(0.8 * n) and n_maxdd_improved >= int(0.8 * n):
        verdict = "GENERALIZES — regime-only is a robust downside-risk overlay across markets"
    elif n_p05_improved >= int(0.6 * n):
        verdict = "PARTIALLY GENERALIZES — downside benefit is common but not universal"
    else:
        verdict = "FRAGILE — regime-only benefit is not consistent across markets"
    print_str += f"  Verdict: {verdict}\n"

    print(print_str)

    # ------------------------------------------------------------------
    # Summary CSV (one row)
    # ------------------------------------------------------------------
    summary_row = {
        "n_markets":           n,
        "n_p05_improved":      n_p05_improved,
        "n_maxdd_improved":    n_maxdd_improved,
        "n_sharpe_improved":   n_sharpe_improved,
        "mean_d_sharpe":       round(mean_d_sharpe,  3),
        "mean_d_p05":          round(mean_d_p05,     4),
        "mean_d_maxdd":        round(mean_d_maxdd,   4),
        "median_d_sharpe":     round(med_d_sharpe,   3),
        "median_d_p05":        round(med_d_p05,      4),
        "median_d_maxdd":      round(med_d_maxdd,    4),
        "p10_d_sharpe":        round(q10, 3),
        "p90_d_sharpe":        round(q90, 3),
        "avg_regime_switches": round(avg_switches, 1),
        "avg_adverse_test_q":  round(avg_adv_test, 1),
        "verdict":             verdict,
    }
    df_summary = pd.DataFrame([summary_row])

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    df_markets.to_csv(OUT_DIR / "market_level_results.csv", index=False)
    df_summary.to_csv(OUT_DIR / "summary.csv",               index=False)
    with open(OUT_DIR / "summary_print.txt", "w") as f:
        f.write(print_str)

    print(f"[MULTIMARKET] Wrote {OUT_DIR}/market_level_results.csv")
    print(f"[MULTIMARKET] Wrote {OUT_DIR}/summary.csv")
    print(f"[MULTIMARKET] Wrote {OUT_DIR}/summary_print.txt")


if __name__ == "__main__":
    main()
