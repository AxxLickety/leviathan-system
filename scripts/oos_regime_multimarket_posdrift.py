"""
OOS Regime Multi-Market Positive-Drift Evaluation — Leviathan attribution test.
================================================================================
Tests whether regime-only Leviathan has true SIGNAL VALUE or primarily
PARTICIPATION-REDUCTION VALUE by placing it in a world where:
  - always_in expected return is positive (opportunity cost of exiting)
  - crashes are de-synchronized across markets (no shared GFC)
  - regime is informative but imperfect (~60–75% of crash-quarters in regime-0)
  - background noise exists in both regimes

Attribution question:
  In the adversarial world, two effects contributed to regime_only's apparent
  robustness:
    1. Signal value: regime-0 still had somewhat higher crash risk (1.3× ratio)
    2. Participation-reduction value: being out ~50% of the time helped because
       always_in had mostly negative expected return

  This world removes effect (2) by raising drift so always_in has positive
  expected return. If regime_only still outperforms on downside metrics, that
  is genuine signal value. If it underperforms on all metrics, the prior
  result was driven by participation reduction, not signal.

Model (unchanged throughout all experiments):
    hold[t] = 1  if  regime[t] == 1  (real_rate < 0)
    hold[t] = 0  if  regime[t] == 0  (real_rate >= 0)

Outputs:
    outputs/oos_regime_multimarket_posdrift/market_level_results.csv
    outputs/oos_regime_multimarket_posdrift/summary.csv
    outputs/oos_regime_multimarket_posdrift/summary_print.txt

Run:
    python scripts/oos_regime_multimarket_posdrift.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.research.path_a.build_multimarket_posdrift import build_all_markets_posdrift
from src.evaluation.backtest import compute_forward_return
from src.evaluation.regime import assign_fragility_regime
from src.backtests.evaluation import summarize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OOS_START = "2008-03-31"
_TRAIN_END  = "2007-12-31"
N_MARKETS   = 12

OUT_DIR = Path("outputs/oos_regime_multimarket_posdrift")

# Reference results from prior worlds (for decay comparison)
_ADV_N_P05_IMPROVED   = 11
_ADV_N_MAXDD_IMPROVED = 12
_ADV_MEAN_D_P05       = 0.0685
_ADV_MEAN_D_MAXDD     = 0.3247
_ADV_MEAN_D_SHARPE    = 0.259


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_invested(hold: pd.Series) -> float:
    return float(hold.astype(float).mean())


def _turnover(hold: pd.Series) -> int:
    return int((hold.astype(int).diff().abs() > 0).sum())


# ---------------------------------------------------------------------------
# Per-market evaluation
# ---------------------------------------------------------------------------

def eval_market(df_mkt: pd.DataFrame, market_id: int) -> dict | None:
    df = compute_forward_return(df_mkt.copy())
    df = assign_fragility_regime(df)

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
        return None

    ret      = test["fwd_return"]
    hold_ai  = pd.Series(True,  index=test.index)
    hold_ro  = test["regime"] == 1
    hold_tr  = hold_ro & (test["trail_4q"] > 0)

    ai_s  = summarize(ret * hold_ai.astype(float), name="always_in")
    ro_s  = summarize(ret * hold_ro.astype(float), name="regime_only")
    tr_s  = summarize(ret * hold_tr.astype(float), name="trend_regime")

    adv_test = int((test["regime"] == 0).sum())

    return {
        "market_id":        market_id,
        "ai_mean":          round(float(ai_s["mean"]),   4),
        "ai_vol":           round(float(ai_s["vol"]),    4),
        "ai_sharpe":        round(float(ai_s["sharpe"]), 3),
        "ai_p05":           round(float(ai_s["p05"]),    4),
        "ai_maxdd":         round(float(ai_s["maxdd"]),  4),
        "ro_mean":          round(float(ro_s["mean"]),   4),
        "ro_sharpe":        round(float(ro_s["sharpe"]), 3),
        "ro_p05":           round(float(ro_s["p05"]),    4),
        "ro_maxdd":         round(float(ro_s["maxdd"]),  4),
        "ro_pct_invested":  round(_pct_invested(hold_ro), 3),
        "ro_turnover":      _turnover(hold_ro),
        "tr_sharpe":        round(float(tr_s["sharpe"]), 3),
        "tr_p05":           round(float(tr_s["p05"]),    4),
        "tr_maxdd":         round(float(tr_s["maxdd"]),  4),
        "d_sharpe":         round(float(ro_s["sharpe"]) - float(ai_s["sharpe"]), 3),
        "d_mean":           round(float(ro_s["mean"])   - float(ai_s["mean"]),   4),
        "d_p05":            round(float(ro_s["p05"])    - float(ai_s["p05"]),    4),
        "d_maxdd":          round(float(ro_s["maxdd"])  - float(ai_s["maxdd"]),  4),
        "adverse_q_test":   adv_test,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[POSDRIFT] Building {N_MARKETS} positive-drift adversarial markets ...")
    df_all, diag_list = build_all_markets_posdrift(n_markets=N_MARKETS, verbose=True)

    market_rows: list[dict] = []
    for mid, grp in df_all.groupby("market_id"):
        row = eval_market(grp.reset_index(drop=True), int(mid))
        if row is not None:
            diag = diag_list[int(mid)]
            row["regime_switches"]    = diag["regime_switches"]
            row["major_onset_date"]   = diag["major_onset_date"]
            row["major_regime"]       = diag["major_regime"]
            row["crash_q_in_r0"]      = diag["crash_q_in_r0"]
            row["crash_q_in_r1"]      = diag["crash_q_in_r1"]
            row["frac_crash_q_in_r0"] = diag["frac_crash_q_in_r0"]
            row["frac_r0_in_crash"]   = diag["frac_r0_in_crash"]
            row["frac_r1_in_crash"]   = diag["frac_r1_in_crash"]
            market_rows.append(row)

    df_m = pd.DataFrame(market_rows).sort_values("market_id").reset_index(drop=True)
    n    = len(df_m)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    n_p05_improved    = int((df_m["d_p05"]    > 0).sum())
    n_maxdd_improved  = int((df_m["d_maxdd"]  > 0).sum())
    n_sharpe_improved = int((df_m["d_sharpe"] > 0).sum())
    n_ai_pos_sharpe   = int((df_m["ai_sharpe"] > 0).sum())
    n_ai_pos_mean     = int((df_m["ai_mean"]   > 0).sum())

    mean_d_sharpe = float(df_m["d_sharpe"].mean())
    mean_d_p05    = float(df_m["d_p05"].mean())
    mean_d_maxdd  = float(df_m["d_maxdd"].mean())
    mean_d_mean   = float(df_m["d_mean"].mean())

    med_d_sharpe  = float(df_m["d_sharpe"].median())
    med_d_p05     = float(df_m["d_p05"].median())
    med_d_maxdd   = float(df_m["d_maxdd"].median())

    avg_frac_r0     = float(df_m["frac_crash_q_in_r0"].mean())
    avg_frac_r0_in  = float(df_m["frac_r0_in_crash"].mean())
    avg_frac_r1_in  = float(df_m["frac_r1_in_crash"].mean())
    n_major_in_r0   = int((df_m["major_regime"] == "r0").sum())
    avg_switches    = float(df_m["regime_switches"].mean())

    q10 = float(df_m["d_sharpe"].quantile(0.10))
    q25 = float(df_m["d_sharpe"].quantile(0.25))
    q75 = float(df_m["d_sharpe"].quantile(0.75))
    q90 = float(df_m["d_sharpe"].quantile(0.90))

    best_idx  = int(df_m["d_sharpe"].idxmax())
    worst_idx = int(df_m["d_sharpe"].idxmin())

    # ------------------------------------------------------------------
    # Print string
    # ------------------------------------------------------------------
    print_str = (
        "OOS Regime Multi-Market Positive-Drift Evaluation\n"
        "===================================================\n"
        f"N markets          : {n}\n"
        f"OOS period         : {_OOS_START}  onward\n"
        f"Model              : regime_only (exit when real_rate >= 0)\n\n"
    )

    # Phase 4 diagnostics
    print_str += "=== Phase 4: World Diagnostics ===\n"
    print_str += (
        f"  always_in Sharpe > 0        : {n_ai_pos_sharpe}/{n} markets  "
        f"(adversarial: 1/{n})\n"
        f"  always_in mean > 0          : {n_ai_pos_mean}/{n} markets\n"
        f"  Avg frac crash-q in r0      : {avg_frac_r0:.2f}  "
        f"(adversarial: 0.58, base: ~0.87)\n"
        f"  Avg frac r0-q crashing      : {avg_frac_r0_in:.2f}  "
        f"(adversarial: 0.28)\n"
        f"  Avg frac r1-q crashing      : {avg_frac_r1_in:.2f}  "
        f"(adversarial: 0.22)\n"
        f"  Major crash in regime-0     : {n_major_in_r0}/{n} markets\n"
        f"  Avg regime switches         : {avg_switches:.1f}\n\n"
    )

    diag_cols = [
        "market_id", "ai_sharpe", "ai_mean",
        "major_onset_date", "major_regime",
        "frac_crash_q_in_r0", "frac_r0_in_crash", "frac_r1_in_crash",
    ]
    print_str += df_m[diag_cols].to_string(index=False) + "\n\n"

    # Market-level results
    print_str += "=== Market-Level Results (OOS) ===\n"
    result_cols = [
        "market_id",
        "ai_sharpe", "ai_mean", "ai_p05", "ai_maxdd",
        "ro_sharpe", "ro_p05",  "ro_maxdd",
        "d_sharpe",  "d_mean",  "d_p05",   "d_maxdd",
        "ro_pct_invested",
    ]
    print_str += df_m[result_cols].to_string(index=False) + "\n\n"

    # Aggregate
    print_str += "=== Aggregate: regime_only vs always_in ===\n"
    print_str += (
        f"  p05 improved   : {n_p05_improved}/{n} markets\n"
        f"  maxdd improved : {n_maxdd_improved}/{n} markets\n"
        f"  Sharpe improved: {n_sharpe_improved}/{n} markets\n\n"
        f"  Mean  Δsharpe  : {mean_d_sharpe:+.3f}\n"
        f"  Mean  Δmean    : {mean_d_mean:+.4f}  "
        f"(positive = regime_only earns more; negative = foregone upside)\n"
        f"  Mean  Δp05     : {mean_d_p05:+.4f}\n"
        f"  Mean  Δmaxdd   : {mean_d_maxdd:+.4f}\n\n"
        f"  Median Δsharpe : {med_d_sharpe:+.3f}\n"
        f"  Median Δp05    : {med_d_p05:+.4f}\n"
        f"  Median Δmaxdd  : {med_d_maxdd:+.4f}\n\n"
    )

    best  = df_m.iloc[best_idx]
    worst = df_m.iloc[worst_idx]
    print_str += (
        f"  Best market  (Δsharpe={best['d_sharpe']:+.3f}): "
        f"market_id={int(best['market_id'])}  "
        f"ai_sharpe={best['ai_sharpe']:.3f}  "
        f"frac_crash_in_r0={best['frac_crash_q_in_r0']:.2f}\n"
        f"  Worst market (Δsharpe={worst['d_sharpe']:+.3f}): "
        f"market_id={int(worst['market_id'])}  "
        f"ai_sharpe={worst['ai_sharpe']:.3f}  "
        f"frac_crash_in_r0={worst['frac_crash_q_in_r0']:.2f}\n\n"
        f"  Δsharpe distribution:\n"
        f"    p10={q10:+.3f}  p25={q25:+.3f}  "
        f"median={med_d_sharpe:+.3f}  "
        f"p75={q75:+.3f}  p90={q90:+.3f}\n\n"
    )

    # Decay vs adversarial world
    print_str += "=== Decay vs Adversarial World ===\n"
    print_str += (
        f"  p05 improved   : {n_p05_improved}/{n}  "
        f"(adversarial: {_ADV_N_P05_IMPROVED}/{n},  "
        f"Δ={n_p05_improved - _ADV_N_P05_IMPROVED:+d})\n"
        f"  maxdd improved : {n_maxdd_improved}/{n}  "
        f"(adversarial: {_ADV_N_MAXDD_IMPROVED}/{n},  "
        f"Δ={n_maxdd_improved - _ADV_N_MAXDD_IMPROVED:+d})\n\n"
        f"  Mean Δp05  : {mean_d_p05:+.4f}  "
        f"(adversarial: {_ADV_MEAN_D_P05:+.4f},  "
        f"decay={mean_d_p05 - _ADV_MEAN_D_P05:+.4f})\n"
        f"  Mean Δmaxdd: {mean_d_maxdd:+.4f}  "
        f"(adversarial: {_ADV_MEAN_D_MAXDD:+.4f},  "
        f"decay={mean_d_maxdd - _ADV_MEAN_D_MAXDD:+.4f})\n\n"
    )

    # Attribution note
    n_mean_sacrifice = int((df_m["d_mean"] < 0).sum())
    print_str += (
        f"  Attribution: regime_only sacrifices mean return in "
        f"{n_mean_sacrifice}/{n} markets\n"
        f"  (these are markets where always_in is profitable in both regimes)\n\n"
    )

    # Verdict
    p05_frac   = n_p05_improved   / n
    maxdd_frac = n_maxdd_improved / n
    if p05_frac >= 0.67 and maxdd_frac >= 0.67:
        verdict = (
            "SIGNAL VALUE CONFIRMED — regime-only downside protection survives "
            "positive-drift world; regime signal is not purely participation reduction"
        )
    elif p05_frac >= 0.50:
        verdict = (
            "MIXED — some downside protection persists but clearly weakened; "
            "prior result had meaningful participation-reduction component"
        )
    else:
        verdict = (
            "PARTICIPATION-REDUCTION ARTIFACT — prior robustness depended on "
            "negative-drift test world; regime signal insufficient in positive-drift world"
        )
    print_str += f"  Verdict: {verdict}\n"

    print(print_str)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    df_m.to_csv(OUT_DIR / "market_level_results.csv", index=False)

    summary_row = {
        "n_markets":              n,
        "n_ai_pos_sharpe":        n_ai_pos_sharpe,
        "n_p05_improved":         n_p05_improved,
        "n_maxdd_improved":       n_maxdd_improved,
        "n_sharpe_improved":      n_sharpe_improved,
        "n_mean_sacrifice":       n_mean_sacrifice,
        "mean_d_sharpe":          round(mean_d_sharpe, 3),
        "mean_d_mean":            round(mean_d_mean,   4),
        "mean_d_p05":             round(mean_d_p05,    4),
        "mean_d_maxdd":           round(mean_d_maxdd,  4),
        "median_d_sharpe":        round(med_d_sharpe,  3),
        "median_d_p05":           round(med_d_p05,     4),
        "median_d_maxdd":         round(med_d_maxdd,   4),
        "p10_d_sharpe":           round(q10, 3),
        "p90_d_sharpe":           round(q90, 3),
        "avg_frac_crash_in_r0":   round(avg_frac_r0,   3),
        "verdict":                verdict,
        "adv_n_p05_improved":     _ADV_N_P05_IMPROVED,
        "adv_mean_d_p05":         _ADV_MEAN_D_P05,
        "adv_mean_d_maxdd":       _ADV_MEAN_D_MAXDD,
    }
    pd.DataFrame([summary_row]).to_csv(OUT_DIR / "summary.csv", index=False)

    with open(OUT_DIR / "summary_print.txt", "w") as f:
        f.write(print_str)

    print(f"[POSDRIFT] Wrote {OUT_DIR}/market_level_results.csv")
    print(f"[POSDRIFT] Wrote {OUT_DIR}/summary.csv")
    print(f"[POSDRIFT] Wrote {OUT_DIR}/summary_print.txt")


if __name__ == "__main__":
    main()
