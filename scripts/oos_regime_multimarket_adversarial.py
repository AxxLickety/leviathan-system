"""
OOS Regime Multi-Market Adversarial Evaluation — Leviathan stress test.
========================================================================
Tests the regime-only overlay in a deliberately harder synthetic world:
  - No synchronized GFC (each market has an idiosyncratic major crash)
  - Crashes occur in both regime==0 and regime==1 (regime is informative
    but imperfect: ~60–70% of crash activity in regime==0)
  - Higher background crash noise in all regimes

Model (unchanged from all prior experiments):
    hold[t] = 1  if  regime[t] == 1  (real_rate < 0)
    hold[t] = 0  if  regime[t] == 0  (real_rate >= 0)

OOS split:
    Train: ≤ 2007-12-31
    Test:  ≥ 2008-03-31

Outputs:
    outputs/oos_regime_multimarket_adversarial/market_level_results.csv
    outputs/oos_regime_multimarket_adversarial/summary.csv
    outputs/oos_regime_multimarket_adversarial/summary_print.txt

Run:
    python scripts/oos_regime_multimarket_adversarial.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.research.path_a.build_multimarket_adversarial import build_all_markets_adversarial
from src.evaluation.backtest import compute_forward_return
from src.evaluation.regime import assign_fragility_regime
from src.backtests.evaluation import summarize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OOS_START = "2008-03-31"
_TRAIN_END  = "2007-12-31"
N_MARKETS   = 12

OUT_DIR     = Path("outputs/oos_regime_multimarket_adversarial")

# Reference results from base multi-market world (for decay comparison)
_BASE_N_P05_IMPROVED    = 11
_BASE_N_MAXDD_IMPROVED  = 10
_BASE_N_SHARPE_IMPROVED = 7
_BASE_MEAN_D_P05        = 0.0926
_BASE_MEAN_D_MAXDD      = 0.3433
_BASE_MEAN_D_SHARPE     = 0.140


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

    adv_train = int((train["regime"] == 0).sum())
    adv_test  = int((test["regime"] == 0).sum())

    return {
        "market_id":         market_id,
        "ai_mean":           round(float(ai_s["mean"]),   4),
        "ai_sharpe":         round(float(ai_s["sharpe"]), 3),
        "ai_p05":            round(float(ai_s["p05"]),    4),
        "ai_maxdd":          round(float(ai_s["maxdd"]),  4),
        "ro_mean":           round(float(ro_s["mean"]),   4),
        "ro_sharpe":         round(float(ro_s["sharpe"]), 3),
        "ro_p05":            round(float(ro_s["p05"]),    4),
        "ro_maxdd":          round(float(ro_s["maxdd"]),  4),
        "ro_pct_invested":   round(_pct_invested(hold_ro), 3),
        "ro_turnover":       _turnover(hold_ro),
        "tr_sharpe":         round(float(tr_s["sharpe"]), 3),
        "tr_p05":            round(float(tr_s["p05"]),    4),
        "tr_maxdd":          round(float(tr_s["maxdd"]),  4),
        "tr_pct_invested":   round(_pct_invested(hold_tr), 3),
        "d_sharpe":          round(float(ro_s["sharpe"]) - float(ai_s["sharpe"]), 3),
        "d_p05":             round(float(ro_s["p05"])    - float(ai_s["p05"]),    4),
        "d_maxdd":           round(float(ro_s["maxdd"])  - float(ai_s["maxdd"]),  4),
        "adverse_q_train":   adv_train,
        "adverse_q_test":    adv_test,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[ADVERSARIAL] Building {N_MARKETS} adversarial synthetic markets ...")
    df_all, diag_list = build_all_markets_adversarial(n_markets=N_MARKETS, verbose=True)

    # ------------------------------------------------------------------
    # Per-market evaluation
    # ------------------------------------------------------------------
    market_rows: list[dict] = []
    for mid, grp in df_all.groupby("market_id"):
        row = eval_market(grp.reset_index(drop=True), int(mid))
        if row is not None:
            # Merge crash diagnostics
            diag = diag_list[int(mid)]
            row["regime_switches"]      = diag["regime_switches"]
            row["major_onset_date"]     = diag["major_onset_date"]
            row["major_regime"]         = diag["major_regime"]
            row["crash_q_in_r0"]        = diag["crash_q_in_r0"]
            row["crash_q_in_r1"]        = diag["crash_q_in_r1"]
            row["frac_crash_q_in_r0"]   = diag["frac_crash_q_in_r0"]
            row["frac_r0_in_crash"]     = diag["frac_r0_in_crash"]
            row["frac_r1_in_crash"]     = diag["frac_r1_in_crash"]
            row["n_crash_events"]       = diag["n_crash_events"]
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

    avg_switches     = float(df_markets["regime_switches"].mean())
    avg_adv_test     = float(df_markets["adverse_q_test"].mean())
    avg_frac_r0      = float(df_markets["frac_crash_q_in_r0"].mean())
    avg_frac_r0_in   = float(df_markets["frac_r0_in_crash"].mean())
    avg_frac_r1_in   = float(df_markets["frac_r1_in_crash"].mean())
    n_major_in_r0    = int((df_markets["major_regime"] == "r0").sum())

    # ------------------------------------------------------------------
    # Build print string
    # ------------------------------------------------------------------
    print_str = (
        "OOS Regime Multi-Market Adversarial Evaluation\n"
        "================================================\n"
        f"N markets          : {n}\n"
        f"OOS period         : {_OOS_START}  onward\n"
        f"Model              : regime_only (exit when real_rate >= 0)\n\n"
    )

    # --- Phase 4: World diagnostics ---
    print_str += "=== Phase 4: Adversarial World Diagnostics ===\n"
    print_str += (
        f"  Avg regime switches (full panel) : {avg_switches:.1f}\n"
        f"  Avg adverse (regime-0) test q    : {avg_adv_test:.1f}\n"
        f"  Avg frac crash-quarters in r0    : {avg_frac_r0:.2f}  "
        f"(base world ~0.85–0.90)\n"
        f"  Avg frac of r0-quarters crashing : {avg_frac_r0_in:.2f}\n"
        f"  Avg frac of r1-quarters crashing : {avg_frac_r1_in:.2f}\n"
        f"  Major crash falls in regime-0    : {n_major_in_r0}/{n} markets\n\n"
    )

    diag_cols = [
        "market_id", "major_onset_date", "major_regime",
        "crash_q_in_r0", "crash_q_in_r1",
        "frac_crash_q_in_r0", "frac_r0_in_crash", "frac_r1_in_crash",
        "regime_switches",
    ]
    print_str += df_markets[diag_cols].to_string(index=False) + "\n\n"

    # --- Market-level results ---
    print_str += "=== Market-Level Results (OOS) ===\n"
    result_cols = [
        "market_id",
        "ai_sharpe", "ai_p05", "ai_maxdd",
        "ro_sharpe", "ro_p05", "ro_maxdd",
        "d_sharpe",  "d_p05",  "d_maxdd",
        "ro_pct_invested",
    ]
    print_str += df_markets[result_cols].to_string(index=False) + "\n\n"

    # --- Aggregate ---
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
        f"frac_crash_in_r0={best['frac_crash_q_in_r0']:.2f}\n"
        f"  Worst market (Δsharpe={worst['d_sharpe']:+.3f}): "
        f"market_id={int(worst['market_id'])}  "
        f"frac_crash_in_r0={worst['frac_crash_q_in_r0']:.2f}\n\n"
    )

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

    # --- Decay vs base world ---
    print_str += "=== Decay vs Base Multi-Market World ===\n"
    decay_p05    = n_p05_improved    - _BASE_N_P05_IMPROVED
    decay_maxdd  = n_maxdd_improved  - _BASE_N_MAXDD_IMPROVED
    decay_sharpe = n_sharpe_improved - _BASE_N_SHARPE_IMPROVED
    print_str += (
        f"  p05 improved   : {n_p05_improved}/{n}  "
        f"(was {_BASE_N_P05_IMPROVED}/{n},  Δ={decay_p05:+d})\n"
        f"  maxdd improved : {n_maxdd_improved}/{n}  "
        f"(was {_BASE_N_MAXDD_IMPROVED}/{n},  Δ={decay_maxdd:+d})\n"
        f"  Sharpe improved: {n_sharpe_improved}/{n}  "
        f"(was {_BASE_N_SHARPE_IMPROVED}/{n},  Δ={decay_sharpe:+d})\n\n"
        f"  Mean Δp05  : {mean_d_p05:+.4f}  (was {_BASE_MEAN_D_P05:+.4f},  "
        f"Δ={mean_d_p05-_BASE_MEAN_D_P05:+.4f})\n"
        f"  Mean Δmaxdd: {mean_d_maxdd:+.4f}  (was {_BASE_MEAN_D_MAXDD:+.4f},  "
        f"Δ={mean_d_maxdd-_BASE_MEAN_D_MAXDD:+.4f})\n"
        f"  Mean Δsharpe:{mean_d_sharpe:+.4f}  (was {_BASE_MEAN_D_SHARPE:+.4f},  "
        f"Δ={mean_d_sharpe-_BASE_MEAN_D_SHARPE:+.4f})\n\n"
    )

    # Decision
    p05_frac   = n_p05_improved   / n
    maxdd_frac = n_maxdd_improved / n
    if p05_frac >= 0.75 and maxdd_frac >= 0.75:
        verdict = "SURVIVES — regime-only downside protection robust even in adversarial world"
    elif p05_frac >= 0.50:
        verdict = "DEGRADED BUT POSITIVE — regime-only has residual but weakened downside value"
    else:
        verdict = "COLLAPSES — prior success depended too heavily on favorable crash design"
    print_str += f"  Verdict: {verdict}\n"

    print(print_str)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    df_markets.to_csv(OUT_DIR / "market_level_results.csv", index=False)

    summary_row = {
        "n_markets":              n,
        "n_p05_improved":         n_p05_improved,
        "n_maxdd_improved":       n_maxdd_improved,
        "n_sharpe_improved":      n_sharpe_improved,
        "mean_d_sharpe":          round(mean_d_sharpe,  3),
        "mean_d_p05":             round(mean_d_p05,     4),
        "mean_d_maxdd":           round(mean_d_maxdd,   4),
        "median_d_sharpe":        round(med_d_sharpe,   3),
        "median_d_p05":           round(med_d_p05,      4),
        "median_d_maxdd":         round(med_d_maxdd,    4),
        "p10_d_sharpe":           round(q10, 3),
        "p90_d_sharpe":           round(q90, 3),
        "avg_frac_crash_in_r0":   round(avg_frac_r0,    3),
        "avg_regime_switches":    round(avg_switches,   1),
        "verdict":                verdict,
        # decay vs base
        "base_n_p05_improved":    _BASE_N_P05_IMPROVED,
        "base_n_maxdd_improved":  _BASE_N_MAXDD_IMPROVED,
        "base_mean_d_p05":        _BASE_MEAN_D_P05,
        "base_mean_d_maxdd":      _BASE_MEAN_D_MAXDD,
    }
    pd.DataFrame([summary_row]).to_csv(OUT_DIR / "summary.csv", index=False)

    with open(OUT_DIR / "summary_print.txt", "w") as f:
        f.write(print_str)

    print(f"[ADVERSARIAL] Wrote {OUT_DIR}/market_level_results.csv")
    print(f"[ADVERSARIAL] Wrote {OUT_DIR}/summary.csv")
    print(f"[ADVERSARIAL] Wrote {OUT_DIR}/summary_print.txt")


if __name__ == "__main__":
    main()
