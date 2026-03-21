"""
OOS Overlay Multi-Baseline Experiment — Leviathan Phase OOS
=============================================================
Tests whether the Leviathan DTI overlay improves downside outcomes across
three independent baselines. Reads frozen dti_cutoff from disk; does NOT
refit any model parameters.

Baselines
---------
1. always_in       : fully invested every quarter
2. valuation_tilt  : invested when DTI ≤ p60(dti, train) — avoids expensive periods
3. trend_tilt      : invested when trailing 4q real price return > 0 — momentum rule

For each baseline, two versions are evaluated:
  raw      : baseline signal only
  overlaid : baseline AND NOT Leviathan gate (zero when regime==0 AND dti > dti_cutoff)

Outputs
-------
  outputs/oos/overlay_multibaseline/comparison.csv
  outputs/oos/overlay_multibaseline/comparison_print.txt

Run:
    python scripts/oos_overlay_multibaseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.path_a.build_dataset import build_master_df
from src.evaluation.backtest import compute_forward_return
from src.evaluation.regime import assign_fragility_regime
from src.backtests.evaluation import summarize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_END  = "2007-12-31"
TEST_START = "2008-03-31"
FROZEN_DIR = Path("outputs/oos/frozen")
OUT_DIR    = Path("outputs/oos/overlay_multibaseline")

_VALUATION_DTI_QUANTILE = 0.60   # train-window DTI cutoff for valuation_tilt
_TREND_LAG              = 4      # quarters of trailing return for trend_tilt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_frozen_cutoff() -> float:
    path = FROZEN_DIR / "dti_cutoff.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen dti_cutoff not found at {path}. Run oos_train.py first."
        )
    with open(path) as f:
        return float(json.load(f)["dti_cutoff"])


def _pct_invested(hold: pd.Series) -> float:
    """Fraction of periods with positive exposure (hold == 1 or True)."""
    return float(hold.astype(float).mean())


def _turnover(hold: pd.Series) -> int:
    """Number of 0→1 or 1→0 transitions."""
    return int((hold.astype(int).diff().abs() > 0).sum())


def _strategy_row(name: str, ret: pd.Series, hold: pd.Series) -> dict:
    s = summarize(ret, name=name)
    s["pct_invested"] = round(_pct_invested(hold), 4)
    s["turnover"]     = _turnover(hold)
    return s


def _leviathan_gate(df: pd.DataFrame, dti_cutoff: float) -> pd.Series:
    """Boolean Series: True = Leviathan permits investment (not in filter zone)."""
    return ~((df["regime"] == 0) & (df["dti"] > dti_cutoff))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load frozen cutoff (firewall: no refitting) -----------------------
    dti_cutoff = _load_frozen_cutoff()
    print(f"[MULTIBASELINE] Loaded frozen dti_cutoff={dti_cutoff:.4f}")

    # --- Build panel --------------------------------------------------------
    df = build_master_df()
    df = compute_forward_return(df)
    df = assign_fragility_regime(df)

    # Training window (for parameter estimation of baselines)
    df_train = df[df["date"] <= pd.Timestamp(TRAIN_END)].copy()

    # Test window — evaluation only (drop last 4 rows: NaN fwd_return from horizon)
    df_test = (
        df[df["date"] >= pd.Timestamp(TEST_START)]
        .dropna(subset=["fwd_return"])
        .copy()
        .reset_index(drop=True)
    )

    # Verify firewall: no test-period data used to parameterise anything below
    assert df_train["date"].max() < pd.Timestamp(TEST_START), "Train/test overlap — abort"

    print(f"[MULTIBASELINE] Test rows: {len(df_test)}")
    print(f"[MULTIBASELINE] Test period: {df_test['date'].min().date()} — "
          f"{df_test['date'].max().date()}")

    # --- Baseline parameter estimation (training window only) ---------------

    # valuation_tilt: DTI threshold = p60 of training DTI
    _val_dti_thr = float(df_train["dti"].quantile(_VALUATION_DTI_QUANTILE))
    print(f"[MULTIBASELINE] valuation_tilt DTI threshold (train p60): {_val_dti_thr:.4f}")

    # --- Build hold series (test period) ------------------------------------

    lev_ok = _leviathan_gate(df_test, dti_cutoff)   # Leviathan gate: True = invest

    # 1. always_in
    hold_ai_raw = pd.Series(True, index=df_test.index)
    hold_ai_ov  = hold_ai_raw & lev_ok

    # 2. valuation_tilt: invest when DTI ≤ train p60 (avoids elevated-DTI periods)
    hold_val_raw = df_test["dti"] <= _val_dti_thr
    hold_val_ov  = hold_val_raw & lev_ok

    # 3. trend_tilt: invest when trailing 4q log return > 0
    #    Computed on the full panel so test rows have a proper lookback.
    log_px = np.log(df["real_price_index"])
    trailing_4q = (log_px - log_px.shift(4)).rename("trailing_4q")
    df_full_with_trend = df.join(trailing_4q, rsuffix="_t")
    trailing_map = df_full_with_trend.set_index("date")["trailing_4q"]
    trend_vals = df_test["date"].map(trailing_map)
    hold_trend_raw = (trend_vals > 0).values
    hold_trend_raw = pd.Series(hold_trend_raw, index=df_test.index)
    hold_trend_ov  = hold_trend_raw & lev_ok

    # Validate no NaN returns
    assert df_test["fwd_return"].notna().all(), "NaN in fwd_return after dropna — unexpected"

    # --- Compute returns and summarize --------------------------------------

    def _ret(hold: pd.Series) -> pd.Series:
        return df_test["fwd_return"] * hold.astype(float)

    rows = [
        _strategy_row("always_in_raw",       _ret(hold_ai_raw),    hold_ai_raw),
        _strategy_row("always_in_overlaid",   _ret(hold_ai_ov),     hold_ai_ov),
        _strategy_row("valuation_raw",        _ret(hold_val_raw),   hold_val_raw),
        _strategy_row("valuation_overlaid",   _ret(hold_val_ov),    hold_val_ov),
        _strategy_row("trend_raw",            _ret(hold_trend_raw), hold_trend_raw),
        _strategy_row("trend_overlaid",       _ret(hold_trend_ov),  hold_trend_ov),
    ]

    df_out = pd.DataFrame(rows)

    # --- Print summary ------------------------------------------------------
    cols_print = ["name", "n", "mean", "sharpe", "p05", "maxdd", "pct_invested", "turnover"]
    print_str = (
        "OOS Overlay Multi-Baseline Comparison\n"
        "======================================\n"
        f"Test period: {df_test['date'].min().date()} — {df_test['date'].max().date()}  "
        f"(n={len(df_test)} quarters)\n"
        f"Frozen dti_cutoff: {dti_cutoff:.4f}\n"
        f"Valuation tilt DTI threshold: {_val_dti_thr:.4f} (train p60)\n\n"
        + df_out[cols_print].to_string(index=False)
        + "\n\n"
        "Overlay lift (overlaid minus raw):\n"
        "-----------------------------------\n"
    )

    pairs = [("always_in_raw", "always_in_overlaid"),
             ("valuation_raw", "valuation_overlaid"),
             ("trend_raw",     "trend_overlaid")]

    lift_rows = []
    for raw_name, ov_name in pairs:
        r_raw = df_out[df_out["name"] == raw_name].iloc[0]
        r_ov  = df_out[df_out["name"] == ov_name].iloc[0]
        label = raw_name.replace("_raw", "")
        lift_rows.append({
            "baseline":    label,
            "Δmean":       round(float(r_ov["mean"])   - float(r_raw["mean"]),   6),
            "Δsharpe":     round(float(r_ov["sharpe"]) - float(r_raw["sharpe"]), 4),
            "Δp05":        round(float(r_ov["p05"])    - float(r_raw["p05"]),    6),
            "Δmaxdd":      round(float(r_ov["maxdd"])  - float(r_raw["maxdd"]),  4),
            "Δpct_invest": round(float(r_ov["pct_invested"]) - float(r_raw["pct_invested"]), 4),
        })

    df_lift = pd.DataFrame(lift_rows)
    print_str += df_lift.to_string(index=False)
    print_str += "\n\nPositive Δ for p05 and Δmaxdd (less negative) → downside improvement.\n"

    # --- Classification verdict --------------------------------------------
    # Rules from experiment spec:
    # overlay → useful if improves p05+maxdd across multiple baselines without destroying mean
    # standalone → improves only always_in
    # alpha-enabling → improves multiple on both downside AND Sharpe with reasonable participation

    n_p05_improved   = (df_lift["Δp05"]   > 0).sum()
    n_maxdd_improved = (df_lift["Δmaxdd"] > 0).sum()
    n_sharpe_pos     = (df_lift["Δsharpe"] > 0).sum()
    n_mean_not_hurt  = (df_lift["Δmean"] > -0.002).sum()   # threshold: <2bp mean drag

    if n_p05_improved >= 2 and n_maxdd_improved >= 2 and n_sharpe_pos >= 2 and n_mean_not_hurt >= 2:
        verdict = "ALPHA-ENABLING"
        verdict_desc = (
            "Leviathan improves both downside metrics AND Sharpe across ≥2 baselines "
            "with tolerable mean drag. The overlay adds signal, not just protection."
        )
    elif n_p05_improved >= 2 and n_maxdd_improved >= 2:
        verdict = "USEFUL OVERLAY"
        verdict_desc = (
            "Leviathan improves downside metrics (p05, maxdd) across ≥2 baselines. "
            "It functions as a risk gate — useful for downside-focused portfolios — "
            "but imposes mean and/or Sharpe cost on momentum or full-exposure strategies."
        )
    elif n_p05_improved <= 1 and n_maxdd_improved <= 1:
        verdict = "STANDALONE FLAG"
        verdict_desc = (
            "Leviathan improves downside for always_in but not across diverse baselines. "
            "It may reflect a feature of the always_in simulation rather than a generalizable "
            "risk indicator."
        )
    else:
        verdict = "MIXED / INCONCLUSIVE"
        verdict_desc = (
            "Results are mixed across baselines. Context-dependent; additional "
            "crash design variants or real data needed."
        )

    print_str += (
        f"\n\nVerdict: {verdict}\n"
        + "-" * (len(verdict) + 9) + "\n"
        + verdict_desc + "\n\n"
        f"  Baselines with p05 improvement  : {n_p05_improved}/3\n"
        f"  Baselines with maxdd improvement: {n_maxdd_improved}/3\n"
        f"  Baselines with Sharpe improvement: {n_sharpe_pos}/3\n"
    )

    print(print_str)

    # --- Write outputs -------------------------------------------------------
    csv_path  = OUT_DIR / "comparison.csv"
    txt_path  = OUT_DIR / "comparison_print.txt"
    lift_path = OUT_DIR / "lift.csv"

    df_out.to_csv(csv_path, index=False)
    df_lift.to_csv(lift_path, index=False)

    with open(txt_path, "w") as f:
        f.write(print_str)

    print(f"\n[MULTIBASELINE] Wrote {csv_path}")
    print(f"[MULTIBASELINE] Wrote {lift_path}")
    print(f"[MULTIBASELINE] Wrote {txt_path}")
    print(f"[MULTIBASELINE] Verdict: {verdict}")


if __name__ == "__main__":
    main()
