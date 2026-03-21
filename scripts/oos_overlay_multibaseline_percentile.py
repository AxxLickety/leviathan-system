"""
OOS Overlay Multi-Baseline Experiment (Percentile-Normalized DTI) — Leviathan Phase OOS
=========================================================================================
Addresses the structural failure of the raw-DTI valuation_tilt baseline in
oos_overlay_multibaseline.py: because DTI is a rising trend (85→135), the training
p60 threshold (101.1) excludes all test-period rows (min test DTI=108.4).

Root cause of expanding-percentile failure (also documented here): the expanding
percentile rank of a monotonically trending series clusters near 1.0 over time — each
new observation is near the historical maximum. With expanding percentile, test dti_pct
ranges 0.70–1.00, so ≤ 0.60 still admits 0 test rows.

Fix: rolling 20-quarter causal percentile rank. Within any 20-quarter window the trend
contribution (~7 DTI points) is smaller than the cross-sectional noise (~8 DTI std), so
the rolling rank has a near-uniform distribution and is a usable signal. Test dti_pct_r20
ranges 0.05–1.00 (mean 0.69); 23/64 test rows fall below 0.60.

Two Leviathan overlay variants are tested:
  lev_abs: frozen absolute dti_cutoff=98.8142 (from oos_train.py; ≡ regime filter in test)
  lev_pct: walk-forward rolling-percentile threshold derived from training only

Three baselines:
  always_in             : fully invested every quarter
  valuation_tilt_roll20 : invest when dti_pct_roll20 ≤ 0.60
  trend_tilt            : invest when trailing 4q log price return > 0

For each baseline × {raw, lev_abs_overlay, lev_pct_overlay} = 9 strategies.

Outputs
-------
  outputs/oos/overlay_multibaseline_percentile/comparison.csv
  outputs/oos/overlay_multibaseline_percentile/comparison_print.txt

Run:
    python scripts/oos_overlay_multibaseline_percentile.py
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
OUT_DIR    = Path("outputs/oos/overlay_multibaseline_percentile")

_ROLL_WINDOW             = 20     # quarters for rolling percentile rank
_VALUATION_PCT_THRESHOLD = 0.60   # invest when dti_pct_roll20 ≤ this

# Walk-forward parameters (percentile version) — mirror oos_train.py structure
_WF_WINDOW_Q     = 20
_WF_QGRID_PCT    = np.arange(0.50, 0.96, 0.05)
_WF_FALLBACK_PCT = 0.80
_WF_MIN_REGIME0  = 10


# ---------------------------------------------------------------------------
# DTI normalization — causal expanding percentile rank
# ---------------------------------------------------------------------------

def compute_expanding_pct_rank(series: pd.Series) -> pd.Series:
    """
    Causal expanding percentile rank.

    For each time t, returns the fraction of values in series[0:t+1] that are
    ≤ series[t]. This is strictly causal: only uses data available up to and
    including time t.

    Returns values in [1/n, 1.0] where n is the number of observations so far.
    """
    arr = series.to_numpy(dtype=float)
    n   = len(arr)
    out = np.empty(n)
    for t in range(n):
        out[t] = np.sum(arr[:t + 1] <= arr[t]) / (t + 1)
    return pd.Series(out, index=series.index, name=series.name + "_pct")


def compute_rolling_pct_rank(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Causal rolling percentile rank over a fixed window.

    For each time t, returns the fraction of values in series[max(0,t-window+1):t+1]
    that are ≤ series[t].
    """
    arr = series.to_numpy(dtype=float)
    n   = len(arr)
    out = np.empty(n)
    for t in range(n):
        start    = max(0, t - window + 1)
        segment  = arr[start: t + 1]
        out[t]   = np.sum(segment <= arr[t]) / len(segment)
    return pd.Series(out, index=series.index, name=series.name + "_roll_pct")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_frozen_cutoff() -> float:
    path = FROZEN_DIR / "dti_cutoff.json"
    if not path.exists():
        raise FileNotFoundError(f"Frozen dti_cutoff not at {path}. Run oos_train.py first.")
    with open(path) as f:
        return float(json.load(f)["dti_cutoff"])


def _pct_invested(hold: pd.Series) -> float:
    return float(hold.astype(float).mean())


def _turnover(hold: pd.Series) -> int:
    return int((hold.astype(int).diff().abs() > 0).sum())


def _strategy_row(name: str, ret: pd.Series, hold: pd.Series) -> dict:
    s = summarize(ret, name=name)
    s["pct_invested"] = round(_pct_invested(hold), 4)
    s["turnover"]     = _turnover(hold)
    return s


def _wf_percentile_cutoff(
    df_wf: pd.DataFrame,
    dti_pct_col: str = "dti_pct",
) -> float:
    """
    Walk-forward search for optimal percentile DTI cutoff on the training window.

    Expanding window; objective = maximize p05 of filtered returns.
    Returns the frozen cutoff (from the final expanding step).
    """
    df_wf = (
        df_wf.dropna(subset=["fwd_return", dti_pct_col, "regime"])
             .sort_values("date")
             .reset_index(drop=True)
    )
    wf_cutoff: float = _WF_FALLBACK_PCT

    for t in range(_WF_WINDOW_Q, len(df_wf)):
        train_wf = df_wf.iloc[:t]
        r0 = train_wf.loc[train_wf["regime"] == 0, dti_pct_col].dropna()

        if len(r0) < _WF_MIN_REGIME0:
            wf_cutoff = _WF_FALLBACK_PCT
            continue

        best_thr = _WF_FALLBACK_PCT
        best_val = -np.inf
        for q in _WF_QGRID_PCT:
            cand = float(q)   # percentile threshold is the quantile level directly
            hold = ~((train_wf["regime"] == 0) & (train_wf[dti_pct_col] > cand))
            val  = float((train_wf["fwd_return"] * hold.astype(float)).quantile(0.05))
            if val > best_val:
                best_val, best_thr = val, cand
        wf_cutoff = best_thr

    return wf_cutoff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Frozen absolute cutoff (from production oos_train.py) -------------
    abs_cutoff = _load_frozen_cutoff()
    print(f"[PCT] Frozen absolute dti_cutoff: {abs_cutoff:.4f}")

    # --- Build full panel ---------------------------------------------------
    df = build_master_df()
    df = compute_forward_return(df)
    df = assign_fragility_regime(df)

    # Rolling 20q causal percentile rank — primary normalization for this experiment
    df["dti_pct_roll"] = compute_rolling_pct_rank(df["dti"], window=_ROLL_WINDOW)

    # Expanding percentile (diagnostic only — degenerate for trending series)
    df["dti_pct_exp"]  = compute_expanding_pct_rank(df["dti"])

    # Train / test split
    df_train = df[df["date"] <= pd.Timestamp(TRAIN_END)].copy()
    df_test  = (
        df[df["date"] >= pd.Timestamp(TEST_START)]
        .dropna(subset=["fwd_return"])
        .copy()
        .reset_index(drop=True)
    )

    assert df_train["date"].max() < pd.Timestamp(TEST_START), "Overlap — abort"

    # --- Walk-forward percentile cutoff on rolling rank (training only) ----
    pct_cutoff = _wf_percentile_cutoff(df_train, dti_pct_col="dti_pct_roll")
    print(f"[PCT] Walk-forward rolling-pct dti_cutoff (train): {pct_cutoff:.4f}")

    # --- Diagnostic: rolling pct distribution in test ----------------------
    print(f"[PCT] Test dti_pct_roll range: "
          f"{df_test['dti_pct_roll'].min():.3f} – {df_test['dti_pct_roll'].max():.3f}  "
          f"(mean={df_test['dti_pct_roll'].mean():.3f})")
    print(f"[PCT] Test expanding pct range: "
          f"{df_test['dti_pct_exp'].min():.3f} – {df_test['dti_pct_exp'].max():.3f}  "
          f"[degenerate — shown for reference]")
    print(f"[PCT] Test rows with dti_pct_roll ≤ {_VALUATION_PCT_THRESHOLD:.2f}: "
          f"{(df_test['dti_pct_roll'] <= _VALUATION_PCT_THRESHOLD).sum()} / {len(df_test)}")
    print(f"[PCT] Test rows with dti_pct_roll > pct_cutoff ({pct_cutoff:.2f}): "
          f"{(df_test['dti_pct_roll'] > pct_cutoff).sum()} / {len(df_test)}")

    # -----------------------------------------------------------------------
    # Gate definitions
    # -----------------------------------------------------------------------

    # Absolute Leviathan gate (same as V1 — frozen production cutoff)
    # Note: abs_cutoff=98.81 < min(test dti)=108.4, so this ≡ (regime != 0) in test
    lev_abs_ok = ~((df_test["regime"] == 0) & (df_test["dti"] > abs_cutoff))
    print(f"[PCT] lev_abs gate: {lev_abs_ok.mean():.3f} fraction invested  "
          f"(DTI condition vacuous — pure regime filter)")

    # Percentile Leviathan gate — rolling 20q pct, walk-forward threshold from training
    lev_pct_ok = ~((df_test["regime"] == 0) & (df_test["dti_pct_roll"] > pct_cutoff))
    print(f"[PCT] lev_pct gate: {lev_pct_ok.mean():.3f} fraction invested  "
          f"(regime==0 AND dti_pct_roll > {pct_cutoff:.2f})")

    # -----------------------------------------------------------------------
    # Baseline hold series (test period)
    # -----------------------------------------------------------------------

    # 1. always_in
    hold_ai = pd.Series(True, index=df_test.index)

    # 2. valuation_tilt_roll20: invest when rolling-20q percentile rank ≤ 0.60
    hold_val_pct = df_test["dti_pct_roll"] <= _VALUATION_PCT_THRESHOLD

    # 3. trend_tilt: invest when trailing 4q log return > 0 (causal lookback)
    log_px = np.log(df["real_price_index"])
    trailing_4q = log_px - log_px.shift(4)
    trailing_map = trailing_4q.set_axis(df["date"]).rename("trailing_4q")
    trend_vals = df_test["date"].map(trailing_map.to_dict())
    hold_trend = (trend_vals > 0).values
    hold_trend = pd.Series(hold_trend, index=df_test.index)

    # -----------------------------------------------------------------------
    # Strategy returns: 3 baselines × {raw, abs_overlay, pct_overlay} = 9
    # -----------------------------------------------------------------------

    def _ret(hold: pd.Series) -> pd.Series:
        return df_test["fwd_return"] * hold.astype(float)

    strategies = [
        # always_in
        ("always_in_raw",             hold_ai,                   hold_ai),
        ("always_in_lev_abs",         hold_ai & lev_abs_ok,      hold_ai & lev_abs_ok),
        ("always_in_lev_pct",         hold_ai & lev_pct_ok,      hold_ai & lev_pct_ok),
        # valuation_tilt_roll20
        ("valuation_roll20_raw",      hold_val_pct,              hold_val_pct),
        ("valuation_roll20_lev_abs",  hold_val_pct & lev_abs_ok, hold_val_pct & lev_abs_ok),
        ("valuation_roll20_lev_pct",  hold_val_pct & lev_pct_ok, hold_val_pct & lev_pct_ok),
        # trend_tilt
        ("trend_raw",                 hold_trend,                hold_trend),
        ("trend_lev_abs",             hold_trend & lev_abs_ok,   hold_trend & lev_abs_ok),
        ("trend_lev_pct",             hold_trend & lev_pct_ok,   hold_trend & lev_pct_ok),
    ]

    rows = [_strategy_row(name, _ret(hold), hold) for name, hold, _ in strategies]
    df_out = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Lift tables (overlay minus raw)
    # -----------------------------------------------------------------------

    def _lift(raw_name: str, ov_name: str, label: str) -> dict:
        r = df_out[df_out["name"] == raw_name].iloc[0]
        o = df_out[df_out["name"] == ov_name].iloc[0]
        return {
            "baseline":    label,
            "Δmean":       round(float(o["mean"])   - float(r["mean"]),   6),
            "Δsharpe":     round(float(o["sharpe"]) - float(r["sharpe"]), 4)
                           if not (np.isnan(float(r["sharpe"])) or np.isnan(float(o["sharpe"])))
                           else float("nan"),
            "Δp05":        round(float(o["p05"])    - float(r["p05"]),    6),
            "Δmaxdd":      round(float(o["maxdd"])  - float(r["maxdd"]),  4),
            "Δpct_invest": round(float(o["pct_invested"]) - float(r["pct_invested"]), 4),
        }

    pairs_abs = [
        ("always_in_raw",        "always_in_lev_abs",        "always_in"),
        ("valuation_roll20_raw", "valuation_roll20_lev_abs", "valuation_roll20"),
        ("trend_raw",            "trend_lev_abs",            "trend"),
    ]
    pairs_pct = [
        ("always_in_raw",        "always_in_lev_pct",        "always_in"),
        ("valuation_roll20_raw", "valuation_roll20_lev_pct", "valuation_roll20"),
        ("trend_raw",            "trend_lev_pct",            "trend"),
    ]

    df_lift_abs = pd.DataFrame([_lift(*p) for p in pairs_abs])
    df_lift_pct = pd.DataFrame([_lift(*p) for p in pairs_pct])

    # -----------------------------------------------------------------------
    # Verdict (applied separately to each overlay variant)
    # -----------------------------------------------------------------------

    def _verdict(df_lift: pd.DataFrame) -> tuple[str, str, dict]:
        valid = df_lift.dropna(subset=["Δsharpe"])
        n_p05  = (df_lift["Δp05"]   > 0).sum()
        n_dd   = (df_lift["Δmaxdd"] > 0).sum()
        n_shp  = (valid["Δsharpe"]  > 0).sum()
        n_mean = (df_lift["Δmean"]  > -0.002).sum()
        if n_p05 >= 2 and n_dd >= 2 and n_shp >= 2 and n_mean >= 2:
            v = "ALPHA-ENABLING"
            d = ("Improves downside AND Sharpe across ≥2 baselines with tolerable mean drag.")
        elif n_p05 >= 2 and n_dd >= 2:
            v = "USEFUL OVERLAY"
            d = ("Improves downside metrics (p05, maxdd) across ≥2 baselines. "
                 "Imposes mean or Sharpe cost on some strategies.")
        elif n_p05 <= 1 and n_dd <= 1:
            v = "STANDALONE FLAG"
            d = ("Improves downside for at most one baseline. "
                 "Does not generalize across strategies.")
        else:
            v = "MIXED / INCONCLUSIVE"
            d = ("Mixed results across baselines.")
        counts = {"n_p05": int(n_p05), "n_dd": int(n_dd), "n_shp": int(n_shp)}
        return v, d, counts

    verdict_abs, desc_abs, cnt_abs = _verdict(df_lift_abs)
    verdict_pct, desc_pct, cnt_pct = _verdict(df_lift_pct)

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------

    cols = ["name", "n", "mean", "sharpe", "p05", "maxdd", "pct_invested", "turnover"]
    print_str = (
        "OOS Overlay Multi-Baseline Experiment — Rolling-Percentile-Normalized DTI\n"
        "===========================================================================\n"
        f"Test period: {df_test['date'].min().date()} — {df_test['date'].max().date()}  "
        f"(n={len(df_test)} quarters)\n"
        f"Frozen absolute dti_cutoff    : {abs_cutoff:.4f}  "
        f"[all test DTI > cutoff → lev_abs ≡ pure regime filter]\n"
        f"Walk-forward roll-pct cutoff  : {pct_cutoff:.4f}  (training only)\n"
        f"Valuation threshold           : dti_pct_roll20 ≤ {_VALUATION_PCT_THRESHOLD:.2f}\n"
        f"Rolling window                : {_ROLL_WINDOW}q\n\n"
        "=== All strategies ===\n"
        + df_out[cols].to_string(index=False)
        + "\n\n"
        "=== Overlay lift: Leviathan ABSOLUTE gate (lev_abs) ===\n"
        + df_lift_abs.to_string(index=False)
        + f"\n  → Verdict: {verdict_abs}\n"
        + f"  → {desc_abs}\n"
        + f"  → p05 improved: {cnt_abs['n_p05']}/3  maxdd: {cnt_abs['n_dd']}/3  "
          f"Sharpe: {cnt_abs['n_shp']}/3\n\n"
        "=== Overlay lift: Leviathan PERCENTILE gate (lev_pct) ===\n"
        + df_lift_pct.to_string(index=False)
        + f"\n  → Verdict: {verdict_pct}\n"
        + f"  → {desc_pct}\n"
        + f"  → p05 improved: {cnt_pct['n_p05']}/3  maxdd: {cnt_pct['n_dd']}/3  "
          f"Sharpe: {cnt_pct['n_shp']}/3\n\n"
        "=== Valuation tilt — train vs test participation ===\n"
    )

    # Participation diagnostics for valuation_tilt_roll20
    train_val_roll = (df_train["dti_pct_roll"] <= _VALUATION_PCT_THRESHOLD).mean()
    test_val_roll  = _pct_invested(hold_val_pct)
    # Expanding pct for comparison
    train_val_exp = (df_train["dti_pct_exp"] <= _VALUATION_PCT_THRESHOLD).mean()
    test_val_exp  = (df_test["dti_pct_exp"] <= _VALUATION_PCT_THRESHOLD).mean()
    print_str += (
        f"  Normalization        | Train invested | Test invested\n"
        f"  ---------------------|----------------|---------------\n"
        f"  Raw DTI (abs p60)    |   1.000 (by def)|   0.000 (degenerate)\n"
        f"  Expanding pct ≤ 0.60 |   {train_val_exp:.3f}        |   {test_val_exp:.3f} (degenerate)\n"
        f"  Rolling-20 pct ≤ 0.60|   {train_val_roll:.3f}        |   {test_val_roll:.3f} (functional)\n"
    )

    print(print_str)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------

    df_out.to_csv(OUT_DIR / "comparison.csv", index=False)
    df_lift_abs.to_csv(OUT_DIR / "lift_abs.csv", index=False)
    df_lift_pct.to_csv(OUT_DIR / "lift_pct.csv", index=False)

    with open(OUT_DIR / "comparison_print.txt", "w") as f:
        f.write(print_str)

    # Save percentile cutoff for reference
    with open(OUT_DIR / "pct_cutoff.json", "w") as f:
        json.dump({
            "wf_pct_cutoff":     pct_cutoff,
            "abs_cutoff":        abs_cutoff,
            "val_pct_threshold": _VALUATION_PCT_THRESHOLD,
            "verdict_abs":       verdict_abs,
            "verdict_pct":       verdict_pct,
        }, f, indent=2)

    print(f"\n[PCT] Wrote {OUT_DIR}/comparison.csv")
    print(f"[PCT] Wrote {OUT_DIR}/comparison_print.txt")
    print(f"[PCT] Verdicts — abs: {verdict_abs}  |  pct: {verdict_pct}")


if __name__ == "__main__":
    main()
