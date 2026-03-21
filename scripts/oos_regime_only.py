"""
OOS Regime-Only Experiment — Leviathan Phase OOS
=================================================
Tests the defensible core of Leviathan: a pure macro-regime overlay that
exits all exposure during adverse rate regimes (regime == 0, positive real
rates) and remains fully invested during accommodative periods (regime == 1).

Background
----------
The full Leviathan hypothesis was: regime==0 AND high DTI jointly predict
elevated crash risk. The DTI dimension was rejected after two experiments:

  1. Absolute DTI (oos_train.py): walk-forward cutoff (98.81) was exceeded
     by all 64 test observations → DTI condition vacuous → gate ≡ regime filter.

  2. Rolling-percentile DTI (oos_train_pct.py): walk-forward collapsed to
     boundary threshold → gate marginally weaker than pure regime exit.
     OOS overlay underperformed regime_only by Δsharpe=−0.24.

Surviving core: regime==0 → exit. No DTI. No learned parameters.

Rule
----
    hold[t] = 1  if  regime[t] == 1  (accommodative: real_rate < 0)
    hold[t] = 0  if  regime[t] == 0  (stressed: real_rate >= 0)

Baselines
---------
  always_in          — fully invested every quarter
  regime_only        — hold when regime == 1
  trend_raw          — hold when trailing 4q log return > 0   (momentum baseline)
  trend_regime       — hold when regime == 1 AND trailing 4q > 0
                       (best non-degenerate baseline from prior multibaseline work)

Stability checks
----------------
  3 train-start dates × 3 OOS-start dates = 9 configurations
  Reports (sharpe, p05, maxdd) for always_in, regime_only, trend_regime.

Outputs
-------
  outputs/oos_regime_only/comparison.csv
  outputs/oos_regime_only/comparison_print.txt
  outputs/oos_regime_only/stability.csv

Run:
    python scripts/oos_regime_only.py
"""
from __future__ import annotations

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

_CANONICAL_TRAIN_START = "1992-01-01"
_CANONICAL_OOS_START   = "2008-03-31"
_CANONICAL_TRAIN_END   = "2007-12-31"

OUT_DIR = Path("outputs/oos_regime_only")

# Stability grid
_TRAIN_STARTS = ["1992-01-01", "1995-01-01", "1999-01-01"]
_OOS_STARTS   = ["2006-03-31", "2008-03-31", "2010-03-31"]


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def _pct_invested(hold: pd.Series) -> float:
    return float(hold.astype(float).mean())


def _turnover(hold: pd.Series) -> int:
    return int((hold.astype(int).diff().abs() > 0).sum())


def _row(ret: pd.Series, hold: pd.Series, name: str) -> dict:
    s = summarize(ret * hold.astype(float), name=name)
    s["pct_invested"] = round(_pct_invested(hold), 4)
    s["turnover"]     = _turnover(hold)
    return s


# ---------------------------------------------------------------------------
# Build the panel (called once; panels with different start dates are sliced)
# ---------------------------------------------------------------------------

def _build_full_panel() -> pd.DataFrame:
    """Build and augment the canonical panel."""
    df = build_master_df(start=_CANONICAL_TRAIN_START)
    df = compute_forward_return(df)
    df = assign_fragility_regime(df)
    # Trailing 4q log return for trend_tilt (causal — only uses past prices)
    log_px          = np.log(df["real_price_index"])
    df["trail_4q"]  = log_px - log_px.shift(4)
    return df


# ---------------------------------------------------------------------------
# Evaluate one OOS split (returns list of metric dicts)
# ---------------------------------------------------------------------------

def _eval_split(df_full: pd.DataFrame, oos_start: str) -> list[dict]:
    test = (
        df_full[df_full["date"] >= pd.Timestamp(oos_start)]
        .dropna(subset=["fwd_return"])
        .copy()
        .reset_index(drop=True)
    )
    ret = test["fwd_return"]

    hold_ai     = pd.Series(True,  index=test.index)
    hold_ro     = test["regime"] == 1
    hold_trend  = test["trail_4q"] > 0
    hold_tr     = hold_ro & hold_trend

    return [
        _row(ret, hold_ai,    "always_in"),
        _row(ret, hold_ro,    "regime_only"),
        _row(ret, hold_trend, "trend_raw"),
        _row(ret, hold_tr,    "trend_regime"),
    ]


# ---------------------------------------------------------------------------
# Stability grid (build panels for each train_start, evaluate at each oos_start)
# ---------------------------------------------------------------------------

def _stability_grid() -> pd.DataFrame:
    rows = []
    for t_start in _TRAIN_STARTS:
        df_full = build_master_df(start=t_start)
        df_full = compute_forward_return(df_full)
        df_full = assign_fragility_regime(df_full)
        log_px         = np.log(df_full["real_price_index"])
        df_full["trail_4q"] = log_px - log_px.shift(4)

        for oos_s in _OOS_STARTS:
            test = (
                df_full[df_full["date"] >= pd.Timestamp(oos_s)]
                .dropna(subset=["fwd_return"])
                .copy()
                .reset_index(drop=True)
            )
            if len(test) < 10:
                continue
            ret      = test["fwd_return"]
            hold_ro  = test["regime"] == 1
            hold_tr  = hold_ro & (test["trail_4q"] > 0)

            def _m(hold, name):
                r = ret * hold.astype(float)
                eq = np.exp(np.cumsum(r))
                dd = float((eq / eq.cummax() - 1).min())
                sh = float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else float("nan")
                return {
                    "train_start": t_start[:4],
                    "oos_start":   oos_s[:4],
                    "strategy":    name,
                    "n":           len(r),
                    "sharpe":      round(sh, 3),
                    "p05":         round(float(r.quantile(0.05)), 4),
                    "maxdd":       round(dd, 4),
                    "pct_inv":     round(float(hold.astype(float).mean()), 3),
                }

            rows += [
                _m(pd.Series(True, index=test.index), "always_in"),
                _m(hold_ro,                            "regime_only"),
                _m(hold_tr,                            "trend_regime"),
            ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Build canonical panel ----------------------------------------------
    df = _build_full_panel()

    # Canonical OOS split
    rows_main = _eval_split(df, _CANONICAL_OOS_START)
    df_main   = pd.DataFrame(rows_main)

    cols = ["name", "n", "mean", "sharpe", "p05", "maxdd", "pct_invested", "turnover"]
    print_str = (
        "OOS Regime-Only Experiment\n"
        "==========================\n"
        f"Canonical OOS: {_CANONICAL_OOS_START}  "
        f"(train ≤ {_CANONICAL_TRAIN_END})\n"
        f"Panel rows: {len(df)}  "
        f"({df['date'].min().date()} – {df['date'].max().date()})\n"
        f"Regime-0 (adverse) quarters in test: "
        f"{(df[df['date'] >= pd.Timestamp(_CANONICAL_OOS_START)]['regime'] == 0).sum()}"
        f" / {(df['date'] >= pd.Timestamp(_CANONICAL_OOS_START)).sum()}\n\n"
        "=== Canonical strategies ===\n"
        + df_main[cols].to_string(index=False)
    )

    # Regime overlay lift vs always_in
    ai_row = df_main[df_main["name"] == "always_in"].iloc[0]
    ro_row = df_main[df_main["name"] == "regime_only"].iloc[0]
    tr_row = df_main[df_main["name"] == "trend_regime"].iloc[0]
    tn_row = df_main[df_main["name"] == "trend_raw"].iloc[0]

    print_str += "\n\n=== Overlay lift (vs always_in) ===\n"
    for row, label in [(ro_row, "regime_only"), (tn_row, "trend_raw"), (tr_row, "trend_regime")]:
        dm = float(row["mean"])  - float(ai_row["mean"])
        ds = float(row["sharpe"])- float(ai_row["sharpe"])
        dp = float(row["p05"])   - float(ai_row["p05"])
        dd = float(row["maxdd"]) - float(ai_row["maxdd"])
        print_str += (
            f"  {label:22s}  Δmean={dm:+.4f}  Δsharpe={ds:+.3f}  "
            f"Δp05={dp:+.4f}  Δmaxdd={dd:+.4f}\n"
        )

    # Regime overlay lift: trend_regime vs trend_raw
    dm2 = float(tr_row["mean"])  - float(tn_row["mean"])
    ds2 = float(tr_row["sharpe"])- float(tn_row["sharpe"])
    dp2 = float(tr_row["p05"])   - float(tn_row["p05"])
    dd2 = float(tr_row["maxdd"]) - float(tn_row["maxdd"])
    print_str += (
        f"\n  Overlay lift on trend_raw:\n"
        f"  {'trend_regime vs trend_raw':22s}  Δmean={dm2:+.4f}  Δsharpe={ds2:+.3f}  "
        f"Δp05={dp2:+.4f}  Δmaxdd={dd2:+.4f}\n"
    )

    # --- Stability grid -----------------------------------------------------
    df_stab = _stability_grid()

    # Pivot to wide format: one row per (train_start, oos_start), columns per strategy
    stab_wide_rows = []
    for (ts, os), grp in df_stab.groupby(["train_start", "oos_start"]):
        base = {"train_start": ts, "oos_start": os}
        for _, r in grp.iterrows():
            st = r["strategy"]
            base[f"{st}_sharpe"] = r["sharpe"]
            base[f"{st}_p05"]    = r["p05"]
            base[f"{st}_maxdd"]  = r["maxdd"]
            base[f"{st}_pct"]    = r["pct_inv"]
        stab_wide_rows.append(base)
    df_stab_wide = pd.DataFrame(stab_wide_rows)

    print_str += (
        "\n\n=== Stability check: regime_only across 9 configurations ===\n"
        "(train_start × oos_start, key metrics for always_in / regime_only / trend_regime)\n\n"
    )
    for _, r in df_stab_wide.iterrows():
        print_str += (
            f"  train={r['train_start']}  oos={r['oos_start']}  │"
            f"  ai_sharpe={r.get('always_in_sharpe',float('nan')):+.3f}"
            f"  ai_maxdd={r.get('always_in_maxdd',float('nan')):+.4f}  │"
            f"  ro_sharpe={r.get('regime_only_sharpe',float('nan')):+.3f}"
            f"  ro_p05={r.get('regime_only_p05',float('nan')):+.4f}"
            f"  ro_maxdd={r.get('regime_only_maxdd',float('nan')):+.4f}  │"
            f"  tr_sharpe={r.get('trend_regime_sharpe',float('nan')):+.3f}"
            f"  tr_maxdd={r.get('trend_regime_maxdd',float('nan')):+.4f}\n"
        )

    # Stability summary counts
    n_ro_sharpe_pos = (df_stab[df_stab["strategy"] == "regime_only"]["sharpe"] > 0).sum()
    n_ro_p05_pos    = (
        df_stab[df_stab["strategy"] == "regime_only"]["p05"]
        > df_stab[df_stab["strategy"] == "always_in"]["p05"].values
    ).sum()

    print_str += (
        f"\n  regime_only Sharpe > 0 in {n_ro_sharpe_pos}/{len(_TRAIN_STARTS)*len(_OOS_STARTS)} configs\n"
        f"  regime_only p05 improves vs always_in in "
        f"{(df_stab[df_stab['strategy']=='regime_only']['p05'].values > df_stab[df_stab['strategy']=='always_in']['p05'].values).sum()}"
        f"/{len(_TRAIN_STARTS)*len(_OOS_STARTS)} configs\n"
    )

    print(print_str)

    # --- Write outputs -------------------------------------------------------
    df_main.to_csv(OUT_DIR / "comparison.csv", index=False)
    df_stab.to_csv(OUT_DIR / "stability.csv",  index=False)
    with open(OUT_DIR / "comparison_print.txt", "w") as f:
        f.write(print_str)

    print(f"\n[REGIME_ONLY] Wrote {OUT_DIR}/comparison.csv")
    print(f"[REGIME_ONLY] Wrote {OUT_DIR}/stability.csv")
    print(f"[REGIME_ONLY] Wrote {OUT_DIR}/comparison_print.txt")


if __name__ == "__main__":
    main()
