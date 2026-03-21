"""
OOS Regime Policy Experiment — Leviathan activation policy comparison.
=======================================================================
Tests five activation policies using the unchanged regime-only Leviathan
signal, evaluated in the positive-drift adversarial multi-market world
(12 markets, always_in Sharpe > 0 in all markets).

Signal is fixed throughout: regime = (real_rate >= 0) → adverse.
Only the ACTIVATION POLICY changes — how the signal translates to exposure.

Policies
--------
1. always_in            — 100% exposure always (benchmark)
2. full_exit            — 100% → 0% when adverse (the prior regime_only)
3. partial_derisk_50    — 100% → 50% when adverse
4. persistent_2q        — exit only after 2 consecutive adverse quarters
5. persistent_3q        — exit only after 3 consecutive adverse quarters
6. partial_persistent   — graduated: 1 adverse → 50%, 2+ adverse → 0%

Key question:
    Does tuning the activation policy recover meaningful Sharpe relative to
    full_exit while retaining most of the p05 / maxdd protection?

Attribution framework:
    For each policy, compute:
      p05_retention_pct  = policy Δp05  / full_exit Δp05  × 100
      cost_pct           = policy Δmean / full_exit Δmean × 100
    A policy is efficient if p05_retention is high and cost_pct is low.

Outputs:
    outputs/oos_regime_policy/market_level_results.csv
    outputs/oos_regime_policy/summary.csv
    outputs/oos_regime_policy/summary_print.txt

Run:
    python scripts/oos_regime_policy_experiment.py
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

OUT_DIR     = Path("outputs/oos_regime_policy")

POLICIES = [
    "always_in",
    "full_exit",
    "partial_derisk_50",
    "persistent_2q",
    "persistent_3q",
    "partial_persistent",
]

POLICY_LABELS = {
    "always_in":          "always_in          (100% always)",
    "full_exit":          "full_exit          (0% when r==0)",
    "partial_derisk_50":  "partial_50         (50% when r==0)",
    "persistent_2q":      "persistent_2q      (exit after 2 consec r==0)",
    "persistent_3q":      "persistent_3q      (exit after 3 consec r==0)",
    "partial_persistent": "partial_persistent (50% after 1q, 0% after 2q)",
}


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

def _apply_policy(ret: np.ndarray, regime: np.ndarray, policy: str) -> np.ndarray:
    """Apply an activation policy to a return series and regime array.

    Returns the held return series (ret * exposure).
    """
    n = len(ret)
    if policy == "always_in":
        exposure = np.ones(n)

    elif policy == "full_exit":
        exposure = (regime == 1).astype(float)

    elif policy == "partial_derisk_50":
        exposure = np.where(regime == 0, 0.5, 1.0)

    elif policy == "persistent_2q":
        # Exit only after adverse regime has persisted for 2 consecutive quarters.
        # First adverse quarter: stay in. Second+ consecutive: exit.
        exposure = np.ones(n)
        consec   = 0
        for t in range(n):
            if regime[t] == 0:
                consec += 1
            else:
                consec = 0
            if consec >= 2:
                exposure[t] = 0.0

    elif policy == "persistent_3q":
        # Exit only after 3 consecutive adverse quarters.
        exposure = np.ones(n)
        consec   = 0
        for t in range(n):
            if regime[t] == 0:
                consec += 1
            else:
                consec = 0
            if consec >= 3:
                exposure[t] = 0.0

    elif policy == "partial_persistent":
        # Graduated derisking:
        #   1 consecutive adverse quarter  → reduce to 50%
        #   2+ consecutive adverse quarters → reduce to 0%
        # On re-entry (regime turns 1): immediate full re-entry.
        exposure = np.ones(n)
        consec   = 0
        for t in range(n):
            if regime[t] == 0:
                consec += 1
            else:
                consec = 0
            if consec >= 2:
                exposure[t] = 0.0
            elif consec == 1:
                exposure[t] = 0.5

    else:
        raise ValueError(f"Unknown policy: {policy}")

    return ret * exposure


def _exposure_series(regime: np.ndarray, policy: str) -> np.ndarray:
    """Return the exposure fraction at each quarter (for pct_invested / turnover)."""
    n = len(regime)
    if policy == "always_in":
        return np.ones(n)
    elif policy == "full_exit":
        return (regime == 1).astype(float)
    elif policy == "partial_derisk_50":
        return np.where(regime == 0, 0.5, 1.0)
    elif policy in ("persistent_2q", "persistent_3q", "partial_persistent"):
        # Derive from dummy return = ones
        dummy = np.ones(n)
        held  = _apply_policy(dummy, regime, policy)
        return held   # exposure = held/dummy = held since dummy=1
    else:
        raise ValueError(f"Unknown policy: {policy}")


def _pct_invested(exp: np.ndarray) -> float:
    return float(exp.mean())


def _turnover(exp: np.ndarray) -> int:
    """Count number of quarters where exposure fraction changes."""
    return int((np.abs(np.diff(exp)) > 1e-9).sum())


# ---------------------------------------------------------------------------
# Per-market evaluation
# ---------------------------------------------------------------------------

def eval_market_policies(
    df_mkt: pd.DataFrame,
    market_id: int,
) -> list[dict] | None:
    """Evaluate all policies for one market. Returns list of metric dicts."""
    df = compute_forward_return(df_mkt.copy())
    df = assign_fragility_regime(df)

    test = (
        df[df["date"] >= pd.Timestamp(_OOS_START)]
        .dropna(subset=["fwd_return"])
        .copy()
        .reset_index(drop=True)
    )

    if len(test) < 10:
        return None

    ret    = test["fwd_return"].values
    regime = test["regime"].values

    rows = []
    for policy in POLICIES:
        held = _apply_policy(ret, regime, policy)
        exp  = _exposure_series(regime, policy)

        s = summarize(pd.Series(held), name=policy)
        rows.append({
            "market_id":    market_id,
            "policy":       policy,
            "mean":         round(float(s["mean"]),   4),
            "vol":          round(float(s["vol"]),    4),
            "sharpe":       round(float(s["sharpe"]), 3),
            "p05":          round(float(s["p05"]),    4),
            "maxdd":        round(float(s["maxdd"]),  4),
            "pct_invested": round(_pct_invested(exp), 3),
            "turnover":     _turnover(exp),
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[POLICY] Building {N_MARKETS} positive-drift markets ...")
    df_all, _ = build_all_markets_posdrift(n_markets=N_MARKETS, verbose=False)

    all_rows: list[dict] = []
    for mid, grp in df_all.groupby("market_id"):
        rows = eval_market_policies(grp.reset_index(drop=True), int(mid))
        if rows:
            all_rows.extend(rows)

    df_all_results = pd.DataFrame(all_rows)

    # ------------------------------------------------------------------
    # Build per-policy summary (vs always_in)
    # ------------------------------------------------------------------
    # For each market, get always_in metrics as reference
    ai_ref = (
        df_all_results[df_all_results["policy"] == "always_in"]
        [["market_id", "mean", "sharpe", "p05", "maxdd"]]
        .rename(columns={"mean": "ai_mean", "sharpe": "ai_sharpe",
                         "p05": "ai_p05",   "maxdd": "ai_maxdd"})
    )
    df_merged = df_all_results.merge(ai_ref, on="market_id")
    df_merged["d_mean"]   = df_merged["mean"]   - df_merged["ai_mean"]
    df_merged["d_sharpe"] = df_merged["sharpe"] - df_merged["ai_sharpe"]
    df_merged["d_p05"]    = df_merged["p05"]    - df_merged["ai_p05"]
    df_merged["d_maxdd"]  = df_merged["maxdd"]  - df_merged["ai_maxdd"]

    # Policy-level aggregates
    policy_summary_rows = []
    for policy in POLICIES:
        sub = df_merged[df_merged["policy"] == policy]
        n   = len(sub)
        row = {
            "policy":            policy,
            "n":                 n,
            "mean_pct_invested": round(float(sub["pct_invested"].mean()), 3),
            "mean_turnover":     round(float(sub["turnover"].mean()),     1),
            "mean_sharpe":       round(float(sub["sharpe"].mean()),       3),
            "mean_p05":          round(float(sub["p05"].mean()),          4),
            "mean_maxdd":        round(float(sub["maxdd"].mean()),        4),
            "n_p05_improved":    int((sub["d_p05"]    > 0).sum()),
            "n_maxdd_improved":  int((sub["d_maxdd"]  > 0).sum()),
            "n_sharpe_improved": int((sub["d_sharpe"] > 0).sum()),
            "mean_d_mean":       round(float(sub["d_mean"].mean()),       4),
            "mean_d_sharpe":     round(float(sub["d_sharpe"].mean()),     3),
            "mean_d_p05":        round(float(sub["d_p05"].mean()),        4),
            "mean_d_maxdd":      round(float(sub["d_maxdd"].mean()),      4),
        }
        policy_summary_rows.append(row)

    df_summary = pd.DataFrame(policy_summary_rows)

    # Attribution framing: relative to full_exit
    fe = df_summary[df_summary["policy"] == "full_exit"].iloc[0]
    fe_d_p05    = float(fe["mean_d_p05"])
    fe_d_mean   = float(fe["mean_d_mean"])
    fe_d_maxdd  = float(fe["mean_d_maxdd"])

    df_summary["p05_retention_pct"]  = (
        df_summary["mean_d_p05"]  / fe_d_p05  * 100
    ).round(1)
    df_summary["maxdd_retention_pct"] = (
        df_summary["mean_d_maxdd"] / fe_d_maxdd * 100
    ).round(1)
    df_summary["mean_cost_pct"] = (
        df_summary["mean_d_mean"] / fe_d_mean  * 100
    ).round(1)

    # ------------------------------------------------------------------
    # Build print string
    # ------------------------------------------------------------------
    print_str = (
        "OOS Regime Policy Experiment\n"
        "==============================\n"
        f"N markets : {N_MARKETS}\n"
        f"World     : positive-drift adversarial  "
        f"(always_in Sharpe > 0 in all markets)\n"
        f"Signal    : regime_only — unchanged throughout\n\n"
    )

    # --- Policy comparison table ---
    print_str += "=== Policy Comparison (aggregated over all markets) ===\n\n"
    header = (
        f"{'Policy':<28}  "
        f"{'pct_inv':>7}  "
        f"{'turnover':>8}  "
        f"{'mean_sh':>7}  "
        f"{'Δmean':>7}  "
        f"{'Δsharpe':>7}  "
        f"{'Δp05':>7}  "
        f"{'Δmaxdd':>8}  "
        f"{'p05_ret%':>8}  "
        f"{'cost%':>6}\n"
    )
    print_str += header
    print_str += "-" * len(header) + "\n"

    for _, row in df_summary.iterrows():
        p05_ret  = row["p05_retention_pct"]
        cost_pct = row["mean_cost_pct"]
        if row["policy"] == "always_in":
            p05_ret_str  = "  ref  "
            cost_pct_str = " ref  "
        else:
            p05_ret_str  = f"{p05_ret:+7.1f}%"
            cost_pct_str = f"{cost_pct:5.1f}%"
        print_str += (
            f"  {POLICY_LABELS[row['policy']]:<26}  "
            f"{row['mean_pct_invested']:>7.3f}  "
            f"{row['mean_turnover']:>8.1f}  "
            f"{row['mean_sharpe']:>+7.3f}  "
            f"{row['mean_d_mean']:>+7.4f}  "
            f"{row['mean_d_sharpe']:>+7.3f}  "
            f"{row['mean_d_p05']:>+7.4f}  "
            f"{row['mean_d_maxdd']:>+8.4f}  "
            f"{p05_ret_str}  "
            f"{cost_pct_str}\n"
        )

    print_str += "\n"

    # --- Win-rate table ---
    print_str += "=== Win Rates vs always_in ===\n"
    print_str += (
        f"  {'Policy':<26}  "
        f"{'p05_impr':>8}  "
        f"{'maxdd_impr':>10}  "
        f"{'sharpe_impr':>11}\n"
    )
    for _, row in df_summary.iterrows():
        print_str += (
            f"  {row['policy']:<26}  "
            f"{row['n_p05_improved']:>3}/{N_MARKETS:>2}        "
            f"{row['n_maxdd_improved']:>3}/{N_MARKETS:>2}          "
            f"{row['n_sharpe_improved']:>3}/{N_MARKETS:>2}\n"
        )
    print_str += "\n"

    # --- Per-market detail for key policies ---
    print_str += "=== Per-Market: full_exit vs partial_50 vs partial_persistent ===\n"
    key_policies = ["always_in", "full_exit", "partial_derisk_50", "partial_persistent"]
    df_key = df_merged[df_merged["policy"].isin(key_policies)].copy()
    pivot_cols = ["market_id", "policy", "sharpe", "p05", "maxdd",
                  "d_sharpe", "d_p05", "d_maxdd", "pct_invested"]
    print_str += (
        df_key[pivot_cols]
        .sort_values(["market_id", "policy"])
        .to_string(index=False)
        + "\n\n"
    )

    # --- Attribution ratios ---
    print_str += "=== Attribution: What fraction of full_exit's effect does each policy retain? ===\n"
    print_str += (
        f"  full_exit reference:  mean Δp05={fe_d_p05:+.4f}  "
        f"mean Δmaxdd={fe_d_maxdd:+.4f}  "
        f"mean Δmean={fe_d_mean:+.4f}\n\n"
        f"  {'Policy':<26}  "
        f"{'p05_retained%':>14}  "
        f"{'maxdd_retained%':>16}  "
        f"{'mean_cost%':>11}\n"
    )
    for _, row in df_summary.iterrows():
        if row["policy"] == "always_in":
            continue
        print_str += (
            f"  {row['policy']:<26}  "
            f"{row['p05_retention_pct']:>13.1f}%  "
            f"{row['maxdd_retention_pct']:>15.1f}%  "
            f"{row['mean_cost_pct']:>10.1f}%\n"
        )
    print_str += "\n"

    # --- Verdict ---
    # Find the most efficient policy (highest p05_retention / |cost|)
    non_ai = df_summary[df_summary["policy"] != "always_in"].copy()
    non_ai["efficiency"] = (
        non_ai["p05_retention_pct"].clip(lower=0)
        / non_ai["mean_cost_pct"].clip(lower=1e-6)
    )
    best_eff_row  = non_ai.loc[non_ai["efficiency"].idxmax()]
    best_p05_row  = non_ai.loc[non_ai["mean_d_p05"].idxmax()]
    best_sh_row   = non_ai.loc[non_ai["mean_d_sharpe"].idxmax()]

    print_str += "=== Policy Recommendations ===\n"
    print_str += (
        f"  Best downside protection (p05): {best_p05_row['policy']}\n"
        f"  Best Sharpe vs always_in      : {best_sh_row['policy']}\n"
        f"  Best efficiency ratio         : {best_eff_row['policy']}  "
        f"(p05_retention={best_eff_row['p05_retention_pct']:.1f}%  "
        f"at cost={best_eff_row['mean_cost_pct']:.1f}% of full_exit's mean drag)\n\n"
    )

    fe_full = df_summary[df_summary["policy"] == "full_exit"].iloc[0]
    pp_full = df_summary[df_summary["policy"] == "partial_persistent"].iloc[0]
    p50_full = df_summary[df_summary["policy"] == "partial_derisk_50"].iloc[0]

    print_str += (
        f"  full_exit vs partial_persistent:\n"
        f"    p05 protection: {fe_full['mean_d_p05']:+.4f} vs {pp_full['mean_d_p05']:+.4f}  "
        f"(retained: {pp_full['p05_retention_pct']:.1f}%)\n"
        f"    mean cost:     {fe_full['mean_d_mean']:+.4f} vs {pp_full['mean_d_mean']:+.4f}  "
        f"(cost retained: {pp_full['mean_cost_pct']:.1f}%)\n"
        f"    Sharpe delta:  {fe_full['mean_d_sharpe']:+.3f} vs {pp_full['mean_d_sharpe']:+.3f}\n\n"
        f"  full_exit vs partial_derisk_50:\n"
        f"    p05 protection: {fe_full['mean_d_p05']:+.4f} vs {p50_full['mean_d_p05']:+.4f}  "
        f"(retained: {p50_full['p05_retention_pct']:.1f}%)\n"
        f"    mean cost:     {fe_full['mean_d_mean']:+.4f} vs {p50_full['mean_d_mean']:+.4f}  "
        f"(cost retained: {p50_full['mean_cost_pct']:.1f}%)\n"
        f"    Sharpe delta:  {fe_full['mean_d_sharpe']:+.3f} vs {p50_full['mean_d_sharpe']:+.3f}\n"
    )

    print(print_str)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    df_all_results.to_csv(OUT_DIR / "market_level_results.csv", index=False)
    df_summary.to_csv(OUT_DIR / "summary.csv",                  index=False)
    with open(OUT_DIR / "summary_print.txt", "w") as f:
        f.write(print_str)

    print(f"\n[POLICY] Wrote {OUT_DIR}/market_level_results.csv")
    print(f"[POLICY] Wrote {OUT_DIR}/summary.csv")
    print(f"[POLICY] Wrote {OUT_DIR}/summary_print.txt")


if __name__ == "__main__":
    main()
