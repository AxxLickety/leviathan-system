from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

SANITY = "outputs/phase2/phase2_sanity.csv"
THRESH = "outputs/path_a/thresholds.csv"

REGIME_FOR_RISK = 0
PROB_FOR_THRESHOLD = 0.10  # match Phase 1 choice

def load_threshold() -> float:
    t = pd.read_csv(THRESH)
    sub = t[(t["regime"] == REGIME_FOR_RISK) & (np.isclose(t["prob"], PROB_FOR_THRESHOLD))]
    if len(sub) == 1:
        return float(sub["dti_threshold"].iloc[0])

    sub2 = t[t["regime"] == REGIME_FOR_RISK].copy()
    sub2["prob_dist"] = (sub2["prob"] - PROB_FOR_THRESHOLD).abs()
    sub2 = sub2.sort_values("prob_dist").head(1)
    return float(sub2["dti_threshold"].iloc[0])

def max_drawdown(log_r: pd.Series) -> float:
    x = log_r.dropna()
    if len(x) == 0:
        return np.nan
    eq = np.exp(x.cumsum())
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    return float(dd.min())

def summarize_strategy(log_r: pd.Series) -> dict:
    x = log_r.dropna()
    if len(x) == 0:
        return {"n": 0, "mean": np.nan, "vol": np.nan, "sharpe": np.nan, "p05": np.nan, "maxdd": np.nan}
    mean = x.mean()
    vol = x.std(ddof=1)
    sharpe = mean / vol if vol > 0 else np.nan
    return {
        "n": int(len(x)),
        "mean": float(mean),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "p05": float(x.quantile(0.05)),
        "maxdd": max_drawdown(x),
    }

def main():
    df = pd.read_csv(SANITY, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    if "real_price_index" not in df.columns:
        raise ValueError("phase2_sanity.csv must include real_price_index")

    threshold_t = load_threshold()
    print(f"[ok] threshold_t={threshold_t:.4f} (regime={REGIME_FOR_RISK}, prob={PROB_FOR_THRESHOLD})")

    # Phase1 risk flag (decision at time t)
    phase1_risk = (df["regime"] == REGIME_FOR_RISK) & (df["dti"] > threshold_t)

    # realized 1Q forward log return: ret(t->t+1)
    lp = np.log(pd.to_numeric(df["real_price_index"], errors="coerce"))
    df["ret_1q_fwd"] = lp.shift(-1) - lp

    # Strategy P1: invested next quarter if NOT phase1 risk at time t
    df["strat_p1"] = np.where(~phase1_risk, df["ret_1q_fwd"], 0.0)

    # Strategy P2: obey hold_unless expression from YAML
    cfg = yaml.safe_load(open("config/phase2/supply_gate.yaml"))
    expr = cfg["rule"]["hold_unless"]
    print("[ok] hold_unless:", expr)

    invested = df.eval(expr).fillna(False)
    df["invested_p2"] = invested.astype(int)
    df["strat_p2"] = df["ret_1q_fwd"].where(invested, 0.0)

    # Save series for inspection
    out = df[["date","ret_1q_fwd","strat_p1","strat_p2","invested_p2","dti","regime","supply_high"]].copy()
    out.to_csv("outputs/phase2/phase2_strategy_series.csv", index=False)
    print("[phase2] wrote outputs/phase2/phase2_strategy_series.csv")

    # Summaries
    res = pd.DataFrame([
        {"strategy":"baseline_always_in", **summarize_strategy(df["ret_1q_fwd"])},
        {"strategy":"phase1_gate", **summarize_strategy(df["strat_p1"])},
        {"strategy":"phase2_gate", **summarize_strategy(df["strat_p2"])},
    ])
    print(res.to_string(index=False))

    res.to_csv("outputs/phase2/phase2_strategy_summary.csv", index=False)
    print("[phase2] wrote outputs/phase2/phase2_strategy_summary.csv")

    # Diagnostics
    print("\n[diag] counts:")
    print("phase1_risk:", int(phase1_risk.sum()))
    print("phase2_not_invested:", int((~invested).sum()))
    print("supply_high:", int((df["supply_high"]==1).sum()))
    print("non_nan_ret_1q_fwd:", int(df["ret_1q_fwd"].notna().sum()))

if __name__ == "__main__":
    main()
