from __future__ import annotations
import numpy as np
import pandas as pd

SANITY = "outputs/phase2/phase2_sanity.csv"
THRESH = "outputs/path_a/thresholds.csv"

REGIME_FOR_RISK = 0
PROB_FOR_THRESHOLD = 0.10  # change to 0.20 if that's what Phase1 used

def load_threshold() -> float | None:
    try:
        t = pd.read_csv(THRESH)
    except FileNotFoundError:
        return None

    # preferred: the path_a thresholds table
    if set(["regime","prob","dti_threshold"]).issubset(t.columns):
        sub = t[(t["regime"] == REGIME_FOR_RISK) & (np.isclose(t["prob"], PROB_FOR_THRESHOLD))]
        if len(sub) == 1:
            return float(sub["dti_threshold"].iloc[0])
        # fallback: take the closest prob within that regime
        sub2 = t[t["regime"] == REGIME_FOR_RISK].copy()
        if len(sub2):
            sub2["prob_dist"] = (sub2["prob"] - PROB_FOR_THRESHOLD).abs()
            sub2 = sub2.sort_values("prob_dist").head(1)
            return float(sub2["dti_threshold"].iloc[0])

    return None

def summarize(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return {"n": 0, "mean": np.nan, "vol": np.nan, "p05": np.nan}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "vol": float(x.std(ddof=1)),
        "p05": float(x.quantile(0.05)),
    }

def main():
    df = pd.read_csv(SANITY, parse_dates=["date"]).sort_values("date")

    if "fwd_ret_4q" not in df.columns:
        raise ValueError("SANITY missing fwd_ret_4q")

    threshold_t = load_threshold()
    if threshold_t is None:
        threshold_t = float(df["dti"].quantile(0.80))
        print(f"[warn] could not read threshold_t from {THRESH}; using dti p80={threshold_t:.4f}")
    else:
        print(f"[ok] threshold_t={threshold_t:.4f} (regime={REGIME_FOR_RISK}, prob={PROB_FOR_THRESHOLD})")

    phase1_risk = (df["regime"] == REGIME_FOR_RISK) & (df["dti"] > threshold_t)
    phase2_risk = phase1_risk & (df["supply_high"] == 1)

    invested_p1 = ~phase1_risk
    invested_p2 = ~phase2_risk

    r = df["fwd_ret_4q"]

    out = []
    out.append(("baseline_all", summarize(r)))
    out.append(("phase1_invested", summarize(r[invested_p1])))
    out.append(("phase2_invested", summarize(r[invested_p2])))
    out.append(("phase1_risk_bucket", summarize(r[phase1_risk])))
    out.append(("phase2_risk_bucket", summarize(r[phase2_risk])))

    rows = []
    for name, stats in out:
        rows.append({"bucket": name, **stats})

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))

    res.to_csv("outputs/phase2/phase2_eval_summary.csv", index=False)
    print("[phase2] wrote outputs/phase2/phase2_eval_summary.csv")

    # quick diagnostics
    print("\n[diag] counts:")
    print("phase1_risk:", int(phase1_risk.sum()))
    print("phase2_risk:", int(phase2_risk.sum()))
    print("supply_high:", int((df['supply_high']==1).sum()))
    print("non_nan_fwd_ret_4q:", int(df['fwd_ret_4q'].notna().sum()))

if __name__ == "__main__":
    main()
