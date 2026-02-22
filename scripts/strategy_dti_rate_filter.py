import pandas as pd
import numpy as np

INFILE = "outputs/path_a/master.csv"
OUTDIR = "outputs/strategy_filter"
WINDOW_Q = 24   # 给一点更稳的起步历史
H = 4

DTI_QGRID = np.arange(0.60, 0.96, 0.05)
DR_QGRID  = np.arange(0.50, 0.91, 0.10)

df = pd.read_csv(INFILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

# forward return
df["log_price"] = np.log(df["real_price_index"])
df["fwd_ret_4q"] = df["log_price"].shift(-H) - df["log_price"]

# rate trend: 4Q change
df["dr4"] = df["real_rate"] - df["real_rate"].shift(H)

df = df.dropna(subset=["fwd_ret_4q", "dti", "real_rate", "regime", "dr4"]).reset_index(drop=True)

def apply_filter(sub: pd.DataFrame, dti_thr: float, dr_thr: float) -> pd.Series:
    """
    Hold unless:
      regime=0 AND dti > dti_thr AND dr4 > dr_thr  => exit (return=0)
    """
    exit_mask = (sub["regime"] == 0) & (sub["dti"] > dti_thr) & (sub["dr4"] > dr_thr)
    hold = ~exit_mask
    return sub["fwd_ret_4q"] * hold.astype(float)

def p05(x: pd.Series) -> float:
    return x.dropna().quantile(0.05)

def summarize(x: pd.Series, name: str) -> dict:
    x = x.dropna()
    return {
        "name": name,
        "n": len(x),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "p05": float(x.quantile(0.05)),
        "min": float(x.min()),
    }

# walk-forward expanding selection (maximize p05 on train)
strat_ret = pd.Series(index=df.index, dtype=float)
chosen = []

for t in range(WINDOW_Q, len(df)):
    train = df.iloc[:t]
    test = df.iloc[t:t+1]

    base0 = train.loc[train["regime"] == 0, ["dti", "dr4"]].dropna()
    if len(base0) < 12:
        # fallback
        dti_thr = train["dti"].quantile(0.8)
        dr_thr = train["dr4"].quantile(0.7)
    else:
        dti_cands = [base0["dti"].quantile(q) for q in DTI_QGRID]
        dr_cands  = [base0["dr4"].quantile(q) for q in DR_QGRID]

        best_val = -1e18
        best_pair = None
        for a in dti_cands:
            for b in dr_cands:
                r = apply_filter(train, a, b)
                val = p05(r)  # tail objective
                if val > best_val:
                    best_val = val
                    best_pair = (float(a), float(b))

        dti_thr, dr_thr = best_pair

    chosen.append((dti_thr, dr_thr))
    strat_ret.iloc[t] = apply_filter(test, dti_thr, dr_thr).iloc[0]

eval_df = df.copy()
eval_df["strategy_ret"] = strat_ret
eval_df["baseline_ret"] = eval_df["fwd_ret_4q"]
eval_df = eval_df.dropna(subset=["strategy_ret"]).reset_index(drop=True)

# diagnostics: filter frequency
exit_mask_eval = (eval_df["regime"] == 0) & (eval_df["strategy_ret"] == 0)
filtered_freq = float(exit_mask_eval.mean())

# summaries
summary = pd.DataFrame([
    summarize(eval_df["baseline_ret"], "baseline"),
    summarize(eval_df["strategy_ret"], "dti+dr4_filter"),
])

summary.to_csv(f"{OUTDIR}/summary_dti_dr4.csv", index=False)
eval_df.to_csv(f"{OUTDIR}/timeseries_dti_dr4.csv", index=False)

print(summary.to_string(index=False))
print("\nFiltered frequency (regime=0 & exit):", filtered_freq)

# also store chosen params time series (aligned to eval_df)
# chosen is from t=WINDOW_Q..end-1 in original df after dropna; align to eval_df dates
# easiest: rebuild a series with NaNs then slice to eval_df length
param_df = pd.DataFrame({
    "date": df["date"].iloc[WINDOW_Q:WINDOW_Q+len(eval_df)].values,
    "dti_thr": [p[0] for p in chosen[:len(eval_df)]],
    "dr4_thr": [p[1] for p in chosen[:len(eval_df)]],
})
param_df.to_csv(f"{OUTDIR}/params_dti_dr4.csv", index=False)
