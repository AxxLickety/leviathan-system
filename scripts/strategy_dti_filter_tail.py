import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INFILE = "outputs/path_a/master.csv"
OUTDIR = "outputs/strategy_filter"
WINDOW_Q = 20
H = 4
QGRID = np.arange(0.60, 0.96, 0.05)

df = pd.read_csv(INFILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

df["log_price"] = np.log(df["real_price_index"])
df["fwd_ret_4q"] = df["log_price"].shift(-H) - df["log_price"]
df = df.dropna(subset=["fwd_ret_4q", "dti", "regime"]).reset_index(drop=True)

def apply_filter(sub, thr):
    hold = ~((sub["regime"] == 0) & (sub["dti"] > thr))
    return sub["fwd_ret_4q"] * hold.astype(float)

def tail_loss(x):
    return x.quantile(0.05)

strat_ret = pd.Series(index=df.index, dtype=float)
thr_used = []

for t in range(WINDOW_Q, len(df)):
    train = df.iloc[:t]
    test = df.iloc[t:t+1]

    base = train.loc[train["regime"] == 0, "dti"].dropna()
    if len(base) < 10:
        thr = train["dti"].quantile(0.8)
    else:
        cands = [base.quantile(q) for q in QGRID]
        best_thr, best_val = None, -1e9
        for thr0 in cands:
            r = apply_filter(train, thr0)
            val = tail_loss(r)
            if val > best_val:
                best_val, best_thr = val, thr0
        thr = best_thr

    thr_used.append(thr)
    strat_ret.iloc[t] = apply_filter(test, thr).iloc[0]

eval_df = df.copy()
eval_df["strategy_ret"] = strat_ret
eval_df["baseline_ret"] = eval_df["fwd_ret_4q"]
eval_df = eval_df.dropna(subset=["strategy_ret"]).reset_index(drop=True)

summary = pd.DataFrame([
    {
        "name": "baseline",
        "mean": eval_df["baseline_ret"].mean(),
        "std": eval_df["baseline_ret"].std(ddof=0),
        "p05": eval_df["baseline_ret"].quantile(0.05),
    },
    {
        "name": "tail_filter",
        "mean": eval_df["strategy_ret"].mean(),
        "std": eval_df["strategy_ret"].std(ddof=0),
        "p05": eval_df["strategy_ret"].quantile(0.05),
    },
])

summary.to_csv(f"{OUTDIR}/summary_tail.csv", index=False)
print(summary.to_string(index=False))
