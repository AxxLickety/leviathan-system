import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INFILE = "outputs/path_a/master.csv"
OUTDIR = "outputs/strategy_filter"
WINDOW_Q = 20          # 至少需要多少个季度的历史才开始walk-forward
H = 4                  # forward horizon in quarters
QGRID = np.arange(0.60, 0.96, 0.05)  # quantile grid for threshold search

df = pd.read_csv(INFILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

# forward return
df["log_price"] = np.log(df["real_price_index"])
df["fwd_ret_4q"] = df["log_price"].shift(-H) - df["log_price"]

# drop rows where target missing
df = df.dropna(subset=["fwd_ret_4q", "dti", "regime"]).reset_index(drop=True)

def apply_filter(sub: pd.DataFrame, thr: float) -> pd.Series:
    """Return strategy return series given threshold thr."""
    hold = ~((sub["regime"] == 0) & (sub["dti"] > thr))
    return sub["fwd_ret_4q"] * hold.astype(float)

def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    if x.std(ddof=0) == 0:
        return np.nan
    return x.mean() / x.std(ddof=0)

rows = []
thr_chosen = []
strat_ret = pd.Series(index=df.index, dtype=float)

# walk-forward: at each t, pick best threshold using only past data
for t in range(WINDOW_Q, len(df)):
    train = df.iloc[:t].copy()
    test = df.iloc[t:t+1].copy()

    # restrict threshold candidates based on train regime=0 dti distribution
    base = train.loc[train["regime"] == 0, "dti"].dropna()
    if len(base) < 10:
        # if not enough regime=0 history, skip optimization
        thr = train["dti"].quantile(0.8)
    else:
        cands = [base.quantile(q) for q in QGRID]
        best = None
        best_metric = -1e18

        # optimize for Sharpe on train (could swap metric later)
        for thr0 in cands:
            r = apply_filter(train, thr0)
            m = sharpe(r)
            # fallback: if sharpe nan, use mean
            if np.isnan(m):
                m = r.mean()
            if m > best_metric:
                best_metric = m
                best = thr0
        thr = float(best)

    thr_chosen.append(thr)
    strat_ret.iloc[t] = apply_filter(test, thr).iloc[0]

# build evaluation frame aligned to dates
eval_df = df.copy()
eval_df["strategy_ret"] = strat_ret
eval_df["baseline_ret"] = eval_df["fwd_ret_4q"]

# only evaluate where strategy_ret exists
eval_df = eval_df.dropna(subset=["strategy_ret"]).reset_index(drop=True)

# summary stats
def summarize(x: pd.Series, name: str) -> dict:
    x = x.dropna()
    return {
        "name": name,
        "n": len(x),
        "mean": x.mean(),
        "std": x.std(ddof=0),
        "sharpe": sharpe(x),
        "min": x.min(),
        "p05": x.quantile(0.05),
        "p50": x.quantile(0.50),
        "p95": x.quantile(0.95),
        "max": x.max(),
    }

sum_baseline = summarize(eval_df["baseline_ret"], "baseline_hold")
sum_strat = summarize(eval_df["strategy_ret"], "filter_strategy")

summary = pd.DataFrame([sum_baseline, sum_strat])
summary.to_csv(f"{OUTDIR}/summary.csv", index=False)

# equity curves (cum log return proxy)
eval_df["baseline_cum"] = eval_df["baseline_ret"].cumsum()
eval_df["strategy_cum"] = eval_df["strategy_ret"].cumsum()
eval_df.to_csv(f"{OUTDIR}/timeseries.csv", index=False)

# plots
plt.figure()
eval_df.set_index("date")[["baseline_cum", "strategy_cum"]].plot()
plt.title("Cumulative (sum) of 1Y Forward Returns: Baseline vs DTI Filter")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/cum.png", dpi=150)

plt.figure()
(eval_df["baseline_ret"]).hist(bins=30, alpha=0.5, label="baseline")
(eval_df["strategy_ret"]).hist(bins=30, alpha=0.5, label="strategy")
plt.legend()
plt.title("Return Distribution: Baseline vs DTI Filter")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/hist.png", dpi=150)

print("Wrote:", f"{OUTDIR}/summary.csv")
print(summary.to_string(index=False))

# quick diagnostic: how often filtered out
filtered = ((eval_df["regime"] == 0) & (eval_df["strategy_ret"] == 0)).mean()
print("\nFiltered frequency (regime=0 & out):", float(filtered))
