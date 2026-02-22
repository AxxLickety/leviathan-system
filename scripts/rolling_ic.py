import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/path_a/master.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# forward return
df["log_price"] = np.log(df["real_price_index"])
df["fwd_ret_4q"] = df["log_price"].shift(-4) - df["log_price"]

# drop NaNs
df = df.dropna(subset=["dti", "fwd_ret_4q", "regime"])

# rolling IC (20 quarters)
rolling_ic_all = df["dti"].rolling(20).corr(df["fwd_ret_4q"])

rolling_ic_reg0 = (
    df[df["regime"] == 0]["dti"]
    .rolling(20)
    .corr(df[df["regime"] == 0]["fwd_ret_4q"])
)

# plot
plt.figure()
rolling_ic_all.plot(label="All")
rolling_ic_reg0.plot(label="Regime 0")
plt.legend()
plt.title("Rolling 20Q IC: DTI vs 1Y Forward Return")
plt.show()
