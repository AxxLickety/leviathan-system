import pandas as pd
import numpy as np

df = pd.read_csv("outputs/path_a/master.csv", parse_dates=["date"])

# forward return
df["log_price"] = np.log(df["real_price_index"])
df["fwd_ret_4q"] = df["log_price"].shift(-4) - df["log_price"]

# drop NaNs
df = df.dropna(subset=["fwd_ret_4q", "dti", "real_rate", "regime"])

def ic(x, y):
    return x.corr(y)

print("IC(dti, fwd_ret_4q):",
      ic(df["dti"], df["fwd_ret_4q"]))

print("IC(real_rate, fwd_ret_4q):",
      ic(df["real_rate"], df["fwd_ret_4q"]))

for r in [0, 1]:
    sub = df[df["regime"] == r]
    print(f"IC(dti, fwd_ret_4q | regime={r}):",
          ic(sub["dti"], sub["fwd_ret_4q"]))
