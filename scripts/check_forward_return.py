import pandas as pd
import numpy as np

df = pd.read_csv("outputs/path_a/master.csv", parse_dates=["date"])

df["log_price"] = np.log(df["real_price_index"])
df["fwd_ret_4q"] = df["log_price"].shift(-4) - df["log_price"]

print(df[["date", "fwd_ret_4q"]].tail(10))
print("\nNaNs in last rows:", df["fwd_ret_4q"].tail(6))
print("\nDescribe:")
print(df["fwd_ret_4q"].describe())
