from __future__ import annotations

import pandas as pd


def excerpt_first_last(df: pd.DataFrame, n=5):
    return pd.concat([df.head(n), df.tail(n)])


def top_risk_quarters(df: pd.DataFrame, k=5):
    return df.sort_values("pred_prob", ascending=False).head(k)
