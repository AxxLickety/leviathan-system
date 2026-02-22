from __future__ import annotations

import pandas as pd


def add_correction_label(
    df: pd.DataFrame,
    *,
    price_col: str = "real_price_index",
    horizon_max_q: int = 20,
    threshold: float = 0.90,
) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    prices = df[price_col].to_numpy()
    y = []

    for t in range(len(prices)):
        if t + horizon_max_q >= len(prices):
            y.append(None)
            continue
        future_min = prices[t + 1 : t + horizon_max_q + 1].min()
        y.append(1 if future_min / prices[t] < threshold else 0)

    df["y"] = y
    return df.dropna(subset=["y"]).assign(y=lambda x: x["y"].astype(int))
