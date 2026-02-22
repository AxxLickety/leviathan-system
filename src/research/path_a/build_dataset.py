from __future__ import annotations

import numpy as np
import pandas as pd


def build_master_df(
    *,
    start: str = "1999-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="Q")
    n = len(dates)

    cycle = 1.5 * np.sin(np.linspace(0, 8 * np.pi, n))
    shocks = rng.normal(0, 0.6, size=n)
    real_rate = cycle + shocks
    regime = (real_rate < 0).astype(int)

    trend = np.linspace(85, 135, n)
    dti = trend + regime * 6 + rng.normal(0, 2.0, size=n)
    dti = np.clip(dti, 60, 180)

    base_growth = 0.008 + regime * 0.004
    noise = rng.normal(0, 0.01, size=n)

    crash = np.zeros(n)
    crash[int(n * 0.30):int(n * 0.38)] = -0.04

    returns = base_growth + noise + crash
    real_price_index = 100 * np.exp(np.cumsum(returns))
    real_price_index = real_price_index / real_price_index[0] * 100

    return pd.DataFrame(
        {
            "date": dates,
            "dti": dti,
            "real_rate": real_rate,
            "regime": regime,
            "real_price_index": real_price_index,
        }
    )
