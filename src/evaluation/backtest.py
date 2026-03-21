# src/evaluation/backtest.py
import numpy as np
import pandas as pd


def compute_forward_return(
    df: pd.DataFrame,
    horizon: int = 4,
    price_col: str = "real_price_index",
) -> pd.DataFrame:
    """
    Compute log forward return over ``horizon`` quarters.

    Formula: log(p[t+H]) - log(p[t])

    This matches the convention used throughout scripts/
    (strategy_dti_filter, calc_ic, rolling_ic, etc.).
    Log returns are additive across time and treat losses
    conservatively — appropriate for a downside-risk system.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the column named by ``price_col``.
    horizon : int
        Number of periods forward.  Default 4 = 1 year on quarterly data.
    price_col : str
        Price level column.  Default "real_price_index".

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with "fwd_return" column added (or replaced).
    """
    out = df.copy()
    log_price = np.log(out[price_col])
    out["fwd_return"] = log_price.shift(-horizon) - log_price
    return out

