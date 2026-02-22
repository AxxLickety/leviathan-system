# src/evaluation/backtest.py
import pandas as pd

def compute_forward_return(
    df: pd.DataFrame,
    horizon: int = 12,
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Compute forward return used for IC / PnL proxy.
    NO side effects, pure function.
    """
    out = df.copy()
    out["fwd_return"] = out[price_col].pct_change(horizon).shift(-horizon)
    return out

