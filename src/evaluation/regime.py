# src/evaluation/regime.py

import numpy as np
import pandas as pd


def assign_regime(
    df: pd.DataFrame,
    *,
    return_col: str = "fwd_return",
    up_label: str = "up",
    down_label: str = "down",
) -> pd.DataFrame:
    """
    Assign market regime based on forward returns.

    Regime definition (ex-post, diagnostic):
        - up   : forward return >= 0
        - down : forward return < 0

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a forward return column.
    return_col : str
        Column name used to define regime.
    up_label : str
        Label for up regime.
    down_label : str
        Label for down regime.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with an additional 'regime' column.
    """

    if return_col not in df.columns:
        raise KeyError(f"'{return_col}' not found in DataFrame")

    out = df.copy()

    out["regime"] = np.where(
        out[return_col] >= 0,
        up_label,
        down_label,
    )

    return out

def ic_by_regime(
    df: pd.DataFrame,
    *,
    signal_col: str,
    return_col: str = "fwd_return",
    min_obs: int = 12,
) -> pd.DataFrame:
    """
    Compute time-series IC by regime for a given signal.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
            - signal_col
            - return_col
            - 'regime' column
    signal_col : str
        Column name of the signal to evaluate.
    return_col : str
        Forward return column.
    min_obs : int
        Minimum observations required per regime.

    Returns
    -------
    pd.DataFrame
        Index: regime
        Columns:
            - ic
            - positive_ratio
            - n_obs
    """

    required = {signal_col, return_col, "regime"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    d = df[[signal_col, return_col, "regime"]].dropna()

    rows = []
    for regime, g in d.groupby("regime"):
        if len(g) < min_obs:
            continue

        ic = g[signal_col].corr(g[return_col])

        rows.append({
            "regime": regime,
            "ic": ic,
            "positive_ratio": (g[signal_col] * g[return_col] > 0).mean(),
            "n_obs": len(g),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("regime")
