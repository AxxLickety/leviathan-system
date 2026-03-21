# src/evaluation/regime.py

import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical operational regime
# ---------------------------------------------------------------------------

def assign_fragility_regime(
    df: pd.DataFrame,
    *,
    rate_col: str = "real_rate",
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Assign the operational fragility regime used throughout Leviathan.

    This is the **ex-ante** regime: it is derived from observable macro inputs
    (real interest rate level) and contains no look-ahead bias.  It is the
    correct regime to use in decision logic, filters, and the interaction logit.

    Convention
    ----------
    regime = 1  →  real_rate < threshold  (negative / accommodative real rates)
    regime = 0  →  real_rate >= threshold  (positive / rate-stressed real rates)

    REGIME_FOR_RISK = 0 throughout scripts and path_a: the DTI filter triggers
    at regime=0 (positive real rates + high DTI = most stressed borrowers).

    Integer encoding is required: the interaction term in fit_logit.py is
    computed as  dti * regime, which requires numeric values.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the column named by ``rate_col``.
    rate_col : str
        Column containing the real interest rate series.
    threshold : float
        Rate level below which the market is considered to be in the
        accommodative (regime=1) state.  Default 0.0 (zero real rate).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an integer 'regime' column added (or replaced).
    """
    if rate_col not in df.columns:
        raise KeyError(f"'{rate_col}' not found in DataFrame")

    out = df.copy()
    out["regime"] = (out[rate_col] < threshold).astype(int)
    return out


# ---------------------------------------------------------------------------
# Descriptive / diagnostic regime
# ---------------------------------------------------------------------------

def assign_directional_regime(
    df: pd.DataFrame,
    *,
    return_col: str = "fwd_return",
    up_label: str = "up",
    down_label: str = "down",
) -> pd.DataFrame:
    """
    Assign a descriptive regime based on the sign of realized forward returns.

    This is an **ex-post** label: it is conditioned on future price outcomes
    and therefore contains look-ahead bias.  It is appropriate only for
    historical attribution and IC decomposition — NOT for decision logic or
    backtest filters.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a forward return column.
    return_col : str
        Column name used to define regime.
    up_label : str
        Label assigned when forward return >= 0.
    down_label : str
        Label assigned when forward return < 0.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with a string 'regime' column added (or replaced).
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


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

def assign_regime(
    df: pd.DataFrame,
    *,
    return_col: str = "fwd_return",
    up_label: str = "up",
    down_label: str = "down",
) -> pd.DataFrame:
    """
    Deprecated alias for assign_directional_regime.

    .. deprecated::
        Use ``assign_directional_regime`` for ex-post diagnostic labeling, or
        ``assign_fragility_regime`` for the operational ex-ante regime.
        This alias exists only for backward compatibility with existing notebooks
        and will be removed in a future cleanup pass.
    """
    warnings.warn(
        "assign_regime is deprecated. "
        "Use assign_fragility_regime (operational, ex-ante) or "
        "assign_directional_regime (diagnostic, ex-post) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return assign_directional_regime(
        df,
        return_col=return_col,
        up_label=up_label,
        down_label=down_label,
    )


# ---------------------------------------------------------------------------
# IC decomposition (works with either regime encoding)
# ---------------------------------------------------------------------------

def ic_by_regime(
    df: pd.DataFrame,
    *,
    signal_col: str,
    return_col: str = "fwd_return",
    min_obs: int = 12,
) -> pd.DataFrame:
    """
    Compute time-series IC by regime for a given signal.

    Works with either integer (fragility) or string (directional) regime
    encodings — it groups by whatever values are present in the 'regime' column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain signal_col, return_col, and a 'regime' column.
    signal_col : str
        Column name of the signal to evaluate.
    return_col : str
        Forward return column.
    min_obs : int
        Minimum observations required per regime.

    Returns
    -------
    pd.DataFrame
        Index: regime values
        Columns: ic, positive_ratio, n_obs
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
