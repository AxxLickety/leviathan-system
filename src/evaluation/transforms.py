"""
Causal time-series feature transforms for the Leviathan OOS pipeline.

All transforms are causal: at time t they use only data available up to and
including t. No future information is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rolling_pct_rank(
    series: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Causal rolling percentile rank.

    At each time t, returns the fraction of values in
    ``series[max(0, t-window+1) : t+1]`` that are ≤ ``series[t]``.

    Properties
    ----------
    - Strictly causal: only uses data up to and including t.
    - Bounded in (0, 1] (minimum is 1/window once window is full).
    - For the first ``window-1`` observations the effective window is shorter
      than ``window``, so values start from 1/1 = 1.0 and converge to
      approximately uniform [1/window, 1.0] once full.
    - For a trending series the rolling rank is stationary (by design):
      it measures rank relative to recent history, not absolute level.

    Parameters
    ----------
    series : pd.Series
        Input time series. Must be float-compatible.
    window : int
        Look-back window in observations. Default 20.

    Returns
    -------
    pd.Series with the same index as ``series``.
    """
    arr = series.to_numpy(dtype=float)
    n   = len(arr)
    out = np.empty(n)
    for t in range(n):
        start    = max(0, t - window + 1)
        segment  = arr[start : t + 1]
        out[t]   = np.sum(segment <= arr[t]) / len(segment)
    return pd.Series(out, index=series.index, name=str(series.name) + "_pct_roll")
