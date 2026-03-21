# src/backtests/evaluation.py
import numpy as np
import pandas as pd


def sharpe(x: pd.Series) -> float:
    """Mean / std using ddof=1 (sample). Returns NaN on insufficient data or zero vol."""
    x = x.dropna()
    if len(x) < 2:
        return np.nan
    vol = x.std(ddof=1)
    if vol == 0:
        return np.nan
    return float(x.mean() / vol)


def max_drawdown(log_r: pd.Series) -> float:
    """Maximum equity-curve drawdown from a log-return series."""
    x = log_r.dropna()
    if len(x) == 0:
        return np.nan
    eq = np.exp(x.cumsum())
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    return float(dd.min())


def summarize(x: pd.Series, name: str = "") -> dict:
    """
    Canonical return-series summary.
    Fields: name, n, mean, vol, sharpe, p05, p50, p95, min, max, maxdd
    """
    x = x.dropna()
    if len(x) == 0:
        return {
            "name": name, "n": 0,
            "mean": np.nan, "vol": np.nan, "sharpe": np.nan,
            "p05": np.nan, "p50": np.nan, "p95": np.nan,
            "min": np.nan, "max": np.nan, "maxdd": np.nan,
        }
    return {
        "name":   name,
        "n":      int(len(x)),
        "mean":   float(x.mean()),
        "vol":    float(x.std(ddof=1)),
        "sharpe": sharpe(x),
        "p05":    float(x.quantile(0.05)),
        "p50":    float(x.quantile(0.50)),
        "p95":    float(x.quantile(0.95)),
        "min":    float(x.min()),
        "max":    float(x.max()),
        "maxdd":  max_drawdown(x),
    }
