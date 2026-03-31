# src/backtests/evaluation.py
from __future__ import annotations

from typing import Callable, Optional

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


def bootstrap_ci(
    x: pd.Series,
    stat: Callable[[np.ndarray], float] = np.mean,
    *,
    weights: Optional[pd.Series] = None,
    B: int = 1000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a statistic.

    Parameters
    ----------
    x : pd.Series
        Observations. NaNs are dropped (and corresponding weights dropped).
    stat : callable
        Statistic to bootstrap. Called as ``stat(sample_array)`` for each
        replicate. Default: ``np.mean``.
    weights : pd.Series or None
        Per-observation sampling probabilities (unnormalised). Must be
        non-negative and align with ``x`` by index. ``None`` → uniform
        (equivalent to all weights = 1.0, backward-compatible).
        Typical usage: ``weight_col`` from the merged data frame, where FRED
        rows carry 1.0 and Zillow rows carry 0.5.
    B : int
        Number of bootstrap replicates. Default 1000.
    ci : float
        Confidence level (0 < ci < 1). The returned interval spans the
        ``(1-ci)/2`` and ``(1+ci)/2`` quantiles. Default 0.95.
    rng : numpy.random.Generator or None
        RNG instance for reproducibility. ``None`` → ``default_rng()``.

    Returns
    -------
    (lower, upper) : tuple[float, float]
        Lower and upper CI bounds.

    Notes
    -----
    Weighted bootstrap resamples observations with replacement, where each
    observation's draw probability is proportional to its weight. FRED
    observations (weight=1.0) are drawn twice as often as Zillow observations
    (weight=0.5) in expectation, reducing the influence of the lower-quality
    source on the CI.

    Sensitivity check: pass ``weights=pd.Series(1.0, index=x.index)`` to
    force uniform weights and compare against the default (no weights) run.
    The two CIs should agree closely; a large divergence indicates that
    down-weighting Zillow observations materially affects the estimate.
    """
    if rng is None:
        rng = np.random.default_rng()

    arr = x.dropna()
    if len(arr) < 2:
        return (np.nan, np.nan)

    if weights is not None:
        w = weights.reindex(arr.index).fillna(0.0)
        if w.sum() == 0:
            raise ValueError("bootstrap_ci: all weights are zero after aligning with x")
        probs = (w / w.sum()).to_numpy()
    else:
        probs = None  # numpy will use uniform

    vals = arr.to_numpy()
    n = len(vals)

    boot_stats = np.empty(B)
    for i in range(B):
        idx = rng.choice(n, size=n, replace=True, p=probs)
        boot_stats[i] = stat(vals[idx])

    alpha = (1.0 - ci) / 2.0
    return (float(np.quantile(boot_stats, alpha)), float(np.quantile(boot_stats, 1.0 - alpha)))


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bootstrap_uniform_weights_match_unweighted() -> None:
    """Weighted bootstrap with all weights=1.0 must agree with unweighted.

    Both use the same sample data and a fixed seed. With uniform weights the
    sampling distribution is identical, so the CIs should agree to within the
    Monte Carlo noise of B=2000 replicates. Tolerance is set to 1% of the
    data range — much larger than the expected Monte Carlo error (~0.1%) but
    tight enough to catch a weight normalisation bug or an accidental bias.
    """
    rng_seed = 0
    data = pd.Series([0.03, -0.05, 0.07, -0.02, 0.04, -0.08, 0.06, 0.01,
                      -0.03, 0.05, -0.01, 0.02, -0.06, 0.08, -0.04, 0.03])
    B = 2000
    uniform_weights = pd.Series(1.0, index=data.index)

    lo_unweighted, hi_unweighted = bootstrap_ci(
        data, np.mean, weights=None, B=B, rng=np.random.default_rng(rng_seed)
    )
    lo_weighted, hi_weighted = bootstrap_ci(
        data, np.mean, weights=uniform_weights, B=B, rng=np.random.default_rng(rng_seed)
    )

    data_range = float(data.max() - data.min())
    tol = 0.01 * data_range  # 1 % of data range

    assert abs(lo_weighted - lo_unweighted) < tol, (
        f"Lower CI mismatch: weighted={lo_weighted:.6f}, "
        f"unweighted={lo_unweighted:.6f}, tol={tol:.6f}"
    )
    assert abs(hi_weighted - hi_unweighted) < tol, (
        f"Upper CI mismatch: weighted={hi_weighted:.6f}, "
        f"unweighted={hi_unweighted:.6f}, tol={tol:.6f}"
    )


if __name__ == "__main__":
    test_bootstrap_uniform_weights_match_unweighted()
    print("test_bootstrap_uniform_weights_match_unweighted  PASSED")
