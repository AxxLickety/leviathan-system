"""
OOS Helper Functions — Leviathan Phase OOS
==========================================
Utility functions used exclusively by scripts/oos_eval.py.
These helpers must not be imported by oos_train.py.

Functions
---------
wilson_ci                  — Wilson confidence interval for a proportion
flag_small_cells           — Mark cells below count threshold
load_frozen_params         — Load all four frozen files from disk into a dict
regime_supply_count_table  — Observation counts by regime × supply × label split
crash_frequency_table      — Crash frequency with Wilson CIs by regime × supply cell
apply_overlay_filter       — Apply DTI/regime overlay; return per-period returns
plot_equity_curves         — Plot cumulative equity curves for multiple strategies
write_verdict              — Write structured evaluation verdict to disk
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtests.evaluation import summarize


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.

    Parameters
    ----------
    k     : number of successes
    n     : total observations
    alpha : significance level (two-sided), default 0.05 → 95% CI

    Returns
    -------
    (lower, upper) — both in [0, 1]
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if not (0 <= k <= n):
        raise ValueError(f"k={k} must be in [0, n={n}]")

    # z for two-sided CI: ppf(1 - alpha/2) ≈ 1.96 for alpha=0.05
    # Computed via standard normal inverse; avoids scipy dependency.
    # For alpha=0.05 → z≈1.959964; use math approximation good to 4 dp.
    z = _normal_ppf(1.0 - alpha / 2.0)

    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _normal_ppf(p: float) -> float:
    """Rational approximation to the standard normal quantile (Abramowitz & Stegun)."""
    assert 0.0 < p < 1.0
    if p < 0.5:
        return -_normal_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)


# ---------------------------------------------------------------------------
# flag_small_cells
# ---------------------------------------------------------------------------

def flag_small_cells(
    df: pd.DataFrame,
    count_col: str,
    threshold: int = 10,
) -> pd.DataFrame:
    """
    Add a boolean 'small_cell' column marking rows where count_col < threshold.

    Parameters
    ----------
    df         : input DataFrame
    count_col  : column containing observation counts
    threshold  : cells strictly below this value are flagged

    Returns
    -------
    Copy of df with 'small_cell' (bool) column added.
    """
    if count_col not in df.columns:
        raise KeyError(f"count_col '{count_col}' not found in DataFrame")
    out = df.copy()
    out["small_cell"] = out[count_col] < threshold
    return out


# ---------------------------------------------------------------------------
# load_frozen_params
# ---------------------------------------------------------------------------

def load_frozen_params(frozen_dir: str | Path) -> dict:
    """
    Load all four frozen files from disk into a single dict.

    Parameters
    ----------
    frozen_dir : path to the directory containing frozen outputs

    Returns
    -------
    dict with keys:
        "coef"           — pd.DataFrame  (index = param name, cols: coef, p_value)
        "thresholds"     — pd.DataFrame  (cols: regime, prob, dti_threshold)
        "dti_cutoff"     — float
        "train_metadata" — dict

    Raises
    ------
    FileNotFoundError if any expected file is absent.
    """
    d = Path(frozen_dir)
    required = {
        "coef.csv":            d / "coef.csv",
        "thresholds.csv":      d / "thresholds.csv",
        "dti_cutoff.json":     d / "dti_cutoff.json",
        "train_metadata.json": d / "train_metadata.json",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen file '{name}' not found at {path}. "
                "Run scripts/oos_train.py first."
            )

    coef_df = pd.read_csv(required["coef.csv"], index_col=0)
    thresholds_df = pd.read_csv(required["thresholds.csv"])

    with open(required["dti_cutoff.json"]) as f:
        cutoff_data = json.load(f)
    dti_cutoff = float(cutoff_data["dti_cutoff"])

    with open(required["train_metadata.json"]) as f:
        train_metadata = json.load(f)

    return {
        "coef":           coef_df,
        "thresholds":     thresholds_df,
        "dti_cutoff":     dti_cutoff,
        "train_metadata": train_metadata,
    }


# ---------------------------------------------------------------------------
# regime_supply_count_table
# ---------------------------------------------------------------------------

def regime_supply_count_table(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    regime_col: str,
    supply_col: Optional[str],
    label_col: str,
) -> pd.DataFrame:
    """
    Observation counts by regime × supply × label for train and test splits.

    supply_col may be None; if so, all rows are grouped into a single 'all' bucket.

    Returns
    -------
    pd.DataFrame with columns: split, regime, supply, label, n
    """
    for name, df in [("train", df_train), ("test", df_test)]:
        for col in [regime_col, label_col]:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' missing from {name} DataFrame")

    rows = []
    for split_name, df in [("train", df_train), ("test", df_test)]:
        df = df.copy()
        if supply_col is not None and supply_col in df.columns:
            group_cols = [regime_col, supply_col, label_col]
        else:
            df["_supply"] = "all"
            group_cols = [regime_col, "_supply", label_col]
        for keys, grp in df.groupby(group_cols):
            regime_val, supply_val, label_val = keys
            rows.append({
                "split":  split_name,
                "regime": regime_val,
                "supply": supply_val,
                "label":  label_val,
                "n":      len(grp),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# crash_frequency_table
# ---------------------------------------------------------------------------

def crash_frequency_table(
    df: pd.DataFrame,
    regime_col: str,
    supply_col: Optional[str],
    label_col: str,
) -> pd.DataFrame:
    """
    Crash frequency and Wilson 95% CI bounds by regime × supply cell.

    Returns
    -------
    pd.DataFrame with columns:
        regime, supply, n, crashes, freq, ci_lo, ci_hi, small_cell
    """
    for col in [regime_col, label_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing from DataFrame")

    df = df.copy()
    if supply_col is not None and supply_col in df.columns:
        group_cols = [regime_col, supply_col]
    else:
        df["_supply"] = "all"
        group_cols = [regime_col, "_supply"]

    rows = []
    for keys, grp in df.groupby(group_cols):
        regime_val = keys[0]
        supply_val = keys[1]
        n = len(grp)
        crashes = int(grp[label_col].sum())
        freq = crashes / n if n > 0 else np.nan
        if n > 0:
            lo, hi = wilson_ci(crashes, n)
        else:
            lo, hi = np.nan, np.nan
        rows.append({
            "regime":  regime_val,
            "supply":  supply_val,
            "n":       n,
            "crashes": crashes,
            "freq":    freq,
            "ci_lo":   lo,
            "ci_hi":   hi,
        })

    result = pd.DataFrame(rows)
    return flag_small_cells(result, count_col="n", threshold=10)


# ---------------------------------------------------------------------------
# apply_overlay_filter
# ---------------------------------------------------------------------------

def apply_overlay_filter(
    df: pd.DataFrame,
    dti_cutoff: float,
    regime_col: str,
    dti_col: str,
    return_col: str,
) -> pd.Series:
    """
    Apply DTI/regime overlay filter.

    Returns fwd_return unless (regime == 0 AND dti > dti_cutoff), in which case 0.0.

    Returns
    -------
    pd.Series of per-period returns (same index as df)
    """
    for col in [regime_col, dti_col, return_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing from DataFrame")

    hold = ~((df[regime_col] == 0) & (df[dti_col] > dti_cutoff))
    return (df[return_col] * hold.astype(float)).rename("overlay")


# ---------------------------------------------------------------------------
# plot_equity_curves
# ---------------------------------------------------------------------------

def plot_equity_curves(
    df: pd.DataFrame,
    strategy_cols: List[str],
    markers: Dict[str, str],
    outpath: str | Path,
) -> None:
    """
    Plot cumulative log-return equity curves for each strategy.

    Parameters
    ----------
    df            : DataFrame with a 'date' column and one column per strategy
    strategy_cols : list of column names to plot
    markers       : dict of col_name → display label
    outpath       : file path to save the figure (PNG)

    Raises
    ------
    KeyError  if any strategy_col or 'date' is missing
    NotADirectoryError if outpath's parent directory does not exist
    """
    outpath = Path(outpath)
    if not outpath.parent.exists():
        raise NotADirectoryError(
            f"Output directory does not exist: {outpath.parent}"
        )
    if "date" not in df.columns:
        raise KeyError("'date' column required in df for plot_equity_curves")
    for col in strategy_cols:
        if col not in df.columns:
            raise KeyError(f"Strategy column '{col}' not found in DataFrame")

    fig, ax = plt.subplots(figsize=(10, 5))

    for col in strategy_cols:
        label = markers.get(col, col)
        cum = df[col].fillna(0).cumsum()
        ax.plot(df["date"], cum, label=label)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative log return")
    ax.set_title("OOS Equity Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# write_verdict
# ---------------------------------------------------------------------------

_REQUIRED_PRIMARY_KEYS = {"always_in", "overlay"}


def write_verdict(
    verdict: str,
    primary_summaries: Dict[str, dict],
    secondary_summaries: Dict[str, dict],
    outpath: str | Path,
) -> None:
    """
    Write a structured evaluation verdict to disk.

    Parameters
    ----------
    verdict            : verdict label string (e.g. "SMOKE_TEST_DEGENERATE_TRAINING")
    primary_summaries  : dict of strategy name → summarize() output dict
                         Must contain keys 'always_in' and 'overlay'.
    secondary_summaries: dict of label → summary dict (sensitivity runs, etc.)
    outpath            : path to write verdict.txt

    Raises
    ------
    ValueError if primary_summaries is missing required keys
    """
    missing = _REQUIRED_PRIMARY_KEYS - set(primary_summaries.keys())
    if missing:
        raise ValueError(
            f"primary_summaries is missing required strategy keys: {missing}. "
            "Cannot write verdict without both 'always_in' and 'overlay' summaries."
        )

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    sep = "=" * 72

    lines += [
        sep,
        "OOS EVALUATION VERDICT",
        sep,
        "",
    ]

    # ---- Smoke-test caveat (always present when verdict contains SMOKE_TEST) ----
    if "SMOKE_TEST" in verdict.upper():
        lines += [
            "*** SMOKE TEST ONLY — training window had zero positive labels       ***",
            "*** This run validates pipeline mechanics, NOT research findings.    ***",
            "*** Logit coefficients are degenerate and economically meaningless.  ***",
            "*** DTI thresholds are extrapolated far outside any valid range.     ***",
            "*** Do NOT interpret strategy metrics as empirically meaningful.     ***",
            "*** A valid OOS run requires a training window with crash events.    ***",
            "",
        ]

    lines += [
        f"Verdict : {verdict}",
        "",
        sep,
        "PRIMARY STRATEGY COMPARISON",
        sep,
        "",
    ]

    # Header
    hdr_fields = ["strategy", "n", "mean", "vol", "sharpe", "p05", "maxdd"]
    col_w = 14
    lines.append("  ".join(f"{h:<{col_w}}" for h in hdr_fields))
    lines.append("-" * (col_w * len(hdr_fields) + 2 * (len(hdr_fields) - 1)))

    def _fmt_row(s: dict) -> str:
        return "  ".join([
            f"{str(s.get('name','')):<{col_w}}",
            f"{s.get('n', 'N/A')!s:<{col_w}}",
            f"{s.get('mean', float('nan')):<{col_w}.6f}",
            f"{s.get('vol',  float('nan')):<{col_w}.6f}",
            f"{s.get('sharpe', float('nan')):<{col_w}.6f}",
            f"{s.get('p05',  float('nan')):<{col_w}.6f}",
            f"{s.get('maxdd', float('nan')):<{col_w}.6f}",
        ])

    for key in ["always_in", "overlay", "always_out"]:
        if key in primary_summaries:
            lines.append(_fmt_row(primary_summaries[key]))

    if secondary_summaries:
        lines += [
            "",
            sep,
            "SECONDARY / SENSITIVITY SUMMARIES",
            sep,
            "",
            "  ".join(f"{h:<{col_w}}" for h in hdr_fields),
            "-" * (col_w * len(hdr_fields) + 2 * (len(hdr_fields) - 1)),
        ]
        for key, s in secondary_summaries.items():
            lines.append(_fmt_row(s))

    lines += [
        "",
        sep,
        "END OF VERDICT",
        sep,
    ]

    outpath.write_text("\n".join(lines) + "\n")
