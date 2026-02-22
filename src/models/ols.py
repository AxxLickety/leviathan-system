# src/models/ols.py

import pandas as pd
import numpy as np
import statsmodels.api as sm


def run_cross_sectional_ols(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
):
    """
    For each date, run a cross-sectional regression:

        y(region, t) ~ X(region, t)

    Returns a DataFrame with:
        - date
        - r2
        - coef_<x>
        - tstat_<x>

    Notes:
        - Rows with NaNs in y or X are dropped per date.
        - If a date has too few observations, that date is skipped.
    """

    results = []

    for dt, tmp in df.groupby("date"):
        tmp = tmp.dropna(subset=[y_col] + x_cols)
        if len(tmp) < len(x_cols) + 2:
            # not enough data points for a meaningful regression
            continue

        y = tmp[y_col].values
        X = tmp[x_cols]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        row = {"date": dt, "r2": model.rsquared}
        # coefficients & t-stats
        for name, val in model.params.items():
            row[f"coef_{name}"] = val
        for name, val in model.tvalues.items():
            row[f"tstat_{name}"] = val

        results.append(row)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results).sort_values("date")
    return out


def run_panel_ols(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    entity_col: str = "region",
    time_col: str = "date",
):
    """
    Simple pooled panel OLS (no fixed effects), mostly as a placeholder.

    y(i,t) ~ X(i,t)

    You can later replace this with a proper FE / RE model using
    linearmodels or a dedicated panel library.
    """

    tmp = df.dropna(subset=[y_col] + x_cols).copy()

    y = tmp[y_col].values
    X = tmp[x_cols]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    return model
