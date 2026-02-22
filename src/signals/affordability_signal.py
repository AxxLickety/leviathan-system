import pandas as pd
import numpy as np


def build_affordability_signal(
    df: pd.DataFrame,
    weights: dict | None = None
) -> pd.DataFrame:
    """
    Build cross-sectional affordability signal.

    Higher score_xs = more affordable (expected higher future return)

    Required columns:
        dti
        pti
        rent_burden
        supply_pressure
        migration_pressure
    """

import pandas as pd
import numpy as np


def ts_zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return s * 0.0
    return (s - s.mean()) / std


def build_affordability_signal(
    df: pd.DataFrame,
    weights: dict | None = None
) -> pd.DataFrame:
    """
    Build time-series affordability signal (Phase 1).
    """

    out = df.copy()

    assert "region" in out.columns, "region column missing before signal construction"

    out["dti_z"] = out.groupby("region")["dti"].transform(ts_zscore)
    out["pti_z"] = out.groupby("region")["pti"].transform(ts_zscore)
    out["rent_burden_z"] = out.groupby("region")["rent_burden"].transform(ts_zscore)
    out["supply_pressure_z"] = out.groupby("region")["supply_pressure"].transform(ts_zscore)
    out["migration_pressure_z"] = out.groupby("region")["migration_pressure"].transform(ts_zscore)

    if weights is None:
        weights = {
            "dti_z": -0.30,
            "pti_z": -0.25,
            "rent_burden_z": -0.20,
            "supply_pressure_z": +0.15,
            "migration_pressure_z": -0.10,
        }

    out["score_xs"] = 0.0
    for k, w in weights.items():
        out["score_xs"] += w * out[k]

    return out

