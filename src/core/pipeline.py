# src/core/pipeline.py

import pandas as pd
from src.evaluation.regime import assign_regime

from src.loaders.housing_loader import load_housing_data
from src.features.affordability import attach_affordability_features
from src.signals.affordability_signal import build_affordability_signal
from src.evaluation.backtest import compute_forward_return


def run_pipeline(region: str) -> pd.DataFrame:
    # 1. load base data
    df = load_housing_data(region)

    # 2. feature engineering
    df = attach_affordability_features(df)

    # 3. signal
    df = build_affordability_signal(df)

    # 4. forward return（必须最后）
    df = compute_forward_return(df, horizon=12)

    df = assign_regime(df)
    df["affordability_active"] = df["regime"] == "down"


    return df
