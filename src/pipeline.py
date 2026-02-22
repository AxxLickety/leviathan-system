# src/pipeline.py

import pandas as pd

from src.loaders.housing_loader import load_housing_data
from src.features.affordability import attach_affordability_features
from src.signals.affordability_signal import build_affordability_signal

def run_pipeline(region: str):
    df = load_housing_data(region)
    df = attach_affordability_features(df)
    df = build_affordability_signal(df)

    # FINAL XS SCORE (single source of truth)
    weights = {
        "dti_z": 0.4,
        "pti_z": 0.3,
        "supply_pressure_z": 0.2,
        "rent_burden_z": 0.1,
    }
    df["score_xs"] = sum(df[k] * w for k, w in weights.items())

    return df


