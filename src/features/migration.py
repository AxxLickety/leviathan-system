# src/features/migration.py
import pandas as pd

def compute_migration_pressure(df: pd.DataFrame) -> pd.Series:
    """
    Migration pressure = net migration rate (normalized)
    """
    if not {"net_migration", "population"}.issubset(df.columns):
        raise ValueError("Missing migration inputs")

    mig_rate = df["net_migration"] / df["population"]
    mig_pressure = (mig_rate - mig_rate.mean()) / mig_rate.std()

    return mig_pressure.rename("mig_pressure")

