# src/features/supply.py
import pandas as pd

def compute_supply_pressure(df: pd.DataFrame) -> pd.Series:
    """
    Supply pressure = tight inventory + weak permitting
    """
    required = {"inventory", "permits"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing supply inputs: {required - set(df.columns)}")

    # Inventory pressure (low inventory = high pressure)
    inv_z = (df["inventory"] - df["inventory"].mean()) / df["inventory"].std()

    # Permit growth (weak growth = high pressure)
    permit_growth = df["permits"].pct_change(12)
    perm_z = (permit_growth - permit_growth.mean()) / permit_growth.std()

    supply_pressure = (-inv_z) + (-perm_z)
    return supply_pressure.rename("supply_pressure")

