# src/features/affordability.py

import pandas as pd
import numpy as np


def monthly_payment(price, rate, years=25):
    """
    Vectorized monthly mortgage payment.
    price: Series or float
    rate: Series or float (annual)
    """
    r = rate / 12
    n = years * 12

    payment = price * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

    # handle zero-rate edge case
    payment = np.where(r == 0, price / n, payment)

    return payment



def attach_affordability_features(df):
    """
    Add all affordability-related structural features.
    """
    out = df.copy()

    # 1. Debt-to-Income
    out["dti"] = out["price"] / out["income"]

    # 2. Payment-to-Income (vectorized, NO apply)
    out["mortgage_payment"] = monthly_payment(
        out["price"],
        out["mortgage_rate"]
    )
    out["pti"] = out["mortgage_payment"] / (out["income"] / 12)

    # 3. Rent burden
    out["rent_burden"] = out["rent"] / (out["income"] / 12)

    # 4. Supply pressure
    out["supply_pressure"] = out["permits"] / out["inventory"]

    # 5. Migration pressure
    out["migration_pressure"] = out["net_migration"] / out["population"]

    return out




