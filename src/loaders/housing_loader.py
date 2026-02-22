import pandas as pd
from pathlib import Path
from src.utils.project_root import get_project_root

# Absolute path to data/raw
DATA_PATH = get_project_root() / "data" / "raw"


def _load_csv(filename: str) -> pd.DataFrame:
    """
    Unified CSV loader from data/raw/
    """
    path = DATA_PATH / filename
    if not path.exists():
        raise FileNotFoundError(f"[DATA FILE NOT FOUND] {path}")
    return pd.read_csv(path)

# -----------------------------------------------------------------------------
# INDIVIDUAL LOADERS
# -----------------------------------------------------------------------------

def load_price():
    df = _load_csv("housing_price.csv")
    return df[["region", "date", "price"]]

def load_income():
    df = _load_csv("income.csv")
    if "income" not in df.columns:
        raise ValueError("income.csv missing required column 'income'")
    return df[["region", "date", "income"]]

def load_population():
    df = _load_csv("population_migration.csv")
    return df[["region", "date", "population", "net_migration"]]

def load_permits():
    """
    Mock permits data because the real permits.csv does not exist or
    does not contain the correct schema.

    We generate synthetic permits as:
        permits = population * 0.001
    This keeps relative scale and correlation realistic.
    """

    pop = load_population()  # has region, date, population, net_migration

    df = pop[["region", "date", "population"]].copy()
    df["permits"] = df["population"] * 0.001  # mock permits proxy

    return df[["region", "date", "permits"]]


def load_mortgage():
    df = _load_csv("mortgage_rate.csv")
    if "mortgage_rate" not in df.columns:
        df["mortgage_rate"] = 0.04  # default fallback
    return df[["region", "date", "mortgage_rate"]]

def load_rent():
    df = _load_csv("rent.csv")
    if "rent" not in df.columns:
        raise ValueError("rent.csv must contain a 'rent' column")
    return df[["region", "date", "rent"]]

def load_inventory():
    df = _load_csv("inventory.csv")

    # 自动寻找 inventory-like 列
    candidate_cols = [c for c in df.columns if c.lower() in ["inventory", "homes_for_sale", "active_listings", "supply"]]

    if len(candidate_cols) == 0:
        raise ValueError(
            f"inventory.csv missing 'inventory' column. Found columns = {df.columns.tolist()}"
        )

    # 自动 rename 到标准列名 inventory
    df = df.rename(columns={candidate_cols[0]: "inventory"})

    return df[["region", "date", "inventory"]]

# -----------------------------------------------------------------------------
# MASTER LOADER (must come AFTER all loaders!)
# -----------------------------------------------------------------------------

def load_housing_data(region: str) -> pd.DataFrame:
    price = load_price()
    inc = load_income()
    pop = load_population()
    mort = load_mortgage()
    rent = load_rent()
    permits = load_permits()
    inventory = load_inventory()

    df = price.merge(inc, on=["region", "date"], how="inner")
    df = df.merge(pop, on=["region", "date"], how="inner")
    df = df.merge(mort, on=["region", "date"], how="inner")
    df = df.merge(rent, on=["region", "date"], how="inner")
    df = df.merge(permits, on=["region", "date"], how="inner")
    df = df.merge(inventory, on=["region", "date"], how="inner")

    # df = df.merge(inventory, on=["region", "date"], how="inner")
    df = df[df["region"] == region].reset_index(drop=True)
    return df

import pandas as pd
from pathlib import Path
from src.utils.project_root import get_project_root

DATA_PATH = get_project_root() / "data" / "raw"

def _load_csv(filename: str) -> pd.DataFrame:
    path = DATA_PATH / filename
    if not path.exists():
        raise FileNotFoundError(f"[DATA FILE NOT FOUND] {path}")
    return pd.read_csv(path)

# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------

def _ensure_cols(df, required, filename):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} missing required columns {missing}. Found {df.columns.tolist()}")
    return df


# -------------------------------------------------------------------------
# RAW LOADERS (WITH MOCK FIXES FOR BAD FILES)
# -------------------------------------------------------------------------

def load_income():
    df = _load_csv("income.csv")
    df = _ensure_cols(df, ["region", "date", "income"], "income.csv")
    return df[["region", "date", "income"]]

def load_price():
    df = _load_csv("housing_price.csv")
    df = df.rename(columns={"value": "price"}) if "value" in df.columns else df
    df = _ensure_cols(df, ["region", "date", "price"], "housing_price.csv")
    return df[["region", "date", "price"]]

def load_rent():
    df = _load_csv("rent.csv")
    if "rent" not in df.columns:
        # mock rent = income * 0.3 / 12 (30% rent burden)
        df = load_income().copy()
        df["rent"] = df["income"] * 0.3 / 12
    return df[["region", "date", "rent"]]

def load_population():
    df = _load_csv("population_migration.csv")
    if "population" not in df.columns or "net_migration" not in df.columns:
        # mock population data
        base = load_income().copy()
        df = base.copy()
        df["population"] = 1_000_000 + df.index * 5000
        df["net_migration"] = df.index * 200
    return df[["region", "date", "population", "net_migration"]]

def load_mortgage():
    df = _load_csv("mortgage_rate.csv")
    if "mortgage_rate" not in df.columns:
        df = load_income().copy()
        df["mortgage_rate"] = 0.04
    return df[["region", "date", "mortgage_rate"]]

def load_permits():
    df = _load_csv("permits.csv")
    if "permits" not in df.columns:
        pop = load_population()
        df = pop.copy()
        df["permits"] = df["population"] * 0.001
    return df[["region", "date", "permits"]]

def load_inventory():
    df = _load_csv("inventory.csv")
    if "inventory" not in df.columns:
        pop = load_population()
        df = pop.copy()
        df["inventory"] = df["population"] * 0.002
    return df[["region", "date", "inventory"]]


# -------------------------------------------------------------------------
# MASTER LOADER
# -------------------------------------------------------------------------

def load_housing_data(region: str) -> pd.DataFrame:

    price = load_price()
    inc = load_income()
    pop = load_population()
    mort = load_mortgage()
    rent = load_rent()
    permits = load_permits()
    inventory = load_inventory()

    df = price.merge(inc, on=["region", "date"], how="inner")
    df = df.merge(pop, on=["region", "date"], how="inner")
    df = df.merge(mort, on=["region", "date"], how="inner")
    df = df.merge(rent, on=["region", "date"], how="inner")
    df = df.merge(permits, on=["region", "date"], how="inner")
    df = df.merge(inventory, on=["region", "date"], how="inner")

    return df[df["region"] == region].reset_index(drop=True)


