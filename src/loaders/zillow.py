"""
Zillow ZHVI loader — Zillow Home Value Index (SFRCONDO, middle tier, SA, monthly).

Source: Zillow Research Data public CSV (no API key required).
URL:    https://files.zillowstatic.com/research/public_csvs/zhvi/
        Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv

The wide CSV has one row per metro and one column per month.
This loader melts it to long format, filters to the requested metros,
and returns a quarterly panel (month-end → quarter-end resampled).

Metro name strings used in the Zillow RegionName column:
  Austin       "Austin, TX"
  Phoenix      "Phoenix, AZ"
  Las Vegas    "Las Vegas, NV"
  Miami        "Miami-Fort Lauderdale, FL"
"""
from __future__ import annotations

from io import StringIO
from typing import Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ZILLOW_ZHVI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)

# Mapping from internal city key → Zillow RegionName string
ZILLOW_METRO_NAMES: dict[str, str] = {
    "austin":    "Austin, TX",
    "phoenix":   "Phoenix, AZ",
    "las_vegas": "Las Vegas, NV",
    "miami":     "Miami, FL",
}

# Non-date columns in the Zillow wide CSV
_META_COLS = {
    "RegionID", "SizeRank", "RegionName", "RegionType",
    "StateName", "State", "City", "Metro", "CountyName",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_zhvi_raw() -> pd.DataFrame:
    """Download the Zillow ZHVI Metro CSV and return as a wide DataFrame."""
    resp = requests.get(ZILLOW_ZHVI_URL, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


def _wide_to_long(raw: pd.DataFrame, city: str) -> pd.DataFrame:
    """Extract one city from the wide Zillow DataFrame and return long format."""
    metro_name = ZILLOW_METRO_NAMES[city]
    row = raw[(raw["RegionName"] == metro_name) & (raw["RegionType"] == "msa")]
    if row.empty:
        raise ValueError(
            f"City '{city}' (metro '{metro_name}') not found in Zillow data. "
            f"Available metros: {raw['RegionName'].tolist()[:20]} ..."
        )
    date_cols = [c for c in raw.columns if c not in _META_COLS]
    series = row[date_cols].iloc[0]
    df = pd.DataFrame({
        "date":  pd.to_datetime(date_cols),
        "price": series.values.astype(float),
    })
    return df.dropna(subset=["price"]).reset_index(drop=True)


def _to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample monthly ZHVI to quarterly (last observation in quarter)."""
    df = df.set_index("date").sort_index()
    df = df.resample("QE").last().dropna()
    return df.reset_index()


# ---------------------------------------------------------------------------
# Single-city loader
# ---------------------------------------------------------------------------

def load_zillow_zhvi(
    city: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    raw: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Pull Zillow ZHVI for one city.

    Parameters
    ----------
    city : str
        One of the keys in ZILLOW_METRO_NAMES (e.g. "austin", "phoenix").
    start : str or None
        ISO date string lower bound (inclusive). None = all available.
    end : str or None
        ISO date string upper bound (inclusive). None = all available.
    raw : pd.DataFrame or None
        Pre-downloaded wide Zillow DataFrame. Pass this when loading multiple
        cities to avoid repeated HTTP requests. None → download fresh.

    Returns
    -------
    DataFrame with columns: region, date, price, source_label, data_weight
        price       : ZHVI index level (nominal USD median home value)
        source_label: "Zillow"
        data_weight : 0.5
    """
    if city not in ZILLOW_METRO_NAMES:
        raise ValueError(
            f"Unknown city '{city}'. "
            f"Available: {sorted(ZILLOW_METRO_NAMES)}"
        )
    if raw is None:
        raw = _fetch_zhvi_raw()

    df = _wide_to_long(raw, city)
    df = _to_quarterly(df)

    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]

    df["region"] = city
    df["source_label"] = "Zillow"
    df["data_weight"] = 0.5
    return df[["region", "date", "price", "source_label", "data_weight"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-city panel loader
# ---------------------------------------------------------------------------

def load_zillow_zhvi_panel(
    cities: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Pull Zillow ZHVI for multiple cities and return a combined panel.

    Downloads the Zillow CSV once and filters per city.

    Parameters
    ----------
    cities : list[str] or None
        Cities to include. None → all cities in ZILLOW_METRO_NAMES.
    start, end : str or None
        Date range passed to each city's extraction.

    Returns
    -------
    Panel DataFrame with columns: region, date, price, source_label, data_weight
    Sorted by region, date.
    """
    if cities is None:
        cities = list(ZILLOW_METRO_NAMES.keys())
    unknown = [c for c in cities if c not in ZILLOW_METRO_NAMES]
    if unknown:
        raise ValueError(f"Unknown cities: {unknown}. Available: {sorted(ZILLOW_METRO_NAMES)}")

    raw = _fetch_zhvi_raw()
    frames = [load_zillow_zhvi(city, start=start, end=end, raw=raw) for city in cities]
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["region", "date"])
        .reset_index(drop=True)
    )
