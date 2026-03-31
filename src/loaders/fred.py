"""
FRED HPI loader — FHFA All-Transactions House Price Index, quarterly (not SA).

Series IDs follow the ATNHPIUS{MSA_CODE}Q pattern from FRED:
  https://fred.stlouisfed.org/categories/44

Metro MSA/MSAD codes used:
  Austin-Round Rock, TX               12420  (MSA)
  Phoenix-Mesa-Scottsdale, AZ         38060  (MSA)
  Las Vegas-Henderson-Paradise, NV    29820  (MSA)
  Miami-Miami Beach-Kendall, FL       33124  (MSAD — core Miami division;
                                             full MSA 33100 not available on FRED)

Uses FRED's public CSV endpoint directly (no API key, no pandas-datareader):
  https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES_ID}
"""
from __future__ import annotations

from io import StringIO
from typing import Optional

import pandas as pd
import requests

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# ---------------------------------------------------------------------------
# Series registry
# ---------------------------------------------------------------------------

FRED_HPI_SERIES: dict[str, str] = {
    "austin":    "ATNHPIUS12420Q",
    "phoenix":   "ATNHPIUS38060Q",
    "las_vegas": "ATNHPIUS29820Q",
    "miami":     "ATNHPIUS33124Q",  # Miami-Miami Beach-Kendall MSAD (full MSA 33100 not on FRED)
}


# ---------------------------------------------------------------------------
# Single-city loader
# ---------------------------------------------------------------------------

def load_fred_hpi(
    city: str,
    start: str = "1990-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Pull FHFA All-Transactions HPI from FRED for one city.

    Parameters
    ----------
    city : str
        One of the keys in FRED_HPI_SERIES (e.g. "austin", "phoenix").
    start : str
        ISO date string for the start of the pull.
    end : str or None
        ISO date string for the end of the pull; None = latest available.

    Returns
    -------
    DataFrame with columns: region, date, price, source_label, data_weight
        price       : raw index level (FHFA HPI, base ~100 at 1995Q1)
        source_label: "FRED"
        data_weight : 1.0
    """
    if city not in FRED_HPI_SERIES:
        raise ValueError(
            f"Unknown city '{city}'. "
            f"Available: {sorted(FRED_HPI_SERIES)}"
        )
    series_id = FRED_HPI_SERIES[city]
    url = _FRED_CSV_URL.format(series_id=series_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    raw = pd.read_csv(StringIO(resp.text))
    raw.columns = ["date", "price"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw = raw.dropna(subset=["price"])
    if start:
        raw = raw[raw["date"] >= pd.Timestamp(start)]
    if end:
        raw = raw[raw["date"] <= pd.Timestamp(end)]
    raw["region"] = city
    raw["source_label"] = "FRED"
    raw["data_weight"] = 1.0
    return raw[["region", "date", "price", "source_label", "data_weight"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-city panel loader
# ---------------------------------------------------------------------------

def load_fred_hpi_panel(
    cities: Optional[list[str]] = None,
    start: str = "1990-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Pull FHFA HPI for multiple cities and return a combined panel.

    Parameters
    ----------
    cities : list[str] or None
        Cities to include. None → all cities in FRED_HPI_SERIES.
    start, end : str
        Date range passed to each city's pull.

    Returns
    -------
    Panel DataFrame with columns: region, date, price, source_label, data_weight
    Sorted by region, date.
    """
    if cities is None:
        cities = list(FRED_HPI_SERIES.keys())
    unknown = [c for c in cities if c not in FRED_HPI_SERIES]
    if unknown:
        raise ValueError(f"Unknown cities: {unknown}. Available: {sorted(FRED_HPI_SERIES)}")
    frames = [load_fred_hpi(city, start=start, end=end) for city in cities]
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["region", "date"])
        .reset_index(drop=True)
    )
