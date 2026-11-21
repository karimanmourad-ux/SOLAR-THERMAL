"""Slice 1 - Weather: NASA POWER hourly data and site lookup.

Pure functions where possible, so they can be tested without internet.
"""
import io

import numpy as np
import pandas as pd
import requests

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMETERS = "ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M"
NASA_MISSING_FLAG = -999.0  # NASA POWER marks missing values with -999

RENAME = {
    "ALLSKY_SFC_SW_DNI": "DNI",   # W/m2 (hourly)
    "ALLSKY_SFC_SW_DWN": "GHI",   # W/m2 (hourly)
    "T2M": "T_air",               # deg C at 2 m
    "RH2M": "RH",                 # % at 2 m
    "WS10M": "Wind",              # m/s at 10 m
}

# Approximate city-centre coordinates (deg N, deg E). For a real plant, use
# the "Coordinates" mode and enter the plant's own location.
PRESET_CITIES = {
    "Aswan, Egypt": (24.09, 32.90),
    "Luxor, Egypt": (25.69, 32.64),
    "Cairo, Egypt": (30.04, 31.24),
    "Alexandria, Egypt": (31.20, 29.92),
    "Suez, Egypt": (29.97, 32.55),
    "Riyadh, Saudi Arabia": (24.71, 46.68),
    "Dubai, UAE": (25.20, 55.27),
    "Amman, Jordan": (31.95, 35.93),
    "Casablanca, Morocco": (33.57, -7.59),
    "Ouarzazate, Morocco": (30.92, -6.89),
    "Tunis, Tunisia": (36.81, 10.18),
}


def validate_site(lat, lon, year):
    if not (-90 <= lat <= 90):
        raise ValueError("Latitude must be between -90 and 90.")
    if not (-180 <= lon <= 180):
        raise ValueError("Longitude must be between -180 and 180.")
    if not (2001 <= int(year) <= 2025):
        raise ValueError("Year must be between 2001 and 2025.")


def parse_power_csv(text):
    """Turn NASA POWER CSV text into a clean hourly DataFrame (Datetime index)."""
    lines = text.splitlines()
    header_idx = next((i for i, l in enumerate(lines) if l.startswith("YEAR")), None)
    if header_idx is None:
        raise ValueError("Unexpected NASA POWER response: no 'YEAR' header line found.")
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    missing_cols = [c for c in RENAME if c not in df.columns]
    if missing_cols:
        raise ValueError(f"NASA POWER response lacks columns: {missing_cols}")
    df = df.replace(NASA_MISSING_FLAG, np.nan)
    df.index = pd.to_datetime({"year": df["YEAR"], "month": df["MO"],
                               "day": df["DY"], "hour": df["HR"]})
    df.index.name = "Datetime"
    return df[list(RENAME)].rename(columns=RENAME)


def fetch_hourly(lat, lon, year, timeout=60):
    """Download one year of hourly data for a point. Raises on any failure."""
    validate_site(lat, lon, year)
    params = {
        "parameters": PARAMETERS, "community": "RE",
        "longitude": lon, "latitude": lat,
        "start": f"{int(year)}0101", "end": f"{int(year)}1231", "format": "CSV",
    }
    try:
        resp = requests.get(NASA_POWER_URL, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach NASA POWER: {exc}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"NASA POWER returned status {resp.status_code}.")
    return parse_power_csv(resp.text)


def summarize(df):
    """Annual and monthly resource numbers from the hourly table."""
    monthly = df[["DNI", "GHI"]].groupby(df.index.month).sum() / 1000.0  # kWh/m2
    monthly.index.name = "Month"
    return {
        "dni_kwh_m2": df["DNI"].sum() / 1000.0,
        "ghi_kwh_m2": df["GHI"].sum() / 1000.0,
        "dni_peak": df["DNI"].max(),
        "t_mean": df["T_air"].mean(),
        "t_min": df["T_air"].min(),
        "t_max": df["T_air"].max(),
        "hours": len(df),
        "missing_dni_hours": int(df["DNI"].isna().sum()),
        "monthly": monthly,
    }


def diurnal_profile(df, month):
    """Average DNI for each hour of the day in a given month (hours as in the data)."""
    sub = df[df.index.month == month]
    return sub["DNI"].groupby(sub.index.hour).mean()


def geocode(query, timeout=10):
    """Look a place name up on OpenStreetMap Nominatim. Returns (lat, lon, name) or None."""
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "limit": 1},
        headers={"User-Agent": "csp-tes-student-project"},
        timeout=timeout,
    )
    resp.raise_for_status()
    hits = resp.json()
    if not hits:
        return None
    return round(float(hits[0]["lat"]), 4), round(float(hits[0]["lon"]), 4), hits[0]["display_name"]
