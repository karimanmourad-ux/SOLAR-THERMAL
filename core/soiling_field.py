"""Slice 3 - Field-average mirror soiling for a tower plant.

Runs the extracted soiling engine (core/soiling.py) for representative heliostats placed on a
ring around the tower, then averages them. The ring and the night-stow position are SIMPLIFYING
CHOICES of this tool, not results of the source papers.
"""
import io
import math

import numpy as np
import pandas as pd

from core import geometry
from core.soiling import (CleaningPolicy, ConstantMeanParams, Notes, daily_loss_stats,
                          simulate_constant_mean)


def constant_pm(index, pm_ug_m3):
    return pd.Series(float(pm_ug_m3), index=index)


def parse_pm_csv(text, index):
    """Read a two-column CSV (time, PM10 in ug/m3) and align it to the weather timestamps."""
    raw = pd.read_csv(io.StringIO(text))
    if raw.shape[1] < 2:
        raise ValueError("The PM10 file needs two columns: time and PM10 (ug/m3).")
    t = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    s = pd.Series(pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy(), index=t).dropna()
    s = s[s.index.notna()].resample("1h").mean().reindex(index)
    missing = int(s.isna().sum())
    if missing:
        raise ValueError(f"The PM10 file does not cover {missing} of the {len(index)} weather hours. "
                         "Use hourly values for the same year as the weather data.")
    if (s < 0).any():
        raise ValueError("PM10 values cannot be negative.")
    return s


def run_field_soiling(df, lat, lon, pm, params, policy, tower_height_m, distance_m,
                      stow_tilt_deg=180.0, n_azimuths=8):
    """Average soiling of mirrors on a ring of radius distance_m around the tower."""
    if tower_height_m <= 2.0:
        raise ValueError("Tower height must be greater than the heliostat height (2 m).")
    if distance_m < 0:
        raise ValueError("Distance from the tower cannot be negative.")
    idx = df.index
    sun = geometry.sun_unit_vectors(idx, lat, lon)
    notes = Notes()
    rho_op, rho_ref, events, daily = [], [], [], []
    for k in range(n_azimuths):
        az = math.radians(360.0 * k / n_azimuths)
        tilt, aoi = geometry.heliostat_tilt_aoi(sun, distance_m * math.sin(az), distance_m * math.cos(az),
                                                tower_height_m, stow_tilt_deg=stow_tilt_deg)
        met = pd.DataFrame({"pm": pm.to_numpy(float), "tilt_deg": tilt, "aoi_deg": aoi}, index=idx)
        ro, rr, ev, _ = simulate_constant_mean(met, params, policy, None, notes)
        rho_op.append(ro); rho_ref.append(rr); events.append(len(ev))
        daily.append(daily_loss_stats(met, params)["mean_loss_pp"])
    rho_op = pd.Series(np.mean(rho_op, axis=0), index=idx)
    rho_ref = pd.Series(np.mean(rho_ref, axis=0), index=idx)
    w = df["DNI"].fillna(0).clip(lower=0)
    sun_up = w > 0
    rho_w = float((rho_op * w).sum() / w.sum()) if w.sum() > 0 else float("nan")
    # daily weighted reflectance for the chart
    num = (rho_op * w).groupby(idx.date).sum()
    den = w.groupby(idx.date).sum()
    daily_rho = (num / den.replace(0, np.nan)).ffill().bfill()
    dl = pd.concat(daily, axis=1).mean(axis=1)
    notes.add("Assumption", f"The field is represented by {n_azimuths} mirrors on a ring of radius {distance_m:g} m around a tower "
                        f"{tower_height_m:g} m high, and the result is their average. This is a simplification of this tool, not of the source papers.")
    notes.add("Assumption", f"Mirror position at night (sun below the horizon): tilt {stow_tilt_deg:g}° "
                        "(180 = face down: the model deposits no dust on them). An operating choice that affects the result.")
    notes.add("Warning", "NASA POWER hours are assumed to be UTC (still to be confirmed in the NASA documentation); sun position and mirror tilt are computed on that basis.")
    notes.add("Warning", "A constant PM10 ignores dust storms; use a measured hourly PM10 series if available.")
    return {
        "rho_op": rho_op, "rho_ref": rho_ref, "daily_rho": daily_rho,
        "rho_weighted": rho_w,
        "loss_pct": (1.0 - rho_w / params.rho0) * 100.0,
        "rho_ref_min": float(rho_ref.min()),
        "cleanings_per_mirror": float(np.mean(events)),
        "daily_loss_pp": dl, "notes": notes, "sun_up_hours": int(sun_up.sum()),
    }
