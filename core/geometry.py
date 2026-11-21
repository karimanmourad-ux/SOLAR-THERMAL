"""Slice 3 - Sun position and heliostat orientation (standard solar geometry).

Sun position: Duffie and Beckman, Solar Engineering of Thermal Processes (declination and
equation of time). Heliostat: the mirror normal bisects the directions to the sun and to the
receiver. No empirical parameters. Timestamps are assumed to be UTC (as believed for the
NASA POWER hourly data; to be confirmed).
"""
import numpy as np
import pandas as pd


def sun_unit_vectors(index, lat_deg, lon_deg):
    """Unit vector towards the sun as (east, north, up) arrays for each timestamp (UTC)."""
    n = index.dayofyear.to_numpy(float)
    b = np.radians(360.0 * (n - 1) / 365.0)
    eq_time_min = 229.2 * (0.000075 + 0.001868 * np.cos(b) - 0.032077 * np.sin(b)
                           - 0.014615 * np.cos(2 * b) - 0.04089 * np.sin(2 * b))
    decl = np.radians(23.45 * np.sin(np.radians(360.0 * (284.0 + n) / 365.0)))
    utc_h = index.hour.to_numpy(float) + index.minute.to_numpy(float) / 60.0
    solar_h = utc_h + (4.0 * lon_deg + eq_time_min) / 60.0       # lon east positive
    omega = np.radians(15.0 * (solar_h - 12.0))                  # hour angle, afternoon positive
    phi = np.radians(lat_deg)
    east = -np.cos(decl) * np.sin(omega)
    north = np.cos(phi) * np.sin(decl) - np.sin(phi) * np.cos(decl) * np.cos(omega)
    up = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(omega)
    return east, north, up


def heliostat_tilt_aoi(sun, x_east_m, y_north_m, tower_height_m, heliostat_height_m=2.0,
                       stow_tilt_deg=180.0):
    """Mirror tilt from horizontal and angle of incidence for a heliostat at (x, y) from the tower.

    While the sun is up the mirror normal n = (s + t) / |s + t|, with s the sun direction and t the
    direction to the receiver. Tilt = angle between n and the vertical; incidence = angle between s
    and n. While the sun is down the mirror sits at the stow tilt (180 = face down, 0 = face up).
    """
    se, sn, su = sun
    dz = tower_height_m - heliostat_height_m
    dist = np.sqrt(x_east_m ** 2 + y_north_m ** 2 + dz ** 2)
    te, tn, tu = -x_east_m / dist, -y_north_m / dist, dz / dist
    ne, nn, nu = se + te, sn + tn, su + tu
    norm = np.sqrt(ne ** 2 + nn ** 2 + nu ** 2)
    norm = np.where(norm < 1e-9, 1e-9, norm)
    ne, nn, nu = ne / norm, nn / norm, nu / norm
    tilt = np.degrees(np.arccos(np.clip(nu, -1, 1)))
    cos_aoi = np.clip(se * ne + sn * nn + su * nu, -1, 1)
    aoi = np.degrees(np.arccos(cos_aoi))
    up = su > 0
    return np.where(up, tilt, stow_tilt_deg), np.where(up, aoi, 0.0)
