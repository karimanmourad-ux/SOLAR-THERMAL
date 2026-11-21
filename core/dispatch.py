# -*- coding: utf-8 -*-
"""Slice 4 - Hourly simulation: solar field heat, thermal storage, backup.

The three pieces below (Plant, field_heat_kw, storage_run) are extracted UNCHANGED from the
group's soiling tool v3. They follow Anderson et al. 2023 (arXiv:2306.06106), Eq. 24-26
(storage) and Eq. 28-30 (field heat), with the plant's heat demand replacing the power-block
demand. Everything after the marker line is new code for this app.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Plant:
    """Plant data: field, storage and factory (general for any factory)."""
    area_m2: float
    eta_opt: float | pd.Series   # optical efficiency **excluding** reflectance (cos, τ, α, γ, IAM, end-loss...)
    q_loss_kw: float   # constant Q_loss [A23] Eq.28
    demand_kw: float | pd.Series   # factory heat demand (replaces Q_pb in [A23] Eq.24)
    s_max_kwh: float   # TES capacity
    q_max_kw: Optional[float] = None
    s0_kwh: float = 0.0

def field_heat_kw(dni_wm2: np.ndarray, rho: np.ndarray, plant: Plant, idx: pd.DatetimeIndex) -> np.ndarray:
    eta = plant.eta_opt.reindex(idx).to_numpy(float) if isinstance(plant.eta_opt, pd.Series) else plant.eta_opt
    q_field = dni_wm2 * plant.area_m2 * eta * rho / 1000.0                     # Eq.30
    q = np.where(q_field <= plant.q_loss_kw, 0.0, q_field - plant.q_loss_kw)   # Eq.28-29 (interpretation)
    if plant.q_max_kw is not None:
        q = np.where(q_field > plant.q_max_kw, plant.q_max_kw - plant.q_loss_kw, q)
    return q

def storage_run(q_rec_kw: np.ndarray, demand_kw: np.ndarray, plant: Plant, dt_h: float):
    n = len(q_rec_kw)
    s = plant.s0_kwh
    q_tes, s_hist, wasted = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        if s / dt_h >= plant.s_max_kwh / dt_h:                                  # Eq.26
            q_eff = 0.0
        elif s / dt_h + q_rec_kw[k] > plant.s_max_kwh / dt_h:
            q_eff = (plant.s_max_kwh - s) / dt_h
        else:
            q_eff = q_rec_kw[k]
        wasted[k] = q_rec_kw[k] - q_eff
        avail = s / dt_h + q_eff   # Eq.24 (Q_pb <- factory demand)
        if avail > demand_kw[k]:
            q_tes[k] = demand_kw[k]
        elif avail > 0.0:
            q_tes[k] = avail
        s = s + (q_eff - q_tes[k]) * dt_h                                       # Eq.25
        s_hist[k] = s
    return q_tes, s_hist, wasted



# ===================================================================== new code below
def operating_demand_kw(index, q_mw, start_hour_local, hours_per_day, days_per_year, utc_offset_h):
    """Heat demand in kW for every hour: constant q_mw during the operating window, 0 otherwise.

    Operating days are spread evenly through the year. Window must end before midnight."""
    if start_hour_local + hours_per_day > 24:
        raise ValueError("Operation would pass midnight. Start earlier or use fewer hours per day.")
    if not (0 < days_per_year <= 365):
        raise ValueError("Operating days per year must be between 1 and 365.")
    local = index + pd.Timedelta(hours=utc_offset_h)
    hour = local.hour.to_numpy()
    day = local.dayofyear.to_numpy().clip(1, 365)
    on_day = np.floor(day * days_per_year / 365.0) != np.floor((day - 1) * days_per_year / 365.0)
    in_window = (hour >= start_hour_local) & (hour < start_hour_local + hours_per_day)
    return np.where(on_day & in_window, q_mw * 1000.0, 0.0)


def simulate_year(df, area_m2, eta_opt_no_refl, rho, availability, eta_rec, storage_kwh, demand_kw):
    """One hourly run. rho = mirror reflectance for each hour (array or Series)."""
    idx = df.index
    dni = df["DNI"].fillna(0).clip(lower=0).to_numpy(float)
    rho_arr = np.asarray(rho, float)
    plant = Plant(area_m2=area_m2, eta_opt=eta_opt_no_refl, q_loss_kw=0.0, demand_kw=demand_kw,
                  s_max_kwh=storage_kwh)
    q_rec = field_heat_kw(dni, rho_arr, plant, idx) * eta_rec * availability
    delivered, level, curtailed = storage_run(q_rec, demand_kw, plant, 1.0)
    return {"q_rec": q_rec, "delivered": delivered, "level": level, "curtailed": curtailed}


def summarize(df, run, demand_kw):
    idx = df.index
    month = idx.month
    t = pd.DataFrame({"gross": run["q_rec"], "delivered": run["delivered"], "demand": demand_kw,
                      "curtailed": run["curtailed"]}, index=idx)
    monthly = t.groupby(month).sum() / 1e6                                   # GWh
    monthly["backup"] = monthly["demand"] - monthly["delivered"]
    dem = float(demand_kw.sum()) / 1e6
    dlv = float(run["delivered"].sum()) / 1e6
    return {"gross_gwh": float(run["q_rec"].sum()) / 1e6, "demand_gwh": dem, "delivered_gwh": dlv,
            "backup_gwh": dem - dlv, "curtailed_gwh": float(run["curtailed"].sum()) / 1e6,
            "solar_fraction": dlv / dem if dem > 0 else float("nan"),
            "storage_end_gwh": float(run["level"][-1]) / 1e6, "monthly": monthly}


def typical_day(df, run, demand_kw, month, utc_offset_h):
    """Average hourly profile (kW) for one month, by local hour."""
    local = df.index + pd.Timedelta(hours=utc_offset_h)
    sel = df.index.month == month
    t = pd.DataFrame({"Solar heat to the system": run["q_rec"], "Delivered to the process": run["delivered"],
                      "Heat demand": demand_kw, "Storage level (kWh)": run["level"]}, index=df.index)[sel]
    return t.groupby(local[sel].hour).mean()
