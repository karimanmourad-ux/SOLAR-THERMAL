# -*- coding: utf-8 -*-
"""Slice 3 - Mirror soiling engine.

Extracted UNCHANGED from the group's soiling tool (csp_soiling_tool_v3): the simplified
constant-mean model of Picotti et al. 2023 (Solar Energy 263:111945, arXiv:2304.11814),
with cleaning logic after Anderson et al. 2023 (arXiv:2306.06106) and Pappa et al. 2025.
Only the pieces needed for the app were kept (no GUI, plotting, calibration or economics).

IMPORTANT: the model was validated on heliostat mirrors at three Australian sites. It has
NOT been validated for Egypt. Its dust parameters (mu, sigma_dep) must be calibrated from a
local reflectance campaign and local PM10 data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


PHI_REF_DEFAULT = 15.0   # [P23] The D&S reflectometer measures at a fixed 15° incidence

PHI_MIN_VALID = 1.0   # [P23] Eq.5 is valid for angles > 1°

RHO_THRESHOLD_DEFAULT = 0.8   # [P25] Reflectance limit for cleaning (an operating choice)

ETA_CLEAN_DEFAULT = 0.99   # [A23] Table 3: high-pressure wash + brush

SOURCES = {
    "P23": "Picotti et al. 2023, Solar Energy 263:111945 — https://arxiv.org/abs/2304.11814",
    "A23": "Anderson et al. 2023, Applied Energy — https://arxiv.org/abs/2306.06106",
    "P25": "Pappa et al. 2025, Atmosphere 16(7):807 — https://doi.org/10.3390/atmos16070807",
}

class Notes:
    ORDER = ["Source", "Assumption", "Gap", "Warning", "Info"]

    def __init__(self) -> None:
        self.items: List[tuple] = []

    def add(self, level: str, text: str) -> None:
        item = (level, text)
        if item not in self.items:
            self.items.append(item)

    def show(self) -> None:
        print("\n" + "=" * 72)
        print("Important notes for the user (read before using the results)")
        print("=" * 72)
        for lvl in self.ORDER:
            group = [t for (l, t) in self.items if l == lvl]
            if not group:
                continue
            print(f"\n[{lvl}]")
            for t in group:
                print(f"  - {t}")
        print("=" * 72 + "\n")

@dataclass
class CleaningPolicy:
    kind: str = "threshold"                   # "threshold" ([P25] ρ<=0.8) | "fixed" ([A23] Eq.17-18)
    rho_threshold: float = RHO_THRESHOLD_DEFAULT
    cleans_per_year: Optional[int] = None
    clean_hour: int = 0   # [A23] 3.3.1: instantaneous, at midnight
    eta_cl: float = ETA_CLEAN_DEFAULT         # [A23] Eq.31

@dataclass
class RainRule:
    """Optional rain rule **chosen by the user**: there is no scientifically established rule (see the notes).
    If the day's rain >= daily_threshold_mm, a share `efficiency` of the soiled area is removed at the first step of the next day."""
    daily_threshold_mm: Optional[float] = None
    efficiency: float = 0.0

@dataclass
class ConstantMeanParams:
    """[P23] The simplified constant-mean model. The values of μ̃ and σ_dep are **per step Δt**, exactly as in the paper."""
    mu: float   # μ̃ (area covered per (unit of PM x step))
    sigma_dep: float   # σ_dep (uncertainty)
    dt_hours: float   # Δt at calibration time: must equal the prediction step
    rho0: float   # ρ0: clean-mirror reflectance (measured)
    second_surface: bool = True
    phi_ref_deg: float = PHI_REF_DEFAULT
    pm_label: str = "PM10"   # dust measure used in calibration (must be the same in prediction)
    mu_ci: Optional[Tuple[float, float]] = None
    sigma_ci: Optional[Tuple[float, float]] = None

def _require(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

def _time_step_hours(index: pd.DatetimeIndex) -> float:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    d = np.diff(index.values).astype("timedelta64[s]").astype(float)
    if len(d) == 0 or not np.allclose(d, d[0]):
        raise ValueError("The time step must be regular. If your data has daylight-saving changes, convert it "
                         "to UTC or to a fixed offset (a repeated or skipped hour breaks the regularity).")
    return float(d[0]) / 3600.0

def incidence_factor_h(phi_deg, second_surface: bool = True) -> np.ndarray:
    """h(φ) from [P23] Eq.5: second-surface -> 2/cosφ (valid for φ>1°); first-surface -> (1+sinφ)/cosφ ([P23] Eq.5 text)."""
    phi = np.minimum(np.asarray(phi_deg, float), 89.0)
    rad = np.radians(phi)
    if second_surface:
        return 2.0 / np.cos(np.radians(np.maximum(phi, PHI_MIN_VALID)))
    return (1.0 + np.sin(rad)) / np.cos(rad)

def _a_series(pm: np.ndarray, tilt_deg: np.ndarray) -> np.ndarray:
    """α_i·cosθ_i from [P23] Eq.17/30 with cosθ clipped at 0 (our assumption for an inverted mirror)."""
    return np.asarray(pm, float) * np.clip(np.cos(np.radians(np.asarray(tilt_deg, float))), 0.0, None)

def simulate_constant_mean(met: pd.DataFrame, p: ConstantMeanParams, pol: CleaningPolicy,
                           rain: Optional[RainRule], notes: Notes):
    _require(met, ["pm", "tilt_deg", "aoi_deg"])
    dt_h = _time_step_hours(met.index)
    if abs(dt_h - p.dt_hours) > 1e-9:
        raise ValueError(f"The prediction step ({dt_h:g}h) must equal the calibration step ({p.dt_hours:g}h). "
                         "Aggregate or resample the PM data so that μ̃ and σ_dep are per step ([P23]).")
    if met[["pm", "tilt_deg", "aoi_deg"]].isna().any().any():
        raise ValueError("NaN found in pm/tilt_deg/aoi_deg")
    nt = len(met)
    a = _a_series(met["pm"].to_numpy(float), met["tilt_deg"].to_numpy(float))
    aoi = met["aoi_deg"].to_numpy(float)
    h_t = incidence_factor_h(aoi, p.second_surface)
    h_ref = float(incidence_factor_h(p.phi_ref_deg, p.second_surface))
    n_clip = int((aoi < PHI_MIN_VALID).sum()) if p.second_surface else 0

    notes.add("Source", f"Reflectance: [P23] Eq.1 ρ = ρ0·(1 − A_soil·h(φ)), h=2/cosφ (second-surface); deposition Eq.17/28: ΔA = PM·cosθ·μ̃ per step.")
    notes.add("Assumption", "The reflectance limit and CF are computed on a reading at a measurement angle of " + f"{p.phi_ref_deg:g}°" +
              " (like a standard reflectometer), while the field heat uses the actual incidence angle aoi_deg. This is our choice.")
    notes.add("Assumption", "cosθ is clipped at 0 (no deposition on an inverted mirror), and there is no roll-off removal mechanism (D_c): in the simplified model it is absorbed inside μ̃.")
    if n_clip:
        notes.add("Warning", f"{n_clip} steps have aoi < {PHI_MIN_VALID:g}° where [P23] Eq.5 is not valid; the angle was clipped at {PHI_MIN_VALID:g}° "
                           "(the loss may be overstated in these steps by up to ~2x).")
    if not p.second_surface:
        notes.add("Assumption", "First-surface mirrors: h=(1+sinφ)/cosφ ([P23]). Check the mirror type of your plant.")

    clean_flag = np.zeros(nt, bool)
    if pol.kind == "fixed":
        if not pol.cleans_per_year:
            raise ValueError("fixed needs cleans_per_year")
        dtd = 24.0 / dt_h
        dk = 365.0 * dtd / pol.cleans_per_year
        for i in {1} | {math.ceil(i * dk) for i in range(1, pol.cleans_per_year)}:
            if 1 <= i <= nt:
                clean_flag[i - 1] = True
        notes.add("Source", f"Fixed-frequency cleaning [A23] Eq.17-18 (F={pol.cleans_per_year}/year).")
        if nt * dt_h < 365 * 24 * 0.95:
            notes.add("Warning", "The simulation is shorter than a year: the Eq.17 schedule assumes a full year.")
    elif pol.kind == "threshold":
        notes.add("Source", f"Cleaning at a reflectance limit ρ ≤ {pol.rho_threshold} ([P25] used 0.8).")
        notes.add("Warning", f"A limit of {pol.rho_threshold} is an operating choice, not a rule; each plant decides by cleaning cost and heat value.")
        notes.add("Assumption", f"Cleaning is carried out instantly at {pol.clean_hour}:00 after the limit is crossed ([A23] 3.3.1), with efficiency η_cl={pol.eta_cl} "
                             "([A23] Eq.31 on the soiled area; mathematically sound because the area is linear in N(D) if η_cl does not depend on size).")
    else:
        raise ValueError("kind: threshold | fixed")

    # Rain rule (optional)
    rain_clean_steps = np.zeros(nt, bool)
    if rain is not None and rain.daily_threshold_mm is not None:
        if "precip_mm" not in met.columns:
            raise KeyError("RainRule needs a precip_mm column")
        daily = met["precip_mm"].resample("D").sum()
        wet_days = set(daily.index[daily >= rain.daily_threshold_mm].date)
        for k, ts in enumerate(met.index):
            if ts.hour == 0 and ts.minute == 0 and (ts - pd.Timedelta(days=1)).date() in wet_days:
                rain_clean_steps[k] = True
        notes.add("Assumption", f"Rain rule chosen by you: a day with ≥ {rain.daily_threshold_mm} mm removes {100 * rain.efficiency:.0f}% of the dust. "
                            "Not a scientifically established rule — the 10.8 mm/day threshold comes from El Boujdaini 2022 (abstract only) for one site in Morocco.")

    A = 0.0
    rho_op = np.empty(nt)
    rho_ref = np.empty(nt)
    events: List[pd.Timestamp] = []
    pending = False
    hours, mins = met.index.hour.to_numpy(), met.index.minute.to_numpy()
    for k in range(nt):
        do_clean = clean_flag[k]
        if pol.kind == "threshold" and pending and hours[k] == pol.clean_hour and mins[k] == 0:
            do_clean, pending = True, False
        if do_clean:
            A *= (1.0 - pol.eta_cl)
            events.append(met.index[k])
        if rain_clean_steps[k]:
            A *= (1.0 - rain.efficiency)
        A += a[k] * p.mu
        rho_op[k] = max(p.rho0 * (1.0 - A * h_t[k]), 0.0)
        rho_ref[k] = max(p.rho0 * (1.0 - A * h_ref), 0.0)
        if pol.kind == "threshold" and rho_ref[k] <= pol.rho_threshold:
            pending = True
    return rho_op, rho_ref, events, a

def daily_loss_stats(met: pd.DataFrame, p: ConstantMeanParams) -> pd.DataFrame:
    """Distribution of the expected daily loss (without cleaning): mean and deviation [P23] Eq.22-23 with σ_r=0 (and Eq.31-32 for the loads)."""
    a = pd.Series(_a_series(met["pm"].to_numpy(float), met["tilt_deg"].to_numpy(float)), index=met.index)
    b = p.rho0 * float(incidence_factor_h(p.phi_ref_deg, p.second_surface))
    s1 = a.resample("D").sum()
    s2 = (a ** 2).resample("D").sum()
    df = pd.DataFrame({"mean_loss_pp": 100.0 * b * p.mu * s1, "sd_loss_pp": 100.0 * p.sigma_dep * b * np.sqrt(s2)})
    df["lo95_pp"] = df["mean_loss_pp"] - 1.96 * df["sd_loss_pp"]
    df["hi95_pp"] = df["mean_loss_pp"] + 1.96 * df["sd_loss_pp"]
    return df
