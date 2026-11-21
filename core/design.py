"""Slice 2 - Solar tower receiver, heliostat field and thermal-storage sizing.

Source: Gamil, Li, Khammash (2023), Energy Conversion and Management 296, 117698.
Equation numbers in the comments refer to that paper.

VALID ONLY for the regime that paper covers: pressurized LIQUID WATER as the heat
transfer fluid, about 50 to 150 C. Not valid for 200 C steam or molten salt.
"""
import math

from core import demand


# ---------------------------------------------------------------- equations
def solar_multiple(t_tes_h, x_sun_h):
    """Eq. 2: SM = (t_TES + X) / X.  X = operating hours with sunlight."""
    if t_tes_h < 0 or x_sun_h <= 0:
        raise ValueError("Storage hours must be >= 0 and sunlight operating hours > 0.")
    return (t_tes_h + x_sun_h) / x_sun_h


def receiver_incident_power_mw(q_load_mw, eta_rec, sm, eta_pipe=1.0):
    """Eq. 1: Q_th,rec = Q_load / (eta_rec * eta_pipe) * SM."""
    return q_load_mw / (eta_rec * eta_pipe) * sm


def htf_power_mw(q_th_rec_mw, eta_rec):
    """Eq. 3: heat absorbed by the fluid, Q_htf = Q_th,rec * eta_rec."""
    return q_th_rec_mw * eta_rec


def receiver_geometry(q_th_rec_mw, q_flux_limit_kw_m2, r_peak_avg, aspect_ratio):
    """Eq. 7-11: allowable flux -> receiver area -> height and diameter."""
    q_allowable = q_flux_limit_kw_m2 / r_peak_avg           # Eq. 7  (kW/m2)
    area = q_th_rec_mw * 1000.0 / q_allowable               # Eq. 8  (m2)
    height = math.sqrt(area * aspect_ratio / math.pi)       # Eq. 11 (m)
    diameter = height / aspect_ratio                        # Eq. 10 (m)
    return {"q_allowable_kw_m2": q_allowable, "a_rec_m2": area,
            "h_rec_m": height, "d_rec_m": diameter}


def storage_tank(q_htf_mw, t_tes_h, cp_kj_kgk, rho_kg_m3, t_hot_c, t_cold_c, eta_sto=1.0):
    """Eq. 4 and 55-57: fluid mass, volume of ONE tank (two tanks: hot and cold)."""
    mdot = q_htf_mw * 1000.0 / (cp_kj_kgk * (t_hot_c - t_cold_c))   # kg/s   (Eq. 4)
    mass = mdot * t_tes_h * 3600.0                                   # kg     (Eq. 55)
    volume = mass / rho_kg_m3 / eta_sto                              # m3     (Eq. 56-57)
    return {"mdot_kg_s": mdot, "htf_mass_kg": mass, "tank_volume_m3": volume}


def heliostat_field(q_th_rec_mw, dni_w_m2, eta_optical_dp, heliostat_area_m2):
    """Eq. 34-35: A_field = Q_th,rec / (DNI * eta_optical);  N = A_field / A_heliostat."""
    area = q_th_rec_mw * 1e6 / (dni_w_m2 * eta_optical_dp)
    return {"a_field_m2": area, "n_helio": math.ceil(area / heliostat_area_m2)}


# ---------------------------------------------------------------- properties
def _method(table, names):
    for name in names:
        fn = getattr(table, name, None)
        if fn is not None:
            return fn
    raise AttributeError(f"The steam-table library has none of the methods {names}.")


def water_properties(p_bar, t_cold_c, t_hot_c, steam_table=None):
    """cp (kJ/kg.K) and density (kg/m3) of liquid water at the AVERAGE fluid temperature
    (the source evaluates fluid properties at (T_hot + T_cold) / 2)."""
    table = steam_table or demand._default_steam_table()
    t_sat = table.tsat_p(p_bar)
    if t_hot_c >= t_sat:
        raise ValueError(f"At {p_bar} bar water boils at {t_sat:.1f} C. Hot temperature must be below "
                         "that so the water stays liquid. Raise the system pressure or lower the temperature.")
    t_avg = (t_hot_c + t_cold_c) / 2.0
    return {"cp": _method(table, ("Cp_pt", "cp_pt"))(p_bar, t_avg),
            "rho": _method(table, ("rho_pt",))(p_bar, t_avg),
            "t_avg_c": t_avg, "t_sat_c": t_sat}


# ---------------------------------------------------------------- whole design
def size_system(q_load_mw, t_tes_h, x_sun_h, t_cold_c, t_hot_c, p_bar,
                eta_rec, eta_opt_dp, dni_design,
                q_flux_limit_kw_m2=600.0, r_peak_avg=2.0, aspect_ratio=1.5,
                heliostat_area_m2=4.0, sm=None, steam_table=None):
    """Size receiver, heliostat field and storage for a given design heat load."""
    if q_load_mw <= 0:
        raise ValueError("Design heat load must be greater than zero.")
    if t_hot_c <= t_cold_c:
        raise ValueError("Hot temperature must be higher than cold temperature.")
    for name, val in (("Receiver efficiency", eta_rec), ("Field optical efficiency", eta_opt_dp)):
        if not 0 < val <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    if min(dni_design, r_peak_avg, aspect_ratio, q_flux_limit_kw_m2, heliostat_area_m2) <= 0:
        raise ValueError("Design DNI, flux limit, flux ratio, aspect ratio and heliostat area must be positive.")

    sm_used = sm if sm is not None else solar_multiple(t_tes_h, x_sun_h)
    q_rec = receiver_incident_power_mw(q_load_mw, eta_rec, sm_used)
    q_htf = htf_power_mw(q_rec, eta_rec)
    props = water_properties(p_bar, t_cold_c, t_hot_c, steam_table)
    out = {"sm": sm_used, "q_th_rec_mw": q_rec, "q_htf_mw": q_htf, "cp": props["cp"],
           "rho": props["rho"], "t_sat_c": props["t_sat_c"]}
    out.update(receiver_geometry(q_rec, q_flux_limit_kw_m2, r_peak_avg, aspect_ratio))
    out.update(storage_tank(q_htf, t_tes_h, props["cp"], props["rho"], t_hot_c, t_cold_c))
    out.update(heliostat_field(q_rec, dni_design, eta_opt_dp, heliostat_area_m2))
    return out
