"""Slice 1 - Process heat demand: from MW directly, or from steam conditions."""


def heat_duty_mw(m_kg_s, h_out_kj_kg, h_in_kj_kg):
    """Q = m * (h_out - h_in).  kg/s * kJ/kg = kW, divided by 1000 -> MW."""
    return m_kg_s * (h_out_kj_kg - h_in_kj_kg) / 1000.0


def tph_to_kg_s(flow_tph):
    return flow_tph * 1000.0 / 3600.0


def _default_steam_table():
    from pyXSteam.XSteam import XSteam  # imported lazily: only needed for steam mode
    return XSteam(XSteam.UNIT_SYSTEM_MKS)  # bar, deg C, kJ/kg, kg/s


def steam_heat_duty(p_bar, t_feedwater_c, t_steam_c, flow_tph, steam_table=None):
    """Heat the steam carries: superheated steam out minus subcooled feedwater in.

    Raises ValueError with a readable message for physically invalid inputs.
    """
    if min(p_bar, flow_tph) <= 0:
        raise ValueError("Pressure and steam flow must be greater than zero.")
    st = steam_table or _default_steam_table()
    t_sat = st.tsat_p(p_bar)
    if t_steam_c <= t_sat:
        raise ValueError(f"Steam temperature must be above saturation ({t_sat:.1f} C at {p_bar} bar) "
                         "for superheated steam.")
    if t_feedwater_c >= t_sat:
        raise ValueError(f"Feedwater must be liquid: its temperature must be below {t_sat:.1f} C at {p_bar} bar.")
    h_in = st.h_pt(p_bar, t_feedwater_c)
    h_out = st.h_pt(p_bar, t_steam_c)
    m = tph_to_kg_s(flow_tph)
    return {"q_mw": heat_duty_mw(m, h_out, h_in), "t_sat_c": t_sat,
            "h_in": h_in, "h_out": h_out, "delta_h": h_out - h_in, "m_kg_s": m}


def annual_demand_mwh(q_mw, hours_per_day, days_per_year):
    """Energy per year, assuming constant load while the plant operates."""
    if not (0 < hours_per_day <= 24):
        raise ValueError("Hours per day must be between 0 and 24.")
    if not (0 < days_per_year <= 366):
        raise ValueError("Days per year must be between 0 and 366.")
    return q_mw * hours_per_day * days_per_year


def load_factor(hours_per_day, days_per_year):
    return hours_per_day * days_per_year / 8760.0
