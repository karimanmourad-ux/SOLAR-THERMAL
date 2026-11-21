"""Solar Process Heat Tool: concentrated solar tower + thermal storage for industrial process heat.

The interface only calls functions from core/. No physics lives in this file.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import demand, design, dispatch, economics, soiling, soiling_field, sources, weather

st.set_page_config(page_title="Solar Process Heat Tool", page_icon="☀️", layout="wide")
st.markdown(
    """<style>
    div[data-testid="stMetric"] {background: rgba(128,128,128,0.08);
        padding: 12px 16px; border-radius: 10px;}
    </style>""",
    unsafe_allow_html=True,
)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@st.cache_data(show_spinner="Downloading hourly data from NASA POWER...", ttl=86400)
def load_weather(lat, lon, year):
    return weather.fetch_hourly(lat, lon, year)


def _soiling_inputs(pm_text, pm_const, mu, sigma, rho0, kind, threshold, cleans, eta_cl, df):
    pm = soiling_field.parse_pm_csv(pm_text, df.index) if pm_text else soiling_field.constant_pm(df.index, pm_const)
    params = soiling.ConstantMeanParams(mu=mu, sigma_dep=sigma, dt_hours=1.0, rho0=rho0)
    policy = soiling.CleaningPolicy(kind=kind, rho_threshold=threshold,
                                    cleans_per_year=int(cleans) if kind == "fixed" else None, eta_cl=eta_cl)
    return pm, params, policy


@st.cache_data(show_spinner="Running the soiling simulation...", ttl=3600)
def run_soiling(lat, lon, year, pm_text, pm_const, mu, sigma, rho0, kind, threshold, cleans, eta_cl,
                tower_h, dist, stow):
    df = load_weather(lat, lon, year)
    pm, params, policy = _soiling_inputs(pm_text, pm_const, mu, sigma, rho0, kind, threshold, cleans, eta_cl, df)
    return soiling_field.run_field_soiling(df, lat, lon, pm, params, policy, tower_h, dist, stow)


@st.cache_data(show_spinner="Comparing cleaning schedules...", ttl=3600)
def soiling_vs_cleaning(lat, lon, year, pm_text, pm_const, mu, sigma, rho0, eta_cl, tower_h, dist, stow):
    df = load_weather(lat, lon, year)
    rows = []
    for n in (1, 4, 12, 26, 52):
        pm, params, policy = _soiling_inputs(pm_text, pm_const, mu, sigma, rho0, "fixed", 0.8, n, eta_cl, df)
        r = soiling_field.run_field_soiling(df, lat, lon, pm, params, policy, tower_h, dist, stow)
        rows.append({"Cleanings per year": n, "Energy-weighted soiling loss (%)": round(r["loss_pct"], 2),
                     "Average reflectance in operation": round(r["rho_weighted"], 3)})
    return pd.DataFrame(rows)


ETA_REF_SOURCE = 0.88  # mirror reflectivity inside the source's field efficiency (Gamil 2023, text after Eq. 42)


@st.cache_data(show_spinner="Running the hourly simulation...", ttl=3600)
def run_simulation(lat, lon, year, area_m2, storage_kwh, q_mw, start_h, hours_day, days_year, tz,
                   eta_w, eta_rec, avail, use_soil, soil_args):
    df = load_weather(lat, lon, year)
    rho0 = soil_args[4]
    clean = None
    if use_soil:
        rho = run_soiling(lat, lon, year, *soil_args)["rho_op"].to_numpy()
    else:
        rho = np.full(len(df), rho0)
    eta_no_refl = eta_w / ETA_REF_SOURCE
    dem = dispatch.operating_demand_kw(df.index, q_mw, start_h, hours_day, days_year, tz)
    run = dispatch.simulate_year(df, area_m2, eta_no_refl, rho, avail, eta_rec, storage_kwh, dem)
    out = {"df": df, "run": run, "demand": dem, "summary": dispatch.summarize(df, run, dem)}
    if use_soil:
        run_c = dispatch.simulate_year(df, area_m2, eta_no_refl, np.full(len(df), rho0), avail, eta_rec,
                                       storage_kwh, dem)
        out["delivered_clean_gwh"] = float(run_c["delivered"].sum()) / 1e6
    return out


@st.cache_data(ttl=3600)
def find_place(query):
    try:
        return weather.geocode(query)
    except Exception:
        return None


st.title("☀️ Solar Process Heat Tool")
st.caption("Concentrated solar tower + thermal storage for industrial process heat in Egypt and MENA. "
           "Preliminary feasibility tool. Every number is traced in the Sources & method tab.")

# ------------------------------------------------------------------ sidebar
st.sidebar.header("📍 Site")
site_mode = st.sidebar.radio("Choose the site by", ["City list", "Search by name", "Coordinates"],
                             help="Use Coordinates for a specific plant location.")
site_name, lat, lon = None, None, None
if site_mode == "City list":
    site_name = st.sidebar.selectbox("City", list(weather.PRESET_CITIES))
    lat, lon = weather.PRESET_CITIES[site_name]
elif site_mode == "Search by name":
    query = st.sidebar.text_input("Place name", help="For example: Aswan, Egypt")
    if query:
        hit = find_place(query)
        if hit:
            lat, lon, site_name = hit
            st.sidebar.success(site_name)
        else:
            st.sidebar.warning("Place not found. Try another spelling or use Coordinates.")
else:
    lat = st.sidebar.number_input("Latitude (deg N)", -90.0, 90.0, 24.09, 0.01, format="%.4f")
    lon = st.sidebar.number_input("Longitude (deg E)", -180.0, 180.0, 32.90, 0.01, format="%.4f")
    site_name = f"{lat:.4f}, {lon:.4f}"

year = st.sidebar.selectbox("Weather year", list(range(2025, 2000, -1)), index=2,
                            help="Hourly data of this year are used as a typical year.")
if st.sidebar.button("Load solar data", type="primary", disabled=lat is None):
    try:
        st.session_state["weather"] = {"df": load_weather(lat, lon, year),
                                       "site": site_name, "lat": lat, "lon": lon, "year": year}
    except Exception as exc:
        st.session_state.pop("weather", None)
        st.sidebar.error(str(exc))

st.sidebar.header("🏭 Process heat load")
load_mode = st.sidebar.radio("Enter the load as", ["Thermal power (MW)", "Steam conditions"],
                             help="If you know the boiler steam data, use Steam conditions.")
steam_res, demand_error = None, None
if load_mode == "Thermal power (MW)":
    q_mw = st.sidebar.number_input("Average heat demand (MWth)", 0.1, 500.0, 10.0, 0.5,
                                   help="Example value - replace with your plant's data.")
else:
    p_bar = st.sidebar.number_input("Steam pressure (bar)", 1.0, 250.0, 10.0, 1.0)
    t_steam = st.sidebar.number_input("Steam temperature (°C)", 50.0, 800.0, 200.0, 5.0)
    t_fw = st.sidebar.number_input("Feedwater temperature (°C)", 5.0, 300.0, 80.0, 5.0)
    flow_tph = st.sidebar.number_input("Steam flow (t/h)", 0.1, 1000.0, 10.0, 0.5)
    q_mw = None
    try:
        steam_res = demand.steam_heat_duty(p_bar, t_fw, t_steam, flow_tph)
        q_mw = steam_res["q_mw"]
    except Exception as exc:
        demand_error = str(exc)

hours_day = st.sidebar.number_input("Operating hours per day", 1.0, 24.0, 16.0, 1.0,
                                    help="Example value - replace with your production schedule.")
days_year = st.sidebar.number_input("Operating days per year", 1, 366, 330, 1,
                                    help="Example value - replace with your production schedule.")

# ------------------------------------------------------------------ tabs
design_result = sim_result = soil_result = econ_result = None
sim_use_soil = False
(tab_summary, tab_site, tab_demand, tab_design, tab_soil, tab_sim, tab_econ, tab_src, tab_about) = st.tabs(
    ["📌 Summary", "☀️ Solar resource", "🏭 Heat demand", "🗼 Tower & storage", "🧹 Soiling",
     "⚡ Hourly simulation", "💰 Economics", "📚 Sources & method", "ℹ️ About & limits"])

with tab_site:
    w = st.session_state.get("weather")
    if w is None:
        st.info("Choose a site in the sidebar and press **Load solar data**.")
    else:
        s = weather.summarize(w["df"])
        st.subheader(f"{w['site']} - {w['year']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annual DNI", f"{s['dni_kwh_m2']:,.0f} kWh/m²",
                  help="Direct normal irradiance, the part of sunlight a solar tower can use.")
        c2.metric("Annual GHI", f"{s['ghi_kwh_m2']:,.0f} kWh/m²")
        c3.metric("Peak DNI", f"{s['dni_peak']:,.0f} W/m²")
        c4.metric("Mean air temperature", f"{s['t_mean']:.1f} °C",
                  f"{s['t_min']:.0f} to {s['t_max']:.0f} °C", delta_color="off")
        if s["missing_dni_hours"]:
            st.warning(f"{s['missing_dni_hours']} hours have no DNI value and count as zero.")

        monthly = s["monthly"]
        fig = go.Figure()
        fig.add_bar(x=MONTHS, y=monthly["DNI"], name="DNI", marker_color="#e67e22")
        fig.add_bar(x=MONTHS, y=monthly["GHI"], name="GHI", marker_color="#3498db")
        fig.update_layout(barmode="group", title="Monthly irradiation", yaxis_title="kWh/m² per month",
                          margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

        month = st.selectbox("Average day of month", range(1, 13), index=5,
                             format_func=lambda m: MONTHS[m - 1])
        prof = weather.diurnal_profile(w["df"], month)
        fig2 = go.Figure(go.Scatter(x=prof.index, y=prof.values, mode="lines+markers",
                                    line=dict(color="#d35400", width=3)))
        fig2.update_layout(title=f"Average hourly DNI - {MONTHS[month - 1]}",
                           xaxis_title="Hour of day (as given in the data)", yaxis_title="DNI (W/m²)",
                           margin=dict(t=50, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Data: NASA POWER, satellite and model derived. Its accuracy for this site has not "
                   "been checked yet (see Sources & method). Hour labels are believed to be UTC.")

with tab_demand:
    if demand_error:
        st.error(demand_error)
    elif q_mw is not None:
        try:
            annual = demand.annual_demand_mwh(q_mw, hours_day, days_year)
        except ValueError as exc:
            st.error(str(exc))
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Heat demand", f"{q_mw:,.2f} MWth")
            c2.metric("Annual heat demand", f"{annual / 1000:,.1f} GWh/yr")
            c3.metric("Operating share of the year", f"{demand.load_factor(hours_day, days_year) * 100:.0f} %")
            if steam_res:
                st.subheader("Steam calculation")
                table = pd.DataFrame({
                    "Quantity": ["Saturation temperature", "Feedwater enthalpy", "Steam enthalpy",
                                 "Enthalpy difference", "Steam mass flow"],
                    "Value": [f"{steam_res['t_sat_c']:.1f}", f"{steam_res['h_in']:.1f}",
                              f"{steam_res['h_out']:.1f}", f"{steam_res['delta_h']:.1f}",
                              f"{steam_res['m_kg_s']:.2f}"],
                    "Unit": ["°C", "kJ/kg", "kJ/kg", "kJ/kg", "kg/s"]})
                st.dataframe(table, hide_index=True, use_container_width=True)
                st.caption("Q = m × (h_steam − h_feedwater). This is the heat the steam carries; "
                           "fuel input would be higher by the boiler losses.")

with tab_design:
    st.subheader("Solar tower and thermal storage")
    st.caption("Method: Gamil et al. (2023), Energy Conversion and Management 296, 117698. Valid only for "
               "pressurized liquid water as the heat transfer fluid, about 50 to 150 °C. "
               "The design heat load is the heat load in the sidebar.")
    if q_mw is None:
        st.info("Enter a valid heat load in the sidebar first.")
    else:
        with st.expander("Design assumptions (defaults = the source's Tucson reference case)"):
            c1, c2, c3 = st.columns(3)
            t_tes = c1.number_input("Storage (hours at full load)", 0.0, 24.0, 3.0, 0.5,
                                    help="Hours of full-load heat the storage tanks can supply.")
            x_sun = c1.number_input("Sunlight operating hours per day", 1.0, 24.0, 9.0, 0.5,
                                    help="Hours per day the plant runs directly from the sun (08:00 to 17:00 in the source).")
            t_cold = c2.number_input("Cold fluid temperature (°C)", 5.0, 140.0, 50.0, 5.0)
            t_hot = c2.number_input("Hot fluid temperature (°C)", 40.0, 200.0, 150.0, 5.0)
            p_sys = c2.number_input("System pressure (bar)", 1.0, 20.0, 6.0, 0.5,
                                    help="Keeps the water liquid at the hot temperature.")
            eta_rec = c3.number_input("Receiver thermal efficiency", 0.3, 0.99, 0.8589, 0.01, format="%.4f",
                                      help="Source value for Tucson, stainless steel 316, Pyromark coating.")
            eta_opt = c3.number_input("Field optical efficiency at design point", 0.3, 0.99, 0.8412, 0.01,
                                      format="%.4f", help="Derived from the source's reference case. Depends on "
                                      "latitude, tower height and layout: replace it for other sites.")
            dni_d = c3.number_input("Design DNI (W/m²)", 400.0, 1200.0, 950.0, 10.0)
        with st.expander("Advanced receiver settings"):
            a1, a2 = st.columns(2)
            q_flux = a1.number_input("Material flux limit (kW/m²)", 100.0, 1500.0, 600.0, 50.0,
                                     help="Stainless steel 316 with water, as cited by the source.")
            r_peak = a1.number_input("Limit-to-average flux ratio", 1.0, 4.0, 2.0, 0.1,
                                     help="Safety factor on the allowable flux. The source explores 1.5 to 3.")
            aspect = a2.number_input("Receiver aspect ratio (height / diameter)", 0.5, 4.0, 1.5, 0.1)
            helio_a = a2.number_input("Area of one heliostat (m²)", 1.0, 150.0, 4.0, 0.5)
        try:
            d = design.size_system(q_mw, t_tes, x_sun, t_cold, t_hot, p_sys, eta_rec, eta_opt, dni_d,
                                  q_flux, r_peak, aspect, helio_a)
        except Exception as exc:
            st.error(str(exc))
        else:
            design_result = {**d, "t_tes": t_tes}
            if t_hot > 150:
                st.warning("The hot temperature is above the 150 °C range validated in the source.")
            if load_mode == "Steam conditions" and t_steam > t_hot - 30:
                st.warning(f"Your steam temperature ({t_steam:.0f} °C) is higher than this water-based design can "
                           f"supply: hot fluid {t_hot:.0f} °C minus about 30 K heat-exchanger difference (as in the source).")
            m1, m2, m3 = st.columns(3)
            m1.metric("Solar multiple", f"{d['sm']:.2f}", help="Receiver size relative to the load: extra size fills the storage.")
            m2.metric("Receiver incident power", f"{d['q_th_rec_mw']:.2f} MWth")
            m3.metric("Heat absorbed by the fluid", f"{d['q_htf_mw']:.2f} MWth")
            f1, f2, f3 = st.columns(3)
            f1.metric("Heliostat field area", f"{d['a_field_m2']:,.0f} m²")
            f2.metric("Number of heliostats", f"{d['n_helio']:,}")
            f3.metric("Receiver size", f"{d['h_rec_m']:.2f} × {d['d_rec_m']:.2f} m",
                      f"{d['a_rec_m2']:.2f} m² area", delta_color="off")
            s1, s2, s3 = st.columns(3)
            s1.metric("Fluid flow", f"{d['mdot_kg_s']:.2f} kg/s")
            s2.metric("Fluid in storage", f"{d['htf_mass_kg'] / 1000:,.1f} t")
            s3.metric("Volume per tank", f"{d['tank_volume_m3']:,.1f} m³", help="Two tanks: one hot, one cold.")
            st.caption("Tower height and the heliostat layout are not designed here. The heliostat count follows from "
                       "the field efficiency you enter. Heat losses from pipes and tanks are ignored, as in the source.")

with tab_soil:
    st.subheader("Mirror soiling (dust)")
    st.warning("**Illustrative parameters.** The dust parameters below are example values from the group's soiling "
               "tool (synthetic demo), not measurements for any site in Egypt. The model (Picotti et al. 2023) was "
               "validated on heliostats at three Australian sites only. Calibrate it with local reflectance and PM10 "
               "data before drawing conclusions.")
    w_s = st.session_state.get("weather")
    if w_s is None:
        st.info("Load solar data first (sidebar). The soiling model uses the hourly weather and sun position of the site.")
    else:
        with st.expander("Dust and mirror inputs", expanded=True):
            d1, d2, d3 = st.columns(3)
            pm_mode = d1.radio("Dust data", ["Constant PM10 level", "Upload hourly PM10 (CSV)"],
                               help="PM10 is the mass of airborne dust particles under 10 µm, in µg/m³.")
            pm_const = d1.number_input("PM10 (µg/m³)", 0.0, 1000.0, 40.0, 5.0,
                                       help="Example value. A constant level ignores dust storms.")
            pm_file = d1.file_uploader("PM10 file (time, PM10)", type=["csv"]) if pm_mode.startswith("Upload") else None
            mu = d2.number_input("Deposition parameter μ̃ (per hour)", 0.0, 1e-3, 4e-6, 1e-7, format="%.2e",
                                 help="Covered mirror area per unit of PM10 per hour. Must come from a local calibration.")
            sigma = d2.number_input("Uncertainty σ_dep (per hour)", 0.0, 1e-3, 1.5e-6, 1e-7, format="%.2e")
            rho0 = d2.number_input("Clean mirror reflectance", 0.5, 1.0, 0.88, 0.01,
                                   help="0.88 is the reflectivity used in Gamil et al. 2023. Use your measured value.")
            tower_h = d3.number_input("Tower height (m)", 5.0, 200.0, 40.0, 5.0)
            dist = d3.number_input("Typical distance of a mirror from the tower (m)", 0.0, 500.0, 25.0, 5.0,
                                   help="Mirrors on a ring of this radius represent the field.")
            stow_sel = d3.selectbox("Mirror position at night", ["Face down (180°)", "Face up (0°)"],
                                    help="Face-down mirrors collect no dust in this model.")
            stow = 180.0 if stow_sel.startswith("Face down") else 0.0
        with st.expander("Cleaning", expanded=True):
            k1, k2, k3 = st.columns(3)
            pol_sel = k1.selectbox("Cleaning rule", ["Clean when reflectance drops to a limit", "Fixed schedule"])
            kind = "threshold" if pol_sel.startswith("Clean when") else "fixed"
            threshold = k2.number_input("Reflectance limit", 0.3, 0.99, 0.80, 0.01,
                                        help="Used with the first rule. A plant choice, not a law.")
            cleans = k2.number_input("Cleanings per year", 1, 365, 12, 1, help="Used with the fixed schedule.")
            eta_cl = k3.number_input("Cleaning effectiveness", 0.5, 1.0, 0.99, 0.01,
                                     help="Share of the dust removed by one cleaning.")
        pm_text = None
        if pm_file is not None:
            pm_text = pm_file.getvalue().decode("utf-8", errors="replace")
        try:
            r_s = run_soiling(w_s["lat"], w_s["lon"], w_s["year"], pm_text, pm_const, mu, sigma, rho0, kind,
                              threshold, cleans, eta_cl, tower_h, dist, stow)
        except Exception as exc:
            st.error(str(exc))
        else:
            soil_result = r_s
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Energy-weighted soiling loss", f"{r_s['loss_pct']:.1f} %",
                      help="Average reflectance loss, weighted by the sunlight available each hour.")
            q2.metric("Average reflectance in operation", f"{r_s['rho_weighted']:.3f}")
            q3.metric("Cleanings per year", f"{r_s['cleanings_per_mirror']:.1f}")
            q4.metric("Lowest reflectance reading", f"{r_s['rho_ref_min']:.3f}")
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=list(r_s["daily_rho"].index), y=list(r_s["daily_rho"].values),
                                       mode="lines", name="Reflectance", line=dict(color="#8e44ad", width=2)))
            if kind == "threshold":
                fig_s.add_hline(y=threshold, line_dash="dash", annotation_text="Cleaning limit")
            fig_s.update_layout(title="Daily average mirror reflectance in operation", yaxis_title="Reflectance",
                                margin=dict(t=50, b=10))
            st.plotly_chart(fig_s, use_container_width=True)
            qs = r_s["daily_loss_pp"].quantile([0.05, 0.5, 0.95])
            st.caption(f"Expected loss per day without cleaning (percentage points per day): "
                       f"p5 = {qs.iloc[0]:.2f}, median = {qs.iloc[1]:.2f}, p95 = {qs.iloc[2]:.2f}.")
            st.markdown("**More cleaning, less loss**")
            try:
                st.dataframe(soiling_vs_cleaning(w_s["lat"], w_s["lon"], w_s["year"], pm_text, pm_const, mu, sigma,
                                                 rho0, eta_cl, tower_h, dist, stow),
                             hide_index=True, use_container_width=True)
            except Exception as exc:
                st.error(str(exc))
            with st.expander("Notes from the soiling model (read before using the results)"):
                for level, text in r_s["notes"].items:
                    st.markdown(f"- **[{level}]** {text}")
            st.caption("The effect of dust on the delivered heat is calculated in the Hourly simulation tab.")

with tab_sim:
    st.subheader("Hourly simulation: solar heat, storage and backup")
    w_h = st.session_state.get("weather")
    if w_h is None:
        st.info("Load solar data first (sidebar).")
    elif q_mw is None or design_result is None:
        st.info("Enter a valid heat load (sidebar) and a valid design in the Tower & storage tab first.")
    else:
        with st.expander("Operation and annual performance (defaults = Gamil et al. Tucson results)", expanded=True):
            h1, h2, h3 = st.columns(3)
            start_h = h1.number_input("Start of operation (local hour)", 0, 23, 8, 1,
                                      help="Hours per day and days per year come from the sidebar.")
            tz = h1.number_input("Time zone (hours from UTC)", -12, 14, int(round(w_h["lon"] / 15)), 1,
                                 help="The weather hours are in UTC (to be confirmed). Egypt is UTC+2 in winter.")
            eta_w = h2.number_input("Field optical efficiency, annual average", 0.3, 0.99, 0.7997, 0.01,
                                    format="%.4f", help="Gamil 2023 Table 7, weighted by sunlight (tower 40 m, Tucson). "
                                    "Includes mirror reflectivity 0.88.")
            eta_rec_y = h2.number_input("Receiver efficiency, annual average", 0.3, 0.99, 0.8098, 0.01,
                                        format="%.4f", help="Gamil 2024 Table 8 (Tucson).")
            avail = h3.number_input("System availability", 0.5, 1.0, 0.96, 0.01, help="Gamil 2024 Table 3 (4 % downtime).")
            use_soil = h3.checkbox("Include mirror soiling (uses the Soiling tab inputs)", value=True,
                                   help="The Soiling tab parameters are illustrative, so this result is too.")
        soil_args = (pm_text, pm_const, mu, sigma, rho0, kind, threshold, cleans, eta_cl, tower_h, dist, stow)
        try:
            sim = run_simulation(w_h["lat"], w_h["lon"], w_h["year"], design_result["a_field_m2"],
                                 design_result["q_htf_mw"] * 1000.0 * design_result["t_tes"], q_mw, start_h,
                                 hours_day, days_year, tz, eta_w, eta_rec_y, avail, use_soil, soil_args)
        except Exception as exc:
            st.error(str(exc))
        else:
            sm = sim["summary"]
            sim_result, sim_use_soil = sim, use_soil
            if use_soil:
                st.warning("The soiling inputs are illustrative (see the Soiling tab), so the dust effect below is too.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Annual heat demand", f"{sm['demand_gwh']:,.1f} GWh")
            m2.metric("Solar heat delivered", f"{sm['delivered_gwh']:,.1f} GWh")
            m3.metric("Solar fraction", f"{sm['solar_fraction'] * 100:.0f} %",
                      help="Share of the process heat supplied by the sun (directly or from storage).")
            m4.metric("Backup heat needed", f"{sm['backup_gwh']:,.1f} GWh")
            n1, n2, n3 = st.columns(3)
            n1.metric("Solar heat produced", f"{sm['gross_gwh']:,.1f} GWh",
                      help="Heat the field puts into the fluid before it is used.")
            n2.metric("Heat not used (storage full)", f"{sm['curtailed_gwh']:,.1f} GWh",
                      f"{sm['curtailed_gwh'] / sm['gross_gwh'] * 100:.0f} % of produced" if sm["gross_gwh"] else None,
                      delta_color="off")
            if use_soil:
                lost = sim["delivered_clean_gwh"] - sm["delivered_gwh"]
                n3.metric("Delivered heat lost to dust", f"{lost:,.2f} GWh",
                          f"{lost / sim['delivered_clean_gwh'] * 100:.1f} %" if sim["delivered_clean_gwh"] else None,
                          delta_color="off")
            mon = sm["monthly"]
            fig_m = go.Figure()
            fig_m.add_bar(x=MONTHS, y=mon["delivered"], name="Solar heat delivered", marker_color="#e67e22")
            fig_m.add_bar(x=MONTHS, y=mon["backup"], name="Backup (fuel)", marker_color="#95a5a6")
            fig_m.update_layout(barmode="stack", title="Heat demand covered, by month", yaxis_title="GWh",
                                margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig_m, use_container_width=True)
            mth = st.selectbox("Average day of month", range(1, 13), index=5, format_func=lambda m: MONTHS[m - 1],
                               key="sim_month")
            td = dispatch.typical_day(sim["df"], sim["run"], sim["demand"], mth, tz)
            fig_d = go.Figure()
            for col, colour in (("Solar heat to the system", "#d35400"), ("Delivered to the process", "#27ae60"),
                                ("Heat demand", "#34495e")):
                fig_d.add_trace(go.Scatter(x=list(td.index), y=list(td[col].values), mode="lines", name=col,
                                           line=dict(color=colour, width=3)))
            fig_d.update_layout(title=f"Average day, {MONTHS[mth - 1]}", xaxis_title="Local hour",
                                yaxis_title="kW", margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_d, use_container_width=True)
            fig_t = go.Figure(go.Scatter(x=list(td.index), y=list(td["Storage level (kWh)"].values), mode="lines",
                                         line=dict(color="#2980b9", width=3)))
            fig_t.update_layout(title="Heat stored in the tanks", xaxis_title="Local hour", yaxis_title="kWh",
                                margin=dict(t=50, b=10))
            st.plotly_chart(fig_t, use_container_width=True)
            with st.expander("How this is calculated, and how to check it"):
                st.markdown(
                    "- Solar heat each hour = sunlight (DNI) × field area × optical efficiency × mirror reflectance × "
                    "receiver efficiency × availability.\n"
                    "- Heat goes to the process first; the surplus charges the storage until it is full (the rest is not "
                    "used); the storage supplies the process when the sun is short; any shortfall is backup (fuel).\n"
                    "- The efficiencies are **constant annual averages** from the source, so the daily shape is only "
                    "approximate (real efficiency peaks around noon, and the receiver is less efficient at low sun). "
                    "Storage heat loss is ignored, as in the source.\n"
                    "- **Check against the source:** choose Tucson (32.25, −110.9) in *Coordinates*, load 1.3 MW, and "
                    "compare *Solar heat produced* with about 5.6 GWh (Gamil 2024, Table 8). Expect differences of "
                    "around 10 %, because the source used a typical year from another database.")
            st.caption("Costs, cost of heat and payback are in the Economics tab.")

with tab_econ:
    st.subheader("Costs, cost of heat and payback")
    if sim_result is None or design_result is None:
        st.info("Run the Tower & storage and Hourly simulation tabs first (load solar data and enter a valid design).")
    else:
        st.warning("**Default costs are for a small relocatable tower plant in the USA** (Gamil et al. 2024, Table 4). "
                   "Replace them with quotes for your project. The thermal storage cost needs special care: see the note "
                   "inside the cost box.")
        with st.expander("Investment costs (defaults = Gamil et al. 2024)"):
            e1, e2, e3 = st.columns(3)
            c_hel = e1.number_input("Heliostat field (USD/m²)", 1.0, 2000.0, 75.0, 5.0)
            c_site = e1.number_input("Site preparation (USD/m²)", 0.0, 500.0, 15.0, 1.0)
            c_tow = e2.number_input("Tower (USD per metre of height)", 10.0, 10000.0, 200.0, 10.0)
            c_rec = e2.number_input("Receiver (USD/m²)", 1.0, 10000.0, 65.0, 5.0)
            c_htf = e3.number_input("Heat transfer fluid (USD/m³)", 0.0, 1000.0, 0.8, 0.1)
            c_tes = e3.number_input("Thermal storage (USD per kWh of heat)", 0.0, 500.0, 32.0, 1.0,
                                    help="Default 32 USD/kWh is the storage cost in Table B.2 of the same paper (Fresnel "
                                    "plant). Table 4 prints 0.004 USD/kWh for water storage, which looks like a unit "
                                    "error (it would make a whole storage system cost about 20 USD).")
            e4, e5 = st.columns(2)
            cont = e4.number_input("Contingency (% of direct cost)", 0.0, 50.0, 7.0, 1.0) / 100.0
            indirect = e5.number_input("Engineering, construction, ownership (% of direct cost)", 0.0, 50.0, 11.0, 1.0) / 100.0
            st.caption(f"Tower height used: {tower_h:g} m (set in the Soiling tab). Storage size: "
                       f"{design_result['q_htf_mw'] * 1000 * design_result['t_tes']:,.0f} kWh of heat.")
        with st.expander("Finance and fuel"):
            f1, f2, f3 = st.columns(3)
            life = f1.number_input("Plant lifetime (years)", 1, 40, 20, 1,
                                   help="The source uses 10 years for a relocatable plant and 20 years for comparisons.")
            rate = f1.number_input("Discount rate (real, % per year)", 0.0, 30.0, 5.31, 0.25,
                                   help="Implied by the financing assumptions in Gamil et al. 2024, Table 5; "
                                   "reproduces the paper's capital recovery factor of 0.131 for 10 years. "
                                   "Taxes and incentives are not modelled.") / 100.0
            om = f2.number_input("Operation and maintenance (% of investment per year)", 0.0, 20.0, 1.6, 0.1) / 100.0
            c_clean = f2.number_input("Cost of one cleaning (USD per m² of mirror)", 0.0, 10.0, 0.0, 0.01,
                                      help="No default: enter your own cleaning cost. Zero means cleaning cost is ignored.")
            fuel = f3.number_input("Fuel price (USD per kWh of fuel)", 0.0, 1.0, 0.027, 0.001, format="%.4f",
                                   help="Default: Arizona industrial gas, 2022 (Gamil et al. 2024). Replace with your price.")
            beff = f3.number_input("Existing boiler efficiency", 0.3, 1.0, 0.80, 0.01, help="Gamil et al. 2024, Eq. 9.")
            esc = st.number_input("Fuel price increase (real, % per year)", 0.0, 20.0, 0.0, 0.5) / 100.0
        unit = {"heliostat_field_usd_m2": c_hel, "site_prep_usd_m2": c_site, "tower_usd_per_m": c_tow,
                "receiver_usd_m2": c_rec, "htf_usd_m3": c_htf}
        n_clean = soil_result["cleanings_per_mirror"] if (sim_use_soil and soil_result is not None) else 0.0
        heat_kwh = sim_result["summary"]["delivered_gwh"] * 1e6

        def run_econ(u=unit, tes=c_tes, fuel_p=fuel, r=rate, om_f=om):
            return economics.evaluate(design_result, tower_h, u, tes, heat_kwh, n_clean, c_clean, fuel_p, beff, r,
                                      life, om_f, esc, cont, indirect)
        try:
            res = run_econ()
        except Exception as exc:
            st.error(str(exc))
        else:
            econ_result = res
            cheaper = res["lcoh_usd_kwh"] < res["fuel_heat_usd_kwh"]
            if cheaper:
                st.success("Solar heat is cheaper than heat from fuel with these inputs.")
            else:
                st.info("Solar heat costs more than heat from fuel with these inputs. Viability depends on the fuel price "
                        "and the investment cost.")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total investment", f"${res['tic']:,.0f}")
            k2.metric("Solar heat cost (LCOH)", f"{res['lcoh_usd_kwh'] * 100:.2f} ¢/kWh",
                      f"${res['lcoh_usd_kwh'] * 1000:,.1f} per MWh", delta_color="off",
                      help="Yearly cost (capital recovery, operation, cleaning) divided by the heat delivered to the process.")
            k3.metric("Heat from fuel", f"{res['fuel_heat_usd_kwh'] * 100:.2f} ¢/kWh",
                      f"${res['fuel_heat_usd_kwh'] * 1000:,.1f} per MWh", delta_color="off")
            k4.metric("Fuel cost saved per year", f"${res['fuel_saved_usd']:,.0f}")
            j1, j2, j3, j4 = st.columns(4)
            j1.metric("Net present value", f"${res['npv']:,.0f}", help="Over the plant lifetime, at the discount rate.")
            j2.metric("Internal rate of return", "n/a" if res["irr"] != res["irr"] else f"{res['irr'] * 100:.1f} %")
            j3.metric("Simple payback", f"{res['payback']:.1f} years" if res["payback"] != float("inf") else "longer than lifetime")
            j4.metric("CO₂ avoided per year", f"{res['co2_avoided_t']:,.0f} t")
            items = res["investment"]["items"]
            fig_c = go.Figure()
            fig_c.add_bar(x=list(items.keys()), y=[round(v) for v in items.values()], marker_color="#e67e22")
            fig_c.update_layout(title="Investment cost by item", yaxis_title="USD", margin=dict(t=50, b=10))
            st.plotly_chart(fig_c, use_container_width=True)
            cum = np.cumsum(res["flows"])
            fig_f = go.Figure(go.Scatter(x=list(range(len(cum))), y=[float(v) for v in cum], mode="lines",
                                         line=dict(color="#27ae60", width=3)))
            fig_f.update_layout(title="Cumulative cash flow", xaxis_title="Year", yaxis_title="USD",
                                margin=dict(t=50, b=10))
            st.plotly_chart(fig_f, use_container_width=True)
            st.markdown("**Sensitivity: change in net present value for ±20 % on one input**")
            rows = []
            for label, key in (("Heliostat field cost", "hel"), ("Thermal storage cost", "tes"), ("Fuel price", "fuel"),
                               ("Discount rate", "rate"), ("Operation and maintenance", "om")):
                out = []
                for k in (0.8, 1.2):
                    kw = {}
                    if key == "hel":
                        kw["u"] = {**unit, "heliostat_field_usd_m2": c_hel * k}
                    elif key == "tes":
                        kw["tes"] = c_tes * k
                    elif key == "fuel":
                        kw["fuel_p"] = fuel * k
                    elif key == "rate":
                        kw["r"] = rate * k
                    else:
                        kw["om_f"] = om * k
                    out.append(run_econ(**kw)["npv"] - res["npv"])
                rows.append({"Input": label, "−20 % (USD)": round(out[0]), "+20 % (USD)": round(out[1])})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("Heat delivered by the simulation is used for every year (no degradation). Taxes, subsidies and "
                       "loans are not modelled. Preliminary feasibility only.")

with tab_src:
    st.subheader("Where every number comes from")
    st.write("Each row is a number or method used in the tool, where it comes from, and how far to trust it. "
             "**NOT USED** rows are things that earlier drafts assumed without a source; they stay out until a source is chosen.")
    src_all = sources.load_sources()
    attention = src_all["status"].str.contains("ILLUSTRATIVE|NOT USED|PARTLY|confirm|Not yet|Derived|Assumption|discrepancy|inconsistency|verify|Scope",
                                               case=False, regex=True)
    view = st.radio("Show", ["Everything", "Only entries that need attention"], horizontal=True)
    st.dataframe(src_all[attention] if view.startswith("Only") else src_all, hide_index=True, use_container_width=True)
    st.download_button("Download the source register (CSV)", src_all.to_csv(index=False),
                       file_name="sources_register.csv", mime="text/csv")

with tab_about:
    st.subheader("How to use this tool")
    st.markdown("""
1. In the sidebar choose the **site**, press **Load solar data**, and enter the **process heat load** (in MW, or from steam conditions).
2. Go through the tabs from left to right: *Tower & storage*, *Soiling*, *Hourly simulation*, *Economics*. Each tab has a
   box of assumptions. Defaults come from the cited papers; change them to your own data.
3. The **Summary** tab collects the main results and lets you download them.
""")
    st.subheader("What this tool cannot do (read before using the results)")
    st.markdown("""
- **Water-based design only.** The tower and storage method is valid for pressurized liquid water at about 50 to 150 °C
  (Gamil et al. 2023). It does not cover 200 °C steam or molten salt, and it does not design the heat exchanger.
- **Efficiencies come from a Tucson, Arizona design.** The field and receiver efficiencies are constant annual values from
  that case. For other sites they should be recalculated with a heliostat-field layout model.
- **Dust parameters are illustrative.** The soiling model was validated on mirrors in Australia. The deposition parameter
  must be calibrated with local reflectance and PM10 data before the dust results mean anything for a real site.
- **Costs are for a small relocatable plant in the USA** and some of the source's own cost figures do not add up (see the
  Sources tab). Use project quotes. Taxes, subsidies and loans are not modelled.
- **Solar data are satellite estimates** (NASA POWER), and their accuracy for the chosen site has not been checked.
- **Preliminary feasibility only.** Not a substitute for a detailed engineering study.
""")
    st.subheader("Main references")
    st.markdown("""
- Gamil, Li, Khammash (2023). *Energy Conversion and Management* 296, 117698 (tower, receiver and storage design).
- Gamil, Li, Ali, Khammash (2024). *Energy* 304, 132085 (costs, performance and economics of the same plant).
- Picotti et al. (2023). *Solar Energy* 263, 111945 (mirror soiling model).
- Anderson et al. (2023), arXiv:2306.06106 (cleaning and plant model used in the soiling tool).
- Pappa et al. (2025). *Atmosphere* 16(7), 807 (cleaning limit).
- NASA POWER (power.larc.nasa.gov) for hourly solar data; pyXSteam for steam tables.
""")

with tab_summary:
    st.subheader("Project summary")
    w_m = st.session_state.get("weather")
    checklist = [("Solar data loaded", w_m is not None), ("Heat load valid", q_mw is not None),
                 ("Tower & storage design valid", design_result is not None),
                 ("Hourly simulation run", sim_result is not None), ("Economics calculated", econ_result is not None)]
    st.markdown("  \n".join(("✅ " if ok else "⬜ ") + name for name, ok in checklist))
    out_rows = []
    if w_m is not None:
        sm0 = weather.summarize(w_m["df"])
        a1, a2, a3 = st.columns(3)
        a1.metric("Site", w_m["site"])
        a2.metric("Annual DNI", f"{sm0['dni_kwh_m2']:,.0f} kWh/m²")
        if q_mw is not None:
            try:
                ann = demand.annual_demand_mwh(q_mw, hours_day, days_year)
                a3.metric("Annual heat demand", f"{ann / 1000:,.1f} GWh")
                out_rows += [("Annual heat demand", ann / 1000, "GWh/yr")]
            except ValueError:
                pass
        out_rows += [("Site", w_m["site"], ""), ("Weather year", w_m["year"], ""),
                     ("Annual DNI", round(sm0["dni_kwh_m2"]), "kWh/m2/yr")]
    if q_mw is not None:
        out_rows += [("Heat load", q_mw, "MWth"), ("Operating hours per day", hours_day, "h"),
                     ("Operating days per year", days_year, "days")]
    if design_result is not None:
        b1, b2, b3 = st.columns(3)
        b1.metric("Heliostat field", f"{design_result['a_field_m2']:,.0f} m²", f"{design_result['n_helio']:,} heliostats",
                  delta_color="off")
        b2.metric("Receiver", f"{design_result['h_rec_m']:.2f} × {design_result['d_rec_m']:.2f} m")
        b3.metric("Volume per storage tank", f"{design_result['tank_volume_m3']:,.1f} m³")
        out_rows += [("Solar multiple", round(design_result["sm"], 3), ""), ("Heliostat field area", round(design_result["a_field_m2"]), "m2"),
                     ("Number of heliostats", design_result["n_helio"], ""), ("Receiver area", round(design_result["a_rec_m2"], 2), "m2"),
                     ("Storage hours", design_result["t_tes"], "h"), ("Volume per tank", round(design_result["tank_volume_m3"], 1), "m3")]
    if sim_result is not None:
        sm1 = sim_result["summary"]
        c1_, c2_, c3_ = st.columns(3)
        c1_.metric("Solar heat delivered", f"{sm1['delivered_gwh']:,.1f} GWh/yr")
        c2_.metric("Solar fraction", f"{sm1['solar_fraction'] * 100:.0f} %")
        c3_.metric("Backup heat needed", f"{sm1['backup_gwh']:,.1f} GWh/yr")
        out_rows += [("Solar heat produced", round(sm1["gross_gwh"], 2), "GWh/yr"), ("Solar heat delivered", round(sm1["delivered_gwh"], 2), "GWh/yr"),
                     ("Solar fraction", round(sm1["solar_fraction"] * 100, 1), "%"), ("Backup heat", round(sm1["backup_gwh"], 2), "GWh/yr"),
                     ("Dust effect included", "yes (illustrative parameters)" if sim_use_soil else "no", "")]
    if econ_result is not None:
        d1_, d2_, d3_, d4_ = st.columns(4)
        d1_.metric("Investment", f"${econ_result['tic']:,.0f}")
        d2_.metric("Solar heat cost", f"{econ_result['lcoh_usd_kwh'] * 100:.2f} ¢/kWh",
                   f"fuel: {econ_result['fuel_heat_usd_kwh'] * 100:.2f} ¢/kWh", delta_color="off")
        d3_.metric("Net present value", f"${econ_result['npv']:,.0f}")
        d4_.metric("CO₂ avoided", f"{econ_result['co2_avoided_t']:,.0f} t/yr")
        out_rows += [("Total investment", round(econ_result["tic"]), "USD"),
                     ("Solar heat cost (LCOH)", round(econ_result["lcoh_usd_kwh"] * 100, 3), "cents/kWh"),
                     ("Heat from fuel", round(econ_result["fuel_heat_usd_kwh"] * 100, 3), "cents/kWh"),
                     ("Net present value", round(econ_result["npv"]), "USD"),
                     ("Simple payback", "longer than lifetime" if econ_result["payback"] == float("inf") else round(econ_result["payback"], 1), "years"),
                     ("CO2 avoided", round(econ_result["co2_avoided_t"]), "t/yr")]
    if not any(ok for _, ok in checklist[2:]):
        st.info("Follow the steps in the sidebar and the tabs to fill this summary.")
    if out_rows:
        st.download_button("Download these results (CSV)",
                           pd.DataFrame(out_rows, columns=["Item", "Value", "Unit"]).to_csv(index=False),
                           file_name="solar_process_heat_results.csv", mime="text/csv")
    reg = sources.load_sources()
    need = reg["status"].str.contains("ILLUSTRATIVE|NOT USED|PARTLY|confirm|Not yet|Derived|Assumption|discrepancy|inconsistency|verify|Scope",
                                      case=False, regex=True).sum()
    st.caption(f"{need} of {len(reg)} entries in the source register are illustrative, derived, assumed or still to be "
               "verified. See the Sources & method tab before presenting any number.")
