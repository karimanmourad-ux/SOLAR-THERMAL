"""Slice 5 - Investment cost, levelized cost of heat (LCOH) and cash flow.

Unit costs and cost structure follow Gamil, Li, Ali, Khammash (2024), Energy 304, 132085,
Table 4 (small relocatable solar tower, USA). The LCOH method is the standard capital-recovery
method written out explicitly here (the source's own LCOH could not be reproduced from its
printed numbers; see data/sources.csv). LCOH is divided by the heat DELIVERED to the process.
"""
import math

# Gamil et al. 2024, Table 4 (as printed) and Table B.2 (TES cost of the Fresnel case)
UNIT_COSTS = {
    "heliostat_field_usd_m2": 75.0,
    "site_prep_usd_m2": 15.0,
    "tower_usd_per_m": 200.0,
    "receiver_usd_m2": 65.0,
    "htf_usd_m3": 0.8,
}
TES_USD_KWH_AS_PRINTED = 0.004      # Table 4 / Table A.2 as printed - looks like a unit error
TES_USD_KWH_TABLE_B2 = 32.0         # Table B.2 (Fresnel plant, thermal oil) - used as the default here
CONTINGENCY = 0.07                  # of direct cost (Table 4, SAM default)
INDIRECT = 0.11                     # of direct cost incl. contingency (Table 4, SAM default)
OM_FRACTION = 0.016                 # annual fixed operating cost / investment (Gamil 2024, p. 8)
BOILER_EFFICIENCY = 0.80            # Gamil 2024, Eq. 9
CO2_KG_PER_KWH_FUEL = 0.198         # Gamil 2024, Eq. 9 / Table 6
# Real discount rate implied by Gamil 2024 Table 5 (50 % debt at 4 %, 13 % equity return, 28 % tax,
# 2.5 % inflation). It reproduces the paper's capital recovery factor 0.131 for 10 years.
DISCOUNT_RATE_GAMIL2024 = 0.0531


def crf(rate, years):
    """Capital recovery factor: share of the investment paid back each year."""
    if years <= 0:
        raise ValueError("Lifetime must be at least 1 year.")
    if rate == 0:
        return 1.0 / years
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def investment(design, tower_height_m, unit, tes_usd_kwh, contingency=CONTINGENCY, indirect=INDIRECT):
    """Investment cost (USD) from the design of the Tower & storage step."""
    tes_kwh = design["q_htf_mw"] * 1000.0 * design["t_tes"]
    items = {
        "Heliostat field": design["a_field_m2"] * unit["heliostat_field_usd_m2"],
        "Site preparation": design["a_field_m2"] * unit["site_prep_usd_m2"],
        "Tower": tower_height_m * unit["tower_usd_per_m"],
        "Receiver": design["a_rec_m2"] * unit["receiver_usd_m2"],
        "Heat transfer fluid": 2.0 * design["tank_volume_m3"] * unit["htf_usd_m3"],
        "Thermal storage": tes_kwh * tes_usd_kwh,
    }
    direct = sum(items.values())
    cont = direct * contingency
    ind = (direct + cont) * indirect
    items["Contingency"] = cont
    items["Engineering, construction, ownership"] = ind
    return {"items": items, "total": direct + cont + ind, "tes_kwh": tes_kwh}


def cash_flows(tic, heat_kwh, fuel_usd_per_kwh, boiler_eff, om_fraction, cleaning_usd, years,
               fuel_escalation=0.0):
    """Year 0 investment, then yearly fuel savings minus operating and cleaning cost."""
    flows = [-tic]
    for t in range(1, int(years) + 1):
        saving = heat_kwh / boiler_eff * fuel_usd_per_kwh * (1 + fuel_escalation) ** (t - 1)
        flows.append(saving - om_fraction * tic - cleaning_usd)
    return flows


def npv(rate, flows):
    return sum(f / (1 + rate) ** t for t, f in enumerate(flows))


def irr(flows):
    lo, hi = -0.99, 5.0
    if npv(lo, flows) * npv(hi, flows) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo, flows) * npv(mid, flows) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def payback_years(flows):
    cum = 0.0
    for t, f in enumerate(flows):
        prev, cum = cum, cum + f
        if t > 0 and cum >= 0:
            return t - 1 + (-prev) / (cum - prev)
    return float("inf")


def evaluate(design, tower_height_m, unit, tes_usd_kwh, heat_kwh, cleanings_per_year, cleaning_usd_per_m2,
             fuel_usd_per_kwh, boiler_eff, rate, years, om_fraction=OM_FRACTION, fuel_escalation=0.0,
             contingency=CONTINGENCY, indirect=INDIRECT):
    """All economic results for one set of inputs. heat_kwh = heat DELIVERED to the process per year."""
    if heat_kwh <= 0:
        raise ValueError("No solar heat is delivered, so costs cannot be compared. Check the design and the weather.")
    inv = investment(design, tower_height_m, unit, tes_usd_kwh, contingency, indirect)
    tic = inv["total"]
    cleaning = cleanings_per_year * design["a_field_m2"] * cleaning_usd_per_m2
    annual_cost = crf(rate, years) * tic + om_fraction * tic + cleaning
    lcoh = annual_cost / heat_kwh
    fuel_heat_cost = fuel_usd_per_kwh / boiler_eff
    flows = cash_flows(tic, heat_kwh, fuel_usd_per_kwh, boiler_eff, om_fraction, cleaning, years, fuel_escalation)
    return {"investment": inv, "tic": tic, "annual_cost": annual_cost, "cleaning_usd": cleaning,
            "lcoh_usd_kwh": lcoh, "fuel_heat_usd_kwh": fuel_heat_cost, "flows": flows,
            "npv": npv(rate, flows), "irr": irr(flows), "payback": payback_years(flows),
            "fuel_saved_usd": heat_kwh / boiler_eff * fuel_usd_per_kwh,
            "co2_avoided_t": heat_kwh / boiler_eff * CO2_KG_PER_KWH_FUEL / 1000.0}
