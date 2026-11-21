"""Run with:  python tests/test_slice5.py   (from the project folder)"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import economics as ec

# Reference design of Gamil 2023/2024 (Tucson): 645 heliostats x 4 m2, receiver 6.783 m2, two 50 m3 tanks,
# 3 h of storage at 1.771 MW fluid heat power, 40 m tower.
REF = {"a_field_m2": 2580.0, "a_rec_m2": 6.783, "tank_volume_m3": 50.0, "q_htf_mw": 1.7708, "t_tes": 3.0}


def test_crf_matches_paper():
    # Gamil 2024: CRF = 0.131 for 10 years with the financing in Table 5 (real rate 5.31 %)
    assert abs(ec.crf(ec.DISCOUNT_RATE_GAMIL2024, 10) - 0.131) < 0.001
    assert abs(ec.crf(0.0, 10) - 0.1) < 1e-12


def test_table4_as_printed_sums_to_what_we_computed():
    inv = ec.investment(REF, 40.0, ec.UNIT_COSTS, ec.TES_USD_KWH_AS_PRINTED)
    # Table 4 as printed gives about 286 000 USD, NOT the paper's 365 521.57 USD (documented discrepancy).
    assert abs(inv["total"] - 285_930) < 600
    assert abs(inv["items"]["Contingency"] / (inv["total"] / 1.07 / 1.11) - 0.07) < 1e-9


def test_alternative_tes_cost_raises_investment():
    a = ec.investment(REF, 40.0, ec.UNIT_COSTS, ec.TES_USD_KWH_AS_PRINTED)["total"]
    b = ec.investment(REF, 40.0, ec.UNIT_COSTS, ec.TES_USD_KWH_TABLE_B2)["total"]
    assert b > a * 1.5


def test_hand_calculation_lcoh_and_cash_flow():
    # Investment 100 000, heat 500 000 kWh/yr, rate 0, 10 years, O&M 2 %, no cleaning
    d = {"a_field_m2": 1.0, "a_rec_m2": 1.0, "tank_volume_m3": 1.0, "q_htf_mw": 0.0, "t_tes": 0.0}
    unit = {"heliostat_field_usd_m2": 100_000.0, "site_prep_usd_m2": 0.0, "tower_usd_per_m": 0.0,
            "receiver_usd_m2": 0.0, "htf_usd_m3": 0.0}
    r = ec.evaluate(d, 0.0, unit, 0.0, 500_000.0, 0, 0.0, 0.05, 0.8, 0.0, 10, om_fraction=0.02,
                    contingency=0.0, indirect=0.0)
    assert abs(r["tic"] - 100_000) < 1e-9
    assert abs(r["lcoh_usd_kwh"] - (100_000 / 10 + 2_000) / 500_000) < 1e-12
    saving = 500_000 / 0.8 * 0.05
    assert abs(r["flows"][1] - (saving - 2_000)) < 1e-9
    assert abs(r["npv"] - (-100_000 + 10 * (saving - 2_000))) < 1e-6
    assert abs(r["payback"] - 100_000 / (saving - 2_000)) < 1e-9
    assert abs(r["co2_avoided_t"] - 500_000 / 0.8 * 0.198 / 1000) < 1e-9


def test_irr_is_consistent_with_npv():
    flows = [-1000.0] + [300.0] * 5
    assert abs(ec.npv(ec.irr(flows), flows)) < 1e-6


def test_bad_input():
    try:
        ec.evaluate(REF, 40.0, ec.UNIT_COSTS, 32.0, 0.0, 0, 0.0, 0.05, 0.8, 0.05, 20)
    except ValueError:
        return
    raise AssertionError("accepted zero heat")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("ok  ", t.__name__)
    print(f"\nAll {len(tests)} tests passed.")
