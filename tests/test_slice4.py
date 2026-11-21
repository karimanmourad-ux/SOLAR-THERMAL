"""Run with:  python tests/test_slice4.py   (from the project folder)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from core import dispatch


def year_index():
    return pd.date_range("2023-01-01", periods=24 * 365, freq="1h")


def test_demand_totals():
    idx = year_index()
    d = dispatch.operating_demand_kw(idx, 10.0, 8, 16, 330, 2)
    assert abs(d.sum() / 1000.0 - 10.0 * 16 * 330) < 1e-6             # MWh = MW x h/day x days
    d2 = dispatch.operating_demand_kw(idx, 1.0, 0, 24, 365, 0)
    assert abs(d2.sum() / 1000.0 - 8760) < 1e-6


def test_demand_validation():
    idx = year_index()
    for args in [(1.0, 10, 16, 300, 0), (1.0, 8, 8, 0, 0), (1.0, 8, 8, 400, 0)]:
        try:
            dispatch.operating_demand_kw(idx, *args)
        except ValueError:
            continue
        raise AssertionError(f"accepted {args}")


def test_hand_calculation_storage_fills():
    idx = pd.date_range("2023-01-01", periods=48, freq="1h")
    df = pd.DataFrame({"DNI": 1000.0}, index=idx)
    # 1000 W/m2 x 1000 m2 x 0.9 x rho 1.0 = 900 kW; x eta_rec 0.8 x availability 1.0 = 720 kW
    r = dispatch.simulate_year(df, 1000.0, 0.9, np.ones(48), 1.0, 0.8, 1e9, np.zeros(48))
    assert abs(r["q_rec"][0] - 720.0) < 1e-9
    assert abs(r["level"][-1] - 720.0 * 48) < 1e-6 and r["curtailed"].sum() == 0


def test_energy_balance_and_limits():
    idx = year_index()
    rng = np.random.default_rng(1)
    dni = np.where((idx.hour >= 6) & (idx.hour <= 17), rng.uniform(0, 900, len(idx)), 0.0)
    df = pd.DataFrame({"DNI": dni}, index=idx)
    dem = dispatch.operating_demand_kw(idx, 1.0, 8, 12, 330, 0)
    rho = np.full(len(idx), 0.88)
    r = dispatch.simulate_year(df, 3000.0, 0.88 / 0.88 * 0.9, rho, 0.96, 0.81, 4000.0, dem)
    s = dispatch.summarize(df, r, dem)
    # solar heat in = delivered + curtailed + change in storage (starts empty)
    assert abs(s["gross_gwh"] - (s["delivered_gwh"] + s["curtailed_gwh"] + s["storage_end_gwh"])) < 1e-9
    assert 0 <= s["solar_fraction"] <= 1 and (r["delivered"] <= dem + 1e-9).all()
    assert abs(s["backup_gwh"] - (s["demand_gwh"] - s["delivered_gwh"])) < 1e-12


def test_dirtier_mirrors_deliver_less():
    idx = year_index()
    df = pd.DataFrame({"DNI": np.where((idx.hour >= 6) & (idx.hour <= 17), 700.0, 0.0)}, index=idx)
    dem = dispatch.operating_demand_kw(idx, 1.0, 8, 12, 330, 0)
    clean = dispatch.simulate_year(df, 3000.0, 0.9, np.full(len(idx), 0.88), 1.0, 0.81, 4000.0, dem)
    dirty = dispatch.simulate_year(df, 3000.0, 0.9, np.full(len(idx), 0.70), 1.0, 0.81, 4000.0, dem)
    assert dirty["delivered"].sum() <= clean["delivered"].sum()


def test_typical_day_shape():
    idx = year_index()
    df = pd.DataFrame({"DNI": np.where((idx.hour >= 6) & (idx.hour <= 17), 700.0, 0.0)}, index=idx)
    dem = dispatch.operating_demand_kw(idx, 1.0, 8, 12, 365, 0)
    r = dispatch.simulate_year(df, 3000.0, 0.9, np.full(len(idx), 0.88), 1.0, 0.81, 4000.0, dem)
    p = dispatch.typical_day(df, r, dem, 6, 0)
    assert len(p) == 24 and p["Heat demand"].iloc[10] == 1000.0 and p["Heat demand"].iloc[3] == 0.0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("ok  ", t.__name__)
    print(f"\nAll {len(tests)} tests passed.")
