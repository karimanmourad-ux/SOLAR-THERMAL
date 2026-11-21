"""Run with:  python tests/test_slice1.py   (from the project folder)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from core import demand, weather, sources


def fake_nasa_csv(hours=48):
    lines = ["-BEGIN HEADER-", "NASA/POWER fake header", "-END HEADER-",
             "YEAR,MO,DY,HR,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M"]
    for h in range(hours):
        dni = -999 if h == 5 else 100.0
        lines.append(f"2023,1,{1 + h // 24},{h % 24},{dni},50.0,20.0,40.0,3.0")
    return "\n".join(lines)


def test_parse_and_missing_flag():
    df = weather.parse_power_csv(fake_nasa_csv())
    assert list(df.columns) == ["DNI", "GHI", "T_air", "RH", "Wind"]
    assert len(df) == 48 and df["DNI"].isna().sum() == 1


def test_summary_units():
    s = weather.summarize(weather.parse_power_csv(fake_nasa_csv()))
    assert abs(s["dni_kwh_m2"] - 47 * 100 / 1000) < 1e-9
    assert s["missing_dni_hours"] == 1 and s["hours"] == 48


def test_bad_inputs():
    for args in [(95, 0, 2023), (0, 200, 2023), (0, 0, 1999)]:
        try:
            weather.validate_site(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted {args}")
    try:
        weather.parse_power_csv("nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted nonsense")


class FakeSteamTable:  # enthalpies copied from the old notebook's printed output (110 bar case)
    def tsat_p(self, p): return 318.08
    def h_pt(self, p, t): return 3491.87 if t > 400 else 511.41


def test_steam_matches_old_notebook():
    r = demand.steam_heat_duty(110, 130, 550, 80, steam_table=FakeSteamTable())
    assert abs(r["m_kg_s"] - 22.22) < 0.01 and abs(r["q_mw"] - 66.23) < 0.01


def test_steam_invalid_conditions():
    for args in [(110, 130, 300, 80), (110, 330, 550, 80), (110, 130, 550, 0)]:
        try:
            demand.steam_heat_duty(*args, steam_table=FakeSteamTable())
        except ValueError:
            continue
        raise AssertionError(f"accepted {args}")


def test_annual_demand():
    assert demand.annual_demand_mwh(10, 16, 330) == 52800
    assert abs(demand.load_factor(24, 365) - 1.0) < 1e-12


def test_sources_file():
    df = sources.load_sources()
    assert {"id", "item", "source", "status"} <= set(df.columns) and len(df) >= 10


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\nAll {len(tests)} tests passed.")
