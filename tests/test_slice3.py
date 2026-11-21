"""Run with:  python tests/test_slice3.py   (from the project folder)"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from core import geometry, soiling, soiling_field


def test_sun_noon_altitude_at_equinox():
    idx = pd.date_range("2023-03-21", periods=24, freq="1h")      # day 80, near the equinox
    e, n, u = geometry.sun_unit_vectors(idx, 24.09, 32.90)
    alt = np.degrees(np.arcsin(u))
    assert abs(alt.max() - (90 - 24.09)) < 1.0                    # noon altitude = 90 - latitude
    assert idx[int(alt.argmax())].hour == 9 or idx[int(alt.argmax())].hour == 10  # solar noon ~ 09:45 UTC at 32.9 E
    assert alt[0] < 0 and alt[12] > 0                             # night at 00 UTC... day at 12 UTC


def test_heliostat_bisects_sun_and_receiver():
    sun = (np.array([0.0]), np.array([0.0]), np.array([1.0]))     # sun at the zenith
    tilt, aoi = geometry.heliostat_tilt_aoi(sun, 0.0, 38.0, 40.0)  # 38 m north, receiver 38 m higher
    half = 0.5 * math.degrees(math.atan2(38.0, 38.0))
    assert abs(tilt[0] - half) < 1e-6 and abs(aoi[0] - half) < 1e-6


def test_night_stow_uses_stow_tilt():
    sun = (np.array([0.0]), np.array([0.0]), np.array([-0.5]))
    tilt, aoi = geometry.heliostat_tilt_aoi(sun, 10.0, 0.0, 40.0, stow_tilt_deg=180.0)
    assert tilt[0] == 180.0


def test_soiling_matches_hand_calculation():
    # 24 h, constant PM10 = 40, mu = 4e-6, mirror flat (tilt 0), incidence 15 deg, clean at start.
    idx = pd.date_range("2023-01-01", periods=24, freq="1h")
    met = pd.DataFrame({"pm": 40.0, "tilt_deg": 0.0, "aoi_deg": 15.0}, index=idx)
    p = soiling.ConstantMeanParams(mu=4e-6, sigma_dep=1.5e-6, dt_hours=1.0, rho0=0.88)
    pol = soiling.CleaningPolicy(kind="fixed", cleans_per_year=1)
    rho_op, rho_ref, events, _ = soiling.simulate_constant_mean(met, p, pol, None, soiling.Notes())
    area = 24 * 40.0 * 4e-6
    expected = 0.88 * (1 - area * 2.0 / math.cos(math.radians(15)))
    assert abs(rho_op[-1] - expected) < 1e-9


def test_field_run_sensible_and_cleaning_helps():
    idx = pd.date_range("2023-01-01", periods=24 * 365, freq="1h")
    dni = np.where((idx.hour >= 7) & (idx.hour <= 15), 700.0, 0.0)
    df = pd.DataFrame({"DNI": dni}, index=idx)
    p = soiling.ConstantMeanParams(mu=4e-6, sigma_dep=1.5e-6, dt_hours=1.0, rho0=0.88)
    pm = soiling_field.constant_pm(idx, 40.0)
    losses = []
    for n in (1, 12, 52):
        pol = soiling.CleaningPolicy(kind="fixed", cleans_per_year=n)
        r = soiling_field.run_field_soiling(df, 24.09, 32.90, pm, p, pol, 40.0, 25.0)
        losses.append(r["loss_pct"])
        assert 0 <= r["loss_pct"] < 100 and r["rho_weighted"] <= 0.88
    assert losses[0] > losses[1] > losses[2]                      # more cleaning -> less loss


def test_pm_csv_alignment_and_errors():
    idx = pd.date_range("2023-01-01", periods=48, freq="1h")
    text = "time,pm\n" + "\n".join(f"{t:%Y-%m-%d %H:%M},{30 + i}" for i, t in enumerate(idx))
    s = soiling_field.parse_pm_csv(text, idx)
    assert len(s) == 48 and s.iloc[0] == 30
    for bad in ("time,pm\n2023-01-01 00:00,30\n", "a\n1\n"):
        try:
            soiling_field.parse_pm_csv(bad, idx)
        except ValueError:
            continue
        raise AssertionError("accepted bad PM file")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    import time
    for t in tests:
        t0 = time.time(); t(); print("ok  ", t.__name__, f"({time.time()-t0:.1f}s)")
    print(f"\nAll {len(tests)} tests passed.")
