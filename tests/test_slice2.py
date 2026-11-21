"""Run with:  python tests/test_slice2.py   (from the project folder)
Checks core/design.py against the reference case printed in Gamil et al. 2023."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import design


class FakeWater:
    """Liquid water at 100 C from standard steam tables (cp 4.216 kJ/kg.K, rho 958.4 kg/m3)."""
    def tsat_p(self, p): return 158.8 if abs(p - 6) < 0.01 else 165.0
    def Cp_pt(self, p, t): return 4.216
    def rho_pt(self, p, t): return 958.4


def close(a, b, tol):
    return abs(a - b) <= tol * abs(b)


def test_solar_multiple_eq2():
    assert abs(design.solar_multiple(3, 9) - 4 / 3) < 1e-12


def test_receiver_geometry_matches_table6():
    # Table 6 gives A=6.873 m2, H=1.811 m, D=1.208 m. Q_th,rec implied by A: 6.873 * 300 kW/m2.
    g = design.receiver_geometry(6.873 * 0.300, 600, 2, 1.5)
    assert close(g["a_rec_m2"], 6.873, 0.001) and close(g["h_rec_m"], 1.811, 0.002) and close(g["d_rec_m"], 1.208, 0.002)


def test_reference_case_within_published_tolerance():
    r = design.size_system(1.3, 3, 9, 50, 150, 6, eta_rec=0.8589, eta_opt_dp=0.8412,
                           dni_design=950, steam_table=FakeWater())
    assert close(r["a_rec_m2"], 6.873, 0.03)        # Eq.2 gives SM 1.333; the paper's own area implies 1.362
    assert close(r["tank_volume_m3"], 50.6, 0.10)   # steam-table water properties; source used its own correlations
    assert close(r["n_helio"], 645, 0.05)


def test_paper_effective_sm_reproduces_table6_area_and_htf_power():
    r = design.size_system(1.3, 3, 9, 50, 150, 6, 0.8589, 0.8412, 950, sm=1.3623, steam_table=FakeWater())
    assert close(r["a_rec_m2"], 6.873, 0.002)
    assert close(r["q_htf_mw"], 1.7708, 0.001)      # 365,521.57 USD / 206.42 USD/kW = 1770.8 kW


def test_bad_inputs():
    bad = [dict(q_load_mw=0), dict(t_hot_c=40), dict(eta_rec=1.2), dict(t_hot_c=170)]
    base = dict(q_load_mw=1.3, t_tes_h=3, x_sun_h=9, t_cold_c=50, t_hot_c=150, p_bar=6,
                eta_rec=0.86, eta_opt_dp=0.84, dni_design=950, steam_table=FakeWater())
    for change in bad:
        args = {**base, **change}
        try:
            design.size_system(**args)
        except ValueError:
            continue
        raise AssertionError(f"accepted {change}")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("ok  ", t.__name__)
    print(f"\nAll {len(tests)} tests passed.")
