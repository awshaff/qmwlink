"""Short-dipole aperture and BBR-equivalent field. (FR-PHY-3, TSD §7.4)"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import tolerances as tol
from qmwlink.constants import C
from qmwlink.thermal import (
    effective_aperture_m2,
    n_th,
    s_e_bbr,
    s_e_bbr_energy_density_rj,
)

F_GRID = np.array([1e9, 5e9, 30e9, 100e9, 300e9])


def test_aperture_formula_and_shape() -> None:
    lam = C / 5e9
    assert float(effective_aperture_m2(5e9)) == pytest.approx(3 * lam**2 / (8 * np.pi), rel=1e-14)
    assert effective_aperture_m2(F_GRID).shape == F_GRID.shape
    assert np.all(np.diff(effective_aperture_m2(F_GRID)) < 0)  # ~ 1/f^2


@given(f=st.floats(1e8, 1e12), T=st.floats(1.0, 1000.0))
def test_two_routes_agree_under_rj(f: float, T: float) -> None:
    """FR-PHY-3 (a): mode-occupation route == energy-density route (algebraic identity)."""
    a = float(s_e_bbr(f, T, planck=False))
    b = float(s_e_bbr_energy_density_rj(f, T))
    assert a == pytest.approx(b, rel=tol.RTOL_BBR_ROUTES)


@pytest.mark.parametrize("f", F_GRID)
def test_planck_vs_rj_field_amplitude_below_1p5_percent(f: float) -> None:
    """FR-PHY-3 (b): amplitude differs < 1.5% for f <= 300 GHz at 300 K."""
    rel = float(s_e_bbr(f, 300.0, planck=False) / s_e_bbr(f, 300.0) - 1.0)
    assert 0.0 < rel < tol.MAX_PLANCK_VS_RJ_FIELD


def test_planck_vs_rj_spot_300ghz() -> None:
    rel = float(s_e_bbr(300e9, 300.0, planck=False) / s_e_bbr(300e9, 300.0) - 1.0)
    assert rel == pytest.approx(0.0121, abs=5e-5)  # 1.21% (FSD FR-PHY-3)
    power = (1.0 + rel) ** 2 - 1.0
    assert power == pytest.approx(0.0244, abs=5e-5)  # power ratio, reported not gated


def test_s_e_bbr_zero_at_zero_temperature_and_scaling() -> None:
    assert float(s_e_bbr(5e9, 0.0)) == 0.0
    assert float(s_e_bbr(5e9, 400.0)) / float(s_e_bbr(5e9, 100.0)) == pytest.approx(
        np.sqrt(float(n_th(5e9, 400.0)) / float(n_th(5e9, 100.0)))
    )


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="f_hz"):
        effective_aperture_m2(0.0)
    with pytest.raises(ValueError, match="T_env_k"):
        s_e_bbr(5e9, -1.0)
