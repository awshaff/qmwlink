"""Unit, limit and property tests for thermal.py. (FR-PHY-1, FR-PHY-2, TSD §3, §10)"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import tolerances as tol
from qmwlink.constants import K_B, H
from qmwlink.thermal import eta_star, n_th, n_th_rj, one_minus_eta_star


def test_n_th_zero_temperature_exact() -> None:
    assert n_th(5e9, 0.0) == 0.0
    assert np.all(n_th([1e9, 1e12], 0.0) == 0.0)


def test_n_th_shape_and_dtype() -> None:
    out = n_th(np.array([1e9, 5e9, 30e9]), 300.0)
    assert out.shape == (3,)
    assert out.dtype == np.float64
    out2 = n_th(np.array([[1e9], [5e9]]), np.array([100.0, 300.0]))
    assert out2.shape == (2, 2)


def test_n_th_huge_frequency_underflows_to_zero_without_warning() -> None:
    with np.errstate(all="raise"):
        assert n_th(1e16, 1.0) == 0.0


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_n_th_rejects_bad_frequency(bad: float) -> None:
    with pytest.raises(ValueError, match="f_hz"):
        n_th(bad, 300.0)


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_n_th_rejects_bad_temperature(bad: float) -> None:
    with pytest.raises(ValueError, match="T_env_k"):
        n_th(5e9, bad)


def test_rj_limit() -> None:
    f, T = 1e6, 300.0  # hf/kT ~ 1.6e-7
    assert n_th(f, T) == pytest.approx(n_th_rj(f, T), rel=H * f / (2 * K_B * T) * 1.01)


@given(f=st.floats(1e8, 1e12), T=st.floats(1.0, 1000.0))
def test_rj_relative_error_identity(f: float, T: float) -> None:
    """FR-PHY-1 AC: (RJ - exact)/exact = expm1(x)/x - 1, x = hf/kT, to rtol 1e-9."""
    x = H * f / (K_B * T)
    exact = float(n_th(f, T))
    rj = float(n_th_rj(f, T))
    assert (rj - exact) / exact == pytest.approx(np.expm1(x) / x - 1.0, rel=tol.RTOL_RJ_REL_ERR)


def test_rj_relative_error_spot_300ghz() -> None:
    """FR-PHY-1 AC spot value 2.438e-2 (vs x/2 = 2.400e-2) at 300 GHz, 300 K."""
    f, T = 300e9, 300.0
    err = float(n_th_rj(f, T)) / float(n_th(f, T)) - 1.0
    assert err == pytest.approx(2.438e-2, abs=tol.ABS_RJ_REL_ERR_300GHZ)
    assert H * f / (K_B * T) / 2 == pytest.approx(2.400e-2, abs=tol.ABS_RJ_REL_ERR_300GHZ)


@given(f=st.floats(1e8, 1e12), T=st.floats(0.0, 1000.0))
def test_eta_star_in_unit_interval_and_window_consistent(f: float, T: float) -> None:
    e = float(eta_star(f, T))
    w = float(one_minus_eta_star(f, T))
    assert 0.0 <= e < 1.0
    assert 0.0 < w <= 1.0
    assert e + w == pytest.approx(1.0, abs=1e-15)


@given(T=st.floats(1.0, 1000.0))
def test_n_th_decreasing_in_frequency(T: float) -> None:
    f = np.array([1e9, 5e9, 30e9, 100e9])
    assert np.all(np.diff(n_th(f, T)) < 0)


def test_window_has_no_cancellation_when_eta_star_near_one() -> None:
    n = float(n_th(1e6, 300.0))  # ~6e6 photons
    assert float(one_minus_eta_star(1e6, 300.0)) == pytest.approx(1.0 / (1.0 + n), rel=1e-14)
