"""TMSV/TMST covariance matrices and physicality. (FR-GAU-1, TSD §5.1)"""

from __future__ import annotations

import numpy as np
import pytest

import tolerances as tol
from qmwlink.gaussian.states import (
    is_physical,
    min_physicality_eigenvalue,
    tmst_cm,
    tmsv_cm,
    two_mode_ac,
)


def test_tmsv_structure_and_vacuum() -> None:
    np.testing.assert_allclose(tmsv_cm(0.0), np.eye(4))
    v = tmsv_cm(0.5)
    a, c = np.cosh(1.0), np.sinh(1.0)
    assert (
        v[0, 0] == pytest.approx(a) and v[0, 2] == pytest.approx(c) and v[1, 3] == pytest.approx(-c)
    )
    np.testing.assert_allclose(v, v.T)


def test_tmsv_physical_for_r_in_0_3() -> None:
    r = np.linspace(0.0, 3.0, 61)
    assert np.all(is_physical(tmsv_cm(r)))


def test_tmsv_is_pure_so_min_eig_is_zero() -> None:
    assert np.allclose(
        min_physicality_eigenvalue(tmsv_cm(np.linspace(0, 1.5, 7))), 0.0, atol=tol.PHYS_EIG_ATOL
    )


def test_unphysical_cm_fails() -> None:
    bad = np.array(tmsv_cm(1.0))
    bad[0, 2] *= 1.1
    bad[1, 3] *= 1.1
    bad[2, 0] *= 1.1
    bad[3, 1] *= 1.1
    assert not bool(is_physical(bad))
    assert not bool(is_physical(0.5 * np.eye(4)))  # sub-vacuum variance


def test_tmsv_tmst_physical_over_spec_grid() -> None:
    """FR-GAU-1 AC: TMSV and TMST pass for r in [0, 3], n_jpa in [0, 1]."""
    r, n = np.meshgrid(np.linspace(0.0, 3.0, 31), np.linspace(0.0, 1.0, 21))
    assert np.all(is_physical(tmst_cm(r, n)))
    assert np.all(is_physical(tmsv_cm(r)))


def test_tmst_c_squared_identity_over_grid() -> None:
    """FR-GAU-1 AC: c^2 = a^2 - (2 n_jpa + 1)^2."""
    r, n = np.meshgrid(np.linspace(0.0, 2.0, 21), np.linspace(0.0, 1.0, 11))
    a, c = two_mode_ac(r, n)
    np.testing.assert_allclose(c**2, a**2 - (2 * n + 1) ** 2, rtol=1e-12, atol=1e-12)


def test_tmst_invariants() -> None:
    a, c = two_mode_ac(0.7, 2.0)
    assert float(a**2 - c**2) == pytest.approx(25.0)  # (2n+1)^2, not 1
    assert bool(is_physical(tmst_cm(0.7, 2.0)))
    np.testing.assert_allclose(tmst_cm(0.7, 0.0), tmsv_cm(0.7))


def test_state_input_errors() -> None:
    with pytest.raises(ValueError, match="r must"):
        tmsv_cm(-0.1)
    with pytest.raises(ValueError, match="n_jpa"):
        tmst_cm(0.5, -1.0)
    with pytest.raises(ValueError, match="shape"):
        min_physicality_eigenvalue(np.eye(3))
