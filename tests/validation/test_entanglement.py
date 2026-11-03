"""Identity and limit tests for entanglement measures. (FR-GAU-4..6, TSD §5, §10)"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qmwlink.gaussian.channels import PIChannel, apply_to_mode_b
from qmwlink.gaussian.entanglement import (
    is_entangled_standard,
    log_negativity,
    negativity,
    nu_minus_general,
    nu_minus_standard,
    standard_form_params,
    teleportation_fidelity,
    tmst_is_entangled,
    tmsv_is_entangled,
    tmsv_through_channel_params,
)
from qmwlink.gaussian.states import tmst_cm, tmsv_cm

R_GRID = (0.1, 0.8, 2.0)
BOUNDARY_GUARD = 1e-9  # skip grid points numerically on the nu = 1 boundary


def test_tmsv_vacuum_pure_eigenvalue() -> None:
    """No channel: TMSV has nu_- = exp(-2r)."""
    r = np.array([0.0, 0.3, 1.0, 2.5])
    # General formula: nu^2 = (D - sqrt(disc))/2 cancels when nu << 1 (Q-2). Hold the
    # 1e-12 absolute claim where nu >= 0.1, and only 1e-8 relative at r = 2.5 (nu ~ 7e-3).
    gen = nu_minus_general(tmsv_cm(r))
    np.testing.assert_allclose(gen[r <= 1.0], np.exp(-2 * r)[r <= 1.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(gen, np.exp(-2 * r), rtol=1e-8)
    a, c = np.cosh(2 * r), np.sinh(2 * r)
    # analytic det(ab - c'^2) = 1 avoids the cosh^2 - sinh^2 cancellation
    np.testing.assert_allclose(nu_minus_standard(a, a, c, det=1.0), np.exp(-2 * r), rtol=1e-12)
    # without it the cancellation costs digits (documented limitation, still < 1e-7 here)
    np.testing.assert_allclose(nu_minus_standard(a, a, c), np.exp(-2 * r), rtol=1e-7)


@given(
    r=st.floats(0.0, 3.0),
    eta=st.floats(0.0, 1.0),
    n=st.floats(0.0, 1e3),
    n_j=st.floats(0.0, 5.0),
)
def test_general_equals_closed_form_random_valid_cms(
    r: float, eta: float, n: float, n_j: float
) -> None:
    """FR-GAU-4 AC: general and closed forms agree to 1e-12 on random valid CMs."""
    v = apply_to_mode_b(tmst_cm(r, n_j), PIChannel.thermal_attenuator(eta, n))
    a, b, cp = standard_form_params(v)
    gen = float(nu_minus_general(v))
    for stable in (True, False):
        cf = float(nu_minus_standard(a, b, cp, stable=stable))
        assert cf == pytest.approx(gen, rel=1e-6, abs=1e-9)
    # the two closed forms agree to 1e-12 relative to the scale a + b
    s = float(nu_minus_standard(a, b, cp, stable=True))
    d = float(nu_minus_standard(a, b, cp, stable=False))
    assert abs(s - d) <= 1e-12 * max(1.0, float(a + b))


def test_general_vs_closed_tight_on_moderate_grid() -> None:
    rng = np.random.default_rng(1234)
    r = rng.uniform(0.05, 2.0, 500)
    eta = rng.uniform(0.0, 1.0, 500)
    n = rng.uniform(0.0, 3.0, 500)
    v = apply_to_mode_b(tmsv_cm(r), PIChannel.thermal_attenuator(eta, n))
    a, b, cp = standard_form_params(v)
    np.testing.assert_allclose(nu_minus_general(v), nu_minus_standard(a, b, cp), rtol=0, atol=1e-12)


def test_stable_form_avoids_cancellation_for_large_a_plus_b() -> None:
    """a + b ~ 1e4: direct form loses digits; stable form matches analytic ab - c'^2 = a y + tau."""
    r, ch = 1.0, PIChannel.thermal_attenuator(0.9, 5e3)
    a, b, cp = tmsv_through_channel_params(r, ch)
    exact_det = a * ch.y_arr + ch.tau_arr
    nu_plus = ((a + b) + np.sqrt((a - b) ** 2 + 4 * cp**2)) / 2
    ref = exact_det / nu_plus
    assert float(nu_minus_standard(a, b, cp, stable=True)) == pytest.approx(float(ref), rel=1e-9)


def test_standard_form_params_rejects_non_standard() -> None:
    v = np.array(tmsv_cm(0.5))
    v[0, 1] = v[1, 0] = 0.1
    with pytest.raises(ValueError, match="standard form"):
        standard_form_params(v)


def test_nu_minus_general_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="shape"):
        nu_minus_general(np.eye(3))
    bad = np.array(  # symmetric but not a covariance matrix: negative discriminant
        [
            [0.251, -0.668, -0.063, -2.22],
            [-0.668, 0.723, 0.039, 0.728],
            [-0.063, 0.039, -1.247, -1.205],
            [-2.22, 0.728, -1.205, -1.465],
        ]
    )
    with pytest.raises(ValueError, match="valid covariance"):
        nu_minus_general(bad)


@pytest.mark.parametrize("r", R_GRID)
def test_tmsv_separability_equivalence_grid(r: float) -> None:
    """FR-GAU-5 AC: nu_- < 1 <=> y < 1 + tau on a 10^4-point grid (no V0-style tuning)."""
    n = 10_000
    rng = np.random.default_rng(7)
    eta = rng.uniform(0.0, 1.0, n)
    nbar = rng.uniform(0.0, 2.0, n)
    ch = PIChannel.thermal_attenuator(eta, nbar)
    # also sweep points straddling the threshold y = 1 + tau for mixed channels
    a, b, cp = tmsv_through_channel_params(r, ch)
    nu = nu_minus_standard(a, b, cp)
    pred = tmsv_is_entangled(ch)
    ok = np.abs(nu - 1.0) > BOUNDARY_GUARD
    assert ok.sum() > 0.9 * n
    assert np.array_equal((nu < 1.0)[ok], pred[ok])
    ok2 = ok & (np.abs((a - 1) * (b - 1) - cp**2) > 1e-9)
    assert np.array_equal(pred[ok2], is_entangled_standard(a, b, cp)[ok2])
    assert pred.any() and (~pred).any()


@pytest.mark.parametrize("r", R_GRID)
def test_tmsv_threshold_independent_of_r(r: float) -> None:
    """Additive noise N on a lossless link: entangled iff 2N < 2, i.e. N < 1, for any r>0."""
    n_vals = np.array([0.5, 0.9, 1.1, 1.5])
    ch = PIChannel.additive_noise(n_vals)
    a, b, cp = tmsv_through_channel_params(r, ch)
    nu = nu_minus_standard(a, b, cp)
    np.testing.assert_array_equal(nu < 1.0, n_vals < 1.0)


def test_nu_equals_one_exactly_on_threshold() -> None:
    ch = PIChannel(0.5, 1.5)  # y = 1 + tau
    a, b, cp = tmsv_through_channel_params(0.8, ch)
    assert float(nu_minus_standard(a, b, cp)) == pytest.approx(1.0, abs=1e-12)
    assert bool(ch.is_entanglement_breaking) and not bool(tmsv_is_entangled(ch))


@pytest.mark.parametrize("r", [0.1, 0.8, 2.0])
@pytest.mark.parametrize("n_j", [0.0, 0.3, 2.0])
def test_tmst_condition_matches_nu_grid(r: float, n_j: float) -> None:
    """FR-GAU-6 AC: condition agrees with direct nu_- evaluation on a grid."""
    rng = np.random.default_rng(11)
    n = 5_000
    eta = rng.uniform(0.0, 1.0, n)
    nbar = rng.uniform(0.0, 3.0, n)
    ch = PIChannel.thermal_attenuator(eta, nbar)
    v = apply_to_mode_b(tmst_cm(r, n_j), ch)
    nu = nu_minus_general(v)
    pred = tmst_is_entangled(r, n_j, ch)
    ok = np.abs(nu - 1.0) > 1e-6
    assert np.array_equal((nu < 1.0)[ok], pred[ok])


def test_tmst_reduces_to_tmsv_at_zero_jpa_noise() -> None:
    eta = np.linspace(0, 1, 1001)
    ch = PIChannel.thermal_attenuator(eta, 1.0)
    ok = np.abs(ch.y_arr - 1.0 - ch.tau_arr) > 1e-9
    np.testing.assert_array_equal(tmst_is_entangled(0.8, 0.0, ch)[ok], tmsv_is_entangled(ch)[ok])


def test_tmst_threshold_not_below_tmsv() -> None:
    """IDEA D1: eta*_TMST >= eta*; a mixed state is never more robust than pure TMSV."""
    eta = np.linspace(0.9, 1.0, 20001)
    ch = PIChannel.thermal_attenuator(eta, 5.0)
    pure = tmsv_is_entangled(ch)
    mixed = tmst_is_entangled(1.0, 0.5, ch)
    assert not np.any(mixed & ~pure)


def test_measures_values_and_limits() -> None:
    nu = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(negativity(nu), [0.5, 0.0, 0.0])
    np.testing.assert_allclose(log_negativity(nu), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(teleportation_fidelity(nu), [1 / 1.5, 0.5, 1 / 3])
    assert np.isinf(negativity(0.0)) and np.isinf(log_negativity(0.0))
    assert float(teleportation_fidelity(1.0)) == 0.5  # classical-limit 1/2 at nu = 1


def test_is_entangled_standard_domain() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        is_entangled_standard(0.5, 1.0, 0.1)
