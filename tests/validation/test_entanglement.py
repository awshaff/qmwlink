"""Identity and limit tests for entanglement measures. (FR-GAU-4..6, TSD §5, §10)"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import tolerances as tol
from qmwlink.gaussian.channels import PIChannel, apply_to_mode_b
from qmwlink.gaussian.entanglement import (
    fidelity_bk,
    fidelity_opt,
    is_entangled_standard,
    log_negativity,
    negativity,
    nu_minus_general,
    nu_minus_standard,
    standard_form_params,
    tmst_eta_star,
    tmst_is_entangled,
    tmst_one_minus_eta_star,
    tmsv_is_entangled,
    tmsv_through_channel_params,
)
from qmwlink.gaussian.states import tmst_cm, tmsv_cm, two_mode_ac
from qmwlink.thermal import eta_star, n_th

R_GRID = (0.1, 0.8, 2.0)


# ---- FR-GAU-4: general vs closed form -----------------------------------------------------
def test_tmsv_pure_eigenvalue_r3_both_forms() -> None:
    """FR-GAU-4 (b): TMSV at r = 3 has nu_- = exp(-6) to rtol 1e-10, in both forms."""
    v = tmsv_cm(3.0)
    assert float(nu_minus_general(v)) == pytest.approx(np.exp(-6.0), rel=tol.RTOL_NU_FORMS)
    a, c = np.cosh(6.0), np.sinh(6.0)
    assert float(nu_minus_standard(a, a, c)) == pytest.approx(np.exp(-6.0), rel=tol.RTOL_NU_FORMS)


def test_tmsv_pure_eigenvalue_over_r() -> None:
    r = np.linspace(0.0, 3.0, 31)
    np.testing.assert_allclose(nu_minus_general(tmsv_cm(r)), np.exp(-2 * r), rtol=tol.RTOL_NU_FORMS)


def test_general_vs_closed_form_random_cms() -> None:
    """FR-GAU-4 (a): 10^4 random valid CMs, b up to 1e4, relative rtol 1e-10."""
    rng = np.random.default_rng(20261103)
    n = 10_000
    r = rng.uniform(0.0, 3.0, n)
    n_j = rng.uniform(0.0, 1.0, n)
    eta = rng.uniform(0.0, 1.0, n)
    n_bar = 10.0 ** rng.uniform(-3.0, np.log10(5000.0), n)
    v = apply_to_mode_b(tmst_cm(r, n_j), PIChannel.thermal_attenuator(eta, n_bar))
    a, b, cp = standard_form_params(v)
    assert float(np.max(b)) > 1e3  # the microwave thermal regime is exercised
    np.testing.assert_allclose(
        nu_minus_general(v), nu_minus_standard(a, b, cp), rtol=tol.RTOL_NU_FORMS, atol=0.0
    )


def test_stable_closed_form_with_analytic_det_large_thermal() -> None:
    """a + b ~ 1e4: stable form with analytic det (= a y + tau for TMSV) vs reference."""
    r, ch = 1.0, PIChannel.thermal_attenuator(0.9, 5e3)
    a, b, cp = tmsv_through_channel_params(r, ch)
    det = a * ch.y_arr + ch.tau_arr
    nu_plus = ((a + b) + np.sqrt((a - b) ** 2 + 4 * cp**2)) / 2
    got = nu_minus_standard(a, b, cp, det=det)
    assert float(got) == pytest.approx(float(det / nu_plus), rel=1e-12)
    assert float(nu_minus_standard(a, b, cp, stable=False)) == pytest.approx(float(got), rel=1e-6)


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


# ---- FR-GAU-5 -----------------------------------------------------------------------------
@pytest.mark.parametrize("r", R_GRID)
def test_tmsv_separability_equivalence_grid(r: float) -> None:
    """nu_- < 1 <=> y < 1 + tau on 10^4 points, excluding |y - (1 + tau)| < 1e-9."""
    n = 10_000
    rng = np.random.default_rng(7)
    half = n // 2
    # half: physical thermal attenuators; half: channels straddling the threshold y = 1 + tau
    ch_a = PIChannel.thermal_attenuator(rng.uniform(0, 1, half), rng.uniform(0, 2, half))
    tau = rng.uniform(0.0, 1.0, half)
    d = rng.choice([-1, 1], half) * 10.0 ** rng.uniform(-10.0, -2.0, half)
    ch_b = PIChannel(tau, 1.0 + tau + d)
    tau_all = np.concatenate([ch_a.tau_arr, ch_b.tau_arr])
    y_all = np.concatenate([ch_a.y_arr, ch_b.y_arr])
    ch = PIChannel(tau_all, y_all)
    a, b, cp = tmsv_through_channel_params(r, ch)
    nu = nu_minus_standard(a, b, cp, det=a * ch.y_arr + ch.tau_arr)
    keep = np.abs(ch.y_arr - (1.0 + ch.tau_arr)) >= tol.BOUNDARY_EXCLUDE_Y
    assert keep.sum() > 0.9 * n
    pred = tmsv_is_entangled(ch)
    assert np.array_equal((nu < 1.0)[keep], pred[keep])
    # (a-1)(b-1) < c'^2 is an independent rewriting of the same condition
    assert np.array_equal(is_entangled_standard(a, b, cp)[keep], pred[keep])
    assert pred.any() and (~pred).any()


@pytest.mark.parametrize("r", R_GRID)
def test_tmsv_threshold_independent_of_r(r: float) -> None:
    """Additive noise on a lossless link: entangled iff N_add < 1, for any r > 0."""
    n_vals = np.array([0.5, 0.9, 1.1, 1.5])
    ch = PIChannel.additive_noise(n_vals)
    a, b, cp = tmsv_through_channel_params(r, ch)
    nu = nu_minus_standard(a, b, cp, det=a * ch.y_arr + ch.tau_arr)
    np.testing.assert_array_equal(nu < 1.0, n_vals < 1.0)


def test_nu_equals_one_exactly_on_threshold() -> None:
    ch = PIChannel(0.5, 1.5)  # y = 1 + tau
    a, b, cp = tmsv_through_channel_params(0.8, ch)
    nu = nu_minus_standard(a, b, cp, det=a * ch.y_arr + ch.tau_arr)
    assert float(nu) == pytest.approx(1.0, abs=1e-12)
    assert bool(ch.is_entanglement_breaking) and not bool(tmsv_is_entangled(ch))


# ---- FR-GAU-6 -----------------------------------------------------------------------------
@pytest.mark.parametrize("r", [0.1, 0.8, 2.0])
@pytest.mark.parametrize("n_j", [0.0, 0.3, 2.0])
def test_tmst_condition_matches_nu_grid(r: float, n_j: float) -> None:
    """FR-GAU-6 (a): condition agrees with direct nu_- evaluation on a grid."""
    rng = np.random.default_rng(11)
    n = 5_000
    ch = PIChannel.thermal_attenuator(rng.uniform(0, 1, n), rng.uniform(0, 3, n))
    nu = nu_minus_general(apply_to_mode_b(tmst_cm(r, n_j), ch))
    pred = tmst_is_entangled(r, n_j, ch)
    ok = np.abs(nu - 1.0) > 1e-6
    assert np.array_equal((nu < 1.0)[ok], pred[ok])


def test_tmst_reduces_to_tmsv_at_zero_jpa_noise() -> None:
    """FR-GAU-6 (b)."""
    ch = PIChannel.thermal_attenuator(np.linspace(0, 1, 1001), 1.0)
    ok = np.abs(ch.y_arr - 1.0 - ch.tau_arr) >= tol.BOUNDARY_EXCLUDE_Y
    np.testing.assert_array_equal(tmst_is_entangled(0.8, 0.0, ch)[ok], tmsv_is_entangled(ch)[ok])


def test_tmst_threshold_not_below_tmsv() -> None:
    """IDEA D1: a mixed state is never more robust than pure TMSV."""
    ch = PIChannel.thermal_attenuator(np.linspace(0.9, 1.0, 20001), 5.0)
    assert not np.any(tmst_is_entangled(1.0, 0.5, ch) & ~tmsv_is_entangled(ch))


N_TH_5GHZ_300K = float(n_th(5e9, 300.0))


@pytest.mark.parametrize(
    ("n_jpa", "r", "ref"),
    [(0.0, 0.5, 0.99920045), (0.05, 0.5, 0.99932075), (0.002, 0.5, 0.99920628)],
)
def test_tmst_threshold_reference_values(n_jpa: float, r: float, ref: float) -> None:
    """FR-GAU-6 (c): 5 GHz, 300 K reference thresholds (+/- 1e-8)."""
    assert tmst_eta_star(N_TH_5GHZ_300K, r, n_jpa) == pytest.approx(
        ref, abs=tol.ATOL_TMST_THRESHOLD
    )


@pytest.mark.parametrize("r", [0.05, 0.5, 1.0, 3.0])
def test_tmst_threshold_pure_state_independent_of_r(r: float) -> None:
    assert tmst_eta_star(N_TH_5GHZ_300K, r, 0.0) == pytest.approx(
        float(eta_star(5e9, 300.0)), abs=1e-12
    )


@pytest.mark.parametrize(("n_jpa", "r"), [(0.0, 0.3), (0.05, 0.5), (0.5, 1.5), (0.2, 3.0)])
@pytest.mark.parametrize("n_th_val", [0.0, 0.7, 20.341, 1249.697])
def test_tmst_threshold_brentq_vs_linear_closed_form(
    n_th_val: float, n_jpa: float, r: float
) -> None:
    """g(omega) is linear in omega = 1 - eta, so brentq must match the explicit root."""
    a_arr, c_arr = two_mode_ac(r, n_jpa)
    a, c, s = float(a_arr), float(c_arr), 2 * n_th_val + 1
    denom = (a - 1.0) * (s - a) + c**2
    expected = 1.0 if denom == 0 else min(1.0, (c**2 - (a - 1.0) ** 2) / denom)
    if (a - 1.0) ** 2 - c**2 >= 0:
        pytest.skip("not entangled at eta = 1")
    got = tmst_one_minus_eta_star(n_th_val, r, n_jpa)
    assert got == pytest.approx(expected, rel=1e-10)


def test_tmst_threshold_matches_direct_nu_root() -> None:
    """Independent check: nu_-(eta) crosses 1 at the returned eta*."""
    n_th_val, r, n_jpa = 20.341, 1.0, 0.1
    eta = tmst_eta_star(n_th_val, r, n_jpa)
    lo = apply_to_mode_b(tmst_cm(r, n_jpa), PIChannel.thermal_attenuator(eta - 1e-6, n_th_val))
    hi = apply_to_mode_b(tmst_cm(r, n_jpa), PIChannel.thermal_attenuator(eta + 1e-6, n_th_val))
    assert float(nu_minus_general(lo)) > 1.0 > float(nu_minus_general(hi))


def test_tmst_threshold_errors() -> None:
    with pytest.raises(ValueError, match="not entangled"):
        tmst_one_minus_eta_star(10.0, 0.0, 0.0)  # r = 0: separable even at eta = 1
    with pytest.raises(ValueError, match="not entangled"):
        tmst_one_minus_eta_star(10.0, 0.1, 5.0)  # too thermal
    with pytest.raises(ValueError, match="n_th"):
        tmst_one_minus_eta_star(-1.0, 0.5, 0.0)
    assert tmst_one_minus_eta_star(0.0, 0.5, 0.0) == 1.0  # T = 0: entangled for all eta > 0


# ---- FR-GAU-4: measures and fidelities -----------------------------------------------------
def test_measures_values_and_limits() -> None:
    nu = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(negativity(nu), [0.5, 0.0, 0.0])
    np.testing.assert_allclose(log_negativity(nu), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(fidelity_opt(nu), [1 / 1.5, 0.5, 1 / 3])
    assert np.isinf(negativity(0.0)) and np.isinf(log_negativity(0.0))
    assert float(fidelity_opt(1.0)) == 0.5


def test_fidelity_reference_values() -> None:
    """FR-GAU-4 (d): TMSV r = 1, 5 GHz / 300 K attenuator, eta = 0.9995."""
    ch = PIChannel.thermal_attenuator(0.9995, N_TH_5GHZ_300K)
    v = apply_to_mode_b(tmsv_cm(1.0), ch)
    a, b, cp = standard_form_params(v)
    nu = nu_minus_standard(a, b, cp, det=a * ch.y_arr + ch.tau_arr)
    assert float(fidelity_opt(nu)) == pytest.approx(0.5858, abs=tol.ATOL_FIDELITY_REF)
    assert float(fidelity_bk(v)) == pytest.approx(0.5681, abs=tol.ATOL_FIDELITY_REF)


def test_fidelity_bk_standard_form_equals_a_plus_b_minus_2c() -> None:
    v = apply_to_mode_b(tmsv_cm(0.7), PIChannel.thermal_attenuator(0.8, 3.0))
    a, b, cp = standard_form_params(v)
    assert float(fidelity_bk(v)) == pytest.approx(float(1.0 / (1.0 + (a + b - 2 * cp) / 2)))


def test_fidelity_opt_ge_bk_grid_and_equality_when_symmetric() -> None:
    """FR-GAU-4 (c)."""
    rng = np.random.default_rng(5)
    n = 5_000
    ch = PIChannel.thermal_attenuator(rng.uniform(0, 1, n), 10 ** rng.uniform(-2, 3, n))
    v = apply_to_mode_b(tmst_cm(rng.uniform(0, 3, n), rng.uniform(0, 1, n)), ch)
    nu = nu_minus_general(v)
    fo, fb = fidelity_opt(nu), fidelity_bk(v)
    assert np.all(fo >= fb * (1.0 - 1e-12))
    sym = tmst_cm(rng.uniform(0, 3, 200), rng.uniform(0, 1, 200))  # a = b
    np.testing.assert_allclose(fidelity_opt(nu_minus_general(sym)), fidelity_bk(sym), rtol=1e-9)


@given(r=st.floats(0.0, 3.0), eta=st.floats(0.0, 1.0), nb=st.floats(0.0, 1e3), nj=st.floats(0, 1))
def test_property_fidelity_opt_ge_bk(r: float, eta: float, nb: float, nj: float) -> None:
    ch = PIChannel.thermal_attenuator(eta, nb)
    v = apply_to_mode_b(tmst_cm(r, nj), ch)
    a, c = (float(x) for x in two_mode_ac(r, nj))
    nu = nu_minus_standard(
        a, float(ch.tau_arr) * a + float(ch.y_arr), np.sqrt(float(ch.tau_arr)) * c
    )
    # Production path (standard-form closed form): the frozen 1e-9 slack.
    assert float(fidelity_opt(nu)) >= float(fidelity_bk(v)) * (1 - 1e-9)
    # General form: its discriminant cancels when nu_+ ~ nu_- (r -> 0), leaving an O(sqrt(eps))
    # absolute error in nu_- (QUESTIONS Q-12), so it is checked with that absolute allowance.
    assert (
        float(fidelity_opt(nu_minus_general(v)))
        >= float(fidelity_bk(v)) - tol.ABS_F_GENERAL_DEGENERATE
    )


def test_fidelity_bk_shape_error_and_is_entangled_domain() -> None:
    with pytest.raises(ValueError, match="shape"):
        fidelity_bk(np.eye(3))
    with pytest.raises(ValueError, match=">= 1"):
        is_entangled_standard(0.5, 1.0, 0.1)
