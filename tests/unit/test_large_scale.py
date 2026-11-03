"""Large-scale PDT: shifted log-gamma, point mass and log-normal baselines. (FR-CH-2, FR-CH-3)"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate, stats

import tolerances as tol
from qmwlink.channel.large_scale import (
    ConditionedLogNormal,
    DeterministicLoss,
    LogNormalV0,
    ShiftedLogGamma,
)
from qmwlink.numerics.mcagree import check_moment, check_probabilities, ecdf_estimate


def mc(law: object) -> np.ndarray:
    rng = np.random.default_rng(tol.MC_SEED)
    return law.sample_eta(rng, tol.MC_N)  # type: ignore[attr-defined]


# ---- ShiftedLogGamma ----------------------------------------------------------------------
def test_validation_and_from_moments() -> None:
    for kwargs in (
        {"psi0": -1, "k": 1, "theta": 1},
        {"psi0": 0, "k": 0, "theta": 1},
        {"psi0": 0, "k": 1, "theta": -1},
        {"psi0": np.nan, "k": 1, "theta": 1},
    ):
        with pytest.raises(ValueError, match=r"psi0|k|theta"):
            ShiftedLogGamma(**kwargs)
    law = ShiftedLogGamma.from_moments(1e-3, 0.04, 4e-4)
    assert isinstance(law, ShiftedLogGamma)
    assert law.k == pytest.approx(4.0) and law.theta == pytest.approx(0.01)
    assert law.mu_g == pytest.approx(0.04) and law.sigma_g**2 == pytest.approx(4e-4)
    with pytest.raises(ValueError, match=">= 0"):
        ShiftedLogGamma.from_moments(0.0, -1.0, 1.0)
    with pytest.raises(ValueError, match="mu_g = 0"):
        ShiftedLogGamma.from_moments(0.0, 0.0, 1.0)


def test_degenerate_sigma_zero_is_point_mass() -> None:
    pm = ShiftedLogGamma.from_moments(0.01, 0.02, 0.0)
    assert isinstance(pm, DeterministicLoss) and pm.psi == pytest.approx(0.03)
    assert isinstance(ShiftedLogGamma.from_moments(0.01, 0.0, 0.0), DeterministicLoss)
    x = np.exp(-0.03)
    assert float(pm.cdf(x)) == 1.0 and float(pm.cdf(x * (1 - 1e-9))) == 0.0
    assert float(pm.sf(x * (1 - 1e-9))) == 1.0 and float(pm.sf(x)) == 0.0
    np.testing.assert_array_equal(pm.sample_eta(np.random.default_rng(0), 5), np.full(5, x))
    assert pm.moment(2) == pytest.approx(np.exp(-0.06))
    with pytest.raises(TypeError, match="point mass"):
        pm.psi_pdf(0.1)


def test_small_sigma_converges_to_point_mass() -> None:
    mu = 0.02
    law = ShiftedLogGamma.from_moments(0.0, mu, (1e-4 * mu) ** 2)
    assert law.moment(1) == pytest.approx(np.exp(-mu), rel=1e-6)


def test_reference_mean_log_gamma() -> None:
    """FR-CH-2 AC: Psi0 = 0, k = 4, theta = 0.01 gives mean 0.96098."""
    assert ShiftedLogGamma(0.0, 4.0, 0.01).moment(1) == pytest.approx(
        0.96098, abs=tol.ATOL_MEAN_LOGGAMMA
    )


@pytest.mark.parametrize("psi0", [0.0, 0.05])
def test_moments_closed_form_and_mc_agree(psi0: float) -> None:
    """FR-CH-2 AC: E[eta_l^m] = exp(-m psi0)(1 + m theta)^-k, m in {1, 2}, MC-agree."""
    law = ShiftedLogGamma(psi0, 4.0, 0.01)
    eta = mc(law)
    for m in (1, 2):
        expected = np.exp(-m * psi0) * (1 + m * 0.01) ** -4.0
        assert law.moment(m) == pytest.approx(expected, rel=1e-14)
        assert check_moment(eta**m, law.moment(m)).ok
    assert law.moment(0) == 1.0


def test_support_is_strictly_zero_to_exp_minus_psi0() -> None:
    law = ShiftedLogGamma(0.05, 4.0, 0.01)
    eta = mc(law)
    assert eta.max() < law.eta_max and eta.min() > 0.0
    assert float(law.cdf(law.eta_max)) == 1.0 and float(law.cdf(law.eta_max * 1.1)) == 1.0
    assert float(law.pdf(law.eta_max * 1.01)) == 0.0 and float(law.pdf(0.0)) == 0.0
    assert float(law.cdf(0.0)) == 0.0 and float(law.sf(0.0)) == 1.0


def test_cdf_matches_mc_and_pdf_consistent() -> None:
    law = ShiftedLogGamma(0.02, 4.0, 0.01)
    eta = mc(law)
    pts = np.quantile(eta, np.linspace(0.02, 0.98, 50))
    p_hat, se = ecdf_estimate(eta, pts)
    assert check_probabilities(p_hat, se, law.cdf(pts)).ok
    np.testing.assert_allclose(law.cdf(pts) + law.sf(pts), 1.0, atol=1e-14)
    # pdf integrates to 1 on (0, eta_max] and is the derivative of the cdf
    val, _ = integrate.quad(
        lambda x: float(law.pdf(x)), 0.3, law.eta_max, points=[0.95], epsabs=0, epsrel=1e-10
    )
    assert val == pytest.approx(1.0, abs=1e-9)
    h = 1e-6
    x0 = 0.96
    dcdf = float(law.cdf(x0 + h) - law.cdf(x0 - h)) / (2 * h)
    assert dcdf == pytest.approx(float(law.pdf(x0)), rel=1e-5)


@given(psi0=st.floats(0.0, 1.0), k=st.floats(0.2, 50.0), theta=st.floats(1e-3, 1.0))
@settings(max_examples=40, deadline=None)
def test_property_cdf_monotone_moments_le_one(psi0: float, k: float, theta: float) -> None:
    law = ShiftedLogGamma(psi0, k, theta)
    x = np.linspace(1e-6, 1.0, 400)
    c = law.cdf(x)
    assert np.all(np.diff(c) >= -1e-12) and np.all((c >= 0) & (c <= 1))
    assert np.all(law.pdf(x) >= 0)
    assert 0.0 < law.moment(2) <= law.moment(1) <= 1.0


def test_sf_tiny_tail_is_accurate() -> None:
    """Upper tail near eta_max uses gammainc, so 1 - cdf cancellation is avoided (rule 1)."""
    law = ShiftedLogGamma(0.0, 4.0, 0.01)
    x = 1.0 - 1e-9
    assert float(law.sf(x)) == pytest.approx(float(stats.gamma(4, scale=0.01).cdf(-np.log(x))))
    assert float(law.sf(x)) == pytest.approx((1e-9 / 0.01) ** 4 / 24.0, rel=1e-6)


# ---- baselines ----------------------------------------------------------------------------
def test_v0_lognormal_mass_above_one_reference() -> None:
    """FR-CH-3 AC: Phi(-1) = 0.15866 for sigma_psi = psi_bar; flagged non-physical."""
    law = LogNormalV0(1.44e-4, 1.44e-4)
    assert law.mass_above_one() == pytest.approx(0.158655, abs=tol.ATOL_MASS_ABOVE_ONE)
    assert law.physical is False
    assert law.mass_above_one() == pytest.approx(float(1 - law.cdf(1.0)), rel=1e-12)
    eta = mc(law)
    assert check_moment((eta > 1.0).astype(float), law.mass_above_one()).ok
    assert law.moment(1) == pytest.approx(np.exp(-1.44e-4 + 0.5 * 1.44e-4**2))


def test_v0_lognormal_mean_can_exceed_one() -> None:
    """IDEA D2: E[eta_l] > 1 when sigma^2 > 2 psi_bar."""
    assert LogNormalV0(0.01, 0.5).moment(1) > 1.0
    assert LogNormalV0(1.0, 0.1).moment(1) < 1.0


def test_v0_lognormal_cdf_mc() -> None:
    law = LogNormalV0(0.2, 0.15)
    eta = mc(law)
    pts = np.quantile(eta, np.linspace(0.02, 0.98, 30))
    p_hat, se = ecdf_estimate(eta, pts)
    assert check_probabilities(p_hat, se, law.cdf(pts)).ok
    val, _ = integrate.quad(
        lambda x: float(law.pdf(x)), 1e-6, 5.0, limit=500, points=[0.8, 1.0, 1.2]
    )
    assert val == pytest.approx(1.0, abs=1e-6)
    assert law.psi_lo_eff() < 0 and len(law.psi_breakpoints()) == 7
    assert check_moment(eta, law.moment(1)).ok


def test_conditioned_lognormal_support_normalizer_and_moments() -> None:
    law = ConditionedLogNormal(1.44e-4, 1.44e-4)
    assert law.physical is True
    assert law.normalizer == pytest.approx(float(stats.norm.cdf(1.0)))
    eta = mc(law)
    assert eta.max() <= 1.0
    assert float(law.cdf(1.0)) == pytest.approx(1.0) and float(law.pdf(1.0001)) == 0.0
    for m in (1, 2):
        assert check_moment(eta**m, law.moment(m)).ok
    law2 = ConditionedLogNormal(0.3, 0.2)
    eta2 = mc(law2)
    pts = np.quantile(eta2, np.linspace(0.02, 0.98, 30))
    p_hat, se = ecdf_estimate(eta2, pts)
    assert check_probabilities(p_hat, se, law2.cdf(pts)).ok
    np.testing.assert_allclose(law2.psi_cdf(0.3) + law2.psi_sf(0.3), 1.0, atol=1e-12)
    assert float(law2.psi_cdf(-1.0)) == 0.0 and float(law2.psi_sf(-1.0)) == 1.0
    val, _ = integrate.quad(lambda p: float(law2.psi_pdf(p)), 0.0, 3.0, points=[0.3], epsabs=0)
    assert val == pytest.approx(1.0, abs=1e-9)
    assert min(law2.psi_breakpoints()) > 0 and law2.psi_lo_eff() == 0.0


def test_baseline_validation() -> None:
    for cls in (LogNormalV0, ConditionedLogNormal):
        with pytest.raises(ValueError, match="psi_bar"):
            cls(0.0, 1.0)
        with pytest.raises(ValueError, match="sigma_psi"):
            cls(1.0, 0.0)
    with pytest.raises(ValueError, match="m must"):
        LogNormalV0(1.0, 1.0).moment(-1)
    with pytest.raises(ValueError, match="m must"):
        ConditionedLogNormal(1.0, 1.0).moment(-1)
    with pytest.raises(ValueError, match="m must"):
        ShiftedLogGamma(0, 1, 1).moment(-1)
    with pytest.raises(ValueError, match="m must"):
        DeterministicLoss(0.1).moment(-1)
    with pytest.raises(ValueError, match="psi"):
        DeterministicLoss(-0.1)


def test_psi_helpers() -> None:
    law = ShiftedLogGamma(0.1, 2.0, 0.5)
    assert law.psi_lo_eff() == 0.1 and all(p > 0.1 for p in law.psi_breakpoints())
    d = DeterministicLoss(0.2)
    assert d.psi_lo_eff() == 0.2 and d.psi_breakpoints() == [0.2]
    assert float(d.psi_cdf(0.2)) == 1.0 and float(d.psi_sf(0.2)) == 1.0


def test_v0_lognormal_psi_cdf_complements_sf() -> None:
    law = LogNormalV0(0.2, 0.15)
    np.testing.assert_allclose(law.psi_cdf([0.0, 0.2, 0.5]) + law.psi_sf([0.0, 0.2, 0.5]), 1.0)
