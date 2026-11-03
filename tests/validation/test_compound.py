"""Compound PDT eta = eta_l * eta_s. (FR-CH-5, TSD §6.4, §10)"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

import tolerances as tol
from qmwlink.channel.compound import CompoundPDT
from qmwlink.channel.large_scale import (
    ConditionedLogNormal,
    DeterministicLoss,
    LogNormalV0,
    ShiftedLogGamma,
)
from qmwlink.channel.small_scale import SmallScale
from qmwlink.numerics.mcagree import check_moment, check_probabilities, ecdf_estimate

pytestmark = pytest.mark.slow

CASES = {
    "k10_loggamma": (SmallScale(10.0, 0.999), ShiftedLogGamma(1e-4, 4.0, 0.01)),
    "rayleigh_loggamma": (SmallScale(0.0, 0.95), ShiftedLogGamma(0.02, 3.0, 0.02)),
    "mrc20_loggamma": (SmallScale(3.0, 0.99, 20), ShiftedLogGamma(0.02, 4.0, 0.01)),
    "k10_v0_lognormal": (SmallScale(10.0, 0.999), LogNormalV0(0.02, 0.02)),
    "k10_cond_lognormal": (SmallScale(10.0, 0.999), ConditionedLogNormal(0.02, 0.02)),
}


@pytest.fixture(scope="module", params=list(CASES))
def mc_case(request: pytest.FixtureRequest) -> tuple[CompoundPDT, np.ndarray]:
    small, large = CASES[request.param]
    c = CompoundPDT(small, large)
    return c, c.sample(np.random.default_rng(tol.MC_SEED), tol.MC_N)


def test_cdf_vs_mc_agree_50_points(mc_case: tuple[CompoundPDT, np.ndarray]) -> None:
    """FR-CH-5 AC: numerical cdf vs MC (1e6), MC-agree (Bonferroni) over 50 points."""
    c, x = mc_case
    pts = np.quantile(x, np.linspace(0.02, 0.98, 50))
    p_hat, se = ecdf_estimate(x, pts)
    res = check_probabilities(p_hat, se, c.cdf(pts))
    assert res.ok, f"max |z| = {res.max_z:.2f} > {res.z_crit:.2f}"


def test_sf_complements_cdf(mc_case: tuple[CompoundPDT, np.ndarray]) -> None:
    c, x = mc_case
    pts = np.quantile(x, [0.05, 0.3, 0.6, 0.95])
    np.testing.assert_allclose(c.cdf(pts) + c.sf(pts), 1.0, atol=1e-9)


def test_moments_are_product_and_match_mc(mc_case: tuple[CompoundPDT, np.ndarray]) -> None:
    c, x = mc_case
    for m in (1, 2):
        assert c.moment(m) == pytest.approx(c.large.moment(m) * c.small.moment(m), rel=1e-14)
        assert check_moment(x**m, c.moment(m)).ok


@pytest.mark.parametrize("name", ["k10_loggamma", "rayleigh_loggamma"])
def test_moments_vs_quadrature_of_compound_pdf(name: str) -> None:
    """FR-CH-5 AC: E[eta^m] = E[eta_l^m] E^QC[eta_s^m] to rtol 1e-8 by quadrature of f(x)."""
    small, large = CASES[name]
    c = CompoundPDT(small, large)
    for m in (0, 1, 2):
        val, _ = integrate.quad(
            lambda t, m=m: t**m * float(c.pdf(t)),
            0.0,
            1.0,
            points=[0.3, 0.6, 0.9, 0.97],
            epsabs=0.0,
            epsrel=1e-10,
            limit=200,
        )
        ref = 1.0 if m == 0 else c.moment(m)
        assert val == pytest.approx(ref, rel=tol.RTOL_MOMENT_QUAD)


def test_pdf_is_derivative_of_cdf() -> None:
    c = CompoundPDT(*CASES["k10_loggamma"])
    h = 1e-5
    for x0 in (0.4, 0.7, 0.9):
        d = float(c.cdf(x0 + h) - c.cdf(x0 - h)) / (2 * h)
        assert d == pytest.approx(float(c.pdf(x0)), rel=1e-5)


# ---- limits: compound reduces to each factor ------------------------------------------------
def test_limit_no_large_scale_loss_equals_small_scale() -> None:
    small = SmallScale(10.0, 0.999)
    c = CompoundPDT(small, DeterministicLoss(0.0))
    x = np.linspace(0.05, 0.999, 25)
    np.testing.assert_allclose(c.cdf(x), small.cdf(x), rtol=1e-12)
    np.testing.assert_allclose(c.sf(x), small.sf(x), rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(c.pdf(x), small.pdf(x), rtol=1e-12)
    # quadrature route with a tiny-variance law converges to the same
    near = CompoundPDT(small, ShiftedLogGamma.from_moments(0.0, 1e-9, (1e-12) ** 2))
    assert near.cdf(0.8)[()] == pytest.approx(float(small.cdf(0.8)), rel=1e-6)


def test_limit_deterministic_loss_is_scaled_small_scale() -> None:
    small = SmallScale(10.0, 0.999)
    psi = 0.05
    c = CompoundPDT(small, DeterministicLoss(psi))
    x = np.linspace(0.05, np.exp(-psi) * 0.999, 20)
    np.testing.assert_allclose(c.cdf(x), small.cdf(x * np.exp(psi)), rtol=1e-12)
    assert float(c.cdf(np.exp(-psi))) == 1.0
    assert c.moment(1) == pytest.approx(np.exp(-psi) * small.moment(1), rel=1e-14)


def test_limit_concentrated_small_scale_equals_large_scale_rescaled() -> None:
    """K -> infinity: eta_s -> eta_bar, so eta ~ eta_bar * eta_l."""
    large = ShiftedLogGamma(0.0, 4.0, 0.01)
    c = CompoundPDT(SmallScale(1e8, 0.9), large)
    x = np.array([0.7, 0.8, 0.85, 0.88])
    np.testing.assert_allclose(c.cdf(x), large.cdf(x / 0.9), atol=2e-3)


def test_compound_never_exceeds_one_for_physical_laws() -> None:
    c = CompoundPDT(SmallScale(10.0, 0.999), ShiftedLogGamma(0.0, 4.0, 0.01))
    assert float(c.cdf(1.0)) == pytest.approx(1.0, abs=1e-12)
    assert float(c.sf(1.0)) == pytest.approx(0.0, abs=1e-12)


def test_v0_lognormal_compound_has_mass_above_one() -> None:
    """Non-physical baseline: eta can exceed 1 through eta_l > 1 (cdf(1) < 1)."""
    c = CompoundPDT(SmallScale(10.0, 0.999), LogNormalV0(0.02, 0.02))
    assert float(c.sf(1.0)) > 0.0


# ---- quantile, edges, errors ---------------------------------------------------------------
def test_ppf_inverts_cdf() -> None:
    c = CompoundPDT(*CASES["k10_loggamma"])
    eps = np.array([1e-4, 0.01, 0.3, 0.9])
    q = c.ppf(eps)
    np.testing.assert_allclose(c.cdf(q), eps, rtol=1e-8)
    assert np.all(np.diff(q) > 0)


def test_ppf_errors() -> None:
    c = CompoundPDT(SmallScale(10.0, 0.999), ShiftedLogGamma(0.5, 4.0, 0.01))  # eta <= e^-0.5
    with pytest.raises(ValueError, match="eps must"):
        c.ppf(0.0)
    with pytest.raises(ValueError, match="eps must"):
        c.ppf(1.0)
    assert float(c.cdf(np.exp(-0.5))) == pytest.approx(1.0, abs=1e-12)  # support edge


def test_edges_and_shapes() -> None:
    c = CompoundPDT(SmallScale(10.0, 0.999), ShiftedLogGamma(1e-4, 4.0, 0.01))
    assert float(c.cdf(0.0)) == 0.0 and float(c.sf(0.0)) == 1.0 and float(c.pdf(-1.0)) == 0.0
    assert c.cdf(np.array([[0.3, 0.6], [0.8, 0.9]])).shape == (2, 2)
    assert c.sample(np.random.default_rng(0), 7).shape == (7,)
    assert np.all(c.pdf(np.linspace(0.01, 0.99, 20)) >= 0)


def test_ppf_above_support_for_nonphysical_baseline() -> None:
    """V0 log-normal has cdf(1) < 1, so high quantiles lie outside [0, 1]."""
    c = CompoundPDT(SmallScale(10.0, 0.999), LogNormalV0(0.02, 0.02))
    assert float(c.cdf(1.0)) < 0.9999
    with pytest.raises(ValueError, match="above the support"):
        c.ppf(0.9999)
