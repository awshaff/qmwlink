"""Small-scale PDT: truncated noncentral chi-square. (FR-CH-1, TSD §6.1, §10)"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import integrate, stats

import tolerances as tol
from qmwlink.channel.small_scale import SmallScale
from qmwlink.numerics.mcagree import check_moment, check_probabilities, ecdf_estimate, ks_agree

ETA_BAR = 0.999


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="K"):
        SmallScale(-1.0, 0.5)
    with pytest.raises(ValueError, match="K"):
        SmallScale(np.nan, 0.5)
    for bad in (0.0, -0.1, 1.0001):
        with pytest.raises(ValueError, match="eta_bar"):
            SmallScale(1.0, bad)
    with pytest.raises(ValueError, match="n_elements"):
        SmallScale(1.0, 0.5, 0)
    with pytest.raises(ValueError, match="n_elements"):
        SmallScale(1.0, 0.5, 2.5)
    SmallScale(1.0, 1.0)  # eta_bar = 1 allowed


@pytest.mark.parametrize(("K", "ref"), [(0.0, 0.63249), (10.0, 0.54404), (100.0, 0.51689)])
def test_normalizer_reference_values(K: float, ref: float) -> None:
    """TSD §6.1 [V]: Z_K at eta_bar = 0.999, N = 1."""
    assert SmallScale(K, ETA_BAR).z == pytest.approx(ref, abs=tol.ATOL_Z_REF)


@pytest.mark.parametrize(
    ("K", "n"), [(0.0, 1), (3.0, 1), (10.0, 1), (100.0, 1), (5.0, 7), (0.0, 50)]
)
def test_pdf_integrates_to_one(K: float, n: int) -> None:
    s = SmallScale(K, ETA_BAR, n)
    mean = float(s.moment(1))
    val, _ = integrate.quad(
        lambda x: float(s.pdf(x)), 0.0, 1.0, points=[mean], epsabs=0, epsrel=1e-12, limit=500
    )
    assert val == pytest.approx(1.0, abs=tol.ATOL_PDF_NORM)


@pytest.mark.parametrize("x", [1.0, 1.5, 10.0, 1e6])
def test_cdf_clamp(x: float) -> None:
    """FR-CH-1 AC: F(x) = 1 for x in {1, 1.5, 10}."""
    s = SmallScale(10.0, ETA_BAR)
    assert float(s.cdf(x)) == 1.0
    assert float(s.sf(x)) == 0.0
    assert float(s.pdf(x)) == 0.0 or x == 1.0


def test_cdf_below_zero_and_edges() -> None:
    s = SmallScale(2.0, 0.8)
    assert float(s.cdf(-1.0)) == 0.0 and float(s.sf(-1.0)) == 1.0 and float(s.pdf(-1.0)) == 0.0
    assert float(s.cdf(0.0)) == 0.0 and float(s.sf(0.0)) == 1.0


@pytest.mark.parametrize("eta_bar", [0.3, 0.9, 1.0])
def test_limit_k0_n1_is_truncated_exponential(eta_bar: float) -> None:
    """K = 0, N = 1: Exp(mean eta_bar) truncated to [0, 1]."""
    s = SmallScale(0.0, eta_bar)
    x = np.linspace(0.0, 1.0, 41)
    expected = -np.expm1(-x / eta_bar) / -np.expm1(-1.0 / eta_bar)
    np.testing.assert_allclose(s.cdf(x), expected, rtol=1e-12, atol=1e-15)
    assert s.z == pytest.approx(-np.expm1(-1.0 / eta_bar), rel=1e-12)


@pytest.mark.parametrize("n", [2, 5, 40])
def test_limit_k0_is_truncated_erlang(n: int) -> None:
    """K = 0, N elements: Gamma(N, scale eta_bar/N) truncated to [0, 1]."""
    s = SmallScale(0.0, 0.9, n)
    g = stats.gamma(n, scale=0.9 / n)
    x = np.linspace(0.05, 0.99, 30)
    np.testing.assert_allclose(s.cdf(x), g.cdf(x) / g.cdf(1.0), rtol=1e-11)
    np.testing.assert_allclose(s.pdf(x), g.pdf(x) / g.cdf(1.0), rtol=1e-11)


def test_limit_large_k_concentrates_at_eta_bar() -> None:
    """K -> infinity: eta_s -> eta_bar with std eta_bar * sqrt((1+2K)/(K+1)^2) ~ eta_bar sqrt(2/K).

    For eta_bar = 0.9 and K >= 1e4 the truncation at 1 is > 7 sd away, so it is negligible.
    """
    for K in (1e4, 1e6, 1e8):
        s = SmallScale(K, 0.9)
        sigma = 0.9 * np.sqrt((1.0 + 2.0 * K) / (K + 1.0) ** 2)
        assert s.moment(1) == pytest.approx(0.9, abs=1e-6 * sigma + 1e-12)
        assert np.sqrt(s.moment(2) - s.moment(1) ** 2) == pytest.approx(sigma, rel=1e-3)
        mass = float(s.cdf(0.9 + 5 * sigma) - s.cdf(0.9 - 5 * sigma))
        assert mass > 1.0 - 1e-5


def test_sf_plus_cdf_is_one_and_monotone() -> None:
    s = SmallScale(10.0, ETA_BAR)
    x = np.linspace(0.0, 1.0, 501)
    np.testing.assert_allclose(s.cdf(x) + s.sf(x), 1.0, atol=1e-12)
    assert np.all(np.diff(s.cdf(x)) >= -1e-15) and np.all(np.diff(s.sf(x)) <= 1e-15)


def test_sf_accurate_near_one() -> None:
    """S(x) for x -> 1 comes from the upper-tail difference, not 1 - F (rule 1)."""
    s = SmallScale(10.0, ETA_BAR)
    x = 1.0 - 1e-9
    direct, _ = integrate.quad(lambda t: float(s.pdf(t)), x, 1.0, epsabs=0, epsrel=1e-10)
    assert float(s.sf(x)) == pytest.approx(direct, rel=1e-5)
    assert float(s.sf(x)) > 0.0


def test_ppf_roundtrip_and_errors() -> None:
    s = SmallScale(10.0, ETA_BAR, 3)
    q = np.array([1e-6, 0.01, 0.3, 0.5, 0.99])
    np.testing.assert_allclose(s.cdf(s.ppf(q)), q, rtol=1e-9)
    assert float(s.ppf(1.0)) == 1.0 and float(s.ppf(0.0)) == 0.0
    with pytest.raises(ValueError, match="q must"):
        s.ppf(1.1)


@given(
    K=st.floats(0.0, 200.0),
    eta_bar=st.floats(0.05, 1.0),
    n=st.integers(1, 30),
)
@settings(max_examples=40, deadline=None)
def test_property_pdf_nonneg_cdf_monotone(K: float, eta_bar: float, n: int) -> None:
    s = SmallScale(K, eta_bar, n)
    x = np.linspace(0.0, 1.0, 101)
    assert np.all(s.pdf(x) >= 0.0)
    c = s.cdf(x)
    assert np.all(np.diff(c) >= -1e-12) and c[0] == 0.0 and c[-1] == 1.0
    assert 0.0 < s.z <= 1.0


def test_moments_basic() -> None:
    s = SmallScale(10.0, ETA_BAR)
    assert s.moment(0) == pytest.approx(1.0, abs=1e-10)
    assert 0.0 < s.moment(2) < s.moment(1) < 1.0
    # truncation: the scale parameter overstates the mean (FR-CH-5 note)
    assert s.moment(1) < ETA_BAR
    with pytest.raises(ValueError, match="m must"):
        s.moment(-1)


@pytest.mark.parametrize(("K", "n"), [(0.0, 1), (10.0, 1), (100.0, 1), (10.0, 20)])
def test_mc_sample_sum_vs_cdf_ks_and_moments(K: float, n: int) -> None:
    """FR-CH-1 AC: MC (1e6) vs cdf KS distance < 2e-3; MC-agree on cdf and moments."""
    s = SmallScale(K, ETA_BAR, n)
    rng = np.random.default_rng(tol.MC_SEED)
    x = s.sample_sum(rng, tol.MC_N)
    assert x.shape == (tol.MC_N,) and x.min() >= 0.0 and x.max() <= 1.0
    ok, d, _ = ks_agree(x, lambda t: s.cdf(t))
    assert d < tol.MAX_KS_D and ok
    pts = np.quantile(x, np.linspace(0.02, 0.98, 50))
    p_hat, se = ecdf_estimate(x, pts)
    assert check_probabilities(p_hat, se, s.cdf(pts)).ok
    assert check_moment(x, s.moment(1)).ok
    assert check_moment(x**2, s.moment(2)).ok


def test_sample_sum_acceptance_rate_is_z() -> None:
    s = SmallScale(100.0, ETA_BAR)  # Z ~ 0.517
    rng = np.random.default_rng(tol.MC_SEED)
    n = 400_000
    raw = rng.noncentral_chisquare(s.nu, s.nc, size=n) * s.sigma2
    rate = float(np.mean(raw <= 1.0))
    se = np.sqrt(s.z * (1 - s.z) / n)
    assert abs(rate - s.z) < 4.4 * se


@pytest.mark.parametrize("n", [1, 4, 16, 64])
def test_sample_elements_matches_sample_sum(n: int) -> None:
    """TSD §10: sample_sum vs sample_elements, two-sample KS at alpha = 1e-3 (N <= 64)."""
    s = SmallScale(5.0, 0.9, n)
    rng = np.random.default_rng(tol.MC_SEED + n)
    elems = s.sample_elements(rng, 100_000)
    assert elems.shape == (100_000, n) and np.all(elems.sum(axis=1) <= 1.0)
    total = s.sample_sum(rng, 100_000)
    assert stats.ks_2samp(elems.sum(axis=1), total).pvalue >= 1e-3


def test_sample_elements_nonzero_phases_are_irrelevant() -> None:
    s = SmallScale(5.0, 0.9, 4)
    rng = np.random.default_rng(3)
    a = s.sample_elements(rng, 100_000, phi0=[0.0, 1.0, 2.0, 3.0]).sum(axis=1)
    b = s.sample_elements(rng, 100_000).sum(axis=1)
    assert stats.ks_2samp(a, b).pvalue >= 1e-3


def test_sample_elements_rejects_large_n_naming_sample_sum() -> None:
    s = SmallScale(5.0, 0.9, 65)
    with pytest.raises(ValueError, match="sample_sum"):
        s.sample_elements(np.random.default_rng(0), 10)
    SmallScale(5.0, 0.9, 65).sample_elements(np.random.default_rng(0), 10, n_max=100)


def test_moment_raises_on_nonfinite_pdf() -> None:
    """The quadrature never integrates a nan silently (TSD §13 rule 6)."""

    class Broken:
        def pdf(self, x: float) -> float:
            return float("nan")

    s = SmallScale(10.0, ETA_BAR)
    s.__dict__["_dist"] = Broken()  # cached_property slot
    with pytest.raises(FloatingPointError, match="non-finite"):
        s.moment(1)
