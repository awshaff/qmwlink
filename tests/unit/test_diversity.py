"""Diversity: array outage, SC tail, Chernoff/Cramer bounds, design law, N_req. (FR-DIV-1..4)"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats

import tolerances as tol
from qmwlink.channel.large_scale import DeterministicLoss
from qmwlink.channel.small_scale import SmallScale
from qmwlink.numerics.mcagree import check_probabilities, ecdf_estimate
from qmwlink.services.diversity import (
    array_outage,
    cramer_rate,
    cramer_rate_legendre,
    design_dimension,
    design_law_holds,
    exponent_bound,
    log_mgf,
    n_req,
    sc_chernoff_bound,
    sc_exact_tail,
    sc_sample,
)
from qmwlink.thermal import n_th

NREQ_TABLE = {  # FSD FR-DIV-3 / TSD §11 [V]
    5: (20_017_832, 3_474_510, 394_473),
    30: (556_629, 96_662, 10_975),
    100: (50_243, 8_737, 993),
    300: (5_630, 983, 112),
}


def _setup(f_ghz: float) -> tuple[float, float]:
    n = float(n_th(f_ghz * 1e9, 300.0))
    es = n / (1.0 + n)
    return es, es + 0.9 * (1.0 - es)


@pytest.mark.parametrize("f_ghz", sorted(NREQ_TABLE))
def test_n_req_reference_table(f_ghz: int) -> None:
    es, eb = _setup(f_ghz)
    for k, ref in zip((0.0, 10.0, 100.0), NREQ_TABLE[f_ghz], strict=True):
        n = n_req(1e-3, k, eb, es)
        assert n == pytest.approx(ref, rel=1e-3), (f_ghz, k)
        # smallest integer: N meets the target, N-1 does not
        assert array_outage(n, k, eb, es) <= 1e-3 < array_outage(n - 1, k, eb, es)


def test_n_req_edge_cases_and_deterministic_loss() -> None:
    es, eb = _setup(300)
    assert n_req(0.99, 0.0, eb, es) == 1
    with pytest.raises(ValueError):
        n_req(1.5, 0.0, eb, es)
    with pytest.raises(ValueError, match="no N"):
        n_req(1e-3, 0.0, eb, es, n_cap=10)
    # an extra deterministic loss (eta_l < 1) never needs fewer elements
    base = n_req(1e-3, 10.0, eb, es)
    assert n_req(1e-3, 10.0, eb, es, DeterministicLoss(1e-4)) >= base


@pytest.mark.parametrize("k", [0.0, 3.0, 10.0])
def test_n1_reduces_to_single_element(k: float) -> None:
    s = SmallScale(k, 0.9)
    for x in (0.52, 0.7, 0.95):
        assert array_outage(1, k, 0.9, x) == float(s.cdf(x))
        assert sc_exact_tail(x, k, 0.9, 1) == pytest.approx(float(s.sf(x)), rel=1e-8)


def test_array_outage_decreases_with_n() -> None:
    es, eb = _setup(30)
    vals = [array_outage(n, 10.0, eb, es) for n in (10, 100, 1000, 10_000)]
    assert all(b < a for a, b in itertools.pairwise(vals))


# ---- selection combining ----------------------------------------------------------------------
def test_sc_domain_error_below_half() -> None:
    with pytest.raises(ValueError, match="x >= 1/2"):
        sc_exact_tail(0.3, 5.0, 0.95, 4)
    assert sc_exact_tail(1.0, 5.0, 0.95, 4) == 0.0
    assert 0.0 <= sc_chernoff_bound(0.3, 5.0, 0.95, 4) <= 4 / SmallScale(5.0, 0.95, 4).z
    with pytest.raises(ValueError):
        sc_chernoff_bound(1.0, 5.0, 0.95, 4)


@pytest.mark.parametrize("n", [2, 4, 16])
@pytest.mark.parametrize("k", [0.0, 5.0])
def test_chernoff_bounds_exact_tail(n: int, k: float) -> None:
    for x in np.linspace(0.5, 0.97, 8):
        ex = sc_exact_tail(float(x), k, 0.98, n)
        assert 0.0 <= ex <= 1.0 + 1e-9
        assert sc_chernoff_bound(float(x), k, 0.98, n) >= ex * (1 - 1e-9)


def test_chernoff_grid_at_300k_threshold() -> None:
    """FR-DIV-2 (a) on x in [max(eta*, 1/2), 1) at 5 GHz: bound >= exact."""
    es, _ = _setup(5)
    for x in np.linspace(max(es, 0.5), 1.0, 6, endpoint=False):
        assert sc_chernoff_bound(float(x), 10.0, 0.9999, 8) >= sc_exact_tail(
            float(x), 10.0, 0.9999, 8
        ) * (1 - 1e-9)


# ---- Cramer / design law ----------------------------------------------------------------------
def test_cramer_rate_properties() -> None:
    for k in (0.0, 1.0, 10.0, 100.0):
        assert cramer_rate(1.0, k) == pytest.approx(0.0, abs=1e-12)
        for a in (0.2, 0.5, 0.9, 1.3, 2.0):
            assert cramer_rate(a, k) >= -1e-12
            assert cramer_rate(a, k) == pytest.approx(
                cramer_rate_legendre(a, k), rel=1e-6, abs=1e-9
            )
    for a in (0.3, 0.7, 2.5):
        assert cramer_rate(a, 0.0) == pytest.approx(a - 1 - math.log(a), rel=1e-13)
    assert log_mgf(0.0, 5.0) == 0.0
    with pytest.raises(ValueError):
        cramer_rate(0.0, 1.0)


def test_exponent_bound_and_design_law() -> None:
    es, eb = _setup(100)
    a = es / eb  # eta_l = 1
    for k in (0.0, 10.0):
        for n in (50, 500, 5000):
            assert exponent_bound(n, k, a, eb) >= array_outage(n, k, eb, es)
    assert design_dimension(10, 0.0) == 10.0 and design_dimension(10, 1.0) == pytest.approx(40 / 3)
    # design law is sufficient: when it holds, the outage is <= eps
    for k in (0.0, 10.0, 100.0):
        for n in (200, 2000, 20000, 200000):
            if design_law_holds(n, k, a, 1e-3, eb):
                assert array_outage(n, k, eb, es) <= 1e-3
    with pytest.raises(ValueError):
        design_law_holds(10, 0.0, 1.2, 1e-3, eb)


# ---- Monte Carlo (FR-DIV-4) -------------------------------------------------------------------
@pytest.mark.parametrize("k", [0.0, 10.0])
def test_mrc_sum_sampling_vs_array_cdf(k: float) -> None:
    n, eb = 10**6, 0.9995
    rng = np.random.default_rng(tol.MC_SEED)
    eta = SmallScale(k, eb, n).sample_sum(rng, 200_000)
    sm = SmallScale(k, eb, n)
    pts = np.quantile(eta, np.linspace(0.02, 0.98, 15))
    p_hat, se = ecdf_estimate(eta, pts)
    assert check_probabilities(p_hat, se, sm.cdf(pts)).ok


def test_sum_vs_per_element_ks() -> None:
    """FR-DIV-4 (b): N <= 64, sum-sampling vs per-element sums (two-sample KS, alpha 1e-3)."""
    n, k, eb = 8, 5.0, 0.9
    rng = np.random.default_rng(tol.MC_SEED)
    sm = SmallScale(k, eb, n)
    a = sm.sample_sum(rng, 60_000)
    b = sm.sample_elements(rng, 60_000).sum(axis=1)
    assert stats.ks_2samp(a, b).pvalue >= 1e-3


def test_sc_exact_tail_vs_mc_and_nmax() -> None:
    n, k, eb = 4, 5.0, 0.95
    rng = np.random.default_rng(tol.MC_SEED)
    eta = sc_sample(rng, 300_000, k, eb, n)
    xs = np.linspace(0.5, 0.62, 7)  # tails large enough for a positive standard error
    tail_hat = np.array([(eta > x).mean() for x in xs])
    assert tail_hat.min() > 1e-4
    se = np.sqrt(tail_hat * (1 - tail_hat) / eta.size)
    exact = np.array([sc_exact_tail(float(x), k, eb, n) for x in xs])
    assert check_probabilities(1.0 - tail_hat, se, 1.0 - exact).ok
    with pytest.raises(ValueError, match="sample_sum"):
        sc_sample(rng, 10, k, eb, 65)
