"""FR-SVC-E2: outage limiting cases, RQ1 identity and MC agreement. (TSD §8.1, §10)"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, special, stats

import tolerances as tol
from qmwlink.channel.compound import CompoundPDT
from qmwlink.channel.large_scale import DeterministicLoss, LogNormalV0, ShiftedLogGamma
from qmwlink.channel.small_scale import SmallScale
from qmwlink.gaussian.channels import PIChannel
from qmwlink.numerics.mcagree import check_moment, check_probabilities, ecdf_estimate
from qmwlink.services.entanglement import EntanglementService
from qmwlink.thermal import n_th

NTH = float(n_th(5e9, 300.0))
ID = PIChannel.identity()
ES = (NTH + 0.0) / (NTH + 1.0)  # eta* (identity front end)


def _service(small: SmallScale, large: object) -> EntanglementService:
    return EntanglementService(CompoundPDT(small, large), NTH, ID, 1.0)  # type: ignore[arg-type]


def _fs_k0(x: float, eta_bar: float) -> float:
    """Clamped truncated-exponential cdf (K = 0, N = 1)."""
    xc = min(max(x, 0.0), 1.0)
    return float(-np.expm1(-xc / eta_bar) / -np.expm1(-1.0 / eta_bar))


def _quad_p_out_k0(pdf, lo: float, hi: float, eta_bar: float, pts: list[float]) -> float:  # type: ignore[no-untyped-def]
    f = lambda p: _fs_k0(ES * np.exp(p), eta_bar) * pdf(p)  # noqa: E731
    return float(integrate.quad(f, lo, hi, points=pts, epsabs=0, epsrel=1e-12, limit=800)[0])


# ---- Case 1: K -> infinity --------------------------------------------------------------------
def test_case1_k_to_infinity_loggamma() -> None:
    """P_out -> Q(k, [ln(eta_bar_s/eta*) - Psi0]/theta) (arg > 0), else 1. [D][V]"""
    eb, psi0, mu, sd = 0.9999, 1e-4, 1e-3, 1e-3
    law = ShiftedLogGamma.from_moments(psi0, mu, sd**2)
    assert isinstance(law, ShiftedLogGamma)
    s = _service(SmallScale(1e9, eb), law)
    arg = (np.log(eb / ES) - psi0) / law.theta
    assert arg > 0
    assert s.p_out() == pytest.approx(float(special.gammaincc(law.k, arg)), abs=2e-3)
    # non-positive argument -> outage 1 (eta_l eta_bar_s <= eta* for every realization)
    s2 = _service(SmallScale(1e9, 0.9990), ShiftedLogGamma.from_moments(1e-4, mu, sd**2))
    assert s2.p_out() == pytest.approx(1.0, abs=1e-9)


def test_case1_k_to_infinity_v0_lognormal() -> None:
    eb, pbar, sg = 0.9999, 1e-3, 1e-3
    s = _service(SmallScale(1e9, eb), LogNormalV0(pbar, sg))
    ref = float(stats.norm.sf(np.log(eb / ES), pbar, sg))  # P(Psi >= ln(eb/eta*))
    assert s.p_out() == pytest.approx(ref, abs=2e-3)


# ---- Case 2: K = 0 ----------------------------------------------------------------------------
def test_case2_k0_loggamma_clamp_active() -> None:
    """Quadrature of the clamped truncated-exponential cdf; the region eta_l < eta* is active."""
    eb, psi0 = 0.9999, 1e-4
    law = ShiftedLogGamma.from_moments(psi0, 2e-3, (1e-3) ** 2)
    assert isinstance(law, ShiftedLogGamma)
    ref = _quad_p_out_k0(
        law.psi_pdf,
        psi0,
        psi0 + 40 * law.sigma_g + law.mu_g,
        eb,
        [*law.psi_breakpoints(), -np.log(ES)],
    )
    mass_clamped = float(law.cdf(ES))  # P(eta_l <= eta*): clamp region carries real weight
    assert 0.05 < mass_clamped < 0.95
    p = _service(SmallScale(0.0, eb), law).p_out()
    assert p == pytest.approx(ref, rel=tol.RTOL_P_OUT_QUAD)


def test_case2_k0_v0_lognormal_clamp_active() -> None:
    eb, pbar, sg = 0.9999, 1e-3, 1e-3
    v0 = LogNormalV0(pbar, sg)
    ref = _quad_p_out_k0(
        lambda p: float(stats.norm.pdf(p, pbar, sg)),
        pbar - 12 * sg,
        pbar + 12 * sg,
        eb,
        [pbar + j * sg for j in (-6, -3, -1, 0, 1, 3, 6)] + [-np.log(ES)],
    )
    assert float(v0.cdf(ES)) > 0.05  # clamp region active
    assert _service(SmallScale(0.0, eb), v0).p_out() == pytest.approx(ref, rel=tol.RTOL_P_OUT_QUAD)


# ---- Case 3: sigma -> 0 -----------------------------------------------------------------------
@pytest.mark.parametrize("psi", [5e-4, 1e-3, 2e-3])
def test_case3_no_shadowing_k0(psi: float) -> None:
    """Deterministic loss: P_out = F_s(min(eta* e^Psi, 1)) with the clamp; psi = 2e-3 clamps."""
    eb = 0.9999
    got = _service(SmallScale(0.0, eb), DeterministicLoss(psi)).p_out()
    assert got == pytest.approx(_fs_k0(ES * np.exp(psi), eb), rel=1e-12)


def test_case3_loggamma_and_v0_collapse_to_same_point_mass() -> None:
    """Narrow laws reproduce the deterministic result (mu_G + Psi0 = Psi_bar)."""
    eb, psi = 0.9999, 1e-3
    ref = _service(SmallScale(0.0, eb), DeterministicLoss(psi)).p_out()
    lg = ShiftedLogGamma.from_moments(0.0, psi, (1e-6) ** 2)
    v0 = LogNormalV0(psi, 1e-6)
    for law in (lg, v0):
        assert _service(SmallScale(0.0, eb), law).p_out() == pytest.approx(ref, rel=1e-4)


# ---- Case 4: deterministic -> step at eta* ------------------------------------------------------
def test_case4_step_function() -> None:
    eb = 0.99999
    above = _service(SmallScale(1e8, eb), DeterministicLoss(1e-5)).p_out()  # eta ~ 0.99998 > eta*
    below = _service(SmallScale(1e8, eb), DeterministicLoss(2e-3)).p_out()  # eta ~ 0.9980 < eta*
    assert above == pytest.approx(0.0, abs=1e-6) and below == pytest.approx(1.0, abs=1e-12)


# ---- RQ1 ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("k", "eb", "n"), [(0.0, 0.9999, 1), (10.0, 0.99999, 1), (100.0, 0.999995, 3)]
)
def test_rq1_identity_deterministic_loss(k: float, eb: float, n: int) -> None:
    """P_out / P_out^surr = 1/Z for a deterministic loss with eta*/eta_l < 1, rtol 1e-12."""
    small = SmallScale(k, eb, n)
    psi = 1e-5
    assert ES / np.exp(-psi) < 1.0
    te = _service(small, DeterministicLoss(psi)).truncation_effect()
    assert te.ratio == pytest.approx(1.0 / small.z, rel=tol.RTOL_RQ1_IDENTITY)
    assert te.removed_mass == pytest.approx(1.0 - small.z, abs=1e-15)
    assert te.delta == pytest.approx(te.p_out - te.p_out_surr, abs=1e-15)
    assert te.physical_surrogate is False and te.p_out > te.p_out_surr


def test_rq1_stochastic_surrogate_is_smaller() -> None:
    law = ShiftedLogGamma.from_moments(1e-5, 1.5e-4, (0.8e-4) ** 2)
    te = _service(SmallScale(10.0, 0.99999), law).truncation_effect()
    assert 0.0 < te.p_out_surr < te.p_out <= 1.0 and te.ratio > 1.0


# ---- MC agreement ------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("k", [0.0, 10.0])
def test_mc_agrees_with_quadrature(k: float) -> None:
    law = ShiftedLogGamma.from_moments(1e-5, 1.5e-4, (0.8e-4) ** 2)
    svc = _service(SmallScale(k, 0.99999), law)
    es = svc.eta_star
    rng = np.random.default_rng(tol.MC_SEED)
    eta = np.concatenate([svc.compound.sample(rng, 10**6) for _ in range(2)])
    # outage at a spread of points around eta* (Bonferroni, FSD §4.3)
    pts = es + np.linspace(-3e-4, 3e-4, 25)
    p_hat, se = ecdf_estimate(eta, pts)
    res = check_probabilities(p_hat, se, svc.compound.cdf(pts))
    assert res.ok, res
    # entanglement metrics as sample means of g(eta) 1{eta > eta*}
    ent = eta > es
    safe = np.where(ent, eta, 1.0)
    metrics = svc.metrics()
    for expected, g in (
        (metrics.e_log_neg, lambda e: np.maximum(0.0, -np.log2(svc.nu_minus(e)))),
        (metrics.e_nu_minus, svc.nu_minus),
        (metrics.e_fidelity_opt, lambda e: 1.0 / (1.0 + svc.nu_minus(e))),
        (metrics.e_fidelity_bk, svc.fidelity_bk_at),
    ):
        vals = np.where(ent, g(safe), 0.0)
        res = check_moment(vals, expected)
        assert res.ok, (expected, res)
