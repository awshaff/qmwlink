"""Entanglement service: FR-SVC-E1 thresholds and E2 metrics. (TSD §8.1)"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

import tolerances as tol
from qmwlink.channel.compound import CompoundPDT
from qmwlink.channel.large_scale import DeterministicLoss, ShiftedLogGamma
from qmwlink.channel.small_scale import SmallScale
from qmwlink.gaussian.channels import PIChannel
from qmwlink.gaussian.entanglement import tmst_eta_star
from qmwlink.receivers.transducer import coherent_transducer
from qmwlink.services.entanglement import (
    EntanglementService,
    chain_channel,
    eta_star_numeric,
    eta_star_rx,
    has_window,
    window_rx,
)
from qmwlink.thermal import eta_star, n_th

NTH = float(n_th(5e9, 300.0))


def _root_nu_one(fe: PIChannel, r: float = 1.0) -> float:
    """Direct root of nu~_-(eta) = 1 using the service's own observable."""
    svc = EntanglementService(CompoundPDT(SmallScale(0.0, 0.9), DeterministicLoss(0.0)), NTH, fe, r)
    return float(
        brentq(lambda om: float(svc.nu_minus(1.0 - om)) - 1.0, 0.0, 1.0, xtol=1e-15, rtol=1e-15)
    )


# ---- FR-SVC-E1 -------------------------------------------------------------------------------
def test_e1_identity_gives_eta_star() -> None:
    assert float(eta_star_rx(NTH, PIChannel.identity())) == pytest.approx(
        float(eta_star(5e9, 300.0)), rel=1e-14
    )


def test_e1_additive_noise_closed_form_and_references() -> None:
    assert float(eta_star_rx(NTH, PIChannel.additive_noise(0.0))) == pytest.approx(
        float(eta_star(5e9, 300.0)), rel=1e-14
    )
    half = float(eta_star_rx(NTH, PIChannel.additive_noise(0.5)))
    assert half == pytest.approx(0.99960, abs=tol.ATOL_ETA_STAR_RX_REF)
    assert half == pytest.approx((NTH + 0.5) / (NTH + 1.0), rel=1e-14)


@pytest.mark.parametrize("n_add", [1.0, 1.2])
def test_e1_no_window_for_n_add_at_least_one(n_add: float) -> None:
    fe = PIChannel.additive_noise(n_add)
    assert not bool(has_window(NTH, fe))
    assert float(eta_star_rx(NTH, fe)) >= 1.0 - 1e-15
    svc = EntanglementService(
        CompoundPDT(SmallScale(10.0, 0.99), DeterministicLoss(0.0)), NTH, fe, 1.0
    )
    assert svc.p_out() == 1.0 and svc.metrics().log_neg == 0.0
    assert svc.truncation_effect().ratio == 1.0


@pytest.mark.parametrize("n_add", [0.0, 0.2, 0.5, 0.9])
def test_e1_closed_form_vs_root_additive(n_add: float) -> None:
    fe = PIChannel.additive_noise(n_add)
    closed = float(eta_star_rx(NTH, fe))
    # closed form vs direct root of nu~_-(eta) = 1, absolute 1e-12 in eta
    assert closed == pytest.approx(1.0 - _root_nu_one(fe), abs=tol.RTOL_ETA_STAR_ROOT)
    assert eta_star_numeric(NTH, fe, 1.0) == pytest.approx(closed, abs=tol.RTOL_ETA_STAR_ROOT)


def test_e1_transducer_reference_and_root() -> None:
    fe = coherent_transducer(0.6, 0.01, 0.01)
    es = float(eta_star_rx(NTH, fe))
    assert es == pytest.approx(0.999219, abs=tol.ATOL_ETA_STAR_RX_REF)
    assert es == pytest.approx(1.0 - _root_nu_one(fe), abs=1e-12)
    assert eta_star_numeric(NTH, fe, 0.3) == pytest.approx(es, abs=1e-12)  # r-independent


def test_e1_window_formula_and_edge_cases() -> None:
    fe = coherent_transducer(0.6, 0.01, 0.01)
    assert float(window_rx(NTH, fe)) == pytest.approx(1.0 - float(eta_star_rx(NTH, fe)), rel=1e-14)
    dead = PIChannel(0.0, 1.0)  # tau_r = 0: no window
    assert float(window_rx(NTH, dead)) == -np.inf and not bool(has_window(NTH, dead))
    with pytest.raises(ValueError, match="n_th"):
        window_rx(-1.0, fe)
    v = np.asarray(eta_star_rx(NTH, PIChannel.additive_noise(np.array([0.0, 0.5]))))
    assert v.shape == (2,)


def test_e1_v0_combining_network_is_attenuator_special_case() -> None:
    """A warm attenuator front end (eta_c, n_c): threshold from §4.3 closed form."""
    eta_c, n_c = 0.8, 0.0
    fe = PIChannel.thermal_attenuator(eta_c, n_c)
    ref = ((2 * NTH + 1) * eta_c + (1 - eta_c) - 1.0) / (2 * eta_c * (NTH + 1.0))
    assert float(eta_star_rx(NTH, fe)) == pytest.approx(ref, rel=1e-14)
    assert eta_star_numeric(NTH, fe, 1.0) == pytest.approx(ref, abs=1e-12)


def test_e1_tmst_numeric_identity_matches_gaussian_module() -> None:
    r, n_jpa = 1.0, 0.3
    got = eta_star_numeric(NTH, PIChannel.identity(), r, n_jpa)
    assert got == pytest.approx(tmst_eta_star(NTH, r, n_jpa), abs=1e-13)
    assert got > float(eta_star(5e9, 300.0))  # JPA noise shrinks the window


def test_e1_numeric_edge_cases() -> None:
    assert np.isnan(eta_star_numeric(NTH, PIChannel.additive_noise(1.5), 1.0))
    assert eta_star_numeric(0.0, PIChannel.identity(), 1.0) == 0.0  # no thermal noise
    with pytest.raises(ValueError, match="scalar"):
        eta_star_numeric(NTH, PIChannel.additive_noise(np.array([0.1, 0.2])), 1.0)


def test_chain_channel_matches_composition() -> None:
    fe = coherent_transducer(0.6, 0.01, 0.01)
    eta = np.array([0.2, 0.9])
    direct = fe @ PIChannel.thermal_attenuator(eta, NTH)
    ch = chain_channel(eta, NTH, fe)
    np.testing.assert_allclose(ch.tau_arr, direct.tau_arr, rtol=1e-14)
    np.testing.assert_allclose(ch.y_arr, direct.y_arr, rtol=1e-14)


# ---- service construction / observables ---------------------------------------------------------
def _svc(
    front: PIChannel | None = None, *, k: float = 10.0, det: bool = False
) -> EntanglementService:
    large = (
        DeterministicLoss(1e-4)
        if det
        else ShiftedLogGamma.from_moments(1e-5, 1.5e-4, (0.8e-4) ** 2)
    )
    return EntanglementService(
        CompoundPDT(SmallScale(k, 0.99999), large), NTH, front or PIChannel.identity(), 1.0
    )


def test_service_validation() -> None:
    c = CompoundPDT(SmallScale(1.0, 0.9), DeterministicLoss(0.0))
    with pytest.raises(ValueError, match="r must"):
        EntanglementService(c, NTH, PIChannel.identity(), 0.0)
    with pytest.raises(ValueError, match="n_jpa"):
        EntanglementService(c, NTH, PIChannel.identity(), 1.0, n_jpa=-1.0)
    with pytest.raises(ValueError, match="scalar"):
        EntanglementService(c, NTH, PIChannel.additive_noise(np.array([0.1, 0.2])), 1.0)


def test_nu_minus_below_one_iff_above_threshold() -> None:
    s = _svc()
    es = s.eta_star
    assert float(s.nu_minus(es + 1e-6)) < 1.0 < float(s.nu_minus(es - 1e-6))
    assert float(s.nu_minus(1.0)) < float(s.nu_minus(0.9995)) < 1.0


def test_metric_ordering_and_bk_below_opt() -> None:
    m = _svc().metrics()
    assert 0.0 < m.fidelity_bk_cond <= m.fidelity_opt_cond  # F_opt >= F_BK pointwise
    assert 0.0 < m.p_out < 1.0 and m.log_neg > 0.0
    assert m.one_minus_p_out == pytest.approx(1.0 - m.p_out, abs=1e-12)
    # FR-SVC-E2 (g): E[F_opt | .] > 1/2 and E[nu~_- | .] < 1
    assert m.fidelity_opt_cond > 0.5 and m.nu_minus_cond < 1.0
    assert m.log_neg_cond == pytest.approx(m.log_neg / m.one_minus_p_out, rel=1e-12)
    assert m.reason == ""


def test_tmst_service_runs_and_uses_numeric_threshold() -> None:
    c = CompoundPDT(SmallScale(10.0, 0.99999), DeterministicLoss(1e-5))
    s = EntanglementService(c, NTH, PIChannel.identity(), 1.0, n_jpa=0.3)
    assert s.eta_star == pytest.approx(tmst_eta_star(NTH, 1.0, 0.3), abs=1e-13)
    assert 0.0 < s.p_out() < 1.0
    assert s.metrics().log_neg > 0.0
    dead = EntanglementService(c, NTH, PIChannel.additive_noise(1.5), 1.0, n_jpa=0.3)
    assert dead.eta_star == 1.0 and dead.p_out() == 1.0


def test_p_out_zero_when_all_entangled() -> None:
    c = CompoundPDT(SmallScale(10.0, 0.9), DeterministicLoss(0.0))
    s = EntanglementService(c, 0.0, PIChannel.identity(), 1.0)  # n_th = 0: eta* = 0
    assert s.eta_star == 0.0 and s.p_out() == 0.0
    assert s.truncation_effect().delta == 0.0


def test_conditional_averages_deterministic_channel() -> None:
    """FR-SVC-E2 (f), near-deterministic: matches the single-realization values
    to the Jensen gap of the residual scatter."""
    # K large: the small-scale scatter is tiny, so eta is nearly deterministic and > eta* = 0.9992
    c = CompoundPDT(SmallScale(1e9, 0.99999999), DeterministicLoss(1e-6))
    s = EntanglementService(c, NTH, PIChannel.identity(), 1.0)
    m = s.metrics()
    eta = c.moment(1)
    assert m.one_minus_p_out == pytest.approx(1.0, abs=1e-9)
    assert m.nu_minus_cond == pytest.approx(
        float(s.nu_minus(eta)), rel=5e-3
    )  # Jensen gap of the residual scatter
    assert m.fidelity_bk_cond == pytest.approx(float(s.fidelity_bk_at(eta)), rel=5e-3)


def test_conditional_nan_when_never_entangled() -> None:
    """FR-SVC-E2 (h): ``1 - P_out < 1e-12`` -> NaN with a reason, never 0."""
    c = CompoundPDT(SmallScale(10.0, 0.9), DeterministicLoss(1e-5))
    dead = EntanglementService(c, NTH, PIChannel.additive_noise(1.5), 1.0)
    m = dead.metrics()
    assert m.reason == "no entangled realizations"
    assert np.isnan([m.nu_minus_cond, m.fidelity_opt_cond, m.fidelity_bk_cond]).all()
    assert m.p_out == 1.0 and m.one_minus_p_out == 0.0 and m.log_neg == 0.0
    # tiny survival (eta well below eta*): also NaN
    far = EntanglementService(
        CompoundPDT(SmallScale(0.0, 1.0), DeterministicLoss(5.0)), NTH, PIChannel.identity(), 1.0
    )
    assert np.isnan(far.metrics().nu_minus_cond)
