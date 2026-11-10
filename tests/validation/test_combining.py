"""Combining-network threshold, V0 eq. (div_netthreshold). (FR-GAU-2c, TSD §4.3)"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

import tolerances as tol
from qmwlink.gaussian.channels import PIChannel, attenuator_cascade_n_eff
from qmwlink.gaussian.entanglement import tmsv_is_entangled
from qmwlink.thermal import eta_star, n_th


def v0_threshold(n_th_val: float, eta_cmb: float, n_cmb: float) -> float:
    """V0 div_netthreshold: [n_th + (1 - eta_cmb) n_cmb / eta_cmb] / (1 + n_th). (TSD §4.3)"""
    return (n_th_val + (1.0 - eta_cmb) * n_cmb / eta_cmb) / (1.0 + n_th_val)


def direct_root(n_th_val: float, eta_cmb: float, n_cmb: float) -> float:
    """Root of y - (1 + tau) = 0 for [cmb] o [attenuator(eta, n_th)], found without the formula."""
    cmb = PIChannel.thermal_attenuator(eta_cmb, n_cmb)

    def h(omega: float) -> float:
        link = PIChannel.thermal_attenuator(one_minus_eta=omega, n_bar=n_th_val)
        net = cmb @ link
        return float(net.y_arr - 1.0 - net.tau_arr)

    return 1.0 - float(brentq(h, 0.0, 1.0, xtol=1e-15))


@pytest.mark.parametrize("f_ghz", [5.0, 30.0, 100.0, 300.0])
@pytest.mark.parametrize(("eta_cmb", "n_cmb"), [(0.9, 0.01), (0.5, 0.2), (0.99, 1.0)])
def test_network_threshold_matches_v0_formula(f_ghz: float, eta_cmb: float, n_cmb: float) -> None:
    nth = float(n_th(f_ghz * 1e9, 300.0))
    formula = v0_threshold(nth, eta_cmb, n_cmb)
    assert formula < 1.0
    assert direct_root(nth, eta_cmb, n_cmb) == pytest.approx(
        formula, abs=tol.ATOL_NETWORK_THRESHOLD
    )


@pytest.mark.parametrize("eta_cmb", [0.3, 0.9, 1.0])
def test_cold_combiner_reduces_to_eta_star_exactly(eta_cmb: float) -> None:
    nth = float(n_th(5e9, 300.0))
    assert v0_threshold(nth, eta_cmb, 0.0) == float(eta_star(5e9, 300.0))  # exact, no tolerance
    assert direct_root(nth, eta_cmb, 0.0) == pytest.approx(
        float(eta_star(5e9, 300.0)), abs=tol.ATOL_NETWORK_THRESHOLD
    )


def test_network_threshold_via_cascade_closed_form() -> None:
    """§4.3: the network is an attenuator (eta eta_cmb, n_eff); check it at the V0 threshold."""
    nth, eta_cmb, n_cmb = 62.011, 0.8, 0.05
    eta = v0_threshold(nth, eta_cmb, n_cmb)
    n_eff = float(attenuator_cascade_n_eff(eta, nth, eta_cmb, n_cmb))
    assert eta * eta_cmb == pytest.approx(n_eff / (1.0 + n_eff), rel=1e-12)
    just_above = PIChannel.thermal_attenuator(eta_cmb, n_cmb) @ PIChannel.thermal_attenuator(
        min(1.0, eta + 1e-6), nth
    )
    just_below = PIChannel.thermal_attenuator(eta_cmb, n_cmb) @ PIChannel.thermal_attenuator(
        eta - 1e-6, nth
    )
    assert bool(tmsv_is_entangled(just_above)) and not bool(tmsv_is_entangled(just_below))
    assert np.isfinite(n_eff)
