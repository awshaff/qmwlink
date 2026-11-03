"""PIChannel, composition, apply_to_mode_b. (FR-GAU-2, FR-GAU-3, TSD §4, §10)"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qmwlink.gaussian.channels import (
    PIChannel,
    apply_to_mode_b,
    attenuator_cascade_n_eff,
)
from qmwlink.gaussian.states import is_physical, tmsv_cm

unit = st.floats(0.0, 1.0)
occ = st.floats(0.0, 1e4)


@st.composite
def physical_channels(draw: st.DrawFn) -> PIChannel:
    kind = draw(st.sampled_from(["att", "noise", "amp"]))
    if kind == "att":
        return PIChannel.thermal_attenuator(draw(unit), draw(occ))
    if kind == "noise":
        return PIChannel.additive_noise(draw(occ))
    return PIChannel.amplifier(draw(st.floats(1.0, 100.0)), draw(occ))


def test_constructors_values() -> None:
    ch = PIChannel.thermal_attenuator(0.7, 2.0)
    assert (float(ch.tau), float(ch.y)) == pytest.approx((0.7, 0.3 * 5.0))
    assert (float(PIChannel.identity().tau), float(PIChannel.identity().y)) == (1.0, 0.0)
    n = PIChannel.additive_noise(3.0)
    assert (float(n.tau), float(n.y)) == (1.0, 6.0)
    a = PIChannel.amplifier(4.0, 1.0)
    assert (float(a.tau), float(a.y)) == (4.0, 9.0)


def test_one_minus_eta_matches_eta_and_is_exact_near_one() -> None:
    a = PIChannel.thermal_attenuator(0.25, 3.0)
    b = PIChannel.thermal_attenuator(one_minus_eta=0.75, n_bar=3.0)
    assert float(a.tau) == pytest.approx(float(b.tau)) and float(a.y) == pytest.approx(float(b.y))
    tiny = PIChannel.thermal_attenuator(one_minus_eta=1e-17, n_bar=1.0)
    assert float(tiny.y) == pytest.approx(3e-17, rel=1e-14)  # eta = 1 - 1e-17 rounds to 1


def test_constructor_argument_errors() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PIChannel.thermal_attenuator()
    with pytest.raises(ValueError, match="exactly one"):
        PIChannel.thermal_attenuator(0.5, one_minus_eta=0.5)
    with pytest.raises(ValueError, match="eta"):
        PIChannel.thermal_attenuator(1.1)
    with pytest.raises(ValueError, match="one_minus_eta"):
        PIChannel.thermal_attenuator(one_minus_eta=-0.1)
    with pytest.raises(ValueError, match="n_bar"):
        PIChannel.thermal_attenuator(0.5, -1.0)
    with pytest.raises(ValueError, match="n_quanta"):
        PIChannel.additive_noise(-1.0)
    with pytest.raises(ValueError, match="gain"):
        PIChannel.amplifier(0.5)
    with pytest.raises(ValueError, match="n_bar"):
        PIChannel.amplifier(2.0, -1.0)


def test_physicality_errors() -> None:
    with pytest.raises(ValueError, match="unphysical"):
        PIChannel(0.5, 0.1)  # y < |1 - tau|
    with pytest.raises(ValueError, match="tau"):
        PIChannel(-0.1, 2.0)
    with pytest.raises(ValueError, match="finite"):
        PIChannel(np.nan, 1.0)
    PIChannel(0.5, 0.5)  # boundary is allowed


def test_matmul_semantics_and_type() -> None:
    a = PIChannel.thermal_attenuator(0.5, 1.0)
    b = PIChannel.amplifier(3.0, 0.5)
    ba = b @ a  # a first
    assert float(ba.tau) == pytest.approx(1.5)
    assert float(ba.y) == pytest.approx(3.0 * 1.5 + 4.0)
    with pytest.raises(TypeError):
        _ = a @ 3  # type: ignore[operator]


@given(physical_channels(), physical_channels(), physical_channels())
def test_composition_associative_and_physical(a: PIChannel, b: PIChannel, c: PIChannel) -> None:
    left = (c @ b) @ a
    right = c @ (b @ a)
    assert float(left.tau) == pytest.approx(float(right.tau), rel=1e-12, abs=1e-300)
    assert float(left.y) == pytest.approx(float(right.y), rel=1e-12, abs=1e-12)


@given(unit, occ, unit, occ)
def test_attenuator_cascade_closed_form(e1: float, n1: float, e2: float, n2: float) -> None:
    """TSD §4.3: cascade of attenuators equals attenuator(eta1 eta2, n_eff)."""
    if 1.0 - e1 * e2 < 1e-6:
        return
    casc = PIChannel.thermal_attenuator(e2, n2) @ PIChannel.thermal_attenuator(e1, n1)
    eta = e1 * e2
    n_eff = float(attenuator_cascade_n_eff(e1, n1, e2, n2))
    ref = PIChannel.thermal_attenuator(eta, n_eff)
    assert float(casc.tau) == pytest.approx(float(ref.tau), rel=1e-12, abs=1e-300)
    assert float(casc.y) == pytest.approx(float(ref.y), rel=1e-9, abs=1e-12)


def test_cascade_errors() -> None:
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        attenuator_cascade_n_eff(1.5, 0.0, 0.5, 0.0)
    with pytest.raises(ValueError, match="< 1"):
        attenuator_cascade_n_eff(1.0, 0.0, 1.0, 0.0)


def test_n_out_and_thermal_attenuator() -> None:
    ch = PIChannel.thermal_attenuator(0.8, 7.0)
    assert float(ch.n_out) == pytest.approx(0.2 * 7.0)  # TSD §4.4
    assert float(PIChannel.identity().n_out) == 0.0
    assert float(PIChannel.amplifier(2.0).n_out) == pytest.approx(1.0)  # (G+(G-1)-1)/2 = G-1


def test_entanglement_breaking_flag() -> None:
    assert bool(PIChannel.additive_noise(1.0).is_entanglement_breaking)  # y = 2 = 1 + tau
    assert not bool(PIChannel.additive_noise(0.4).is_entanglement_breaking)


def test_array_channel_broadcast() -> None:
    eta = np.linspace(0, 1, 5)
    ch = PIChannel.thermal_attenuator(eta, 2.0)
    assert ch.tau_arr.shape == (5,)
    assert PIChannel(eta, 1.0 + 0 * eta).y_arr.shape == (5,)


def test_apply_to_mode_b_block_form_and_physical() -> None:
    v = tmsv_cm(0.8)
    eta, nb = 0.6, 3.0
    out = apply_to_mode_b(v, PIChannel.thermal_attenuator(eta, nb))
    a, c = np.cosh(1.6), np.sinh(1.6)
    z = np.diag([1.0, -1.0])
    expected = np.block(
        [
            [a * np.eye(2), np.sqrt(eta) * c * z],
            [np.sqrt(eta) * c * z, (eta * a + (1 - eta) * (2 * nb + 1)) * np.eye(2)],
        ]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-14)
    assert bool(is_physical(out))


def test_apply_to_mode_b_batches_and_shape_error() -> None:
    eta = np.linspace(0.1, 0.9, 4)
    out = apply_to_mode_b(tmsv_cm(1.0), PIChannel.thermal_attenuator(eta, 1.0))
    assert out.shape == (4, 4, 4)
    out2 = apply_to_mode_b(tmsv_cm([0.5, 1.0]), PIChannel.identity())
    assert out2.shape == (2, 4, 4)
    np.testing.assert_allclose(out2, tmsv_cm([0.5, 1.0]))
    with pytest.raises(ValueError, match="shape"):
        apply_to_mode_b(np.eye(3), PIChannel.identity())


@given(physical_channels(), st.floats(0.0, 3.0))
def test_physical_channel_preserves_physicality(ch: PIChannel, r: float) -> None:
    out = apply_to_mode_b(tmsv_cm(r), ch)
    scale = max(1.0, float(np.max(np.abs(out))))
    assert (
        float(np.min(np.linalg.eigvalsh(out + 1j * np.kron(np.eye(2), [[0, 1], [-1, 0]]))))
        >= -1e-12 * scale
    )
