"""Receivers: FR-RX-1..5 acceptance criteria. (TSD §7)"""

from __future__ import annotations

import numpy as np
import pytest

import tolerances as tol
from qmwlink.constants import K_B, H
from qmwlink.gaussian.channels import PIChannel
from qmwlink.receivers.base import (
    IdealHomodyne,
    MeasuringReceiver,
    ideal_heterodyne,
    quantum_limited_amplifier,
)
from qmwlink.receivers.electronic import (
    electronic_receiver,
    n_amp_from_tn,
    quantum_limit_tn,
)
from qmwlink.receivers.rydberg import RydbergReceiver, n_amp_from_s_e
from qmwlink.receivers.transducer import coherent_transducer
from qmwlink.thermal import n_th, s_e_bbr

F5 = 5e9
S_E_10NV = 1e-6  # 10 nV/cm/sqrt(Hz) in V/m/sqrt(Hz)


# ---- FR-RX-1 ---------------------------------------------------------------------------------
def test_rx1_below_caves_bound_raises() -> None:
    with pytest.raises(ValueError, match="n_amp"):
        MeasuringReceiver(0.49, "ssb")
    with pytest.raises(ValueError, match="n_amp"):
        MeasuringReceiver(float("nan"), "ssb")


def test_rx1_sideband_required_and_validated() -> None:
    with pytest.raises(TypeError):
        MeasuringReceiver(0.5)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="sideband"):
        MeasuringReceiver(0.5, "usb")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bandwidth"):
        MeasuringReceiver(0.5, "ssb", bandwidth_hz=-1.0)


def test_rx1_image_band_occupation() -> None:
    nth = float(n_th(F5, 300.0))
    ssb, dsb = MeasuringReceiver(1.0, "ssb"), MeasuringReceiver(1.0, "dsb")
    eta = np.array([0.0, 0.3, 1.0])
    assert np.all(ssb.n_img(eta, nth) == 0.0)
    assert dsb.n_img(0.0, nth) == pytest.approx(nth)  # AC (c): eta = 0 -> n_th
    np.testing.assert_allclose(dsb.n_img(eta, nth), (1 - eta) * nth)
    np.testing.assert_allclose(dsb.n_img(eta, nth, 2.0), (1 - eta) * nth + eta * 2.0)
    np.testing.assert_allclose(dsb.n_img(0.3, nth, one_minus_eta=0.7), 0.7 * nth)
    with pytest.raises(ValueError, match=">= 0"):
        dsb.n_img(0.5, -1.0)


def test_rx1_noise_photons_heterodyne_identity() -> None:
    """``ssb`` with ``N_amp = 1/2`` gives the textbook ``(1-eta) n_th + 1``."""
    nth = float(n_th(F5, 300.0))
    het = ideal_heterodyne(1e6)
    eta = np.linspace(0.0, 1.0, 7)
    np.testing.assert_allclose(het.noise_photons(eta, nth), (1 - eta) * nth + 1.0, rtol=1e-14)
    assert het.bandwidth_hz == 1e6


def test_rx2_ideal_receivers() -> None:
    assert ideal_heterodyne().n_amp == 0.5 and ideal_heterodyne().sideband == "ssb"
    assert IdealHomodyne().bandwidth_hz is None
    amp = quantum_limited_amplifier(4.0)
    assert float(amp.tau_arr) == 4.0 and float(amp.y_arr) == 3.0


# ---- FR-RX-3 ---------------------------------------------------------------------------------
def test_rx3_rj_unit() -> None:
    assert n_amp_from_tn(H * F5 / K_B, F5, "rj") == pytest.approx(1.0, rel=1e-12)


def test_rx3_conventions_at_planck_limit() -> None:
    t = H * F5 / (K_B * np.log(3.0))
    assert t == pytest.approx(0.21842, abs=5e-6)
    got = [n_amp_from_tn(t, F5, c) for c in ("rj", "planck", "callen_welton")]  # type: ignore[arg-type]
    np.testing.assert_allclose(got, [0.9102, 0.5000, 1.0000], atol=tol.ATOL_TN_REF)


def test_rx3_below_quantum_limit_and_validation() -> None:
    with pytest.raises(ValueError, match="quantum limit"):
        n_amp_from_tn(0.5 * H * F5 / K_B * 0.9, F5, "rj")
    with pytest.raises(ValueError, match="quantum limit"):
        n_amp_from_tn(0.0, F5, "planck")
    assert n_amp_from_tn(0.0, F5, "callen_welton") == 0.5  # T_n = 0 is the CW quantum limit
    with pytest.raises(ValueError, match="tn_convention"):
        n_amp_from_tn(100.0, F5, "none")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="t_n_k"):
        n_amp_from_tn(-1.0, F5, "rj")
    with pytest.raises(ValueError, match="f_hz"):
        n_amp_from_tn(100.0, 0.0, "rj")


def test_rx3_quantum_limit_tn_and_receiver() -> None:
    for conv in ("rj", "planck", "callen_welton"):
        t = quantum_limit_tn(F5, conv)  # type: ignore[arg-type]
        assert n_amp_from_tn(t, F5, conv) == pytest.approx(0.5, abs=1e-9)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tn_convention"):
        quantum_limit_tn(F5, "x")  # type: ignore[arg-type]
    rx = electronic_receiver(50.0, F5, tn_convention="rj", bandwidth_hz=1e9)
    assert rx.sideband == "ssb" and rx.n_amp == pytest.approx(K_B * 50.0 / (H * F5))
    with pytest.raises(TypeError):
        electronic_receiver(50.0, F5)  # type: ignore[call-arg]


def test_rx3_high_tn_conventions_agree_to_first_order() -> None:
    """RJ and Callen-Welton agree to O(x); Planck is lower by 1/2 (TSD §7.3)."""
    t = 1e4  # hf/kT = 2.4e-5
    rj, pl, cw = (n_amp_from_tn(t, F5, c) for c in ("rj", "planck", "callen_welton"))  # type: ignore[arg-type]
    assert cw - rj == pytest.approx(0.0, abs=1e-4 * rj)
    assert rj - pl == pytest.approx(0.5, abs=1e-3)


# ---- FR-RX-4 ---------------------------------------------------------------------------------
def test_rx4_references() -> None:
    rx = RydbergReceiver.from_sensitivity(S_E_10NV, F5, field_convention="rms")
    assert rx.n_amp == pytest.approx(3.438e5, rel=tol.RTOL_N_AMP_REF)
    assert rx.t_n_k == pytest.approx(8.25e4, rel=tol.RTOL_T_N_REF)
    assert rx.sideband == "dsb"  # default
    pk = RydbergReceiver.from_sensitivity(S_E_10NV, F5, field_convention="peak")
    assert pk.n_amp == pytest.approx(1.719e5, rel=tol.RTOL_N_AMP_REF)
    assert pk.n_amp == pytest.approx(rx.n_amp / 2.0, rel=1e-12)


def test_rx4_field_convention_required() -> None:
    with pytest.raises(TypeError):
        RydbergReceiver.from_sensitivity(S_E_10NV, F5)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="field_convention"):
        RydbergReceiver.from_sensitivity(S_E_10NV, F5, field_convention="avg")  # type: ignore[arg-type]


def test_rx4_sub_quantum_limit_raises_never_clamps() -> None:
    with pytest.raises(ValueError, match="quantum limit"):
        RydbergReceiver.from_sensitivity(1e-9, F5, field_convention="rms")
    with pytest.raises(ValueError, match="s_e"):
        n_amp_from_s_e(-1.0, F5)
    with pytest.raises(ValueError, match="f_hz"):
        n_amp_from_s_e(1e-6, 0.0)


def test_rx4_includes_bbr_subtracts_in_power() -> None:
    s_bbr = float(s_e_bbr(F5, 300.0))
    meas = np.hypot(S_E_10NV, s_bbr)
    a = RydbergReceiver.from_sensitivity(meas, F5, field_convention="rms", includes_bbr=True)
    b = RydbergReceiver.from_sensitivity(S_E_10NV, F5, field_convention="rms")
    assert a.n_amp == pytest.approx(b.n_amp, rel=1e-12)
    with pytest.raises(ValueError, match="BBR floor"):
        RydbergReceiver.from_sensitivity(0.5 * s_bbr, F5, field_convention="rms", includes_bbr=True)


def test_rx4_linear_range_flag_peak_vs_peak() -> None:
    """AC (e): E_rx,rms < E_1dB,pk < E_rx,pk gives linear_regime = False."""
    a_r, p = 2.0, 1e-9
    probe = RydbergReceiver.from_n_amp(1e5, F5)
    e_pk = probe.e_rx_pk(p, a_r)
    e_rms = e_pk / np.sqrt(2.0)
    e_1db = 0.5 * (e_rms + e_pk)
    assert e_rms < e_1db < e_pk
    rx = RydbergReceiver.from_n_amp(1e5, F5, e_1db_pk_v_per_m=e_1db)
    assert rx.linear_regime(p, a_r) is False
    assert rx.n_amp == 1e5  # numbers unchanged
    ok = RydbergReceiver.from_n_amp(1e5, F5, e_1db_pk_v_per_m=1.1 * e_pk)
    assert ok.linear_regime(p, a_r) is True
    assert RydbergReceiver.from_n_amp(1e5, F5).linear_regime(p, a_r) is True  # no limit given
    with pytest.raises(ValueError, match="a_r_m2"):
        probe.e_rx_pk(p, 0.0)


def test_rx4_aperture_invariance() -> None:
    """AC (f): N_amp is identical for any G_enh / A_r; the mapping takes no aperture argument."""
    import inspect

    assert list(inspect.signature(n_amp_from_s_e).parameters) == ["s_e_rms", "f_hz"]
    sig = inspect.signature(RydbergReceiver.from_sensitivity).parameters
    assert not {"a_r_m2", "g_enh", "a_r"} & set(sig)
    vals = {
        RydbergReceiver.from_sensitivity(S_E_10NV, F5, field_convention="rms").n_amp
        for _ in (1, 10, 100)
    }
    assert len(vals) == 1


def test_rx4_alternative_constructors_and_roundtrip() -> None:
    rx = RydbergReceiver.from_sensitivity(S_E_10NV, F5, field_convention="rms")
    assert rx.s_e_rms == pytest.approx(S_E_10NV, rel=1e-12)
    via_t = RydbergReceiver.from_noise_temperature(rx.t_n_k, F5)
    assert via_t.n_amp == pytest.approx(rx.n_amp, rel=1e-12)
    with pytest.raises(NotImplementedError, match="M7"):
        RydbergReceiver.from_physics()


# ---- FR-RX-5 ---------------------------------------------------------------------------------
def test_rx5_transducer_channel() -> None:
    ch = coherent_transducer(0.6, 0.01, 0.01)
    assert isinstance(ch, PIChannel)
    assert float(ch.tau_arr) == 0.6
    assert float(ch.y_arr) == pytest.approx(0.4 * 1.02 + 0.02, rel=1e-14)
    with pytest.raises(ValueError, match="eta_t"):
        coherent_transducer(1.2)
    with pytest.raises(ValueError, match=">= 0"):
        coherent_transducer(0.5, -0.1)
