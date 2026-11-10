"""T-LIN-1..12 (TSD §14.11): weak-probe vs full Lindblad, analytic limits, readout units."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy.signal import find_peaks

import tolerances as tol
from qmwlink.constants import ETA0
from qmwlink.receivers.rydberg import RydbergReceiver
from qmwlink.receivers.rydberg_physics import lindblad, readout, weak_probe
from qmwlink.receivers.rydberg_physics.doppler import (
    DopplerSettings,
    doppler_average,
    sigma_v,
    velocity_grid,
)
from qmwlink.receivers.rydberg_physics.model import (
    GAMMA2_RB87_PER_S,
    SPECIES_MASS_KG,
    LindbladRARModel,
    RARParams,
    evaluate_sweep,
    with_overrides,
)
from qmwlink.receivers.rydberg_physics.readout import HBAR
from qmwlink.receivers.rydberg_physics.scheme import GAMMA2_RB87_D2, Decay, Drive, LadderScheme

TP = 2.0 * np.pi
K_P, K_C = TP / 780.241e-9, TP / 480e-9
RYD = TP * 1e-3  # Rydberg decay 2 pi x 1 kHz (placeholder scale; unspecified in TSD §14.11)
GT = TP * 0.1  # transit rate of the TSD §14.11 cases
COARSE = DopplerSettings(n_v_start=513, rtol=1e-3)


def scheme4(gamma_t: float = GT) -> LadderScheme:
    return LadderScheme(4, gamma3=RYD, gamma4=RYD, gamma_t=gamma_t)


def full_rho21(scheme: LadderScheme, drive: Drive, v: np.ndarray) -> np.ndarray:
    lv = lindblad.build(scheme, drive)
    return lindblad.rho21(lv, lindblad.steady_state(lv, v))


def params(**over: float) -> RARParams:
    """TSD §14.11 optical case with placeholder SI inputs (weak probe: Omega_p/2pi = 0.084 MHz)."""
    d12 = readout.cycling_dipole_cm(GAMMA2_RB87_PER_S, K_P)
    d34 = 1.0e-26
    base = RARParams(
        mass_kg=SPECIES_MASS_KG["Rb87"], t_vapor_k=300.0, n0_per_m3=1e16, cell_length_m=0.05,
        probe_wavelength_m=780.241e-9, probe_power_w=1e-8, probe_waist_m=1e-3, d12_cm=d12,
        coupling_wavelength_m=480e-9, f_mw_hz=5e9, d34_cm=d34,
        e_lo_pk_v_per_m=HBAR * TP * 20e6 / d34, gamma_t_per_s=TP * 0.1e6,
        gamma3_per_s=TP * 1e3, gamma4_per_s=TP * 1e3, omega_c_rad_s=TP * 5e6, eta_d=0.9,
    )  # fmt: skip
    return with_overrides(base, **over)


# ---- T-LIN-1 ------------------------------------------------------------------------------------
def test_tlin1_cold_three_level_full_vs_continued_fraction() -> None:
    sc = LadderScheme(3, gamma3=RYD, gamma_t=GT)
    v0 = np.array([0.0])
    for dp in TP * np.linspace(-10.0, 10.0, 21):
        dr = Drive(2e-3 * GAMMA2_RB87_D2, TP * 5.0, delta_p=dp)
        np.testing.assert_allclose(
            weak_probe.rho21(sc, dr, v0), full_rho21(sc, dr, v0), rtol=tol.RTOL_TLIN1
        )


def test_weak_probe_error_scales_as_omega_p_squared() -> None:
    """The weak-probe/full gap is the expected saturation correction (TSD §14.4), not a bug."""
    sc, v0 = LadderScheme(3, gamma3=RYD, gamma_t=GT), np.array([0.0])
    err = []
    for frac in (2.5e-3, 5e-3, 1e-2):
        dr = Drive(frac * GAMMA2_RB87_D2, TP * 5.0, delta_p=TP * 1.0)
        f = full_rho21(sc, dr, v0)[0]
        err.append(abs(weak_probe.rho21(sc, dr, v0)[0] - f) / abs(f))
    assert err[1] / err[0] == pytest.approx(4.0, rel=0.02)
    assert err[2] / err[1] == pytest.approx(4.0, rel=0.02)


# ---- T-LIN-2 ------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_levels", [3, 4])
def test_tlin2_full_steady_state_is_a_density_matrix(n_levels: int) -> None:
    rng = np.random.default_rng(tol.MC_SEED + n_levels)
    for _ in range(25):
        dephasing = tuple(TP * rng.uniform(0, 0.5, n_levels - 1)) if rng.random() < 0.5 else ()
        decays = None
        if rng.random() < 0.5:  # explicit channels, including a branch 3 -> 1
            decays = (Decay(2, 1, GAMMA2_RB87_D2), Decay(3, 2, RYD), Decay(3, 1, 2 * RYD))
            if n_levels == 4:
                decays += (Decay(4, 3, RYD), Decay(4, 1, RYD))
        sc = LadderScheme(
            n_levels, gamma3=TP * rng.uniform(0, 0.1), gamma4=TP * rng.uniform(0, 0.1),
            gamma_t=TP * rng.uniform(0.01, 1.0), decays=decays, dephasing=dephasing,
        )  # fmt: skip
        dr = Drive(
            TP * rng.uniform(0.01, 10), TP * rng.uniform(0, 20), TP * rng.uniform(0, 30),
            *(TP * rng.uniform(-10, 10, 3)), K_P, K_C,
        )  # fmt: skip
        v = rng.uniform(-800, 800, 7)
        lv = lindblad.build(sc, dr)
        rho = lindblad.density_matrix(lindblad.steady_state(lv, v), n_levels)
        np.testing.assert_allclose(np.trace(rho, axis1=-2, axis2=-1), 1.0, atol=tol.ATOL_TLIN2_EIG)
        np.testing.assert_allclose(rho, rho.conj().swapaxes(-1, -2), atol=tol.ATOL_TLIN2_EIG)
        assert np.linalg.eigvalsh(rho).min() >= -tol.ATOL_TLIN2_EIG
        resid = np.einsum("vij,vj->vi", lv.at(v), lindblad.steady_state(lv, v))
        assert np.abs(resid).max() < 1e-10


# ---- T-LIN-3 ------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_levels", [3, 4])
def test_tlin3_two_level_limit(n_levels: int) -> None:
    sc = LadderScheme(n_levels, gamma3=RYD, gamma4=RYD, gamma_t=GT)
    v0 = np.array([0.0])
    for dp in TP * np.array([-8.0, -1.0, 0.0, 0.5, 6.0]):
        dr = Drive(1e-3 * GAMMA2_RB87_D2, 0.0, 0.0, delta_p=dp)
        exact = 0.5 * dr.omega_p / (dp + 1j * sc.coherence_rate(2))
        np.testing.assert_allclose(weak_probe.rho21(sc, dr, v0)[0], exact, rtol=1e-14)
        np.testing.assert_allclose(full_rho21(sc, dr, v0)[0], exact, rtol=tol.RTOL_TLIN3)


# ---- T-LIN-4, T-LIN-5 ----------------------------------------------------------------------------
def at_splitting(absorption: np.ndarray, detuning_mhz: np.ndarray) -> float:
    """Separation of the two most prominent absorption minima."""
    idx, prop = find_peaks(-absorption, prominence=0.0)
    top = idx[np.argsort(prop["prominences"])[::-1][:2]]
    return float(abs(detuning_mhz[top[0]] - detuning_mhz[top[1]]))


def test_tlin4_cold_autler_townes_splitting() -> None:
    sc, v0 = scheme4(), np.array([0.0])
    scan = np.linspace(-6.0, 6.0, 2401)
    absorb = np.array(
        [
            -full_rho21(sc, Drive(TP * 0.05, TP * 5, TP * 4, delta_p=TP * d), v0)[0].imag
            for d in scan
        ]
    )
    split = at_splitting(absorb, scan)
    assert split == pytest.approx(4.0, rel=tol.RTOL_TLIN4_SPLIT)
    assert split == pytest.approx(3.93, abs=0.011)  # TSD §14.11 measured value


def test_tlin5_doppler_averaged_autler_townes_splitting() -> None:
    sc = scheme4()
    v, w = velocity_grid(sigma_v(300.0, SPECIES_MASS_KG["Rb87"]), 2**15 + 1)
    scan = np.linspace(-15.0, 15.0, 1201)

    def absorption(d_mhz: float) -> float:
        dr = Drive(TP * 0.05, TP * 5, TP * 20, TP * d_mhz, k_p=K_P, k_c=K_C)
        return float(-(weak_probe.rho21(sc, dr, v) @ w).imag)

    absorb = np.array([absorption(d) for d in scan])
    predicted = 480.0 / 780.241 * 20.0
    assert predicted == pytest.approx(12.30, abs=0.005)
    assert at_splitting(absorb, scan) == pytest.approx(predicted, rel=tol.RTOL_TLIN5_SPLIT)


# ---- T-LIN-6 ------------------------------------------------------------------------------------
@pytest.mark.parametrize("omega_p_mhz", [0.01, 2.0])
def test_tlin6_h0_equals_finite_difference(omega_p_mhz: float) -> None:
    sc = scheme4()
    dr = Drive(TP * omega_p_mhz, TP * 5, TP * 20, k_p=K_P, k_c=K_C)
    v = np.array([0.0, 3.0, 30.0, -120.0])
    h = 1e-2  # rad/us; round-off of the difference stays below 1e-7 at |v| = 120 m/s
    up, dn = dr.with_omega_m(dr.omega_m + h), dr.with_omega_m(dr.omega_m - h)
    fd_w = (weak_probe.rho21(sc, up, v) - weak_probe.rho21(sc, dn, v)) / (2 * h)
    np.testing.assert_allclose(
        weak_probe.transfer_function(sc, dr, v, np.array([0.0]))[0], fd_w, rtol=tol.RTOL_TLIN6_FD
    )
    fd_f = (full_rho21(sc, up, v) - full_rho21(sc, dn, v)) / (2 * h)
    lv = lindblad.build(sc, dr)
    np.testing.assert_allclose(lindblad.transfer_function(lv, v, 0.0), fd_f, rtol=tol.RTOL_TLIN6_FD)
    np.testing.assert_allclose(  # rho0 pass-through gives the same answer
        lindblad.transfer_function(lv, v, 0.0, lindblad.steady_state(lv, v)),
        lindblad.transfer_function(lv, v, 0.0), rtol=1e-13,
    )  # fmt: skip


# ---- T-LIN-7 ------------------------------------------------------------------------------------
def test_tlin7_weak_probe_vs_full_transfer_function() -> None:
    """Omega_p/2pi = 0.01 MHz (Omega_p/Gamma_2 = 1.6e-3), TSD §14.4 verified point."""
    sc = scheme4()
    dr = Drive(TP * 0.01, TP * 5, TP * 20, k_p=K_P, k_c=K_C)
    assert dr.omega_p / GAMMA2_RB87_D2 <= 2e-3
    lv = lindblad.build(sc, dr)
    v = np.array([0.0, 100.0])
    for f_mhz in (0.0, 0.3, 1.0, 3.0):
        got = weak_probe.transfer_function(sc, dr, v, np.array([TP * f_mhz]))[0]
        ref = lindblad.transfer_function(lv, v, TP * f_mhz)
        np.testing.assert_allclose(got, ref, rtol=tol.RTOL_TLIN7)
    np.testing.assert_allclose(  # steady state at the same point
        weak_probe.rho21(sc, dr, v), full_rho21(sc, dr, v), rtol=tol.RTOL_TLIN7
    )
    one = np.array([100.0])  # TSD §14.4: <= 1e-8 at v = 100 m/s
    np.testing.assert_allclose(
        weak_probe.transfer_function(sc, dr, one, np.array([TP * 1.0]))[0],
        lindblad.transfer_function(lv, one, TP * 1.0), rtol=2e-8,
    )  # fmt: skip


# ---- T-LIN-8 ------------------------------------------------------------------------------------
def test_tlin8_doppler_loop_terminates_and_is_monotone_in_tolerance() -> None:
    sc = scheme4()
    dr = Drive(TP * 0.05, TP * 5, TP * 20, k_p=K_P, k_c=K_C)
    sig = sigma_v(300.0, SPECIES_MASS_KG["Rb87"])
    n_v, values = [], []
    for rtol in (1e-2, 1e-3, 1e-4, 1e-5):
        res = doppler_average(
            lambda v: weak_probe.rho21(sc, dr, v), sig, DopplerSettings(rtol=rtol)
        )
        assert res.converged and res.rel_change <= rtol
        n_v.append(res.n_v)
        values.append(complex(res.value))
    assert n_v == sorted(n_v) and n_v[-1] > n_v[0]
    assert abs(values[-2] - values[-1]) / abs(values[-1]) < 1e-4
    with pytest.raises(RuntimeError, match="not converged"):
        doppler_average(
            lambda v: weak_probe.rho21(sc, dr, v), sig,
            DopplerSettings(n_v_start=65, rtol=1e-9, n_v_max=257),
        )  # fmt: skip


# ---- T-LIN-9 (optional) --------------------------------------------------------------------------
def test_tlin9_full_vs_qutip() -> None:
    qutip = pytest.importorskip("qutip")
    rng = np.random.default_rng(tol.MC_SEED)
    for _ in range(20):
        sc = LadderScheme(
            4, gamma3=TP * rng.uniform(0.001, 0.1), gamma4=TP * rng.uniform(0.001, 0.1),
            gamma_t=TP * rng.uniform(0.05, 1.0),
        )  # fmt: skip
        dr = Drive(
            TP * rng.uniform(0.1, 5), TP * rng.uniform(1, 10), TP * rng.uniform(1, 20),
            *(TP * rng.uniform(-5, 5, 3)), K_P, K_C,
        )  # fmt: skip
        v = float(rng.uniform(-300, 300))
        lv = lindblad.build(sc, dr)
        rho = lindblad.density_matrix(lindblad.steady_state(lv, np.array([v])), 4)[0]
        h = lindblad.hamiltonian(sc, dr) + v * lindblad.doppler_hamiltonian(sc, dr)
        ref = qutip.steadystate(qutip.Qobj(h), [qutip.Qobj(c) for c in sc.collapse_operators()])
        np.testing.assert_allclose(rho, ref.full(), rtol=tol.RTOL_TLIN9_QUTIP, atol=1e-12)


# ---- T-LIN-10 ------------------------------------------------------------------------------------
def test_tlin10_two_level_resonant_cross_section() -> None:
    gamma2_si = GAMMA2_RB87_PER_S
    d12 = readout.cycling_dipole_cm(gamma2_si, K_P)
    lam = TP / K_P
    for engine in ("weak_probe", "full"):
        # gamma3 > 0 only makes the decoupled level relax (unique full steady state)
        sc = LadderScheme(3, gamma3=RYD, gamma_t=0.0)
        dr = Drive(1e-4 * GAMMA2_RB87_D2, 0.0)
        v0 = np.array([0.0])
        fn = weak_probe.rho21 if engine == "weak_probe" else full_rho21
        rho = fn(sc, dr, v0)[0]
        n0 = 1e15
        alpha = readout.absorption_coefficient(rho.imag, dr.omega_p * 1e6, K_P, n0, d12)
        rtol = 1e-14 if engine == "weak_probe" else tol.RTOL_TLIN10
        assert alpha / n0 == pytest.approx(3.0 * lam**2 / TP, rel=rtol)


# ---- T-LIN-11 ------------------------------------------------------------------------------------
def test_tlin11_auto_recomputes_with_full_on_a_strong_probe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    strong = [
        LindbladRARModel(params(probe_power_w=1e-6, omega_c_rad_s=TP * oc * 1e6), COARSE)
        for oc in (5.0, 10.0)
    ]
    rng = np.random.default_rng(tol.MC_SEED)
    with caplog.at_level(logging.WARNING, logger="qmwlink.receivers.rydberg_physics.model"):
        res = evaluate_sweep(strong, "auto", weak_probe_tol=0.01, rng=rng, bandwidth=False)
    assert res.engine_used == "full" and res.recomputed_with_full
    assert res.max_discrepancy > 0.01 and set(res.checkpoints) == {0, 1}
    assert all(r.engine == "full" for r in res.results)
    assert "recomputing the sweep with the full Lindblad engine" in caplog.text
    weak_only = evaluate_sweep(strong, "weak_probe", bandwidth=False)
    assert weak_only.engine_used == "weak_probe" and np.isnan(weak_only.max_discrepancy)
    assert weak_only.results[0].s_e_rms != pytest.approx(res.results[0].s_e_rms, rel=0.01)

    weak = [LindbladRARModel(params(omega_c_rad_s=TP * oc * 1e6), COARSE) for oc in (5.0, 10.0)]
    caplog.clear()
    ok = evaluate_sweep(weak, "auto", rng=np.random.default_rng(tol.MC_SEED), bandwidth=False)
    assert ok.engine_used == "weak_probe" and not ok.recomputed_with_full
    assert 0.0 < ok.max_discrepancy < 0.01 and "recomputing" not in caplog.text


# ---- T-LIN-12 ------------------------------------------------------------------------------------
def test_tlin12_readout_units_rms_sensitivity_and_peak_linear_range() -> None:
    p = params()
    res = LindbladRARModel(p, COARSE).evaluate("weak_probe")
    # S_E is rms: a tone of rms amplitude S_E sqrt(1 Hz) gives an output rms equal to sqrt(S_P).
    s_p = readout.power_noise_psd(res.p_out_w, 299792458.0 / p.probe_wavelength_m, p.eta_d, 0.0)
    assert abs(res.kappa_w_per_v_m) * res.s_e_rms == pytest.approx(np.sqrt(s_p), rel=1e-12)
    assert res.s_e_rms != pytest.approx(np.sqrt(2 * s_p) / abs(res.kappa_w_per_v_m), rel=0.1)
    # E_1dB is a peak field: hbar Omega_sig / d34, below the LO peak field.
    assert res.e_1db_case == "compression"
    assert 0.0 < res.e_1db_pk_v_per_m < p.e_lo_pk_v_per_m
    omega_1db = res.e_1db_pk_v_per_m * p.d34_cm / HBAR
    assert 0.0 < omega_1db < res.omega_lo_rad_s
    # The §7.4 flag compares peak with peak: pick P_rx with E_rx,rms < E_1dB,pk < E_rx,pk.
    rx = RydbergReceiver.from_physics(p, doppler=COARSE)
    assert rx.sideband == "dsb" and rx.e_1db_pk_v_per_m == pytest.approx(res.e_1db_pk_v_per_m)
    assert rx.n_amp == pytest.approx(res.n_amp) and rx.bandwidth_hz == pytest.approx(res.b_3db_hz)
    a_r = 2.0
    e_rx_pk = 1.2 * res.e_1db_pk_v_per_m
    p_rx = e_rx_pk**2 * a_r / (2.0 * ETA0)
    assert rx.e_rx_pk(p_rx, a_r) == pytest.approx(e_rx_pk)
    assert e_rx_pk / np.sqrt(2.0) < res.e_1db_pk_v_per_m < e_rx_pk
    assert rx.linear_regime(p_rx, a_r) is False
    assert rx.linear_regime(p_rx / 4.0, a_r) is True
