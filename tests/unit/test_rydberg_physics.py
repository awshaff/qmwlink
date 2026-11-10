"""M7 unit tests: scheme, engines, Doppler grid, response, readout, model (FR-LIN-1..6, FR-RX-6)."""

from __future__ import annotations

import numpy as np
import pytest

import tolerances as tol
from qmwlink.constants import EPS0, C, H
from qmwlink.receivers.rydberg import RydbergReceiver, n_amp_from_s_e
from qmwlink.receivers.rydberg_physics import lindblad, readout, weak_probe
from qmwlink.receivers.rydberg_physics.doppler import (
    DopplerSettings,
    doppler_average,
    sigma_v,
    velocity_grid,
)
from qmwlink.receivers.rydberg_physics.model import (
    GAMMA2_RB87_PER_S,
    QUASI_STATIC_FRACTION,
    SPECIES_MASS_KG,
    LindbladRARModel,
    RARParams,
    beam_peak_field,
    evaluate_sweep,
    rabi_rad_s,
    select_checkpoints,
    with_overrides,
)
from qmwlink.receivers.rydberg_physics.readout import HBAR
from qmwlink.receivers.rydberg_physics.response import bandwidth_3db
from qmwlink.receivers.rydberg_physics.scheme import GAMMA2_RB87_D2, Decay, Drive, LadderScheme

TP = 2.0 * np.pi
K_P, K_C = TP / 780.241e-9, TP / 480e-9
COARSE = DopplerSettings(n_v_start=513, rtol=1e-3)


def params(**over: float) -> RARParams:
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


# ---- scheme (FR-LIN-1) ---------------------------------------------------------------------------
def test_scheme_rejects_other_level_counts() -> None:
    for n in (2, 5):
        with pytest.raises(ValueError, match="n_levels must be 3 or 4"):
            LadderScheme(n)


def test_coherence_rates_hand_case() -> None:
    """gamma_j1 = Gamma_j^out/2 + gamma_t + gamma_phi,j (TSD §14.3)."""
    sc = LadderScheme(
        4, gamma2=6.0, gamma3=0.4, gamma4=0.2, gamma_t=0.1, dephasing=(0.0, 0.05, 0.03)
    )
    assert sc.coherence_rate(2) == pytest.approx(3.0 + 0.1)
    assert sc.coherence_rate(3) == pytest.approx(0.2 + 0.1 + 0.05)
    assert sc.coherence_rate(4) == pytest.approx(0.1 + 0.1 + 0.03)
    assert [(d.from_level, d.to_level) for d in sc.channels] == [(2, 1), (3, 2), (4, 1)]
    assert sc.gamma_out(1) == 0.0 and sc.gamma_phi(1) == 0.0
    branched = LadderScheme(3, decays=(Decay(2, 1, 6.0), Decay(3, 2, 0.3), Decay(3, 1, 0.1)))
    assert branched.gamma_out(3) == pytest.approx(0.4)
    assert branched.coherence_rate(3) == pytest.approx(0.2)
    default = LadderScheme(3, gamma_t=0.1)
    assert default.coherence_rate(2) == pytest.approx(GAMMA2_RB87_D2 / 2 + 0.1)
    assert pytest.approx(TP * 6.0666) == GAMMA2_RB87_D2
    with pytest.raises(ValueError, match="levels >= 2"):
        sc.coherence_rate(1)
    with pytest.raises(ValueError, match=r"1\.\.4"):
        sc.gamma_out(5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"gamma_t": -1.0}, "gamma_t"),
        ({"gamma3": float("nan")}, "gamma3"),
        ({"dephasing": (0.1,)}, "one rate per level"),
        ({"dephasing": (0.1, -0.1, 0.0)}, "dephasing rates"),
        ({"decays": (Decay(1, 2, 1.0),)}, "not a valid channel"),
        ({"decays": (Decay(3, 3, 1.0),)}, "not a valid channel"),
        ({"decays": (Decay(2, 1, -1.0),)}, "decay rates"),
    ],
)
def test_scheme_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LadderScheme(4, **kwargs)  # type: ignore[arg-type]


def test_transit_collapse_operators_are_the_reset_map() -> None:
    """sum_k D[sqrt(gt)|1><k|] rho = gt (|1><1| - rho) for a unit-trace rho (TSD §14.3)."""
    n, gt = 4, 0.37
    sc = LadderScheme(n, gamma2=0.0, gamma_t=gt)
    ops = sc.collapse_operators()
    assert len(ops) == n
    rng = np.random.default_rng(3)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    out = sum(
        op @ rho @ op.conj().T - 0.5 * (op.conj().T @ op @ rho + rho @ op.conj().T @ op)
        for op in ops
    )
    ground = np.zeros((n, n))
    ground[0, 0] = 1.0
    np.testing.assert_allclose(out, gt * (ground - rho), atol=1e-14)
    deph = LadderScheme(3, gamma2=0.0, dephasing=(0.2, 0.0)).collapse_operators()
    assert len(deph) == 1 and deph[0][1, 1] == pytest.approx(np.sqrt(0.4))


def test_drive_validation() -> None:
    with pytest.raises(ValueError, match="omega_p"):
        Drive(-1.0, 1.0)
    with pytest.raises(ValueError, match="delta_c"):
        Drive(1.0, 1.0, delta_c=float("inf"))
    assert Drive(1.0, 2.0, 3.0, k_p=5.0).with_omega_m(4.0) == Drive(1.0, 2.0, 4.0, k_p=5.0)


# ---- weak-probe engine (FR-LIN-2) ----------------------------------------------------------------
def test_weak_probe_matches_matrix_solve_and_continued_fraction() -> None:
    sc = LadderScheme(4, gamma3=0.01, gamma4=0.02, gamma_t=0.5)
    dr = Drive(0.3, TP * 5, TP * 20, 1.0, -2.0, 0.7, K_P, K_C)
    v = np.array([-200.0, 0.0, 13.0])
    x = weak_probe.steady_state(sc, dr, v)
    for i, vi in enumerate(v):
        m = weak_probe.m_matrix(sc, dr, float(vi))
        np.testing.assert_allclose(m @ x[i], [0.5 * dr.omega_p, 0, 0], atol=1e-13)
        d2 = dr.delta_p - 1e-6 * K_P * vi
        d3 = d2 + dr.delta_c + 1e-6 * K_C * vi
        d4 = d3 + dr.delta_m
        cf = (0.5 * dr.omega_p) / (
            d2 + 1j * sc.coherence_rate(2)
            - (dr.omega_c**2 / 4)
            / (d3 + 1j * sc.coherence_rate(3)
               - (dr.omega_m**2 / 4) / (d4 + 1j * sc.coherence_rate(4)))
        )  # fmt: skip
        assert weak_probe.rho21(sc, dr, v)[i] == pytest.approx(cf, rel=1e-12)
        w = TP * 1.3
        h_ref = -np.linalg.solve(m - w * np.eye(3), np.array([0, -x[i, 2] / 2, -x[i, 1] / 2]))[0]
        got = weak_probe.transfer_function(sc, dr, v, np.array([w]))[0, i]
        assert got == pytest.approx(h_ref, rel=1e-11)
    assert weak_probe.transfer_function(sc, dr, v, np.array([0.0, 1.0])).shape == (2, 3)


def test_weak_probe_three_level_and_degenerate_limits() -> None:
    sc3 = LadderScheme(3, gamma3=0.01, gamma_t=0.2)
    dr = Drive(0.1, TP * 3, 0.0, 0.4, 0.0)
    x = weak_probe.steady_state(sc3, dr, np.array([0.0, 50.0]))
    assert x.shape == (2, 3) and np.all(x[:, 2] == 0)
    assert weak_probe.m_matrix(sc3, dr).shape == (2, 2)
    with pytest.raises(ValueError, match="n_levels = 4"):
        weak_probe.transfer_function(sc3, dr, 0.0, 0.0)
    # ideal EIT (no Rydberg dephasing, on resonance): exactly zero absorption, no warning
    ideal = LadderScheme(3, gamma3=0.0, gamma_t=0.0)
    assert weak_probe.rho21(ideal, Drive(0.1, TP * 3), np.array([0.0]))[0] == 0.0
    # Omega_c = 0: two-level atom, and no response to the microwave field
    sc4 = LadderScheme(4, gamma_t=0.0)
    off = Drive(0.1, 0.0, TP * 20)
    assert weak_probe.rho21(sc4, off, np.array([0.0]))[0] == pytest.approx(
        0.05 / (1j * GAMMA2_RB87_D2 / 2)
    )
    np.testing.assert_array_equal(weak_probe.steady_state(sc4, off, np.array([0.0]))[0, 1:], 0)
    np.testing.assert_array_equal(
        weak_probe.transfer_function(sc4, off, np.array([0.0, 9.0]), np.array([0.0])), 0
    )


@pytest.mark.parametrize("n_levels", [3, 4])
def test_weak_probe_dynamics(n_levels: int) -> None:
    """dx/dt = i(Mx - s): starts at x(0), relaxes to the steady state, satisfies the ODE."""
    sc = LadderScheme(n_levels, gamma3=0.01, gamma4=0.01, gamma_t=TP * 0.1)
    dr = Drive(0.3, TP * 5, TP * 20, 0.5, k_p=K_P, k_c=K_C)
    n = n_levels - 1
    x_ss = weak_probe.steady_state(sc, dr, 7.0)[:n]
    x = weak_probe.evolve(sc, dr, np.array([0.0, 0.1, 200.0]), v=7.0)
    np.testing.assert_allclose(x[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(x[2], x_ss, rtol=1e-10)
    m = weak_probe.m_matrix(sc, dr, 7.0)
    s = np.zeros(n, dtype=complex)
    s[0] = 0.5 * dr.omega_p
    h = 1e-6
    xp, xm = (weak_probe.evolve(sc, dr, np.array([0.1 + e]), v=7.0)[0] for e in (h, -h))
    np.testing.assert_allclose((xp - xm) / (2 * h), 1j * (m @ x[1] - s), rtol=1e-6, atol=1e-9)
    assert np.all(np.linalg.eigvals(1j * m).real < 0)
    custom = weak_probe.evolve(sc, dr, np.array([0.0]), v=7.0, x_init=x_ss)
    np.testing.assert_allclose(custom[0], x_ss, rtol=1e-12)
    with pytest.raises(ValueError, match="x_init"):
        weak_probe.evolve(sc, dr, np.array([0.0]), x_init=np.zeros(5))


# ---- full engine (FR-LIN-3) ----------------------------------------------------------------------
def test_liouvillian_is_affine_in_velocity_and_matches_direct_construction() -> None:
    sc = LadderScheme(4, gamma3=0.05, gamma4=0.02, gamma_t=0.3, dephasing=(0.1, 0.0, 0.2))
    dr = Drive(1.5, TP * 5, TP * 20, 0.3, -0.2, 0.1, K_P, K_C)
    lv = lindblad.build(sc, dr)
    assert lv.dim == 16 and lv.at(np.zeros((2, 3))).shape == (2, 3, 16, 16)
    rng = np.random.default_rng(1)
    rho = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    for v in (0.0, 123.0):
        h = lindblad.hamiltonian(sc, dr) + v * lindblad.doppler_hamiltonian(sc, dr)
        direct = -1j * (h @ rho - rho @ h)
        for op in sc.collapse_operators():
            ldl = op.conj().T @ op
            direct += op @ rho @ op.conj().T - 0.5 * (ldl @ rho + rho @ ldl)
        np.testing.assert_allclose(
            (lv.at(np.array(v)) @ rho.reshape(-1)).reshape(4, 4), direct, atol=1e-11
        )
    hv = lindblad.doppler_hamiltonian(sc, dr)
    np.testing.assert_allclose(
        np.diag(hv).real, 1e-6 * np.array([0.0, K_P, K_P - K_C, K_P - K_C]), rtol=1e-15
    )
    h0 = lindblad.hamiltonian(sc, dr)
    np.testing.assert_allclose(h0, h0.conj().T)
    assert h0[1, 1] == -0.3 and h0[2, 2] == pytest.approx(-0.1) and h0[3, 3] == pytest.approx(-0.2)
    assert h0[1, 0] == 0.75 and h0[3, 2] == pytest.approx(TP * 10)


def test_full_engine_three_level_has_no_microwave_response() -> None:
    lv = lindblad.build(LadderScheme(3, gamma3=0.01, gamma_t=0.1), Drive(0.1, 1.0))
    assert lv.l1 is None and lv.dim == 9
    with pytest.raises(ValueError, match="n_levels = 4"):
        lindblad.transfer_function(lv, np.array([0.0]), 0.0)


# ---- Doppler (FR-LIN-4) --------------------------------------------------------------------------
def test_sigma_v_and_grid() -> None:
    sig = sigma_v(300.0, SPECIES_MASS_KG["Rb87"])
    assert sig == pytest.approx(169.4, rel=tol.REL_SIGMA_V_RB87)
    v, w = velocity_grid(sig, 4097)
    assert v[0] == pytest.approx(-5 * sig) and v[-1] == pytest.approx(5 * sig)
    assert w.sum() == pytest.approx(1.0, abs=1e-14) and np.all(w > 0)
    assert v @ w == pytest.approx(0.0, abs=1e-10)
    assert (v**2) @ w == pytest.approx(sig**2, rel=2e-4)  # 5 sigma truncation of the Maxwellian
    for bad in ((0.0, 5), (1.0, 2)):
        with pytest.raises(ValueError):
            velocity_grid(*bad)
    with pytest.raises(ValueError):
        sigma_v(0.0, 1.0)


def test_doppler_average_fixed_grid_vector_output_and_errors() -> None:
    res = doppler_average(lambda v: np.stack([v**2, np.ones_like(v)]), 2.0, n_v=2001)
    np.testing.assert_allclose(res.value, [4.0, 1.0], rtol=2e-4)
    assert res.n_v == 2001 and res.converged and np.isnan(res.rel_change)
    conv = doppler_average(lambda v: np.cos(v / 3.0) + 0j, 2.0, DopplerSettings(n_v_start=9))
    assert conv.value == pytest.approx(np.exp(-(4 / 9) / 2), rel=1e-4)
    zero = doppler_average(lambda v: np.zeros_like(v), 2.0, DopplerSettings(n_v_start=9))
    assert zero.n_v == 17 and zero.rel_change == 0.0
    with pytest.raises(FloatingPointError, match="non-finite"):
        doppler_average(lambda v: np.full_like(v, np.nan), 2.0, n_v=11)
    for kw in ({"n_sigma": 0.0}, {"n_v_start": 2}, {"n_v_start": 9, "n_v_max": 5}, {"rtol": 0.0}):
        with pytest.raises(ValueError):
            DopplerSettings(**kw)  # type: ignore[arg-type]


# ---- response (FR-LIN-5) -------------------------------------------------------------------------
def test_bandwidth_single_pole_and_peaking() -> None:
    fc = 3.7  # MHz
    bw = bandwidth_3db(lambda w: 1.0 / np.sqrt(1.0 + (w / (TP * fc)) ** 2))
    assert bw.b_3db_hz == pytest.approx(fc * 1e6, rel=1e-9)
    assert bw.peaking == 1.0 and bw.f_peak_hz == 0.0 and not bw.reached_scan_limit

    f0, q = 5.0, 4.0  # resonant second-order response: peaks near f0

    def resonant(w: np.ndarray) -> np.ndarray:
        x = w / (TP * f0)
        return np.asarray(1.0 / np.sqrt((1 - x**2) ** 2 + (x / q) ** 2))

    bw2 = bandwidth_3db(resonant)
    x_3db = np.sqrt((1 - 1 / (2 * q**2)) + np.sqrt((1 - 1 / (2 * q**2)) ** 2 + 1))
    assert bw2.b_3db_hz == pytest.approx(x_3db * f0 * 1e6, rel=1e-9)
    assert bw2.peaking == pytest.approx(q / np.sqrt(1 - 1 / (4 * q**2)), rel=1e-9)
    assert bw2.f_peak_hz == pytest.approx(f0 * np.sqrt(1 - 1 / (2 * q**2)) * 1e6, rel=1e-5)


def test_bandwidth_edge_cases() -> None:
    flat = bandwidth_3db(lambda w: np.ones_like(w))
    assert flat.reached_scan_limit and flat.b_3db_hz == pytest.approx(1e9)
    with pytest.raises(ValueError, match="below the scan range"):
        bandwidth_3db(lambda w: 1.0 / np.sqrt(1.0 + (w / (TP * 1e-5)) ** 2))
    with pytest.raises(ValueError, match=r"\|H\(0\)\|"):
        bandwidth_3db(lambda w: np.zeros_like(w))
    with pytest.raises(ValueError, match="f_min_mhz"):
        bandwidth_3db(lambda w: np.ones_like(w), f_min_mhz=2.0, f_max_mhz=1.0)


def test_model_bandwidth_refinement_agrees_with_dense_scan() -> None:
    """FR-LIN-5 AC: refined B_3dB vs a dense-grid scan to 1e-3 relative."""
    model = LindbladRARModel(params(omega_c_rad_s=TP * 10e6, e_lo_pk_v_per_m=0.33), COARSE)
    res = model.evaluate("weak_probe", linear_range=False)
    f = np.geomspace(1e-3, 1e3, 60001)
    habs = np.abs(model.h_avg("weak_probe", TP * f, res.n_v))
    h0 = abs(model.h_avg("weak_probe", np.array([0.0]), res.n_v)[0])
    dense = f[np.nonzero(habs >= h0 / np.sqrt(2))[0][-1]] * 1e6
    assert res.b_3db_hz == pytest.approx(dense, rel=tol.RTOL_B3DB_DENSE)
    assert res.peaking == pytest.approx(max(habs.max() / h0, 1.0), rel=5e-3)
    assert not res.b_3db_is_lower_bound


# ---- readout (FR-LIN-6) --------------------------------------------------------------------------
def test_readout_si_conversions() -> None:
    assert pytest.approx(H / TP) == HBAR
    k_p, n0, d12 = K_P, 2e15, 2.5e-29
    pref = readout.absorption_prefactor(k_p, n0, d12)
    assert pref == pytest.approx(2 * k_p * n0 * d12**2 / (EPS0 * HBAR))
    alpha = readout.absorption_coefficient(-1e-3, 5e5, k_p, n0, d12)
    assert alpha == pytest.approx(pref * 1e-3 / 5e5) and alpha > 0
    assert readout.transmitted_power(2e-6, 30.0, 0.05) == pytest.approx(2e-6 * np.exp(-1.5))
    assert readout.slope_kappa(1e-6, 0.05, 2e-9, 1e-26) == pytest.approx(
        -1e-6 * 0.05 * 2e-9 * 1e-26 / HBAR
    )
    nu = C / 780.241e-9
    s_p = readout.power_noise_psd(1e-6, nu, 0.5, 3e-13)
    assert s_p == pytest.approx(2 * H * nu * 1e-6 / 0.5 + 9e-26)
    assert readout.sensitivity_rms(4e-24, -2e-9) == pytest.approx(1e-3)
    d_cyc = readout.cycling_dipole_cm(GAMMA2_RB87_PER_S, K_P)
    assert d_cyc / 8.4783536e-30 == pytest.approx(2.989, rel=1e-3)  # e a0 (Steck: 2.989 e a0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: readout.absorption_prefactor(0.0, 1.0, 1.0),
        lambda: readout.absorption_coefficient(-1.0, 0.0, 1.0, 1.0, 1.0),
        lambda: readout.transmitted_power(-1.0, 1.0, 1.0),
        lambda: readout.power_noise_psd(1.0, 1.0, 0.0, 0.0),
        lambda: readout.power_noise_psd(-1.0, 1.0, 1.0, 0.0),
        lambda: readout.sensitivity_rms(1.0, 0.0),
        lambda: readout.linear_range(lambda w: w, 0.0, 1.0),
        lambda: readout.linear_range(lambda w: w, 1.0, 0.0),
    ],
)
def test_readout_validation(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]


def test_linear_range_synthetic_compression_expansion_and_none() -> None:
    """FR-LIN-6 (c): P(O_LO + x) = k x -/+ b x^3 has c = 1 -/+ (3b/4k) s^2 exactly."""
    lo, k, b = 10.0, 2.0, 0.05

    def comp(w: np.ndarray) -> np.ndarray:
        return np.asarray(1.0 + k * (w - lo) - b * (w - lo) ** 3)

    def expa(w: np.ndarray) -> np.ndarray:
        return np.asarray(1.0 + k * (w - lo) + b * (w - lo) ** 3)

    assert readout.first_harmonic(comp, lo, 2.0) == pytest.approx(k * 2 - 0.75 * b * 8, rel=1e-13)
    assert readout.compression(comp, lo, 2.0, k) == pytest.approx(1 - 0.75 * b / k * 4, rel=1e-13)
    r_c = readout.linear_range(comp, lo, k)
    assert r_c.case == "compression" and r_c.reason == ""
    assert r_c.omega_sig_1db == pytest.approx(
        np.sqrt((1 - 10 ** (-0.05)) * 4 * k / (3 * b)), rel=1e-8
    )
    r_e = readout.linear_range(expa, lo, k)
    assert r_e.case == "expansion"
    assert r_e.omega_sig_1db == pytest.approx(np.sqrt((10**0.05 - 1) * 4 * k / (3 * b)), rel=1e-8)
    r_n = readout.linear_range(lambda w: 3.0 * w, lo, 3.0)
    assert r_n.case == "none" and np.isnan(r_n.omega_sig_1db) and "up to the LO" in r_n.reason
    hard = readout.linear_range(lambda w: np.tanh(5e3 * (w - lo)), lo, 5e3)
    assert hard.case == "none" and "smallest scanned" in hard.reason
    sign = readout.linear_range(lambda w: np.sin(0.9 * (w - lo)), lo, 0.9)  # A_1 changes sign
    assert sign.case == "compression" and 0 < sign.omega_sig_1db < lo


# ---- model (FR-RX-6) -----------------------------------------------------------------------------
def test_si_rabi_conversions() -> None:
    e = beam_peak_field(1e-6, 1e-3)
    intensity = 2 * 1e-6 / (np.pi * 1e-6)
    assert e == pytest.approx(np.sqrt(2 * intensity / (C * EPS0)))
    assert rabi_rad_s(1e-29, 3.0) == pytest.approx(3e-29 / HBAR)
    p = params()
    assert p.omega_p_rad_s == pytest.approx(rabi_rad_s(p.d12_cm, beam_peak_field(1e-8, 1e-3)))
    assert p.omega_lo_rad_s == pytest.approx(TP * 20e6)
    dr, sc = p.drive(), p.scheme()
    assert dr.omega_c == pytest.approx(TP * 5.0) and dr.omega_m == pytest.approx(TP * 20.0)
    assert dr.k_p == pytest.approx(K_P) and dr.k_c == pytest.approx(K_C)
    assert sc.gamma2 == pytest.approx(GAMMA2_RB87_D2) and sc.gamma_t == pytest.approx(TP * 0.1)
    beam = params(omega_c_rad_s=None, coupling_power_w=0.02, coupling_waist_m=2e-3, d23_cm=1e-31)  # type: ignore[arg-type]
    assert beam.omega_c_si == pytest.approx(rabi_rad_s(1e-31, beam_peak_field(0.02, 2e-3)))
    custom = params(decays_per_s=((2, 1, 3e7), (3, 1, 1e4), (4, 3, 2e4)))  # type: ignore[arg-type]
    assert custom.scheme().gamma_out(3) == pytest.approx(1e-2)
    assert custom.scheme().gamma_out(4) == pytest.approx(2e-2)
    with pytest.raises(ValueError):
        beam_peak_field(1.0, 0.0)
    with pytest.raises(ValueError):
        rabi_rad_s(0.0, 1.0)


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"mass_kg": 0.0}, "mass_kg"),
        ({"n0_per_m3": -1.0}, "n0_per_m3"),
        ({"eta_d": 1.5}, "eta_d"),
        ({"omega_c_rad_s": None}, "give omega_c_rad_s"),
        ({"omega_c_rad_s": -1.0}, "omega_c_rad_s must be"),
        ({"d34_cm": 0.0}, "d34_cm"),
    ],
)
def test_params_validation(over: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        params(**over)


@pytest.fixture(scope="module")
def baseline() -> tuple[LindbladRARModel, object]:
    model = LindbladRARModel(params(), COARSE)
    return model, model.evaluate("weak_probe")


def test_model_slope_matches_finite_difference_of_output_power(
    baseline: tuple[LindbladRARModel, object],
) -> None:
    """kappa = dP_out/dE_sig end to end: difference P_out at E_LO +/- dE (same Doppler grid)."""
    model, res = baseline
    p = model.params
    d_e = 1e-4 * p.e_lo_pk_v_per_m
    d_omega = p.d34_cm * d_e / HBAR * 1e-6
    lo = model.drive.omega_m
    up, dn = (
        model.p_out(model.rho21_avg("weak_probe", n_v=res.n_v, omega_m=lo + s * d_omega)[0])  # type: ignore[attr-defined]
        for s in (1, -1)
    )
    assert res.kappa_w_per_v_m == pytest.approx((up - dn) / (2 * d_e), rel=1e-6)  # type: ignore[attr-defined]


def test_model_outputs_are_consistent(baseline: tuple[LindbladRARModel, object]) -> None:
    model, res = baseline
    p = model.params
    r = res  # type: ignore[assignment]
    assert r.engine == "weak_probe" and r.n_v >= 1025  # type: ignore[attr-defined]
    assert r.alpha_per_m > 0 and 0 < r.p_out_w < p.probe_power_w  # type: ignore[attr-defined]
    assert r.p_out_w == pytest.approx(p.probe_power_w * np.exp(-r.alpha_per_m * p.cell_length_m))  # type: ignore[attr-defined]
    assert r.n_amp == pytest.approx(n_amp_from_s_e(r.s_e_rms, p.f_mw_hz))  # type: ignore[attr-defined]
    assert r.omega_lo_rad_s == pytest.approx(TP * 20e6) and r.f_mw_hz == 5e9  # type: ignore[attr-defined]
    assert 1e6 < r.b_3db_hz < 1e8 and r.peaking >= 1.0  # type: ignore[attr-defined]
    assert model.discrepancy() < 0.01
    # photon-shot-noise scaling: 4x the probe power halves S_E in the weak-probe model
    quad = LindbladRARModel(params(probe_power_w=4e-8), COARSE).evaluate(
        "weak_probe", bandwidth=False, linear_range=False
    )
    assert quad.s_e_rms == pytest.approx(r.s_e_rms / 2.0, rel=1e-9)  # type: ignore[attr-defined]
    assert np.isnan(quad.b_3db_hz) and quad.e_1db_case == "not computed"


def test_model_errors_and_quasi_static_check(baseline: tuple[LindbladRARModel, object]) -> None:
    model, res = baseline
    b = res.b_3db_hz  # type: ignore[attr-defined]
    ok = model.evaluate("weak_probe", linear_range=False, delta_hz=0.5 * QUASI_STATIC_FRACTION * b)
    assert ok.b_3db_hz == pytest.approx(b)
    with pytest.raises(ValueError, match="delta << B_3dB"):
        model.evaluate("weak_probe", linear_range=False, delta_hz=b)
    with pytest.raises(ValueError, match="needs bandwidth=True"):
        model.evaluate("weak_probe", bandwidth=False, linear_range=False, delta_hz=1.0)
    with pytest.raises(ValueError, match="4-level"):
        LindbladRARModel(params(n_levels=3), COARSE).evaluate()
    with pytest.raises(ValueError, match="without an LO"):
        LindbladRARModel(params(e_lo_pk_v_per_m=0.0), COARSE).evaluate()
    three = LindbladRARModel(params(n_levels=3), COARSE)
    assert three.params.omega_lo_rad_s == 0.0 and three.discrepancy() < 0.01
    # a huge photon flux maps below the quantum limit: raise, never clamp (FR-RX-4)
    with pytest.raises(ValueError, match="below the quantum limit"):
        LindbladRARModel(
            params(d34_cm=1e-19, e_lo_pk_v_per_m=HBAR * TP * 20e6 / 1e-19), COARSE
        ).evaluate("weak_probe", bandwidth=False, linear_range=False)


def test_full_engine_h_avg_and_evaluate_agree_with_weak_probe_at_weak_drive() -> None:
    small = DopplerSettings(n_v_start=129, rtol=5e-2)
    model = LindbladRARModel(params(), small)
    om = TP * np.array([0.0, 1.0])
    n_v = 257
    np.testing.assert_allclose(
        model.h_avg("full", om, n_v), model.h_avg("weak_probe", om, n_v), rtol=0.01
    )
    full = model.evaluate("full", bandwidth=False, linear_range=False)
    weak = model.evaluate("weak_probe", bandwidth=False, linear_range=False)
    assert full.engine == "full" and full.s_e_rms == pytest.approx(weak.s_e_rms, rel=0.01)


def test_checkpoints_and_sweep_arguments() -> None:
    rng = np.random.default_rng(5)
    cps = select_checkpoints([0.1, 0.5, 0.2, 0.3, 0.4, 0.05, 0.01, 0.02], rng)
    assert cps[0] == 1 and len(cps) == 5 and len(set(cps)) == 5
    assert cps == select_checkpoints(
        [0.1, 0.5, 0.2, 0.3, 0.4, 0.05, 0.01, 0.02], np.random.default_rng(5)
    )
    assert select_checkpoints([1.0], rng) == (0,)
    assert set(select_checkpoints([1.0, 2.0, 0.5], rng)) == {0, 1, 2}
    with pytest.raises(ValueError, match="empty sweep"):
        select_checkpoints([], rng)
    models = [LindbladRARModel(params(), COARSE)]
    with pytest.raises(ValueError, match="method must be"):
        evaluate_sweep(models, "fast")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weak_probe_tol"):
        evaluate_sweep(models, "auto", weak_probe_tol=0.0)
    with pytest.raises(ValueError, match="explicit rng"):
        evaluate_sweep(models, "auto")


def test_from_physics_without_linear_range() -> None:
    rx = RydbergReceiver.from_physics(params(), doppler=COARSE, linear_range=False)
    assert rx.e_1db_pk_v_per_m is None and rx.sideband == "dsb" and rx.f_hz == 5e9
    assert rx.linear_regime(1.0, 1.0) is True
