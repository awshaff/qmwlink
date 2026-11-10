"""``LindbladRARModel``: SI atomic/optical inputs to receiver figures. (FR-RX-6, TSD §14.6-14.10)

SI in -> ``kappa``, ``S_E`` (rms), ``N_amp``, ``B_3dB``, ``E_1dB`` (peak), with the engine used and
the weak-probe/full discrepancy. This is the only module that converts between SI and the core
units of TSD §14.2 (us, rad/us).

Field conventions (TSD §2): Rabi frequencies use **peak** fields, ``Omega = d E_pk / hbar``;
``S_E`` is **rms**; ``E_1dB`` is **peak**. The superheterodyne readout responds to the envelope, so
a receiver built from this model is ``sideband="dsb"``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import constants as _sc

from qmwlink.constants import EPS0, C
from qmwlink.receivers.rydberg import n_amp_from_s_e
from qmwlink.receivers.rydberg_physics import lindblad, readout, weak_probe
from qmwlink.receivers.rydberg_physics.doppler import (
    DopplerSettings,
    doppler_average,
    sigma_v,
    velocity_grid,
)
from qmwlink.receivers.rydberg_physics.readout import HBAR
from qmwlink.receivers.rydberg_physics.response import Bandwidth, bandwidth_3db
from qmwlink.receivers.rydberg_physics.scheme import Decay, Drive, LadderScheme

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Engine = Literal["weak_probe", "full"]
Method = Literal["auto", "weak_probe", "full"]

PER_S_TO_CORE = 1e-6
"""1/s (or rad/s) -> rad/us."""

SPECIES_MASS_KG: dict[str, float] = {"Rb87": 86.909180520 * float(_sc.atomic_mass)}
"""Atomic masses [L] (87Rb: Steck, *Rubidium 87 D Line Data*, rev. 2.3)."""

GAMMA2_RB87_PER_S = 2.0 * np.pi * 6.0666e6
"""87Rb D2 natural decay rate [1/s] [L] (Steck, rev. 2.3)."""

_FULL_CHUNK = 8192
_MAX_ELEMENTS = 4_000_000
BANDWIDTH_GRID: dict[str, int] = {"weak_probe": 121, "full": 61}
"""Log-scan points of the bandwidth search (the full engine pays one batched solve per point)."""
QUASI_STATIC_FRACTION = 0.1
"""``delta <= 0.1 B_3dB`` is accepted as ``delta << B_3dB`` for the quasi-static §14.9 [A]."""


def beam_peak_field(power_w: float, waist_m: float) -> float:
    """On-axis peak field of a Gaussian beam: ``sqrt(2 I/(c eps_0))`` with ``I = 2P/(pi w0^2)``."""
    if power_w < 0 or waist_m <= 0:
        raise ValueError("power_w must be >= 0 and waist_m > 0")
    intensity = 2.0 * power_w / (np.pi * waist_m**2)
    return float(np.sqrt(2.0 * intensity / (C * EPS0)))


def rabi_rad_s(dipole_cm: float, e_pk_v_per_m: float) -> float:
    """``Omega = d E_pk / hbar`` [rad/s] (peak field, TSD §2)."""
    if dipole_cm <= 0 or e_pk_v_per_m < 0:
        raise ValueError("dipole must be > 0 and the field >= 0")
    return float(dipole_cm * e_pk_v_per_m / HBAR)


@dataclass(frozen=True)
class RARParams:
    """SI inputs of the receiver physics model (TSD §12 ``lindblad`` block, §14.10).

    Rates are natural decay rates in 1/s. The coupling Rabi frequency is given either directly
    (``omega_c_rad_s``) or through ``coupling_power_w``, ``coupling_waist_m`` and ``d23_cm``.
    ``n_levels = 3`` has no microwave transition: ``d34_cm``/``e_lo_pk_v_per_m`` are ignored and
    only the optical steady state is available.
    """

    mass_kg: float
    t_vapor_k: float
    n0_per_m3: float
    cell_length_m: float
    probe_wavelength_m: float
    probe_power_w: float
    probe_waist_m: float
    d12_cm: float
    coupling_wavelength_m: float
    f_mw_hz: float
    d34_cm: float
    e_lo_pk_v_per_m: float
    gamma_t_per_s: float
    gamma3_per_s: float
    gamma4_per_s: float
    gamma2_per_s: float = GAMMA2_RB87_PER_S
    omega_c_rad_s: float | None = None
    coupling_power_w: float | None = None
    coupling_waist_m: float | None = None
    d23_cm: float | None = None
    n_levels: int = 4
    decays_per_s: tuple[tuple[int, int, float], ...] | None = None
    """Optional explicit ``(from, to, rate_per_s)`` channels replacing the §14.3 defaults."""
    delta_p_rad_s: float = 0.0
    delta_c_rad_s: float = 0.0
    delta_m_rad_s: float = 0.0
    eta_d: float = 1.0
    nep_w_per_rthz: float = 0.0

    def __post_init__(self) -> None:
        positive = ("mass_kg", "t_vapor_k", "cell_length_m", "probe_wavelength_m", "probe_power_w",
                    "probe_waist_m", "d12_cm", "coupling_wavelength_m", "f_mw_hz")  # fmt: skip
        for name in positive:
            v = getattr(self, name)
            if not (np.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("n0_per_m3", "gamma_t_per_s", "gamma2_per_s", "gamma3_per_s", "gamma4_per_s",
                     "nep_w_per_rthz"):  # fmt: skip
            v = getattr(self, name)
            if not (np.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be finite and >= 0")
        if not 0 < self.eta_d <= 1:
            raise ValueError("eta_d must be in (0, 1]")
        beam = (self.coupling_power_w, self.coupling_waist_m, self.d23_cm)
        if self.omega_c_rad_s is None and any(b is None for b in beam):
            raise ValueError(
                "give omega_c_rad_s, or all of coupling_power_w, coupling_waist_m and d23_cm"
            )
        if self.omega_c_rad_s is not None and self.omega_c_rad_s < 0:
            raise ValueError("omega_c_rad_s must be >= 0")
        if self.n_levels == 4 and not (self.d34_cm > 0 and self.e_lo_pk_v_per_m >= 0):
            raise ValueError("d34_cm must be > 0 and e_lo_pk_v_per_m >= 0")

    # ---- SI Rabi frequencies ---------------------------------------------------------------
    @property
    def omega_p_rad_s(self) -> float:
        return rabi_rad_s(self.d12_cm, beam_peak_field(self.probe_power_w, self.probe_waist_m))

    @property
    def omega_c_si(self) -> float:
        if self.omega_c_rad_s is not None:
            return self.omega_c_rad_s
        assert self.coupling_power_w is not None and self.coupling_waist_m is not None
        assert self.d23_cm is not None
        return rabi_rad_s(
            self.d23_cm, beam_peak_field(self.coupling_power_w, self.coupling_waist_m)
        )

    @property
    def omega_lo_rad_s(self) -> float:
        """LO Rabi frequency ``d_34 E_LO,pk / hbar`` (0 for 3 levels)."""
        return rabi_rad_s(self.d34_cm, self.e_lo_pk_v_per_m) if self.n_levels == 4 else 0.0

    @property
    def k_p(self) -> float:
        return 2.0 * np.pi / self.probe_wavelength_m

    @property
    def k_c(self) -> float:
        return 2.0 * np.pi / self.coupling_wavelength_m

    # ---- core-unit objects -------------------------------------------------------------------
    def scheme(self) -> LadderScheme:
        s = PER_S_TO_CORE
        decays = None
        if self.decays_per_s is not None:
            decays = tuple(Decay(a, b, r * s) for a, b, r in self.decays_per_s)
        return LadderScheme(
            self.n_levels, gamma2=self.gamma2_per_s * s, gamma3=self.gamma3_per_s * s,
            gamma4=self.gamma4_per_s * s, gamma_t=self.gamma_t_per_s * s, decays=decays,
        )  # fmt: skip

    def drive(self) -> Drive:
        s = PER_S_TO_CORE
        return Drive(
            self.omega_p_rad_s * s, self.omega_c_si * s, self.omega_lo_rad_s * s,
            self.delta_p_rad_s * s, self.delta_c_rad_s * s, self.delta_m_rad_s * s,
            self.k_p, self.k_c,
        )  # fmt: skip


@dataclass(frozen=True)
class RARModelResult:
    """Model outputs in SI: ``s_e_rms`` is rms single-sided, ``e_1db_pk_v_per_m`` a peak field."""

    engine: Engine
    n_v: int
    omega_p_rad_s: float
    omega_c_rad_s: float
    omega_lo_rad_s: float
    rho21: complex
    h0_per_rad_s: complex
    """Doppler-averaged ``d<rho_21>/dOmega_m`` [1/(rad/s)]."""
    alpha_per_m: float
    p_out_w: float
    kappa_w_per_v_m: float
    s_e_rms: float
    n_amp: float
    b_3db_hz: float
    peaking: float
    b_3db_is_lower_bound: bool
    e_1db_pk_v_per_m: float
    e_1db_case: str
    e_1db_reason: str
    f_mw_hz: float


class LindbladRARModel:
    """Receiver physics model for one parameter point. (FR-RX-6, TSD §14.10)"""

    def __init__(self, params: RARParams, doppler: DopplerSettings | None = None) -> None:
        self.params = params
        self.doppler = doppler or DopplerSettings()
        self.scheme = params.scheme()
        self.drive = params.drive()
        self.sigma_v = sigma_v(params.t_vapor_k, params.mass_kg)

    # ---- Doppler-averaged engine calls (core units) ---------------------------------------------
    def _rho21_v(self, engine: Engine, drive: Drive, v: FloatArray) -> ComplexArray:
        if engine == "weak_probe":
            return weak_probe.rho21(self.scheme, drive, v)
        lv = lindblad.build(self.scheme, drive)
        parts = [
            lindblad.rho21(lv, lindblad.steady_state(lv, v[i : i + _FULL_CHUNK]))
            for i in range(0, v.size, _FULL_CHUNK)
        ]
        return np.concatenate(parts)

    def rho21_avg(
        self, engine: Engine, *, n_v: int | None = None, omega_m: float | None = None
    ) -> tuple[complex, int]:
        """Doppler-averaged ``<rho_21>`` and the ``N_v`` used (refined unless ``n_v`` is given)."""
        drive = self.drive if omega_m is None else self.drive.with_omega_m(omega_m)
        res = doppler_average(
            lambda v: self._rho21_v(engine, drive, v), self.sigma_v, self.doppler, n_v=n_v
        )
        return complex(res.value), res.n_v

    def h_avg(self, engine: Engine, omega: FloatArray, n_v: int) -> ComplexArray:
        """Doppler-averaged transfer function at angular frequencies ``omega`` [rad/us]."""
        v, w = velocity_grid(self.sigma_v, n_v, self.doppler.n_sigma)
        om = np.atleast_1d(np.asarray(omega, dtype=np.float64))
        out = np.empty(om.shape, dtype=np.complex128)
        if engine == "weak_probe":
            step = max(1, _MAX_ELEMENTS // n_v)
            for i in range(0, om.size, step):
                out[i : i + step] = (
                    weak_probe.transfer_function(self.scheme, self.drive, v, om[i : i + step]) @ w
                )
            return out
        lv = lindblad.build(self.scheme, self.drive)
        out[:] = 0.0
        for j in range(0, n_v, _FULL_CHUNK):  # chunks of velocity classes, then frequencies
            sl = slice(j, j + _FULL_CHUNK)
            rho0 = lindblad.steady_state(lv, v[sl])
            for i, o in enumerate(om):
                out[i] += complex(lindblad.transfer_function(lv, v[sl], float(o), rho0) @ w[sl])
        return out

    # ---- SI readout ----------------------------------------------------------------------------
    def _alpha(self, rho21: complex) -> float:
        p = self.params
        return readout.absorption_coefficient(
            rho21.imag, p.omega_p_rad_s, p.k_p, p.n0_per_m3, p.d12_cm
        )

    def p_out(self, rho21: complex) -> float:
        p = self.params
        return readout.transmitted_power(p.probe_power_w, self._alpha(rho21), p.cell_length_m)

    def evaluate(
        self,
        engine: Engine = "weak_probe",
        *,
        bandwidth: bool = True,
        linear_range: bool = True,
        delta_hz: float | None = None,
    ) -> RARModelResult:
        """All receiver figures with one engine. (FR-RX-6, FR-LIN-5,6)

        ``delta_hz`` is the IF offset: the quasi-static linear range of §14.9 needs
        ``delta << B_3dB`` and raises otherwise. A mapped ``N_amp < 1/2`` raises (never clamped).
        """
        p = self.params
        if p.n_levels != 4:
            raise ValueError("the receiver figures need the 4-level scheme (microwave transition)")
        if p.omega_lo_rad_s <= 0:
            raise ValueError("e_lo_pk_v_per_m must be > 0: no small-signal response without an LO")
        rho, n_v = self.rho21_avg(engine)
        h0_core = complex(self.h_avg(engine, np.array([0.0]), n_v)[0])
        alpha = self._alpha(rho)
        p_out = self.p_out(rho)
        pref = readout.absorption_prefactor(p.k_p, p.n0_per_m3, p.d12_cm) / p.omega_p_rad_s
        dalpha_core = pref * (-h0_core.imag)  # 1/m per rad/us
        kappa = readout.slope_kappa(p_out, p.cell_length_m, dalpha_core * PER_S_TO_CORE, p.d34_cm)
        s_p = readout.power_noise_psd(p_out, C / p.probe_wavelength_m, p.eta_d, p.nep_w_per_rthz)
        s_e = readout.sensitivity_rms(s_p, kappa)
        n_amp = n_amp_from_s_e(s_e, p.f_mw_hz)
        bw = Bandwidth(float("nan"), float("nan"), float("nan"), False)
        if bandwidth:
            n_grid = BANDWIDTH_GRID[engine]
            bw = bandwidth_3db(lambda om: np.abs(self.h_avg(engine, om, n_v)), n_grid=n_grid)
        if delta_hz is not None:
            if not bandwidth:
                raise ValueError("the delta_hz check needs bandwidth=True")
            if delta_hz > QUASI_STATIC_FRACTION * bw.b_3db_hz:
                raise ValueError(
                    f"quasi-static linear range needs delta << B_3dB: delta = {delta_hz:.4g} Hz, "
                    f"B_3dB = {bw.b_3db_hz:.4g} Hz"
                )
        e_1db, case, reason = float("nan"), "not computed", ""
        if linear_range:
            lo_core = self.drive.omega_m
            kappa_omega = -p_out * p.cell_length_m * dalpha_core  # W per rad/us

            def p_ss(omega_m: FloatArray) -> FloatArray:
                vals = [
                    self.p_out(self.rho21_avg(engine, n_v=n_v, omega_m=float(o))[0])
                    for o in np.atleast_1d(omega_m)
                ]
                return np.asarray(vals, dtype=np.float64)

            lr = readout.linear_range(p_ss, lo_core, kappa_omega)
            case, reason = lr.case, lr.reason
            e_1db = HBAR * (lr.omega_sig_1db / PER_S_TO_CORE) / p.d34_cm
        return RARModelResult(
            engine, n_v, p.omega_p_rad_s, p.omega_c_si, p.omega_lo_rad_s, rho,
            h0_core * PER_S_TO_CORE, alpha, p_out, kappa, s_e, n_amp, bw.b_3db_hz, bw.peaking,
            bw.reached_scan_limit, e_1db, case, reason, p.f_mw_hz,
        )  # fmt: skip

    def discrepancy(self) -> float:
        """Max relative weak-probe vs full difference in ``<rho_21>`` and ``H(0)``. (TSD §14.6)

        Both engines use the velocity grid on which the weak-probe average converged, so the
        number isolates the model difference from grid error.
        """
        rho_w, n_v = self.rho21_avg("weak_probe")
        rho_f, _ = self.rho21_avg("full", n_v=n_v)
        worst = abs(rho_w - rho_f) / abs(rho_f)
        if self.params.n_levels == 4 and self.drive.omega_m > 0:
            zero = np.array([0.0])
            h_w, h_f = self.h_avg("weak_probe", zero, n_v)[0], self.h_avg("full", zero, n_v)[0]
            worst = max(worst, abs(h_w - h_f) / abs(h_f))
        return float(worst)


@dataclass(frozen=True)
class SweepResult:
    """A sweep with the engine actually used and the ``auto`` bookkeeping (FR-RX-6 AC)."""

    results: tuple[RARModelResult, ...]
    method: Method
    engine_used: Engine
    max_discrepancy: float
    """Max weak-probe/full relative discrepancy over the checkpoints (``NaN`` if not checked)."""
    checkpoints: tuple[int, ...] = field(default=())
    recomputed_with_full: bool = False
    weak_probe_tol: float = 0.01


def select_checkpoints(
    omega_p: Sequence[float], rng: np.random.Generator, n_random: int = 4
) -> tuple[int, ...]:
    """The largest-``Omega_p`` point plus up to ``n_random`` seeded random others. (TSD §14.6)"""
    n = len(omega_p)
    if n == 0:
        raise ValueError("empty sweep")
    first = int(np.argmax(np.asarray(omega_p)))
    rest = [i for i in range(n) if i != first]
    k = min(n_random, len(rest))
    extra = [rest[int(i)] for i in rng.choice(len(rest), size=k, replace=False)] if k else []
    return (first, *sorted(extra))


def evaluate_sweep(
    models: Sequence[LindbladRARModel],
    method: Method = "auto",
    *,
    weak_probe_tol: float = 0.01,
    rng: np.random.Generator | None = None,
    bandwidth: bool = True,
    linear_range: bool = False,
) -> SweepResult:
    """Evaluate a sweep with ``method`` in ``auto | weak_probe | full``. (FR-LIN-4, TSD §14.6)

    ``auto`` runs the weak-probe engine, checks it against the full engine at >= 5 checkpoints
    (all points if the sweep is shorter) and, if the max relative discrepancy exceeds
    ``weak_probe_tol``, recomputes the sweep with the full engine and logs a warning.
    """
    if method not in ("auto", "weak_probe", "full"):
        raise ValueError(f"method must be auto, weak_probe or full, got {method!r}")
    if not 0 < weak_probe_tol < 1:
        raise ValueError("weak_probe_tol must be in (0, 1)")

    def run(engine: Engine) -> tuple[RARModelResult, ...]:
        return tuple(
            m.evaluate(engine, bandwidth=bandwidth, linear_range=linear_range) for m in models
        )

    if method != "auto":
        return SweepResult(run(method), method, method, float("nan"), weak_probe_tol=weak_probe_tol)
    if rng is None:
        raise ValueError("method='auto' needs an explicit rng for the checkpoint draw")
    cps = select_checkpoints([m.drive.omega_p for m in models], rng)
    worst = max(models[i].discrepancy() for i in cps)
    if worst > weak_probe_tol:
        logger.warning(
            "weak-probe/full discrepancy %.3g exceeds weak_probe_tol %.3g: recomputing the sweep "
            "with the full Lindblad engine", worst, weak_probe_tol,
        )  # fmt: skip
        return SweepResult(run("full"), method, "full", worst, cps, True, weak_probe_tol)
    return SweepResult(run("weak_probe"), method, "weak_probe", worst, cps, False, weak_probe_tol)


def with_overrides(params: RARParams, **changes: Any) -> RARParams:
    """Copy of ``params`` with fields replaced (e.g. ``omega_c_rad_s``, ``e_lo_pk_v_per_m``)."""
    return replace(params, **changes)
