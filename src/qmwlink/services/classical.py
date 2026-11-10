"""Classical service: per-realization capacities, ergodic/outage capacity, throughput.

(FR-SVC-C1..C3, TSD §8.2.) Coherent-state Gaussian encoding with ``N_s`` photons per mode.
All capacities are in bit per mode and increasing in ``eta``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from qmwlink.channel.compound import CompoundPDT
from qmwlink.channel.large_scale import DeterministicLoss
from qmwlink.constants import H
from qmwlink.numerics.integrate import checked_quad
from qmwlink.receivers.base import MeasuringReceiver

FloatArray = NDArray[np.float64]

GL_NODES = 32
_INNER_QUANTILES = (1e-12, 1e-8, 1e-4, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0 - 1e-4)
_PSI_TAIL = 1e-16
"""Upper Psi limit is where the large-scale survival function falls below this."""


def _f(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def g_entropy(x: ArrayLike) -> FloatArray:
    """Bosonic entropy ``g(x) = (x+1) log2(x+1) - x log2 x``, ``g(0) = 0``, evaluated as
    ``log2(1+x) + x log2(1+1/x)``. (TSD §8.2)"""
    v = _f(x)
    if np.any(v < 0):
        raise ValueError("g(x) needs x >= 0")
    # g = log2(1+x) + x log2(1+1/x): no catastrophic cancellation at large x (TSD §13 rule).
    safe = np.where(v > 0, v, 1.0)
    tail = np.where(v > 0, v * np.log1p(1.0 / safe), 0.0)
    return np.asarray((np.log1p(v) + tail) / np.log(2.0), dtype=np.float64)


def n_out_fading(eta: ArrayLike, n_th: float, one_minus_eta: ArrayLike | None = None) -> FloatArray:
    """Output noise ``N_out = (1 - eta) n_th`` of the fading channel alone."""
    e = _f(eta)
    om = 1.0 - e if one_minus_eta is None else _f(one_minus_eta)
    return np.asarray(om * n_th, dtype=np.float64)


def c_holevo(eta: ArrayLike, n_s: float, n_th: float) -> FloatArray:
    """``g(eta N_s + N_out) - g(N_out)`` with ``N_out = (1-eta) n_th``."""
    e = _f(eta)
    n_out = n_out_fading(e, n_th)
    return np.asarray(g_entropy(e * n_s + n_out) - g_entropy(n_out), dtype=np.float64)


def c_het(eta: ArrayLike, n_s: float, n_th: float) -> FloatArray:
    """Ideal SSB heterodyne ``log2(1 + eta N_s/(N_out + 1))``."""
    e = _f(eta)
    return np.asarray(np.log2(1.0 + e * n_s / (n_out_fading(e, n_th) + 1.0)), dtype=np.float64)


def c_hom(eta: ArrayLike, n_s: float, n_th: float) -> FloatArray:
    """Ideal homodyne ``(1/2) log2(1 + 4 eta N_s/(2 N_out + 1))``."""
    e = _f(eta)
    return np.asarray(
        0.5 * np.log2(1.0 + 4.0 * e * n_s / (2.0 * n_out_fading(e, n_th) + 1.0)),
        dtype=np.float64,
    )


def c_rx(eta: ArrayLike, n_s: float, n_th: float, rx: MeasuringReceiver) -> FloatArray:
    """Measuring receiver ``log2(1 + eta N_s / D)``, ``D = rx.noise_photons`` (TSD §8.2)."""
    e = _f(eta)
    return np.asarray(np.log2(1.0 + e * n_s / rx.noise_photons(e, n_th)), dtype=np.float64)


def tx_photons_per_mode(p_t_w: float, f_hz: float, bandwidth_hz: float) -> float:
    """``N_s = P_t/(h f B)``. (FR-SVC-C3, TSD §8.2)"""
    if p_t_w < 0 or f_hz <= 0 or bandwidth_hz <= 0:
        raise ValueError("need P_t >= 0, f > 0, B > 0")
    return float(p_t_w / (H * f_hz * bandwidth_hz))


def throughput_bps(capacity_bit_per_mode: ArrayLike, bandwidth_hz: float) -> FloatArray:
    """``R_b = B C`` [bit/s]. (FR-SVC-C3)"""
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be > 0")
    return np.asarray(bandwidth_hz * _f(capacity_bit_per_mode), dtype=np.float64)


@dataclass(frozen=True)
class ClassicalService:
    """Capacity metrics of ``cap(eta)`` (vectorized, increasing) over the compound PDT.

    ``cap`` maps ``eta`` to bit per mode. (FR-SVC-C2)
    """

    compound: CompoundPDT
    cap: Callable[[FloatArray], FloatArray]

    def ergodic(self) -> float:
        """``E[C(eta)]`` by nested quadrature (outer ``checked_quad`` over Psi)."""
        small, large = self.compound.small, self.compound.large
        q_pts = small.ppf(np.array(_INNER_QUANTILES))
        xg, wg = np.polynomial.legendre.leggauss(GL_NODES)

        def inner(eta_l: float) -> float:
            edges = np.unique(np.concatenate(([0.0], q_pts[(q_pts > 0) & (q_pts < 1.0)], [1.0])))
            a, b = edges[:-1, None], edges[1:, None]
            x = (0.5 * (b - a) * xg[None, :] + 0.5 * (a + b)).ravel()
            w = (0.5 * (b - a) * wg[None, :]).ravel()
            return float(np.sum(w * self.cap(eta_l * x) * small.pdf(x)))

        if isinstance(large, DeterministicLoss):
            return inner(float(np.exp(-large.psi)))
        lo = large.psi_lo_eff()
        hi = lo + 1.0
        while float(large.psi_sf(hi)) > _PSI_TAIL and hi < lo + 1e4:
            hi = lo + 2.0 * (hi - lo)
        return checked_quad(
            lambda p: inner(float(np.exp(-p))) * float(large.psi_pdf(p)),
            lo,
            hi,
            points=[p for p in large.psi_breakpoints() if lo < p < hi],
        )[0]

    def outage_capacity(self, eps: float) -> float:
        """``C_eps = C(F^{-1}(eps))``. (FR-SVC-C2)"""
        return float(self.cap(self.compound.ppf(np.float64(eps))))

    def rate_outage(self, rate: float) -> float:
        """``P(C < R) = F(C^{-1}(R))`` with ``C^{-1}`` by ``brentq``; 1 if ``R >= C(1)``,
        0 if ``R <= 0``."""
        if rate <= 0.0:
            return 0.0
        if rate >= float(self.cap(np.ones(1))[0]):
            return 1.0
        eta = brentq(
            lambda e: float(self.cap(np.full(1, e))[0]) - rate,
            0.0, 1.0, xtol=1e-14, rtol=4 * np.finfo(float).eps,
        )  # fmt: skip
        return float(self.compound.cdf(eta))

    def throughput(self, bandwidth_hz: float) -> float:
        """Ergodic ``R_b = B E[C]`` [bit/s]. (FR-SVC-C3)"""
        return float(throughput_bps(self.ergodic(), bandwidth_hz))
