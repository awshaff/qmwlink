"""Entanglement service over the compound PDT. (FR-SVC-E1, FR-SVC-E2, TSD §8.1)

Chain per realization: fading attenuator ``(eta, n_th)`` then front end ``(tau_r, y_r)``,
``tau = tau_r eta``, ``y = tau_r (1 - eta)(2 n_th + 1) + y_r``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from qmwlink.channel.compound import CompoundPDT
from qmwlink.channel.large_scale import DeterministicLoss
from qmwlink.gaussian.channels import PIChannel
from qmwlink.gaussian.entanglement import (
    fidelity_opt,
    log_negativity,
    nu_minus_standard,
)
from qmwlink.gaussian.states import two_mode_ac
from qmwlink.numerics.integrate import checked_quad

FloatArray = NDArray[np.float64]

GL_NODES = 32
"""Gauss-Legendre nodes per inner panel."""
_INNER_QUANTILES = (1e-12, 1e-8, 1e-4, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0 - 1e-4)
ETA_FLOOR = 1e-12
"""Lower clip for ``eta*`` when it is 0 (all realizations entangled)."""


# ---- E1: receiver-referred threshold ---------------------------------------------------------
def chain_channel(eta: ArrayLike, n_th: float, front_end: PIChannel) -> PIChannel:
    """Fading attenuator then ``front_end``: ``(tau_r eta, tau_r (1-eta)(2 n_th+1) + y_r)``."""
    e = np.asarray(eta, dtype=np.float64)
    return PIChannel(
        front_end.tau_arr * e, front_end.tau_arr * (1.0 - e) * (2.0 * n_th + 1.0) + front_end.y_arr
    )


def window_rx(n_th: float, front_end: PIChannel) -> FloatArray:
    """Window ``1 - eta*_rx = (1 + tau_r - y_r)/(2 tau_r (n_th + 1))`` (cancellation-free).

    Entangled iff ``eta > eta*_rx``; a window exists iff this is ``> 0``. ``tau_r = 0`` has no
    window (returned as ``-inf``). (TSD §8.1)
    """
    if n_th < 0:
        raise ValueError("n_th must be >= 0")
    t, y = front_end.tau_arr, front_end.y_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (1.0 + t - y) / (2.0 * t * (n_th + 1.0))
    return np.asarray(np.where(t > 0, w, -np.inf), dtype=np.float64)


def eta_star_rx(n_th: float, front_end: PIChannel) -> FloatArray:
    """TMSV receiver-referred threshold ``(tau_r(2n+1) + y_r - 1)/(2 tau_r (n+1))``. (FR-SVC-E1)

    Not clipped: values ``>= 1`` mean "no window" (``+inf`` for ``tau_r = 0``).
    """
    return np.asarray(1.0 - window_rx(n_th, front_end), dtype=np.float64)


def has_window(n_th: float, front_end: PIChannel) -> NDArray[np.bool_]:
    """True iff some ``eta < 1`` is entangled (``eta*_rx < 1``)."""
    return np.asarray(window_rx(n_th, front_end) > 0)


def eta_star_numeric(n_th: float, front_end: PIChannel, r: float, n_jpa: float = 0.0) -> float:
    """Threshold from the §5.4 condition by ``brentq`` in ``omega = 1 - eta`` (TMSV or TMST).

    Returns ``nan`` when even ``eta = 1`` is separable (no window) and ``0`` when every ``eta``
    is entangled. (FR-SVC-E1, TSD §8.1)
    """
    if front_end.tau_arr.size != 1:
        raise ValueError("eta_star_numeric needs a scalar front end")
    a, c = (float(v) for v in two_mode_ac(r, n_jpa))
    tr, yr = float(front_end.tau_arr), float(front_end.y_arr)

    def margin(omega: float) -> float:  # > 0  <=>  entangled
        tau = tr * (1.0 - omega)
        y = tr * omega * (2.0 * n_th + 1.0) + yr
        return tau * c * c - (a - 1.0) * (tau * a + y - 1.0)

    if margin(0.0) <= 0.0:
        return float("nan")
    if margin(1.0) > 0.0:
        return 0.0
    omega = brentq(margin, 0.0, 1.0, xtol=1e-14, rtol=4 * np.finfo(float).eps)
    return 1.0 - float(omega)


# ---- E2: metrics over the compound PDT ------------------------------------------------------
NO_ENTANGLED_REASON = "no entangled realizations"
ONE_MINUS_P_OUT_MIN = 1e-12
"""Below this ``1 - P_out`` the conditional averages are ``NaN`` (TSD §8.1, Q-8)."""


@dataclass(frozen=True)
class EntanglementMetrics:
    """Entanglement metrics over a compound PDT. (FR-SVC-E2, TSD §8.1, Q-8)

    ``log_neg`` is the unconditional mean ``E[E_N]``. ``nu_minus_cond``, ``fidelity_opt_cond``
    and ``fidelity_bk_cond`` are conditional on the link working,
    ``E[g | eta > eta*_rx] = (1/(1-P_out)) int_{eta*}^1 g f``; they are ``NaN`` with
    ``reason`` when ``1 - P_out < 1e-12``. ``one_minus_p_out`` is always reported next to them.
    The bare restricted integrals ``e_*`` are internal (never reported).
    """

    eta_star: float
    p_out: float
    one_minus_p_out: float
    log_neg: float
    log_neg_cond: float
    nu_minus_cond: float
    fidelity_opt_cond: float
    fidelity_bk_cond: float
    reason: str
    e_nu_minus: float
    e_fidelity_opt: float
    e_fidelity_bk: float


@dataclass(frozen=True)
class TruncationEffect:
    """RQ1: truncated vs untruncated-surrogate outage next to the removed mass. (TSD §8.1)"""

    p_out: float
    p_out_surr: float
    delta: float
    ratio: float
    removed_mass: float
    physical_surrogate: bool = field(default=False)


@dataclass(frozen=True)
class EntanglementService:
    """TMSV (``n_jpa = 0``) or TMST source through fading + front end over a compound PDT."""

    compound: CompoundPDT
    n_th: float
    front_end: PIChannel
    r: float
    n_jpa: float = 0.0

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError("r must be > 0")
        if self.n_jpa < 0 or self.n_th < 0:
            raise ValueError("n_jpa and n_th must be >= 0")
        if self.front_end.tau_arr.size != 1:
            raise ValueError("front_end must be a scalar PIChannel")

    # ---- threshold --------------------------------------------------------------------------
    @property
    def eta_star(self) -> float:
        """Closed form for TMSV, numeric root for TMST; ``>= 1`` (or nan) means no window."""
        if self.n_jpa == 0.0:
            return float(eta_star_rx(self.n_th, self.front_end))
        v = eta_star_numeric(self.n_th, self.front_end, self.r, self.n_jpa)
        return 1.0 if np.isnan(v) else v

    def p_out(self) -> float:
        """``P_out = F(eta*_rx)`` (1 if no window, 0 if every ``eta`` is entangled)."""
        es = self.eta_star
        if es >= 1.0:
            return 1.0
        if es <= 0.0:
            return 0.0
        return float(self.compound.cdf(es))

    def truncation_effect(self) -> TruncationEffect:
        """``Delta P_out`` and ``P_out/P_out^surr`` next to ``1 - Z_{K,N}``. (RQ1, E1)"""
        es = self.eta_star
        p = self.p_out()
        degenerate = es >= 1.0 or es <= 0.0
        ps = p if degenerate else float(self.compound.cdf_surrogate(es))
        ratio = p / ps if ps > 0 else float("nan")
        return TruncationEffect(p, ps, p - ps, ratio, 1.0 - self.compound.small.z)

    # ---- per-realization observables ----------------------------------------------------------
    def nu_minus(self, eta: ArrayLike) -> FloatArray:
        """``nu~_-(eta)`` for the source through the chain (cancellation-free det)."""
        e = np.asarray(eta, dtype=np.float64)
        ch = chain_channel(e, self.n_th, self.front_end)
        a, c = two_mode_ac(self.r, self.n_jpa)
        tau, y = ch.tau_arr, ch.y_arr
        det = a * y + tau * (2.0 * self.n_jpa + 1.0) ** 2  # a y + tau (a^2 - c^2)
        return nu_minus_standard(a, tau * a + y, np.sqrt(tau) * c, det=det)

    def fidelity_bk_at(self, eta: ArrayLike) -> FloatArray:
        """Braunstein-Kimble fidelity at ``eta`` (standard form: ``Vx- = Vp+ = a + b - 2c'``)."""
        e = np.asarray(eta, dtype=np.float64)
        ch = chain_channel(e, self.n_th, self.front_end)
        a, c = two_mode_ac(self.r, self.n_jpa)
        v = a + (ch.tau_arr * a + ch.y_arr) - 2.0 * np.sqrt(ch.tau_arr) * c
        return np.asarray(1.0 / (1.0 + v / 2.0), dtype=np.float64)

    def one_minus_p_out(self) -> float:
        """``1 - P_out`` from the survival function (no subtraction near 1)."""
        es = self.eta_star
        if es >= 1.0:
            return 0.0
        if es <= 0.0:
            return 1.0
        return float(self.compound.sf(es))

    def metrics(self) -> EntanglementMetrics:
        """All E2 metrics (nested quadrature in Psi-space; see :meth:`expect_entangled`)."""
        es = self.eta_star
        p = self.p_out()
        q = self.one_minus_p_out()
        nan = float("nan")
        if es >= 1.0 or q < ONE_MINUS_P_OUT_MIN:
            return EntanglementMetrics(
                es, p, q, 0.0, nan, nan, nan, nan, NO_ENTANGLED_REASON, nan, nan, nan
            )
        en = self.expect_entangled(lambda e: log_negativity(self.nu_minus(e)))
        nu = self.expect_entangled(self.nu_minus)
        fo = self.expect_entangled(lambda e: fidelity_opt(self.nu_minus(e)))
        fb = self.expect_entangled(self.fidelity_bk_at)
        return EntanglementMetrics(es, p, q, en, en / q, nu / q, fo / q, fb / q, "", nu, fo, fb)

    # ---- nested quadrature -------------------------------------------------------------------
    def expect_entangled(self, g: Callable[[FloatArray], FloatArray]) -> float:
        """``int_{eta*}^1 g(eta) f(eta) d eta`` as ``int f_Psi int_{eta*/eta_l}^1 g f_s``.

        Outer ``checked_quad`` over ``Psi`` in ``[Psi_lo, -ln eta*]`` (breakpoints from the
        law); inner vectorized Gauss-Legendre panels split at small-scale quantiles.
        ``g`` must be a vectorized function of ``eta``.
        """
        es = max(self.eta_star, ETA_FLOOR)
        if es >= 1.0:
            return 0.0
        small, large = self.compound.small, self.compound.large
        q_pts = small.ppf(np.array(_INNER_QUANTILES))
        xg, wg = np.polynomial.legendre.leggauss(GL_NODES)

        def inner(eta_l: float) -> float:
            lo = es / eta_l
            if lo >= 1.0:
                return 0.0
            edges = np.unique(np.concatenate(([lo], q_pts[(q_pts > lo) & (q_pts < 1.0)], [1.0])))
            a, b = edges[:-1, None], edges[1:, None]
            x = (0.5 * (b - a) * xg[None, :] + 0.5 * (a + b)).ravel()
            w = (0.5 * (b - a) * wg[None, :]).ravel()
            return float(np.sum(w * g(eta_l * x) * small.pdf(x)))

        if isinstance(large, DeterministicLoss):
            return inner(float(np.exp(-large.psi)))
        lo_psi = large.psi_lo_eff()
        hi_psi = -float(np.log(es))
        if hi_psi <= lo_psi:
            return 0.0
        return checked_quad(
            lambda p: inner(float(np.exp(-p))) * float(large.psi_pdf(p)),
            lo_psi,
            hi_psi,
            points=large.psi_breakpoints(),
        )[0]
