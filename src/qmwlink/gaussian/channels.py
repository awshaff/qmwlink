"""Phase-insensitive Gaussian channels in (tau, y) form. (FR-GAU-2, FR-GAU-3, TSD §4)

Convention: vacuum variance 1, covariance ordering (x_A, p_A, x_B, p_B).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

PHYS_TOL = 1e-12
"""Absolute slack on the physicality check y >= |1 - tau| (rounding after composition)."""


def _as_float(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class PIChannel:
    """Channel ``V -> tau V + y I`` on one mode, displacement ``d -> sqrt(tau) d``.

    ``tau`` and ``y`` may be arrays (broadcast together), e.g. over a transmissivity axis;
    after construction both are ``float64`` ndarrays.
    Physicality requires ``tau >= 0`` and ``y >= |1 - tau|``. (FR-GAU-2, TSD §4)
    """

    tau: ArrayLike
    y: ArrayLike

    def __post_init__(self) -> None:
        tau, y = np.broadcast_arrays(_as_float(self.tau), _as_float(self.y))
        if not (np.all(np.isfinite(tau)) and np.all(np.isfinite(y))):
            raise ValueError("tau and y must be finite")
        if np.any(tau < 0):
            raise ValueError("tau must be >= 0")
        if np.any(y < np.abs(1.0 - tau) - PHYS_TOL):
            raise ValueError("unphysical channel: y must satisfy y >= |1 - tau|")
        object.__setattr__(self, "tau", tau.copy())
        object.__setattr__(self, "y", y.copy())

    # ---- constructors (TSD §4.1) -------------------------------------------------------
    @classmethod
    def identity(cls) -> PIChannel:
        """Identity channel (tau, y) = (1, 0). (FR-GAU-2, TSD §4.1)"""
        return cls(np.float64(1.0), np.float64(0.0))

    @classmethod
    def thermal_attenuator(
        cls,
        eta: ArrayLike | None = None,
        n_bar: ArrayLike = 0.0,
        *,
        one_minus_eta: ArrayLike | None = None,
    ) -> PIChannel:
        """Thermal attenuator ``(eta, (1 - eta)(2 n_bar + 1))``. (FR-GAU-2, TSD §4.1)

        Give exactly one of ``eta`` or ``one_minus_eta``; the latter keeps full precision
        when eta -> 1 (TSD §13 rule 3).
        """
        if (eta is None) == (one_minus_eta is None):
            raise ValueError("give exactly one of eta or one_minus_eta")
        n = _as_float(n_bar)
        if np.any(n < 0):
            raise ValueError("n_bar must be >= 0")
        if one_minus_eta is None:
            e = _as_float(eta)  # type: ignore[arg-type]
            if np.any((e < 0) | (e > 1)):
                raise ValueError("eta must be in [0, 1]")
            om = 1.0 - e
        else:
            om = _as_float(one_minus_eta)
            if np.any((om < 0) | (om > 1)):
                raise ValueError("one_minus_eta must be in [0, 1]")
            e = 1.0 - om
        return cls(e, om * (2.0 * n + 1.0))

    @classmethod
    def additive_noise(cls, n_add: ArrayLike) -> PIChannel:
        """Additive noise of ``N_add`` quanta, ``(1, 2 N_add)``. (FR-GAU-2, TSD §4.1)"""
        n = _as_float(n_add)
        if np.any(n < 0):
            raise ValueError("n_add must be >= 0")
        return cls(np.float64(1.0), 2.0 * n)

    @classmethod
    def amplifier(cls, gain: ArrayLike, n_bar: ArrayLike = 0.0) -> PIChannel:
        """Amplifier of gain ``G >= 1`` with idler occupation n_bar, ``(G, (G-1)(2 n_bar+1))``.

        (FR-GAU-2, TSD §4.1)
        """
        g = _as_float(gain)
        n = _as_float(n_bar)
        if np.any(g < 1):
            raise ValueError("gain must be >= 1")
        if np.any(n < 0):
            raise ValueError("n_bar must be >= 0")
        return cls(g, (g - 1.0) * (2.0 * n + 1.0))

    # ---- algebra -----------------------------------------------------------------------
    @property
    def tau_arr(self) -> FloatArray:
        return np.asarray(self.tau, dtype=np.float64)

    @property
    def y_arr(self) -> FloatArray:
        return np.asarray(self.y, dtype=np.float64)

    def __matmul__(self, other: PIChannel) -> PIChannel:
        """``B @ A`` applies A first, then B: ``(tau_B tau_A, tau_B y_A + y_B)``. (TSD §4.2)"""
        if not isinstance(other, PIChannel):
            return NotImplemented
        return PIChannel(self.tau_arr * other.tau_arr, self.tau_arr * other.y_arr + self.y_arr)

    @property
    def n_out(self) -> FloatArray:
        """Output noise photons for vacuum input, ``(tau + y - 1)/2``. (FR-GAU-2, TSD §4.4)"""
        return np.asarray((self.tau_arr + self.y_arr - 1.0) / 2.0, dtype=np.float64)

    @property
    def is_entanglement_breaking(self) -> NDArray[np.bool_]:
        """``y >= 1 + tau`` (Holevo 2008). (FR-GAU-5, TSD §5.4)"""
        return np.asarray(self.y_arr >= 1.0 + self.tau_arr)


def attenuator_cascade_n_eff(
    eta1: ArrayLike, n1: ArrayLike, eta2: ArrayLike, n2: ArrayLike
) -> FloatArray:
    """Effective occupation of attenuator ``(eta1, n1)`` followed by ``(eta2, n2)``.

    ``n_eff = [eta2 (1-eta1) n1 + (1-eta2) n2] / (1 - eta1 eta2)``; the cascade is an
    attenuator with ``eta = eta1 eta2``. (FR-GAU-2, TSD §4.3 [L])
    """
    e1, e2, a, b = (_as_float(v) for v in (eta1, eta2, n1, n2))
    if np.any((e1 < 0) | (e1 > 1) | (e2 < 0) | (e2 > 1)):
        raise ValueError("eta1 and eta2 must be in [0, 1]")
    denom = 1.0 - e1 * e2
    if np.any(denom <= 0):
        raise ValueError("eta1 * eta2 must be < 1 for n_eff to be defined")
    return np.asarray((e2 * (1.0 - e1) * a + (1.0 - e2) * b) / denom, dtype=np.float64)


def apply_to_mode_b(V: ArrayLike, channel: PIChannel) -> FloatArray:
    """Apply ``channel`` to mode B of a two-mode CM ``[[A, C], [C^T, B]]``. (FR-GAU-3, TSD §4.5)

    Result blocks: ``[[A, sqrt(tau) C], [sqrt(tau) C^T, tau B + y I]]``. ``V`` has shape
    ``(..., 4, 4)``; batch axes broadcast against the channel's axes.
    """
    v = _as_float(V)
    if v.shape[-2:] != (4, 4):
        raise ValueError("V must have shape (..., 4, 4)")
    batch = np.broadcast_shapes(v.shape[:-2], channel.tau_arr.shape)
    out = np.array(np.broadcast_to(v, (*batch, 4, 4)))
    tau = np.broadcast_to(channel.tau_arr, batch)[..., None, None]
    y = np.broadcast_to(channel.y_arr, batch)[..., None, None]
    c = v[..., 0:2, 2:4]
    b = v[..., 2:4, 2:4]
    out[..., 0:2, 2:4] = np.sqrt(tau) * c
    out[..., 2:4, 0:2] = np.sqrt(tau) * np.swapaxes(c, -1, -2)
    out[..., 2:4, 2:4] = tau * b + y * np.eye(2)
    return out
