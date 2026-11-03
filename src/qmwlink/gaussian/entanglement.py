"""PT symplectic eigenvalue, entanglement measures and separability conditions.

(FR-GAU-4, FR-GAU-5, FR-GAU-6, TSD §5.2-5.4)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qmwlink.gaussian.channels import PIChannel
from qmwlink.gaussian.states import two_mode_ac

FloatArray = NDArray[np.float64]


def _f(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def nu_minus_general(V: ArrayLike) -> FloatArray:
    """Smallest PT symplectic eigenvalue of a two-mode CM, general formula. (FR-GAU-4, TSD §5.2)

    ``nu^2 = (D - sqrt(D^2 - 4 det V)) / 2`` with ``D = det A + det B - 2 det C``.
    """
    v = _f(V)
    if v.shape[-2:] != (4, 4):
        raise ValueError("V must have shape (..., 4, 4)")
    a, c, b = v[..., 0:2, 0:2], v[..., 0:2, 2:4], v[..., 2:4, 2:4]
    delta = np.linalg.det(a) + np.linalg.det(b) - 2.0 * np.linalg.det(c)
    disc = delta**2 - 4.0 * np.linalg.det(v)
    if np.any(disc < -1e-9 * np.maximum(delta**2, 1.0)):
        raise ValueError("negative discriminant: V is not a valid covariance matrix")
    nu2 = (delta - np.sqrt(np.maximum(disc, 0.0))) / 2.0
    return np.asarray(np.sqrt(np.maximum(nu2, 0.0)), dtype=np.float64)


def nu_minus_standard(
    a: ArrayLike,
    b: ArrayLike,
    c_prime: ArrayLike,
    *,
    stable: bool = True,
    det: ArrayLike | None = None,
) -> FloatArray:
    """Standard-form ``nu_-`` for ``[[aI, c'Z], [c'Z, bI]]``. (FR-GAU-4, TSD §5.2)

    ``stable=True`` uses ``2(ab - c'^2) / ((a+b) + sqrt((a-b)^2 + 4c'^2))``, algebraically
    identical to the direct form ``((a+b) - sqrt(...))/2`` but free of cancellation for
    ``a + b >> 1``.

    ``det`` optionally supplies ``ab - c'^2`` computed analytically; otherwise it is formed
    as a difference, which loses digits when ``ab ~ c'^2`` (e.g. a pure TMSV at large r,
    where ``cosh^2 - sinh^2 = 1``). For TMSV through ``(tau, y)`` it is ``a y + tau``.
    """
    aa, bb, cc = _f(a), _f(b), _f(c_prime)
    root = np.sqrt((aa - bb) ** 2 + 4.0 * cc**2)
    if stable:
        d = aa * bb - cc**2 if det is None else _f(det)
        return np.asarray(2.0 * d / ((aa + bb) + root), dtype=np.float64)
    return np.asarray(((aa + bb) - root) / 2.0, dtype=np.float64)


def standard_form_params(
    V: ArrayLike, atol: float = 1e-12
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Extract ``(a, b, c')`` from a standard-form CM; raise if not in standard form. (TSD §5.2)"""
    v = _f(V)
    a, b, cp = v[..., 0, 0], v[..., 2, 2], v[..., 0, 2]
    expected = np.zeros_like(v)
    eye = np.eye(2)
    z = np.diag([1.0, -1.0])
    expected[..., 0:2, 0:2] = a[..., None, None] * eye
    expected[..., 2:4, 2:4] = b[..., None, None] * eye
    expected[..., 0:2, 2:4] = cp[..., None, None] * z
    expected[..., 2:4, 0:2] = cp[..., None, None] * z
    if not np.allclose(v, expected, rtol=0.0, atol=atol):
        raise ValueError("V is not in standard form [[aI, c'Z], [c'Z, bI]]")
    return a, b, cp


def negativity(nu_minus: ArrayLike) -> FloatArray:
    """``max(0, (1 - nu)/(2 nu))``; infinite at ``nu = 0``. (FR-GAU-4, TSD §5.3)"""
    nu = _f(nu_minus)
    with np.errstate(divide="ignore"):
        return np.asarray(np.maximum(0.0, (1.0 - nu) / (2.0 * nu)), dtype=np.float64)


def log_negativity(nu_minus: ArrayLike) -> FloatArray:
    """``E_N = max(0, -log2 nu)``; infinite at ``nu = 0``. (FR-GAU-4, TSD §5.3)"""
    nu = _f(nu_minus)
    with np.errstate(divide="ignore"):
        return np.asarray(np.maximum(0.0, -np.log2(nu)), dtype=np.float64)


def teleportation_fidelity(nu_minus: ArrayLike) -> FloatArray:
    """``F = 1/(1 + nu_-)``. (FR-GAU-4, TSD §5.3 [L])"""
    return np.asarray(1.0 / (1.0 + _f(nu_minus)), dtype=np.float64)


def is_entangled_standard(a: ArrayLike, b: ArrayLike, c_prime: ArrayLike) -> NDArray[np.bool_]:
    """Standard form with ``a, b >= 1``: entangled iff ``(a-1)(b-1) < c'^2``. (TSD §5.4)"""
    aa, bb, cc = _f(a), _f(b), _f(c_prime)
    if np.any(aa < 1) or np.any(bb < 1):
        raise ValueError("a and b must be >= 1")
    return np.asarray((aa - 1.0) * (bb - 1.0) < cc**2)


def tmsv_through_channel_params(
    r: ArrayLike, channel: PIChannel
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(a, b, c')`` of TMSV(r) after ``channel`` on mode B: ``b = tau a + y``,
    ``c'^2 = tau (a^2 - 1)``. (FR-GAU-5, TSD §5.4)"""
    a, c = two_mode_ac(r)
    return a, channel.tau_arr * a + channel.y_arr, np.sqrt(channel.tau_arr) * c


def tmsv_is_entangled(channel: PIChannel) -> NDArray[np.bool_]:
    """TMSV (any ``r > 0``) through ``channel`` is entangled iff ``y < 1 + tau``.

    (FR-GAU-5, TSD §5.4)
    """
    return np.asarray(channel.y_arr < 1.0 + channel.tau_arr)


def tmst_is_entangled(r: ArrayLike, n_jpa: ArrayLike, channel: PIChannel) -> NDArray[np.bool_]:
    """TMST through ``channel``: entangled iff ``(a-1)(tau a + y - 1) < tau c^2``.

    (FR-GAU-6, TSD §5.4)
    """
    a, c = two_mode_ac(r, n_jpa)
    return np.asarray(
        (a - 1.0) * (channel.tau_arr * a + channel.y_arr - 1.0) < channel.tau_arr * c**2
    )
