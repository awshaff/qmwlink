"""Full Lindblad engine: Liouvillian ``A + v B``, batched steady state. (FR-LIN-3, TSD §14.5)

Row-major vectorization, ``vec(A X B) = (A kron B^T) vec(X)``, so ``vec(rho)[i n + j] = rho_ij``.
The Liouvillian is affine in velocity; ``A`` and ``B`` are built once per parameter point and the
batch over velocity classes is formed by broadcasting. Written independently of ``weak_probe.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qmwlink.receivers.rydberg_physics.scheme import Drive, LadderScheme

ComplexArray = NDArray[np.complex128]

_DOPPLER_SCALE = 1e-6


def hamiltonian(scheme: LadderScheme, drive: Drive) -> ComplexArray:
    """RWA Hamiltonian at ``v = 0`` (``hbar = 1``, rad/us). (TSD §14.3)"""
    n = scheme.n_levels
    h = np.zeros((n, n), dtype=np.complex128)
    h[1, 1] = -drive.delta_p
    h[2, 2] = -(drive.delta_p + drive.delta_c)
    h[1, 0] = h[0, 1] = 0.5 * drive.omega_p
    h[2, 1] = h[1, 2] = 0.5 * drive.omega_c
    if n == 4:
        h[3, 3] = -(drive.delta_p + drive.delta_c + drive.delta_m)
        h[3, 2] = h[2, 3] = 0.5 * drive.omega_m
    return h


def doppler_hamiltonian(scheme: LadderScheme, drive: Drive) -> ComplexArray:
    """``H_v`` per unit velocity: ``1e-6 (k_p|2><2| + (k_p - k_c)(|3><3| + |4><4|))``."""
    n = scheme.n_levels
    hv = np.zeros((n, n), dtype=np.complex128)
    hv[1, 1] = _DOPPLER_SCALE * drive.k_p
    for j in range(2, n):
        hv[j, j] = _DOPPLER_SCALE * (drive.k_p - drive.k_c)
    return hv


def _commutator_super(h: ComplexArray) -> ComplexArray:
    eye = np.eye(h.shape[0])
    return -1j * (np.kron(h, eye) - np.kron(eye, h.T))


@dataclass(frozen=True)
class Liouvillian:
    """``L(v) = A + v B`` and the microwave derivative ``L_1 = dL/dOmega_m``. (TSD §14.5)"""

    a: ComplexArray
    b: ComplexArray
    l1: ComplexArray | None
    n_levels: int

    @property
    def dim(self) -> int:
        return self.n_levels**2

    def at(self, v: ArrayLike) -> ComplexArray:
        """Batch ``L(v)``, shape ``v.shape + (dim, dim)``."""
        vv = np.asarray(v, dtype=np.float64)
        return np.asarray(self.a + vv[..., None, None] * self.b, dtype=np.complex128)


def build(scheme: LadderScheme, drive: Drive) -> Liouvillian:
    """Build ``A``, ``B`` and ``L_1`` once for a parameter point. (TSD §14.5)"""
    n = scheme.n_levels
    eye = np.eye(n)
    a = _commutator_super(hamiltonian(scheme, drive))
    for op in scheme.collapse_operators():
        ldl = op.conj().T @ op
        a = a + np.kron(op, op.conj()) - 0.5 * np.kron(ldl, eye) - 0.5 * np.kron(eye, ldl.T)
    b = _commutator_super(doppler_hamiltonian(scheme, drive))
    l1 = None
    if n == 4:
        h1 = np.zeros((n, n), dtype=np.complex128)
        h1[3, 2] = h1[2, 3] = 0.5
        l1 = _commutator_super(h1)
    return Liouvillian(a, b, l1, n)


def _bordered(lv: Liouvillian, v: ArrayLike) -> ComplexArray:
    """``L(v)`` with row 0 replaced by ``vec(I)^T`` (trace condition)."""
    n = lv.n_levels
    a = lv.a.copy()
    b = lv.b.copy()
    a[0, :] = np.eye(n).reshape(-1)
    b[0, :] = 0.0
    vv = np.asarray(v, dtype=np.float64)
    return np.asarray(a + vv[..., None, None] * b, dtype=np.complex128)


def steady_state(lv: Liouvillian, v: ArrayLike) -> ComplexArray:
    """Steady-state ``vec(rho)`` per velocity class, shape ``v.shape + (dim,)``. (TSD §14.5)

    Bordered system: row 0 of ``L`` replaced by the trace row, right-hand side ``e_0``; one batched
    ``numpy.linalg.solve`` over ``v``.
    """
    mats = _bordered(lv, v)
    rhs = np.zeros((*mats.shape[:-1], 1), dtype=np.complex128)
    rhs[..., 0, 0] = 1.0
    return np.asarray(np.linalg.solve(mats, rhs)[..., 0], dtype=np.complex128)


def density_matrix(vec_rho: ComplexArray, n_levels: int) -> ComplexArray:
    """Reshape ``vec(rho)`` (row-major) to ``(..., n, n)``."""
    return vec_rho.reshape((*vec_rho.shape[:-1], n_levels, n_levels))


def rho21(lv: Liouvillian, vec_rho: ComplexArray) -> ComplexArray:
    """``rho_21 = <2|rho|1>`` from ``vec(rho)``."""
    return np.asarray(vec_rho[..., lv.n_levels], dtype=np.complex128)


def transfer_function(
    lv: Liouvillian, v: ArrayLike, omega: float, rho0: ComplexArray | None = None
) -> ComplexArray:
    """``d rho_21/d Omega_m`` at angular frequency ``omega`` [rad/us], per velocity class.

    ``delta rho(s) = (s I - L_0)^{-1} L_1 rho_0`` at ``s = i omega``; at ``omega = 0`` the bordered
    system (trace row, zero right-hand side entry) is used. ``rho0`` may pass the steady state of
    the same ``v`` to avoid solving for it again. (TSD §14.5)
    """
    if lv.l1 is None:
        raise ValueError("the microwave transfer function needs n_levels = 4")
    if rho0 is None:
        rho0 = steady_state(lv, v)
    rhs = np.einsum("ij,...j->...i", lv.l1, rho0)[..., None]
    if omega == 0.0:
        mats = -_bordered(lv, v)
        rhs[..., 0, 0] = 0.0
    else:
        mats = 1j * omega * np.eye(lv.dim) - lv.at(v)
    return np.asarray(np.linalg.solve(mats, rhs)[..., lv.n_levels, 0], dtype=np.complex128)
