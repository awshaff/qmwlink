"""Coherent transducer front end. (FR-RX-5, TSD §7.5)"""

from __future__ import annotations

from qmwlink.gaussian.channels import PIChannel


def coherent_transducer(eta_t: float, n_t: float = 0.0, n_add: float = 0.0) -> PIChannel:
    """Attenuator ``(eta_t, n_t)`` followed by additive noise ``N_t``.

    ``tau_r = eta_t``, ``y_r = (1 - eta_t)(2 n_t + 1) + 2 N_t``. Parameter values belong in
    config with a ``source:``.
    """
    if not (0.0 <= eta_t <= 1.0):
        raise ValueError("eta_t must be in [0, 1]")
    if n_t < 0 or n_add < 0:
        raise ValueError("n_t and n_add must be >= 0")
    return PIChannel(eta_t, (1.0 - eta_t) * (2.0 * n_t + 1.0) + 2.0 * n_add)
