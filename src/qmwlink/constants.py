"""Physical constants (SI, CODATA via ``scipy.constants``). (FR-PHY-1, TSD §3.1)"""

from __future__ import annotations

from scipy import constants as _sc

H: float = float(_sc.h)
"""Planck constant [J s]."""

K_B: float = float(_sc.k)
"""Boltzmann constant [J/K]."""

C: float = float(_sc.c)
"""Speed of light in vacuum [m/s]."""

MU0: float = float(_sc.mu_0)
"""Vacuum permeability [H/m]."""

EPS0: float = float(_sc.epsilon_0)
"""Vacuum permittivity [F/m]."""

ETA0: float = (MU0 / EPS0) ** 0.5
"""Free-space impedance sqrt(mu0/eps0) [ohm]."""
