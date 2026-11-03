"""Regression against TSD §11 reference table, T = 300 K. (FR-PHY-1, FR-PHY-2)"""

from __future__ import annotations

import pytest

import tolerances as tol
from qmwlink.thermal import eta_star, n_th, one_minus_eta_star

# f [GHz], n_th, eta*, 1 - eta*   (TSD §11)
TABLE = [
    (5.0, 1249.697, 0.99920, 7.996e-4),
    (30.0, 207.867, 0.99521, 4.788e-3),
    (100.0, 62.011, 0.98413, 1.587e-2),
    (300.0, 20.341, 0.95314, 4.686e-2),
]


@pytest.mark.parametrize(("f_ghz", "n_ref", "eta_ref", "win_ref"), TABLE)
def test_tsd_section_11_table(f_ghz: float, n_ref: float, eta_ref: float, win_ref: float) -> None:
    f = f_ghz * 1e9
    assert float(n_th(f, 300.0)) == pytest.approx(n_ref, abs=tol.ABS_N_TH_5GHZ)
    assert float(eta_star(f, 300.0)) == pytest.approx(eta_ref, abs=tol.ABS_ETA_STAR)
    assert float(one_minus_eta_star(f, 300.0)) == pytest.approx(
        win_ref, rel=tol.REL_ONE_MINUS_ETA_STAR
    )
