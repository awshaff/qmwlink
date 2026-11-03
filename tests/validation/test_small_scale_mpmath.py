"""Small-scale cdf/sf vs mpmath at large N, down to 1e-8 tails. (TSD §6.1, NFR-2)"""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

import tolerances as tol
from mp_ref import ncx2_cdf_mp
from qmwlink.channel.small_scale import SmallScale

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(("n", "K"), [(10**5, 10.0), (10**7, 0.0), (10**7, 1.0)])
def test_cdf_sf_vs_mpmath(n: int, K: float) -> None:
    """3 points per function for N in {1e5, 1e7}: 1% relative, including a ~1e-8 lower tail."""
    s = SmallScale(K, 0.9999, n)
    nu, lam = s.nu, s.nc
    mean, sd = nu + lam, float(np.sqrt(2.0 * (nu + 2.0 * lam)))
    f_xmax = ncx2_cdf_mp(s.x_max, nu, lam)
    # lower tail (~2e-8), the bulk, and a point just below the truncation edge (small sf)
    for x_sur in (mean - 5.5 * sd, mean, s.x_max - 1e-3 * sd):
        eta = x_sur * s.sigma2
        f_x = ncx2_cdf_mp(x_sur, nu, lam)
        cdf_ref = float(f_x / f_xmax)
        sf_ref = float((f_xmax - f_x) / f_xmax)
        assert float(s.cdf(eta)) == pytest.approx(cdf_ref, rel=tol.RTOL_MPMATH_TAIL)
        assert float(s.sf(eta)) == pytest.approx(sf_ref, rel=tol.RTOL_MPMATH_TAIL)
    mp.mp.dps = 15


def test_lower_tail_1e8_accuracy_k0_large_n() -> None:
    """Regression for Q-3: central chi2.cdf was off by 2.3% here; the ncx2 path must not be."""
    n = 10**7
    s = SmallScale(0.0, 0.9999, n)
    sd = float(np.sqrt(2.0 * s.nu))
    x_sur = s.nu - 5.5 * sd
    ref = float(ncx2_cdf_mp(x_sur, s.nu, 0.0) / ncx2_cdf_mp(s.x_max, s.nu, 0.0))
    assert float(s.cdf(x_sur * s.sigma2)) == pytest.approx(ref, rel=1e-6)
    assert ref < 1e-7
