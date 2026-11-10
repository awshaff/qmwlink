"""Pinned scipy guard for the ncx2 floor helper. (FR-CH-1 (b), TSD §6.1, §11)

A scipy upgrade that fails this test is blocked: the fix relies on an implementation detail.
"""

from __future__ import annotations

import numpy as np
import pytest

from qmwlink.channel.small_scale import SmallScale
from qmwlink.numerics.special import NC_FLOOR, ncx2_cdf, ncx2_sf

# N: (lower-tail cdf at 2N - 5.5 sqrt(4N), upper-tail sf at 2N + 5.5 sqrt(4N)); TSD §11
REF = {
    10**5: (1.590617e-8, 2.256714e-8),
    10**6: (1.796389e-8, 2.006462e-8),
    10**7: (1.865997e-8, 1.932410e-8),
    10**8: (1.888480e-8, 1.909481e-8),
}


@pytest.mark.parametrize("n", sorted(REF))
def test_ncx2_floor_tail(n: int) -> None:
    nu = 2.0 * n
    sd = np.sqrt(4.0 * n)
    lo, hi = REF[n]
    assert float(ncx2_cdf(nu - 5.5 * sd, nu, 0.0)) == pytest.approx(lo, rel=1e-6)
    assert float(ncx2_sf(nu + 5.5 * sd, nu, 0.0)) == pytest.approx(hi, rel=1e-6)


def test_floor_is_central_at_small_n() -> None:
    """lam = 0 and lam = NC_FLOOR agree with scipy's central chi2 where chi2 is accurate."""
    from scipy import stats

    x = np.linspace(0.5, 30.0, 40)
    np.testing.assert_allclose(ncx2_cdf(x, 6.0, 0.0), stats.chi2.cdf(x, 6.0), rtol=1e-12)
    np.testing.assert_allclose(ncx2_sf(x, 6.0, NC_FLOOR), stats.chi2.sf(x, 6.0), rtol=1e-12)


def test_small_scale_k0_uses_helper_at_large_n() -> None:
    """The K = 0 SmallScale lower tail at N = 1e7 hits the reference (chi2 would be 2.3% off)."""
    n = 10**7
    s = SmallScale(0.0, 0.9999, n)
    x_sur = 2.0 * n - 5.5 * np.sqrt(4.0 * n)
    got = float(s.surrogate_cdf(x_sur * s.sigma2))
    assert got == pytest.approx(REF[n][0], rel=1e-6)
