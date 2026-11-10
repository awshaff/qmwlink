"""Link budget. (FR-CH-6, TSD §6.5)"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import tolerances as tol
from qmwlink.channel.link_budget import (
    GAMMA_ABS_5GHZ,
    absorption_range_m,
    aperture_from_gain,
    fresnel_product,
    goubau_efficiency,
    link_budget,
    symmetric_aperture_for_efficiency,
    tau2_for_efficiency,
)
from qmwlink.constants import C
from qmwlink.thermal import n_th


@given(tau2=st.floats(0.0, 1e6))
def test_goubau_never_exceeds_one_by_construction(tau2: float) -> None:
    """FR-CH-6 (a): eta_geo <= 1 with no clipping in the code path."""
    assert 0.0 <= goubau_efficiency(tau2) <= 1.0


@given(tau2=st.floats(1e-12, 1e-2))
def test_goubau_friis_limit(tau2: float) -> None:
    """FR-CH-6 (b): relative difference from Friis <= tau^2/2 for tau^2 <= 1e-2."""
    rel = abs(goubau_efficiency(tau2) - tau2) / tau2
    assert (
        rel <= tau2 / 2.0 * (1 + 1e-9) + 4 * np.finfo(float).eps
    )  # + float rounding of the difference


def test_reference_5ghz_10m_1m2() -> None:
    """FR-CH-6 (c)."""
    b = link_budget(5e9, 10.0, 1.0, 1.0, gamma_abs_per_m=GAMMA_ABS_5GHZ)
    assert b.tau2 == pytest.approx(2.7816, abs=5e-5)
    assert b.eta_geo == pytest.approx(0.93806, abs=tol.ATOL_ETA_GEO)
    assert b.physical and b.coupling_model == "goubau"
    assert b.psi0 == pytest.approx(GAMMA_ABS_5GHZ * 10.0)
    c = link_budget(5e9, 10.0, 1.0, 1.0, gamma_abs_per_m=0.0, coupling_model="friis_clipped")
    assert c.eta_geo == 1.0 and c.physical is False
    small = link_budget(5e9, 1000.0, 1.0, 1.0, gamma_abs_per_m=0.0, coupling_model="friis_clipped")
    assert small.eta_geo == small.tau2 < 1.0


def test_absorption_range_at_threshold() -> None:
    """FR-CH-6 (d): 555.47 m +/- 0.5 m at 5 GHz, 300 K."""
    d = absorption_range_m(GAMMA_ABS_5GHZ, float(n_th(5e9, 300.0)))
    assert d == pytest.approx(555.47, abs=tol.ATOL_RANGE_M)
    eta_star = float(n_th(5e9, 300.0)) / (1 + float(n_th(5e9, 300.0)))
    assert np.exp(-GAMMA_ABS_5GHZ * d) == pytest.approx(eta_star, rel=1e-12)


def test_other_tsd_references() -> None:
    lam = C / 5e9
    # eta_bar = 0.9999 at 10 m needs A_t = A_r ~ 1.82 m^2
    assert symmetric_aperture_for_efficiency(0.9999, 5e9, 10.0) == pytest.approx(1.82, abs=5e-3)
    assert tau2_for_efficiency(goubau_efficiency(2.5)) == pytest.approx(2.5, rel=1e-12)
    # 30 dBi horn to a bare dipole cell: tau^2 = 3.4e-4 at 10 m -> eta_geo ~ tau^2 (outage ~ 1)
    a_t = aperture_from_gain(10**3.0, lam)
    a_r = 3 * lam**2 / (8 * np.pi)
    assert fresnel_product(a_t, a_r, lam, 10.0) == pytest.approx(3.4e-4, rel=0.02)


def test_g_enh_scales_receive_aperture_only() -> None:
    """FR-CH-6 (e): tau^2 linear in G_enh; goubau eta_geo increases and stays < 1."""
    base = link_budget(5e9, 10.0, 0.1, 0.1, gamma_abs_per_m=0.0)
    prev = base.eta_geo
    for g in (2.5, 10.0, 1e3, 1e6):
        enh = link_budget(5e9, 10.0, 0.1, 0.1, gamma_abs_per_m=0.0, g_enh=g)
        assert enh.tau2 == pytest.approx(g * base.tau2, rel=1e-12)
        assert enh.a_t_m2 == pytest.approx(0.1) and enh.a_r_m2 == pytest.approx(0.1 * g)
        assert prev < enh.eta_geo < 1.0 or enh.eta_geo == pytest.approx(1.0)
        prev = enh.eta_geo


def test_no_transmit_side_enhancement_argument() -> None:
    import inspect

    params = inspect.signature(link_budget).parameters
    assert "g_enh" in params and not any(p.startswith("g_enh_") for p in params)
    with pytest.raises(TypeError):
        link_budget(5e9, 10.0, 1.0, 1.0, gamma_abs_per_m=0.0, g_enh_tx=2.0)  # type: ignore[call-arg]


def test_validation_errors() -> None:
    kw = {"gamma_abs_per_m": 0.0}
    with pytest.raises(ValueError, match="f_hz"):
        link_budget(0.0, 10.0, 1.0, 1.0, **kw)
    with pytest.raises(ValueError, match="g_enh"):
        link_budget(5e9, 10.0, 1.0, 1.0, g_enh=0.5, **kw)
    with pytest.raises(ValueError, match="gamma_abs"):
        link_budget(5e9, 10.0, 1.0, 1.0, gamma_abs_per_m=-1.0)
    with pytest.raises(ValueError, match="coupling_model"):
        link_budget(5e9, 10.0, 1.0, 1.0, coupling_model="bogus", **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="> 0"):
        fresnel_product(1.0, 1.0, 0.06, 0.0)
    with pytest.raises(ValueError, match="> 0"):
        aperture_from_gain(0.0, 0.06)
    with pytest.raises(ValueError, match="tau2"):
        goubau_efficiency(-1.0)
    with pytest.raises(ValueError, match="eta_geo"):
        tau2_for_efficiency(1.0)
    with pytest.raises(ValueError, match="> 0"):
        absorption_range_m(0.0, 10.0)
