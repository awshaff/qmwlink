"""Classical service and crossover (FR-SVC-C1..C4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qmwlink.channel.compound import CompoundPDT
from qmwlink.channel.large_scale import DeterministicLoss, ShiftedLogGamma
from qmwlink.channel.small_scale import SmallScale
from qmwlink.receivers.base import MeasuringReceiver, ideal_heterodyne
from qmwlink.services.classical import (
    ClassicalService,
    c_het,
    c_holevo,
    c_hom,
    c_rx,
    g_entropy,
    throughput_bps,
    tx_photons_per_mode,
)
from qmwlink.services.crossover import (
    crossover_capacity,
    crossover_closed_form,
    crossover_numeric,
    dsb_image_floor_db,
)
from qmwlink.services.throughput_map import interp_loglog, throughput_ratio
from qmwlink.thermal import n_th

NTH = float(n_th(5e9, 300.0))
ETAS = np.linspace(1e-6, 1.0, 200)


def test_g_entropy_values() -> None:
    assert g_entropy(0.0) == 0.0
    assert g_entropy(1.0) == pytest.approx(2.0)  # 2 log2 2 - 0
    assert g_entropy(1e-12) > 0


@pytest.mark.parametrize("ns", [1e-2, 1.0, 1e3])
@pytest.mark.parametrize("nth", [0.0, 0.5, NTH])
def test_ordering_and_monotone(ns: float, nth: float) -> None:
    ch = c_holevo(ETAS, ns, nth)
    assert np.all(ch >= c_het(ETAS, ns, nth) - 1e-12)
    assert np.all(ch >= c_hom(ETAS, ns, nth) - 1e-12)
    for c in (ch, c_het(ETAS, ns, nth), c_hom(ETAS, ns, nth)):
        assert np.all(np.diff(c) > -1e-12)
        assert c[0] < 1e-3 * max(ns, 1) + 1e-6  # -> 0 as eta -> 0 (C1c)


def test_c_rx_special_cases_and_monotone_in_namp() -> None:
    het = ideal_heterodyne()
    assert np.allclose(c_rx(ETAS, 50.0, NTH, het), c_het(ETAS, 50.0, NTH), rtol=1e-13)
    prev = c_rx(ETAS, 50.0, NTH, MeasuringReceiver(0.5, "ssb"))
    for n in (1.0, 10.0, 100.0):
        cur = c_rx(ETAS, 50.0, NTH, MeasuringReceiver(n, "ssb"))
        assert np.all(cur <= prev + 1e-15)
        prev = cur
    ssb = c_rx(ETAS, 50.0, NTH, MeasuringReceiver(3.0, "ssb"))
    dsb = c_rx(ETAS, 50.0, NTH, MeasuringReceiver(3.0, "dsb"))
    assert np.all(dsb <= ssb + 1e-15)


def _compound(k: float = 10.0) -> CompoundPDT:
    return CompoundPDT(SmallScale(k, 0.9), ShiftedLogGamma.from_moments(0.5, 1e-3, 2.5e-7))


def test_ergodic_matches_mc_and_deterministic() -> None:
    comp, ns = _compound(), 1e3
    svc = ClassicalService(comp, lambda e: c_het(e, ns, NTH))
    s = comp.sample(np.random.default_rng(7), 400_000)
    v = c_het(s, ns, NTH)
    assert abs(svc.ergodic() - v.mean()) < 5 * v.std() / math.sqrt(v.size)
    det = ClassicalService(
        CompoundPDT(SmallScale(0.0, 1.0), DeterministicLoss(-math.log(0.2))),
        lambda e: c_het(e, ns, NTH),
    )
    # eta = 0.2 * eta_s with eta_s <= 1, so the ergodic capacity is below C(0.2)
    assert det.ergodic() < float(c_het(0.2, ns, NTH))


def test_outage_capacity_and_rate_outage_roundtrip() -> None:
    comp, ns = _compound(), 1e3
    svc = ClassicalService(comp, lambda e: c_het(e, ns, NTH))
    for eps in (1e-3, 1e-2, 1e-1):
        c = svc.outage_capacity(eps)
        assert c == pytest.approx(float(c_het(comp.ppf(eps), ns, NTH)), rel=1e-6)
        assert svc.rate_outage(c) == pytest.approx(eps, rel=1e-6)
    assert svc.rate_outage(0.0) == 0.0
    assert svc.rate_outage(1e9) == 1.0


def test_throughput_hand_case() -> None:
    assert throughput_bps(2.5, 4e6) == pytest.approx(1e7)
    assert tx_photons_per_mode(6.62607015e-34 * 1e9 * 1e6 * 3.0, 1e9, 1e6) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        throughput_bps(1.0, 0.0)


# ---- crossover -------------------------------------------------------------------------------
def test_crossover_references() -> None:
    ssb = crossover_closed_form(3.0, 5e9, 300.0, "ssb")
    assert ssb.ratio_to_bbr == pytest.approx(0.99823, abs=1e-4)
    assert ssb.n_amp_req == pytest.approx(1245.27, abs=0.01)
    d4 = crossover_closed_form(4.0, 5e9, 300.0, "dsb")
    assert d4.ratio_to_bbr == pytest.approx(0.71659, abs=1e-4)
    assert d4.n_amp_req == pytest.approx(641.71, abs=0.01)
    assert crossover_closed_form(6.0, 5e9, 300.0, "dsb").ratio_to_bbr == pytest.approx(
        1.40849, abs=1e-4
    )


def test_crossover_dsb_floor_nan_with_reason() -> None:
    r = crossover_closed_form(3.0, 5e9, 300.0, "dsb")
    assert math.isnan(r.s_e_req) and not r.attainable
    assert r.reason is not None and "image floor" in r.reason and "3.009" in r.reason
    assert dsb_image_floor_db(NTH) == pytest.approx(3.009, abs=5e-4)
    # floors at higher frequency (TSD §8.3)
    for f, db in ((30e9, 3.000), (100e9, 2.976), (300e9, 2.907)):
        assert dsb_image_floor_db(float(n_th(f, 300.0))) == pytest.approx(db, abs=1e-3)
    # unattainable SSB (negative penalty)
    s = crossover_closed_form(-1.0, 5e9, 300.0, "ssb")
    assert math.isnan(s.s_e_req) and s.reason is not None


@pytest.mark.parametrize("sb", ["ssb", "dsb"])
@pytest.mark.parametrize("delta", [1.0, 3.0, 4.0, 6.0, 10.0])
@pytest.mark.parametrize("eta_ref", [0.0, 0.3, 0.9])
def test_closed_form_vs_numeric(sb: str, delta: float, eta_ref: float) -> None:
    a = crossover_closed_form(delta, 30e9, 300.0, sb, eta_ref)  # type: ignore[arg-type]
    b = crossover_numeric(delta, 30e9, 300.0, sb, eta_ref)  # type: ignore[arg-type]
    if math.isnan(a.s_e_req):
        assert math.isnan(b.s_e_req) and b.reason is not None
    else:
        assert b.n_amp_req == pytest.approx(a.n_amp_req, rel=1e-6)
        assert b.s_e_req == pytest.approx(a.s_e_req, rel=1e-6)


def test_crossover_capacity_variant() -> None:
    comp, ns = _compound(), 1e3
    het = ClassicalService(comp, lambda e: c_het(e, ns, NTH))
    r = crossover_capacity(het, ns, NTH, "ssb", 0.1, 5e9, 300.0)
    rx = MeasuringReceiver(r.n_amp_req, "ssb")
    got = ClassicalService(comp, lambda e: c_rx(e, ns, NTH, rx)).ergodic()
    assert got == pytest.approx(0.9 * het.ergodic(), rel=1e-8)
    assert r.n_amp_req > 0.5
    d = crossover_capacity(het, ns, NTH, "dsb", 1e-6, 5e9, 300.0)
    assert math.isnan(d.s_e_req) and d.reason is not None and "image floor" in d.reason
    with pytest.raises(ValueError):
        crossover_capacity(het, ns, NTH, "ssb", 0.0, 5e9, 300.0)


# ---- throughput map --------------------------------------------------------------------------
def test_interp_loglog_masks_and_is_exact_at_nodes() -> None:
    f, v = [5e9, 30e9, 100e9], [2.0, 8.0, 50.0]
    assert interp_loglog(30e9, f, v) == pytest.approx(8.0)
    assert math.isnan(interp_loglog(4e9, f, v)) and math.isnan(interp_loglog(200e9, f, v))
    mid = interp_loglog(math.sqrt(5e9 * 30e9), f, v)
    assert mid == pytest.approx(4.0)  # geometric mean of the end values on a log-log line
    with pytest.raises(ValueError):
        interp_loglog(1e9, [1e9], [-1.0])


def test_power_limited_asymptote_and_equal_bandwidth() -> None:
    eta_l = 1e-3
    comp = CompoundPDT(SmallScale(0.0, 1.0), DeterministicLoss(-math.log(eta_l)))
    rar = MeasuringReceiver(400.0, "dsb", 1e6)
    lna = MeasuringReceiver(900.0, "ssb", 1e6)
    p = throughput_ratio(comp, rar, lna, NTH, 1e-18, 5e9)
    eta = 1e-3 * comp.small.moment(1)  # mean eta_s only matters at O(eta N_s/D)
    assert eta * p.n_s_rar / (1000.0) < 1e-4
    d_rar = float(rar.noise_photons(eta, NTH))
    d_lna = float(lna.noise_photons(eta, NTH))
    assert p.rho_db == pytest.approx(10 * math.log10(d_lna / d_rar), rel=1e-3)
    # equal bandwidths and D_RAR >= D_LNA  =>  rho <= 0
    rar2 = MeasuringReceiver(1500.0, "dsb", 1e6)
    for pt in (1e-18, 1e-12, 1e-6):
        assert throughput_ratio(comp, rar2, lna, NTH, pt, 5e9).rho_db <= 1e-12
    with pytest.raises(ValueError):
        throughput_ratio(comp, MeasuringReceiver(1.0, "dsb"), lna, NTH, 1e-9, 5e9)
