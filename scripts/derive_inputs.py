#!/usr/bin/env python3
"""Regenerate the ITU-derived inputs of TSD §16.1-16.2 (QUESTIONS Q-16).

Outputs, per band f in {5, 30, 100, 300} GHz, all in 1/m (power attenuation, nepers per metre):
  * gamma_abs  : deterministic O2-only specific attenuation (TSD §16.1)       -> config gamma_abs_per_m
  * m1, s1     : mean and std of the weather-driven part G per metre (§16.2) -> config stochastic_per_m
  * k          : log-gamma shape m1^2 / s1^2 (distance-independent, §6.2)

Integration rule (binding; TSD §16.2 values must be reproduced to 3 significant figures):
  * Pressure 1013.25 hPa and temperature 300 K everywhere (equal to T_env).
  * Psi_0: ITU-R P.676-12 Annex 1 dry-air (O2) attenuation, gamma0_exact(f, p, rho=7.5 g/m^3, T).
  * Water vapour: ITU-R P.836 surface water-vapour density rho(p) exceeded for p % of an average year, at the
    18 standard levels P836_LEVELS. rho is linear in p between levels. gamma_w(rho) from P.676-12 Annex 1
    (gammaw_exact) is evaluated on a uniform grid of N_VAP points in p over [0.1, 99] % and averaged with
    the trapezoid rule; the result is the mean conditional on p in [0.1, 99] % (the omitted 1.1 % of the
    year is neglected). Variance likewise.
  * Rain: ITU-R P.837-7 rain rate R(p) exceeded for p %, on N_RAIN log-spaced points from 0.001 % to P0
    (rain probability, P.837-7); zero rain for p > P0. gamma_R(R) from ITU-R P.838-3, horizontal path
    (elevation 0 deg), circular polarisation (tau = 45 deg). E[gamma_R] and E[gamma_R^2] by the trapezoid
    rule in p (as a fraction); p < 0.001 % is neglected.
  * Water vapour and rain are independent: m1 = mean_w + mean_R, s1^2 = var_w + var_R.
  * Unit conversion: gamma[1/m] = gamma[dB/km] * ln(10) / 10 / 1000.

Location: Kyung Hee University, Seoul campus (37.597 N, 127.052 E), altitude 0.05 km.

Requires the optional dependency itur==0.4.0 (scripts only; never imported by src/).
Usage: python scripts/derive_inputs.py [--lat 37.597 --lon 127.052] [--yaml]
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

try:
    import itur
    from itur.models import itu676, itu836, itu837, itu838
except ImportError as exc:  # pragma: no cover
    raise SystemExit("derive_inputs.py needs the optional dependency itur==0.4.0") from exc

ITUR_VERSION = "0.4.0"
BANDS_GHZ = (5.0, 30.0, 100.0, 300.0)
P_HPA = 1013.25
T_K = 300.0
RHO_STD = 7.5
ALT_KM = 0.05
P836_LEVELS = np.array([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 60, 70, 80, 90, 95, 99], float)
N_VAP = 2000
N_RAIN = 200
P_RAIN_MIN = 1e-3  # percent
DBKM_TO_PER_M = np.log(10.0) / 10.0 / 1000.0


def _val(q) -> np.ndarray:
    return np.asarray(getattr(q, "value", q), dtype=float)


def derive(lat: float, lon: float) -> dict:
    warnings.filterwarnings("ignore")
    if itur.__version__ != ITUR_VERSION:
        warnings.warn(f"itur {itur.__version__} != pinned {ITUR_VERSION}; values may differ")

    # Water-vapour density distribution at the standard P.836 levels, linear in p in between.
    rho_lv = np.array([float(_val(itu836.surface_water_vapour_density(lat, lon, p, ALT_KM))) for p in P836_LEVELS])
    p_vap = np.linspace(P836_LEVELS[0], P836_LEVELS[-1], N_VAP)
    rho_vap = np.interp(p_vap, P836_LEVELS, rho_lv)

    # Rain-rate distribution.
    p0 = float(_val(itu837.rainfall_probability(lat, lon)))
    p_rain = np.logspace(np.log10(P_RAIN_MIN), np.log10(p0), N_RAIN)
    r_rain = np.array([float(_val(itu837.rainfall_rate(lat, lon, p))) for p in p_rain])

    out = {"lat": lat, "lon": lon, "P0_percent": p0,
           "R_0.01": float(_val(itu837.rainfall_rate(lat, lon, 0.01))),
           "rho_median": float(np.interp(50.0, P836_LEVELS, rho_lv)),
           "rho_1pct": float(np.interp(1.0, P836_LEVELS, rho_lv)), "bands": {}}
    span = p_vap[-1] - p_vap[0]
    for f in BANDS_GHZ:
        g0 = float(_val(itu676.gamma0_exact(f, P_HPA, RHO_STD, T_K)))
        gw = np.array([float(_val(itu676.gammaw_exact(f, P_HPA, r, T_K))) for r in rho_vap])
        mw = np.trapezoid(gw, p_vap) / span
        vw = np.trapezoid((gw - mw) ** 2, p_vap) / span
        gr = np.array([float(_val(itu838.rain_specific_attenuation(r, f, 0.0, 45.0))) if r > 0 else 0.0
                       for r in r_rain])
        w = p_rain / 100.0
        mr = np.trapezoid(gr, w)
        vr = np.trapezoid(gr ** 2, w) - mr ** 2
        m1 = (mw + mr) * DBKM_TO_PER_M
        s1 = np.sqrt(vw + vr) * DBKM_TO_PER_M
        out["bands"][f] = {"gamma_abs": g0 * DBKM_TO_PER_M, "m1": m1, "s1": s1, "k": (m1 / s1) ** 2,
                           "rain_var_share": vr / (vw + vr),
                           "gamma_std_atm_dbkm": g0 + float(_val(itu676.gammaw_exact(f, P_HPA, RHO_STD, T_K)))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lat", type=float, default=37.597)
    ap.add_argument("--lon", type=float, default=127.052)
    ap.add_argument("--yaml", action="store_true", help="print config blocks instead of a table")
    a = ap.parse_args()
    d = derive(a.lat, a.lon)
    if a.yaml:
        print("gamma_abs_per_m: {" + ", ".join(f"{int(f)}: {b['gamma_abs']:.4g}" for f, b in d["bands"].items()) + "}")
        print("stochastic_per_m:")
        for f, b in d["bands"].items():
            print(f"  {int(f)}: {{mean: {b['m1']:.4g}, std: {b['s1']:.4g}}}")
        return
    print(f"site ({d['lat']}, {d['lon']}): P0 = {d['P0_percent']:.3g} %, R0.01 = {d['R_0.01']:.3g} mm/h, "
          f"rho median = {d['rho_median']:.3g} g/m3, rho 1 % = {d['rho_1pct']:.3g} g/m3")
    print("band | gamma_abs [1/m] | std atm [dB/km] | m1 [1/m] | s1 [1/m] | k | rain var share")
    for f, b in d["bands"].items():
        print(f"{f:5.0f} | {b['gamma_abs']:.4g} | {b['gamma_std_atm_dbkm']:.4g} | {b['m1']:.4g} | {b['s1']:.4g} | "
              f"{b['k']:.3g} | {b['rain_var_share']:.2f}")


if __name__ == "__main__":
    main()