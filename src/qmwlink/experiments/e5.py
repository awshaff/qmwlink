"""E5: classical ergodic capacity and RAR-vs-LNA throughput map (RQ5, FR-SVC-C5, TSD §8.4)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from qmwlink.channel.compound import CompoundPDT
from qmwlink.channel.large_scale import LargeScaleLaw, ShiftedLogGamma
from qmwlink.channel.link_budget import LinkBudget, link_budget
from qmwlink.channel.small_scale import SmallScale
from qmwlink.experiments.common import save_fig, write_csv, write_meta
from qmwlink.experiments.rx_cfg import capacity_fn
from qmwlink.experiments.schema import BandRx, E5Config, load_config
from qmwlink.plotting.style import COLUMN_WIDTH_IN, OKABE_ITO, apply_style
from qmwlink.receivers.base import MeasuringReceiver
from qmwlink.receivers.electronic import electronic_receiver
from qmwlink.receivers.rydberg import RydbergReceiver
from qmwlink.services.classical import ClassicalService, c_rx
from qmwlink.services.throughput_map import interp_loglog, throughput_ratio
from qmwlink.thermal import n_th

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


class CapRow(NamedTuple):
    band_ghz: float
    K: float
    n_s: float
    d_m: float
    receiver: str
    c_ergodic_bit_per_mode: float


class MapRow(NamedTuple):
    f_ghz: float
    d_m: float
    K: float
    rate_rar_bps: float
    rate_lna_bps: float
    rho_db: float
    n_s_rar: float
    n_s_lna: float


def _law(cfg: E5Config, lb: LinkBudget) -> LargeScaleLaw:
    st = cfg.large_scale.stochastic
    return ShiftedLogGamma.from_moments(lb.psi0, st.mean, st.std**2)


def _gamma_at(cfg: E5Config, f_ghz: float) -> float:
    """Absorption interpolated linearly in log f between the configured bands (clamped)."""
    bands = sorted(cfg.large_scale.gamma_abs_per_m)
    vals = [cfg.large_scale.gamma_abs_per_m[b] for b in bands]
    return float(np.interp(np.log(f_ghz), np.log(bands), vals))


def _interp(pts: list[BandRx], f_hz: float, attr: str) -> float:
    return interp_loglog(f_hz, [p.f_ghz * 1e9 for p in pts], [getattr(p, attr) for p in pts])


def _rx_capacity(
    rx: MeasuringReceiver, n_s: float, n_th_val: float
) -> Callable[[FloatArray], FloatArray]:
    return lambda e: c_rx(e, n_s, n_th_val, rx)


def _receivers(cfg: E5Config, f_hz: float) -> tuple[RydbergReceiver, MeasuringReceiver]:
    """RAR and LNA at ``f_hz`` from the per-band tables (``ValueError`` outside their range)."""
    s_e, b_rar = _interp(cfg.rar_bands, f_hz, "value"), _interp(cfg.rar_bands, f_hz, "bandwidth_hz")
    t_n, b_lna = _interp(cfg.lna_bands, f_hz, "value"), _interp(cfg.lna_bands, f_hz, "bandwidth_hz")
    if any(np.isnan(v) for v in (s_e, b_rar, t_n, b_lna)):
        raise ValueError(f"{f_hz / 1e9:g} GHz is outside the configured receiver bands")
    rar = RydbergReceiver.from_sensitivity(
        s_e, f_hz, field_convention=cfg.rar_field_convention,
        sideband=cfg.rar_sideband, bandwidth_hz=b_rar,
    )  # fmt: skip
    lna = electronic_receiver(
        t_n, f_hz, tn_convention=cfg.lna_tn_convention,
        sideband=cfg.lna_sideband, bandwidth_hz=b_lna,
    )  # fmt: skip
    return rar, lna


def compute_capacity(cfg: E5Config) -> list[CapRow]:
    rows: list[CapRow] = []
    lbc = cfg.link_budget
    for band in cfg.bands_ghz:
        f = band * 1e9
        nth = float(n_th(f, cfg.environment.T_env_k))
        rar, lna = _receivers(cfg, f)
        for d in cfg.distance_m.values():
            lb = link_budget(f, float(d), lbc.A_t_m2, lbc.A_r_m2,
                             gamma_abs_per_m=cfg.large_scale.gamma_abs_per_m[band],
                             coupling_model=lbc.coupling_model, g_enh=lbc.G_enh)  # fmt: skip
            law = _law(cfg, lb)
            for k in cfg.small_scale["K"]:
                comp = CompoundPDT(SmallScale(k, lb.eta_geo), law)
                for ns in cfg.n_s:
                    caps = {spec.label: capacity_fn(spec, f, ns, nth) for spec in cfg.receivers}
                    caps["LNA"] = _rx_capacity(lna, ns, nth)
                    caps["RAR"] = _rx_capacity(rar, ns, nth)
                    for label, cap in caps.items():
                        c = ClassicalService(comp, cap).ergodic()
                        rows.append(CapRow(band, k, ns, float(d), label, c))
    return rows


def compute_map(cfg: E5Config) -> list[MapRow]:
    rows: list[MapRow] = []
    lbc = cfg.link_budget
    nan = float("nan")
    for fg in cfg.f_ghz.values():
        f = float(fg) * 1e9
        s_e = _interp(cfg.rar_bands, f, "value")
        b_rar = _interp(cfg.rar_bands, f, "bandwidth_hz")
        t_n = _interp(cfg.lna_bands, f, "value")
        b_lna = _interp(cfg.lna_bands, f, "bandwidth_hz")
        masked = any(np.isnan(v) for v in (s_e, b_rar, t_n, b_lna))
        nth = float(n_th(f, cfg.environment.T_env_k))
        if not masked:
            rar, lna = _receivers(cfg, f)
        for d in cfg.distance_m.values():
            if masked:
                rows += [MapRow(float(fg), float(d), k, nan, nan, nan, nan, nan)
                         for k in cfg.small_scale["K"]]  # fmt: skip
                continue
            lb = link_budget(f, float(d), lbc.A_t_m2, lbc.A_r_m2,
                             gamma_abs_per_m=_gamma_at(cfg, float(fg)),
                             coupling_model=lbc.coupling_model, g_enh=lbc.G_enh)  # fmt: skip
            law = _law(cfg, lb)
            for k in cfg.small_scale["K"]:
                comp = CompoundPDT(SmallScale(k, lb.eta_geo), law)
                p = throughput_ratio(comp, rar, lna, nth, cfg.p_t_w, f)
                rows.append(MapRow(float(fg), float(d), k, p.rate_rar_bps, p.rate_lna_bps,
                                   p.rho_db, p.n_s_rar, p.n_s_lna))  # fmt: skip
    return rows


def _fig_capacity(rows: list[CapRow], cfg: E5Config) -> Figure:
    apply_style()
    k0, n0 = cfg.small_scale["K"][-1], cfg.n_s[0]
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_WIDTH_IN, 3.3), sharex=True)
    labels = [s.label for s in cfg.receivers] + ["LNA", "RAR"]
    for ax, band in zip(axes.ravel(), cfg.bands_ghz, strict=False):
        for i, lab_ in enumerate(labels):
            sel = [r for r in rows if r.band_ghz == band and k0 == r.K and r.n_s == n0
                   and r.receiver == lab_]  # fmt: skip
            ax.plot([r.d_m for r in sel], [r.c_ergodic_bit_per_mode for r in sel],
                    color=OKABE_ITO[i % len(OKABE_ITO)],
                    label=lab_ if band == cfg.bands_ghz[0] else None)  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.text(0.05, 0.08, f"{band:g} GHz", transform=ax.transAxes, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("distance (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\bar C$ (bit/mode)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def _fig_map(rows: list[MapRow], cfg: E5Config) -> Figure:
    apply_style()
    ks = cfg.small_scale["K"]
    fs, ds = cfg.f_ghz.values(), cfg.distance_m.values()
    fig, axes = plt.subplots(len(ks), 1, figsize=(COLUMN_WIDTH_IN, 2.0 * len(ks)), sharex=True,
                             squeeze=False)  # fmt: skip
    for ax, k in zip(axes[:, 0], ks, strict=True):
        z = np.full((len(ds), len(fs)), np.nan)
        for r in rows:
            if k == r.K:
                z[int(np.argmin(abs(ds - r.d_m))), int(np.argmin(abs(fs - r.f_ghz)))] = r.rho_db
        m = ax.pcolormesh(fs, ds, np.ma.masked_invalid(z), shading="nearest", cmap="RdBu")
        if np.nanmin(z) < 0 < np.nanmax(z):
            cs = ax.contour(fs, ds, np.nan_to_num(z), levels=[0.0], colors="k", linewidths=0.9)
            ax.clabel(cs, fmt=r"$\rho=0$", fontsize=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("distance (m)")
        ax.text(0.03, 0.85, f"$K={k:g}$", transform=ax.transAxes, fontsize=7)
        fig.colorbar(m, ax=ax, label=r"$\rho$ (dB)")
    axes[-1, 0].set_xlabel("frequency (GHz)")
    fig.tight_layout()
    return fig


def run(config_path: str | Path, out_root: str | Path = "results") -> Path:
    cfg, h = load_config(config_path, E5Config)
    assert isinstance(cfg, E5Config)
    out = Path(out_root) / cfg.experiment / h
    out.mkdir(parents=True, exist_ok=True)
    cap, mp = compute_capacity(cfg), compute_map(cfg)
    write_csv(out / "capacity.csv", CapRow._fields, cap)
    write_csv(out / "map.csv", MapRow._fields, mp)
    save_fig(_fig_capacity(cap, cfg), out / "fig_capacity")
    save_fig(_fig_map(mp, cfg), out / "fig_rho_map")
    n_masked = sum(bool(np.isnan(r.rho_db)) for r in mp)
    write_meta(out / "meta.json", cfg.experiment, h,
               {"n_cap_rows": len(cap), "n_map_rows": len(mp), "n_masked": n_masked,
                "rar_sources": [b.source for b in cfg.rar_bands],
                "lna_sources": [b.source for b in cfg.lna_bands]})  # fmt: skip
    logger.info("E5 written to %s", out)
    return out
