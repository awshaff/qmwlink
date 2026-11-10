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
from qmwlink.channel.link_budget import link_budget
from qmwlink.channel.small_scale import SmallScale
from qmwlink.experiments.common import save_fig, write_csv, write_meta
from qmwlink.experiments.inputs import LnaTable, RarTable
from qmwlink.experiments.rx_cfg import capacity_fn
from qmwlink.experiments.schema import E5Config, load_config
from qmwlink.plotting.style import COLUMN_WIDTH_IN, OKABE_ITO, apply_style
from qmwlink.receivers.base import MeasuringReceiver
from qmwlink.services.classical import ClassicalService, c_rx
from qmwlink.services.throughput_map import throughput_ratio
from qmwlink.thermal import n_th

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


class CapRow(NamedTuple):
    f_ghz: float
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
    rho_equal_b_db: float
    b_rar_hz: float
    b_lna_hz: float
    n_s_rar: float
    n_s_lna: float


def _tables(cfg: E5Config) -> tuple[RarTable, LnaTable]:
    tm = cfg.throughput_map
    return RarTable.from_cfg(tm.rar, cfg.environment.T_env_k), LnaTable(tm.lna)


def _rx_capacity(
    rx: MeasuringReceiver, n_s: float, n_th_val: float
) -> Callable[[FloatArray], FloatArray]:
    return lambda e: c_rx(e, n_s, n_th_val, rx)


def compute_capacity(cfg: E5Config) -> list[CapRow]:
    """Panel (a): ergodic bit/mode of the ideal receivers and, where defined, LNA and RAR."""
    rows: list[CapRow] = []
    lbc, ls = cfg.link_budget, cfg.large_scale
    rar_t, lna_t = _tables(cfg)
    for fg in cfg.bands_ghz:
        f = fg * 1e9
        nth = float(n_th(f, cfg.environment.T_env_k))
        rar, lna = rar_t.receiver_at(f), lna_t.receiver_at(f)
        for d in cfg.throughput_map.distance_m.values():
            lb = link_budget(f, float(d), lbc.A_t_m2, lbc.A_r_m2,
                             gamma_abs_per_m=ls.gamma_abs_at(fg),
                             coupling_model=lbc.coupling_model, g_enh=lbc.G_enh)  # fmt: skip
            law = ls.law(fg, float(d), lb.psi0)
            for k in cfg.throughput_map.K:
                comp = CompoundPDT(SmallScale(k, lb.eta_geo), law)
                for ns in cfg.n_s:
                    caps = {sp.label: capacity_fn(sp, f, ns, nth) for sp in cfg.receivers}
                    if lna is not None:
                        caps["LNA"] = _rx_capacity(lna, ns, nth)
                    if rar is not None:
                        caps["RAR"] = _rx_capacity(rar, ns, nth)
                    for label, cap in caps.items():
                        c = ClassicalService(comp, cap).ergodic()
                        rows.append(CapRow(fg, k, ns, float(d), label, c))
    return rows


def compute_map(cfg: E5Config) -> list[MapRow]:
    """``rho`` on the (f, d, K) grid, plus the equal-bandwidth variant; masked cells are NaN."""
    rows: list[MapRow] = []
    tm, lbc, ls = cfg.throughput_map, cfg.link_budget, cfg.large_scale
    rar_t, lna_t = _tables(cfg)
    nan = float("nan")
    for fg in tm.f_ghz.values():
        f = float(fg) * 1e9
        nth = float(n_th(f, cfg.environment.T_env_k))
        rar, lna = rar_t.receiver_at(f), lna_t.receiver_at(f)
        lna_eq = (
            lna_t.receiver_at(f, rar.bandwidth_hz)
            if (rar is not None and tm.equal_bandwidth_case)
            else None
        )
        for d in tm.distance_m.values():
            if rar is None or lna is None:
                rows += [MapRow(float(fg), float(d), k, nan, nan, nan, nan, nan, nan, nan, nan)
                         for k in tm.K]  # fmt: skip
                continue
            lb = link_budget(f, float(d), lbc.A_t_m2, lbc.A_r_m2,
                             gamma_abs_per_m=ls.gamma_abs_at(float(fg)),
                             coupling_model=lbc.coupling_model, g_enh=lbc.G_enh)  # fmt: skip
            law = ls.law(float(fg), float(d), lb.psi0)
            for k in tm.K:
                comp = CompoundPDT(SmallScale(k, lb.eta_geo), law)
                p = throughput_ratio(comp, rar, lna, nth, tm.P_t_w, f)
                rho_eq = nan
                if lna_eq is not None:
                    rho_eq = throughput_ratio(comp, rar, lna_eq, nth, tm.P_t_w, f).rho_db
                rows.append(
                    MapRow(
                        float(fg), float(d), k, p.rate_rar_bps, p.rate_lna_bps, p.rho_db, rho_eq,
                        float(rar.bandwidth_hz or nan), float(lna.bandwidth_hz or nan),
                        p.n_s_rar, p.n_s_lna,
                    )
                )  # fmt: skip
    return rows


def _fig_capacity(rows: list[CapRow], cfg: E5Config) -> Figure:
    apply_style()
    ks = cfg.throughput_map.K
    k0, n0 = ks[-1], cfg.n_s[0]
    n_pan = len(cfg.bands_ghz)
    fig, axes = plt.subplots(
        (n_pan + 1) // 2, 2, figsize=(COLUMN_WIDTH_IN, 3.3), sharex=True, squeeze=False
    )
    labels = [s.label for s in cfg.receivers] + ["LNA", "RAR"]
    for ax, band in zip(axes.ravel(), cfg.bands_ghz, strict=False):
        for i, lab_ in enumerate(labels):
            sel = [r for r in rows if r.f_ghz == band and k0 == r.K and r.n_s == n0
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


def _fig_map(rows: list[MapRow], cfg: E5Config, col: str) -> Figure:
    """``rho`` map (``col`` = ``rho_db`` or ``rho_equal_b_db``); masked cells are blank."""
    apply_style()
    tm = cfg.throughput_map
    ks, fs, ds = tm.K, tm.f_ghz.values(), tm.distance_m.values()
    fig, axes = plt.subplots(len(ks), 1, figsize=(COLUMN_WIDTH_IN, 2.0 * len(ks)), sharex=True,
                             squeeze=False)  # fmt: skip
    for ax, k in zip(axes[:, 0], ks, strict=True):
        z = np.full((len(ds), len(fs)), np.nan)
        for r in rows:
            if k == r.K:
                z[int(np.argmin(abs(ds - r.d_m))), int(np.argmin(abs(fs - r.f_ghz)))] = getattr(
                    r, col
                )
        if np.all(np.isnan(z)):
            continue
        m = ax.pcolormesh(fs, ds, np.ma.masked_invalid(z), shading="nearest", cmap="RdBu",
                          vmin=-np.nanmax(abs(z)), vmax=np.nanmax(abs(z)))  # fmt: skip
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
    save_fig(_fig_map(mp, cfg, "rho_db"), out / "fig_rho_map")
    if cfg.throughput_map.equal_bandwidth_case:
        save_fig(_fig_map(mp, cfg, "rho_equal_b_db"), out / "fig_rho_map_equal_b")
    rar_t, lna_t = _tables(cfg)
    n_masked = sum(bool(np.isnan(r.rho_db)) for r in mp)
    write_meta(
        out / "meta.json", cfg.experiment, h,
        {"n_cap_rows": len(cap), "n_map_rows": len(mp), "n_masked": n_masked,
                "per_link": cfg.large_scale.per_link,
                "P_t_w": cfg.throughput_map.P_t_w,
                "rar_f_range_hz": rar_t.f_range_hz, "lna_f_range_hz": lna_t.f_range_hz,
                "rar_entries": [e.label for e in rar_t.entries],
                "rar_sources": [e.source for e in rar_t.entries],
                "lna_sources": [p.source for p in cfg.throughput_map.lna.bands],
                "lna_bandwidth_sources": [b.source for b in cfg.throughput_map.lna.B_hz]},
    )  # fmt: skip
    logger.info("E5 written to %s", out)
    return out
