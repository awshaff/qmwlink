"""E8: outage capacity ``C_eps`` per receiver (FR-SVC-C2, TSD §8.2)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from qmwlink.channel.compound import CompoundPDT
from qmwlink.channel.link_budget import link_budget
from qmwlink.channel.small_scale import SmallScale
from qmwlink.experiments.common import save_fig, write_csv, write_meta
from qmwlink.experiments.inputs import LnaTable, RarTable
from qmwlink.experiments.rx_cfg import capacity_fn
from qmwlink.experiments.schema import E8Config, load_config
from qmwlink.plotting.style import COLUMN_WIDTH_IN, OKABE_ITO, apply_style
from qmwlink.services.classical import ClassicalService, c_rx
from qmwlink.thermal import n_th

logger = logging.getLogger(__name__)


class E8Row(NamedTuple):
    band_ghz: float
    K: float
    d_m: float
    receiver: str
    eps: float
    c_eps: float
    c_ergodic: float


COLUMNS = E8Row._fields


def compute(cfg: E8Config) -> list[E8Row]:
    rows: list[E8Row] = []
    f = cfg.band_ghz * 1e9
    nth = float(n_th(f, cfg.environment.T_env_k))
    gamma = cfg.large_scale.gamma_abs_per_m[cfg.band_ghz]
    ls, lbc = cfg.large_scale, cfg.link_budget
    eps_grid = cfg.eps.values()
    for d in cfg.distance_m:
        lb = link_budget(f, d, lbc.A_t_m2, lbc.A_r_m2, gamma_abs_per_m=gamma,
                         coupling_model=lbc.coupling_model, g_enh=lbc.G_enh)  # fmt: skip
        law = ls.law(cfg.band_ghz, d, lb.psi0)
        for k in cfg.small_scale["K"]:
            comp = CompoundPDT(SmallScale(k, lb.eta_geo), law)
            quantiles = comp.ppf(eps_grid)  # shared across receivers (monotone cap)
            caps = {sp.label: capacity_fn(sp, f, cfg.n_s, nth) for sp in cfg.receivers}
            lna = LnaTable(cfg.lna).receiver_at(f)
            rar = RarTable.from_cfg(cfg.rar, cfg.environment.T_env_k).receiver_at(f)
            if lna is not None:
                caps["LNA"] = lambda e, r=lna: c_rx(e, cfg.n_s, nth, r)  # type: ignore[misc]
            if rar is not None:
                caps["RAR"] = lambda e, r=rar: c_rx(e, cfg.n_s, nth, r)  # type: ignore[misc]
            for label, cap in caps.items():
                svc = ClassicalService(comp, cap)
                erg = svc.ergodic()
                for eps, q in zip(eps_grid, quantiles, strict=True):
                    rows.append(
                        E8Row(cfg.band_ghz, k, d, label, float(eps), float(svc.cap(q)), erg)
                    )  # fmt: skip
    return rows


def _figure(rows: list[E8Row], cfg: E8Config) -> Figure:
    apply_style()
    ds, ks = cfg.distance_m, cfg.small_scale["K"]
    fig, axes = plt.subplots(len(ds), 1, figsize=(COLUMN_WIDTH_IN, 1.9 * len(ds)), sharex=True,
                             squeeze=False)  # fmt: skip
    labels = [s.label for s in cfg.receivers] + ["LNA", "RAR"]
    for ax, d in zip(axes[:, 0], ds, strict=True):
        for i, lab_ in enumerate(labels):
            for k, ls in zip(ks, ("-", "--", ":", "-."), strict=False):
                sel = [r for r in rows if r.d_m == d and k == r.K and r.receiver == lab_]
                ax.plot([r.eps for r in sel], [r.c_eps for r in sel], ls=ls,
                        color=OKABE_ITO[i % len(OKABE_ITO)],
                        label=f"{lab_}, $K={k:g}$" if d == ds[0] else None)  # fmt: skip
        ax.set_xscale("log")
        ax.set_ylabel(r"$C_\epsilon$ (bit/mode)")
        ax.text(0.04, 0.82, f"$d={d:g}$ m", transform=ax.transAxes, fontsize=7)
    axes[-1, 0].set_xlabel(r"outage probability $\epsilon$")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def run(config_path: str | Path, out_root: str | Path = "results") -> Path:
    cfg, h = load_config(config_path, E8Config)
    assert isinstance(cfg, E8Config)
    out = Path(out_root) / cfg.experiment / h
    out.mkdir(parents=True, exist_ok=True)
    rows = compute(cfg)
    write_csv(out / "data.csv", COLUMNS, rows)
    save_fig(_figure(rows, cfg), out / "fig_outage_capacity")
    write_meta(
        out / "meta.json",
        cfg.experiment,
        h,
        {"seed": cfg.seed, "n_rows": len(rows), "per_link": cfg.large_scale.per_link},
    )
    logger.info("E8 written to %s", out)
    return out
