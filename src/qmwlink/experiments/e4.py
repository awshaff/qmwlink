"""E4: receiver-referred entanglement window (RQ3, FR-SVC-E1, TSD §8.1, §16.5).

``eta*_rx`` and the window ``1 - eta*_rx`` vs the input-referred added noise ``N_add_in`` for
each band, with the published front-end classes marked at their own ``N_add_in``. The window
depends only on ``N_add_in = N_t/eta_t``; ``eta_t`` affects rate, not the window.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from qmwlink.experiments.common import save_fig, write_csv, write_meta
from qmwlink.experiments.schema import E4Config, load_config
from qmwlink.plotting.style import COLUMN_WIDTH_IN, OKABE_ITO, apply_style
from qmwlink.receivers.transducer import coherent_transducer
from qmwlink.services.entanglement import eta_star_rx, window_rx
from qmwlink.thermal import n_th

logger = logging.getLogger(__name__)


class E4Row(NamedTuple):
    band_ghz: float
    front_end: str
    is_class_point: bool
    eta_t: float
    n_add_in: float
    eta_star_rx: float
    window: float
    has_window: bool


COLUMNS = E4Row._fields
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def _row(band: float, label: str, point: bool, eta_t: float, n_t: float, n_add_in: float,
         n_th_val: float) -> E4Row:  # fmt: skip
    fe = coherent_transducer(eta_t, n_t, eta_t * n_add_in)
    w = float(window_rx(n_th_val, fe))
    return E4Row(band, label, point, eta_t, n_add_in, float(eta_star_rx(n_th_val, fe)), w, w > 0)


def compute(cfg: E4Config) -> list[E4Row]:
    rows: list[E4Row] = []
    for band in cfg.bands_ghz:
        nth = float(n_th(band * 1e9, cfg.environment.T_env_k))
        for na in cfg.n_add.values():  # eta_t = 1 sweep: the window depends only on N_add_in
            rows.append(_row(band, "sweep", False, 1.0, 0.0, float(na), nth))
        for c in cfg.front_end_classes:
            rows.append(_row(band, c.label, True, c.eta_t, c.n_t, c.N_add_in, nth))
        for ref in cfg.references_outside_axis:
            rows.append(_row(band, str(ref["label"]), True, 1.0, 0.0, float(ref["n_add_in"]), nth))
    return rows


def _figure(rows: list[E4Row], cfg: E4Config) -> Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 3.0))
    for j, band in enumerate(cfg.bands_ghz):
        sel = [r for r in rows if r.band_ghz == band and not r.is_class_point and r.has_window]
        ax.plot([r.n_add_in for r in sel], [r.window for r in sel], color=OKABE_ITO[j],
                label=f"{band:g} GHz")  # fmt: skip
    band0 = cfg.bands_ghz[0]
    xmax = cfg.n_add.stop
    for i, c in enumerate(cfg.front_end_classes):
        if c.N_add_in > xmax:
            continue
        r = next(
            r for r in rows if r.band_ghz == band0 and r.is_class_point and r.front_end == c.label
        )
        if r.has_window:
            ax.plot(r.n_add_in, r.window, MARKERS[i % len(MARKERS)], color="k", ms=4,
                    label=c.label)  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel(r"input-referred added noise $N_{\rm add}$ (quanta)")
    ax.set_ylabel(r"window $1-\eta^*_{\rm rx}$")
    ax.legend(frameon=False, fontsize=5.5, loc="upper right")
    fig.tight_layout()
    return fig


def run(config_path: str | Path, out_root: str | Path = "results") -> Path:
    cfg, h = load_config(config_path, E4Config)
    assert isinstance(cfg, E4Config)
    out = Path(out_root) / cfg.experiment / h
    out.mkdir(parents=True, exist_ok=True)
    rows = compute(cfg)
    write_csv(out / "data.csv", COLUMNS, rows)
    save_fig(_figure(rows, cfg), out / "fig")
    write_meta(out / "meta.json", cfg.experiment, h,
               {"class_sources": {c.label: c.source for c in cfg.front_end_classes},
                "references_outside_axis": cfg.references_outside_axis})  # fmt: skip
    logger.info("E4 written to %s", out)
    return out
