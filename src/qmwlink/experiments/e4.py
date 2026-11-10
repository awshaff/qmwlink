"""E4: receiver-referred entanglement window (RQ3, FR-SVC-E1, TSD §8.1).

``eta*_rx`` and the window width ``1 - eta*_rx`` vs additive noise ``N_add`` for several front-end
classes (attenuator ``(eta_t, n_t)`` followed by ``N_add``) and bands.
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
    n_add: float
    eta_star_rx: float
    window: float
    has_window: bool


COLUMNS = E4Row._fields


def compute(cfg: E4Config) -> list[E4Row]:
    rows: list[E4Row] = []
    n_add = cfg.n_add.values()
    for band in cfg.bands_ghz:
        nth = float(n_th(band * 1e9, cfg.environment.T_env_k))
        for cls in cfg.classes:
            for na in n_add:
                fe = coherent_transducer(cls.eta_t, cls.n_t, float(na))
                w = float(window_rx(nth, fe))
                rows.append(
                    E4Row(band, cls.label, float(na), float(eta_star_rx(nth, fe)), w, w > 0)
                )
    return rows


def _figure(rows: list[E4Row], cfg: E4Config) -> Figure:
    apply_style()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 4.2), sharex=True)
    band0 = cfg.bands_ghz[0]
    for i, cls in enumerate(cfg.classes):
        sel = [r for r in rows if r.band_ghz == band0 and r.front_end == cls.label]
        a1.plot([r.n_add for r in sel], [1.0 - r.eta_star_rx for r in sel], color=OKABE_ITO[i],
                label=cls.label)  # fmt: skip
    a1.axhline(0.0, color="k", lw=0.6, ls=":")  # below: no window
    a1.set_ylabel(rf"$1-\eta^*_{{\rm rx}}$ ({band0:g} GHz)")
    a1.legend(frameon=False)
    cls0 = cfg.classes[0]
    for j, band in enumerate(cfg.bands_ghz):
        sel = [r for r in rows if r.band_ghz == band and r.front_end == cls0.label and r.has_window]
        a2.plot([r.n_add for r in sel], [r.window for r in sel], color=OKABE_ITO[j],
                label=f"{band:g} GHz")  # fmt: skip
    a2.set_yscale("log")
    a2.set_xlabel(r"additive noise $N_{\rm add}$ (quanta)")
    a2.set_ylabel(rf"window $1-\eta^*_{{\rm rx}}$ ({cls0.label})")
    a2.legend(frameon=False)
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
               {"class_sources": {c.label: c.source for c in cfg.classes}})  # fmt: skip
    logger.info("E4 written to %s", out)
    return out
