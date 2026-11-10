"""Experiment registry: id -> config schema, runner and default config. (FR-EXP-1,2, TSD §12)"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from qmwlink.experiments import e1, e2, e3, e4, e5, e6, e7, e8, e9, schema


@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment: ``run(config_path, out_root) -> output directory``."""

    exp_id: str
    name: str
    model: type[BaseModel]
    run: Callable[[str | Path, str | Path], Path]
    question: str

    @property
    def default_config(self) -> str:
        return f"configs/{self.exp_id}.yaml"


_SPECS = (
    ExperimentSpec("e1", "e1_truncation", schema.E1Config, e1.run,
                   "Small-scale PDT shape and truncation effect (RQ1)"),
    ExperimentSpec("e2", "e2_law_comparison", schema.E2Config, e2.run,
                   "Large-scale law comparison (RQ2)"),
    ExperimentSpec("e3", "e3_entanglement_outage", schema.E3Config, e3.run,
                   "Entanglement outage vs distance"),
    ExperimentSpec("e4", "e4_receiver_window", schema.E4Config, e4.run,
                   "Receiver-referred window (RQ3)"),
    ExperimentSpec("e5", "e5_classical_throughput", schema.E5Config, e5.run,
                   "Classical capacity and RAR-vs-LNA throughput (RQ5)"),
    ExperimentSpec("e6", "e6_bbr_crossover", schema.E6Config, e6.run,
                   "BBR crossover map (RQ4)"),
    ExperimentSpec("e7", "e7_diversity", schema.E7Config, e7.run,
                   "Diversity hardening and N_req"),
    ExperimentSpec("e8", "e8_outage_capacity", schema.E8Config, e8.run,
                   "Outage capacity"),
    ExperimentSpec("e9", "e9_rar_validity", schema.E9Config, e9.run,
                   "Validity of the Gaussian RAR model and model-derived points"),
)  # fmt: skip

EXPERIMENTS: dict[str, ExperimentSpec] = {s.exp_id: s for s in _SPECS}
"""Keyed by short id (``e1`` ... ``e9``)."""

_BY_NAME = {s.name: s for s in _SPECS}


def get(key: str) -> ExperimentSpec:
    """Look up by short id (``e3``, case-insensitive) or by the config's ``experiment`` name."""
    k = key.strip().lower()
    if k in EXPERIMENTS:
        return EXPERIMENTS[k]
    if k in _BY_NAME:
        return _BY_NAME[k]
    known = ", ".join(f"{s.exp_id} ({s.name})" for s in _SPECS)
    raise KeyError(f"unknown experiment {key!r}; known: {known}")
