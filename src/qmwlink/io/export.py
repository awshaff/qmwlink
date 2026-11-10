"""Paper export: collect figures, CSV and metadata of E1-E9 into one folder. (FR-EXP-2,3,5)"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable
from pathlib import Path

from qmwlink.experiments import registry
from qmwlink.experiments.runner import RunResult, run_config
from qmwlink.io.latex import write_tables

logger = logging.getLogger(__name__)

_EXPORT_SUFFIXES = (".pdf", ".png", ".csv", ".json")


def export_run(result: RunResult, paper_dir: Path) -> list[Path]:
    """Copy one run's outputs to ``paper_dir/<experiment>/``."""
    dest = paper_dir / result.experiment
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(result.out_dir.iterdir()):
        if src.suffix in _EXPORT_SUFFIXES:
            copied.append(Path(shutil.copy2(src, dest / src.name)))
    return copied


def run_all(
    out_root: str | Path = "results",
    *,
    config_dir: str | Path = "configs",
    force: bool = False,
    ids: Iterable[str] | None = None,
) -> list[RunResult]:
    """Run the default config of every experiment (or of ``ids``), with the cache."""
    results = []
    for exp_id in ids if ids is not None else registry.EXPERIMENTS:
        spec = registry.get(exp_id)
        results.append(run_config(Path(config_dir) / f"{spec.exp_id}.yaml", out_root, force=force))
    return results


def paper_figs(
    out_root: str | Path = "results",
    *,
    config_dir: str | Path = "configs",
    force: bool = True,
    ids: Iterable[str] | None = None,
) -> Path:
    """Force-run E1-E9, then export PDFs/CSV/metadata and the LaTeX tables to ``<out>/paper/``."""
    paper = Path(out_root) / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    results = run_all(out_root, config_dir=config_dir, force=force, ids=ids)
    nreq = None
    for res in results:
        export_run(res, paper)
        if res.experiment == "e7_diversity":
            nreq = res.out_dir / "nreq.csv"
    cfg = Path(config_dir)
    write_tables(
        paper / "tables", defaults_path=cfg / "defaults.yaml",
        literature_path=cfg / "literature_rar.yaml", nreq_csv=nreq,
    )  # fmt: skip
    logger.info("paper outputs in %s", paper)
    return paper
