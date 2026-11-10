"""``qmw`` command line. (FR-EXP-2,4, TSD §12)

``qmw run configs/e3.yaml [--force]``, ``qmw list``, ``qmw figs``, ``qmw paper-figs``,
``qmw tables``, ``qmw bench``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from qmwlink.experiments import registry
from qmwlink.experiments.runner import ConfigError, run_config


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qmw", description="qmwlink experiment runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one experiment config (cached)")
    run.add_argument("config", help="path to a YAML config, or an experiment id such as e3")
    run.add_argument("--force", action="store_true", help="recompute even on a cache hit")
    run.add_argument("--out", default="results")
    sub.add_parser("list", help="list experiments")
    for name, text in (("figs", "run all experiments with the cache"),
                       ("paper-figs", "force-run E1-E9 and export to <out>/paper/")):  # fmt: skip
        p = sub.add_parser(name, help=text)
        p.add_argument("--out", default="results")
        p.add_argument("--configs", default="configs")
        p.add_argument("--only", nargs="*", default=None, help="subset of experiment ids")
    tab = sub.add_parser("tables", help="write the LaTeX tables")
    tab.add_argument("--out", default="results/paper/tables")
    tab.add_argument("--configs", default="configs")
    bench = sub.add_parser("bench", help="per-velocity-class cost of both receiver engines")
    bench.add_argument("--results-md", default="docs/RESULTS.md")
    bench.add_argument("--no-write", action="store_true", help="print only")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("qmwlink").setLevel(logging.INFO)
    try:
        if args.cmd == "list":
            for spec in registry.EXPERIMENTS.values():
                print(f"{spec.exp_id}  {spec.name:<26} {spec.default_config:<18} {spec.question}")
        elif args.cmd == "run":
            cfg = args.config
            if not Path(cfg).is_file() and cfg.lower() in registry.EXPERIMENTS:
                cfg = registry.get(cfg).default_config
            res = run_config(cfg, args.out, force=args.force)
            state = "cache hit" if res.cache_hit else f"computed in {res.wall_time_s:.1f} s"
            print(f"{res.experiment}: {state} -> {res.out_dir}")
        elif args.cmd in ("figs", "paper-figs"):
            from qmwlink.io.export import paper_figs, run_all

            if args.cmd == "figs":
                for res in run_all(args.out, config_dir=args.configs, ids=args.only):
                    state = "cache hit" if res.cache_hit else f"{res.wall_time_s:.1f} s"
                    print(f"{res.experiment}: {state} -> {res.out_dir}")
            else:
                print(paper_figs(args.out, config_dir=args.configs, ids=args.only))
        elif args.cmd == "tables":
            from qmwlink.io.latex import write_tables

            cfg_dir = Path(args.configs)
            for p in write_tables(
                args.out, defaults_path=cfg_dir / "defaults.yaml",
                literature_path=cfg_dir / "literature_rar.yaml",
            ):  # fmt: skip
                print(p)
        else:
            from qmwlink.receivers.rydberg_physics.bench import run_bench

            print(run_bench(None if args.no_write else args.results_md))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
