"""``python -m qmwlink.experiments <id> configs/<file>.yaml [--out results]``.

Uncached direct entry point (kept from M3-M5); the cached runner is ``qmw run`` (``qmwlink.cli``).
"""

from __future__ import annotations

import argparse
import logging

from qmwlink.experiments import registry

RUNNERS = {k: s.run for k, s in registry.EXPERIMENTS.items()}


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m qmwlink.experiments")
    ap.add_argument("experiment", choices=sorted(RUNNERS))
    ap.add_argument("config")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("qmwlink").setLevel(logging.INFO)
    print(RUNNERS[args.experiment](args.config, args.out))


if __name__ == "__main__":
    main()
