"""``python -m qmwlink.experiments {e3|e4|e5|e6|e8} configs/<file>.yaml [--out results]``.

Temporary entry point for M3-M4; the cached runner and ``qmw`` CLI arrive in M6.
"""

from __future__ import annotations

import argparse
import logging

from qmwlink.experiments import e3, e4, e5, e6, e8

RUNNERS = {"e3": e3.run, "e4": e4.run, "e5": e5.run, "e6": e6.run, "e8": e8.run}


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
