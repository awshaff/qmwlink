"""Per-velocity-class cost of both receiver engines (FR-LIN-2,3 benchmarks; TSD §14.12).

Recorded, not CI-gated. ``run_bench`` prints the table and, given a path, rewrites the benchmark
section of ``docs/RESULTS.md`` between the markers below.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np

from qmwlink.receivers.rydberg_physics import lindblad, weak_probe
from qmwlink.receivers.rydberg_physics.scheme import Drive, LadderScheme

START, END = "<!-- BENCH:START -->", "<!-- BENCH:END -->"
TWO_PI = 2.0 * np.pi
TARGET_WEAK_NS = 50.0
TARGET_FULL_US = 20.0


def _case() -> tuple[LadderScheme, Drive]:
    """TSD §14.11 case: ``2 pi x (0.05, 5, 20, 0.1)`` MHz, 780/480 nm."""
    scheme = LadderScheme(4, gamma3=TWO_PI * 1e-3, gamma4=TWO_PI * 1e-3, gamma_t=TWO_PI * 0.1)
    drive = Drive(TWO_PI * 0.05, TWO_PI * 5.0, TWO_PI * 20.0,
                  k_p=TWO_PI / 780.241e-9, k_c=TWO_PI / 480e-9)  # fmt: skip
    return scheme, drive


def time_weak_probe(n_v: int = 10**4, repeats: int = 200) -> float:
    """Best-of-5 time per velocity class of the weak-probe steady state [ns]."""
    scheme, drive = _case()
    v = np.linspace(-850.0, 850.0, n_v)
    best = np.inf
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(repeats):
            weak_probe.rho21(scheme, drive, v)
        best = min(best, (time.perf_counter() - t0) / repeats)
    return float(best / n_v * 1e9)


def time_full(n_v: int = 10**3, repeats: int = 20) -> float:
    """Best-of-5 time per velocity class of the batched full steady state [us]."""
    scheme, drive = _case()
    lv = lindblad.build(scheme, drive)
    v = np.linspace(-850.0, 850.0, n_v)
    best = np.inf
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(repeats):
            lindblad.steady_state(lv, v)
        best = min(best, (time.perf_counter() - t0) / repeats)
    return float(best / n_v * 1e6)


def run_bench(results_md: str | Path | None = "docs/RESULTS.md") -> str:
    """Measure both engines; write the section to ``results_md`` unless it is ``None``."""
    weak, full = time_weak_probe(), time_full()
    lines = [
        START,
        "| Engine | $N_v$ | Cost per velocity class | Target (not gated) | Met |",
        "|---|---|---|---|---|",
        f"| Weak-probe closed form (`weak_probe.rho21`) | $10^4$ | {weak:.1f} ns | "
        f"≤ {TARGET_WEAK_NS:g} ns | {'yes' if weak <= TARGET_WEAK_NS else 'no'} |",
        f"| Full Lindblad, batched `solve` (`lindblad.steady_state`) | $10^3$ | {full:.2f} µs | "
        f"≤ {TARGET_FULL_US:g} µs | {'yes' if full <= TARGET_FULL_US else 'no'} |",
        "",
        f"Measured with `make bench` on {platform.machine()} / {platform.system()}, "
        f"Python {platform.python_version()}, NumPy {np.__version__}; best of 5 repetitions.",
        END,
    ]
    text = "\n".join(lines)
    if results_md is not None:
        path = Path(results_md)
        doc = path.read_text(encoding="utf-8") if path.is_file() else ""
        if START in doc and END in doc:
            doc = doc[: doc.index(START)] + text + doc[doc.index(END) + len(END) :]
        else:
            doc = doc.rstrip("\n") + "\n\n## Receiver engine benchmarks (FR-LIN-2, FR-LIN-3)\n\n"
            doc += text + "\n"
        path.write_text(doc, encoding="utf-8")
    return text
