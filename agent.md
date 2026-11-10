# agent.md — Instructions for the coding agent

You are implementing `qmwlink`, a Python research toolkit that generates every result of a journal paper on quantum-consistent stochastic microwave channels with Rydberg atomic receivers. The researcher (Abdurrahman) owns the physics and the paper; you own clean, tested, reproducible code. Correctness beats speed and beats cleverness.

## 1. Read before writing any code

1. `docs/IDEA.md` — why the project exists and what the paper claims.
2. `docs/FSD.md` — what to build: requirement IDs (`FR-*`), acceptance criteria, experiments E1–E9.
3. `docs/TSD.md` — how to build it: every formula, algorithm, numerical rule, layout. **The TSD is the single source of truth for math.**

Re-read the relevant TSD section before starting each module.

## 2. Golden rules

1. **No invented physics.** Implement only formulas written in the TSD. If you need a formula that is not there, stop and add an entry to `QUESTIONS.md` instead of guessing or recalling one from memory.
2. **No invented numbers.** Every literature parameter in configs needs a `source:` field. If none is given, use a clearly marked placeholder (`# [A]`) and log it in `QUESTIONS.md`.
3. **Test first for math.** For each formula: write the limiting-case test and the cross-method test (TSD §10) before or together with the implementation.
4. **Disagreement is a finding, not a bug to hide.** If analytic and Monte Carlo results disagree beyond tolerance, or a V0 reference value (TSD §11, "unverified") is not reproduced, do **not** tune tolerances or parameters to make it pass. Record the evidence in `QUESTIONS.md` with label `finding` and continue with other work.
5. **Tolerances are frozen.** They live in `tests/tolerances.py`. Changing one requires a `QUESTIONS.md` entry explaining why.
6. **Follow the numerical rules in TSD §13** (survival functions near 1, `1 - eta` as a primary variable, scaled Bessel, explicit RNG, checked root finding and quadrature).
7. **Conventions are binding** (FSD §4, TSD §2): vacuum variance 1, SI units at interfaces, bits per mode, CM ordering $(x_A,p_A,x_B,p_B)$.
8. **Traceability.** Every public function's docstring cites its FR ID and TSD section, e.g. `"""... (FR-CH-1, TSD §6.1)"""`.
9. **Do not edit `docs/IDEA.md`, `docs/FSD.md`, `docs/TSD.md`** unless Abdurrahman asks. Propose changes in `QUESTIONS.md`.
10. **Do not touch any LaTeX paper source** in this repo or elsewhere.

## 3. Performance policy and the receiver physics model

- **Pure Python.** NumPy/SciPy vectorization only. Do not add Numba, Cython, C/C++ or GPU code. If a figure breaks the 5-minute budget (FSD NFR-3), profile it and raise a `QUESTIONS.md` entry with the profile before changing anything (FSD NFR-7).
- **Vectorize over velocity classes.** Never loop over $v$ in Python to build matrices. Build the Liouvillian once as $\mathcal A+v\mathcal B$ and broadcast (TSD §14.5). The measured cost gap between a naive loop and the batched form is ~5–9× (TSD §14.12).
- **The weak-probe and full engines are two independent implementations of the same physics** (TSD §14.4, §14.5). Do not derive one from the other's code. Their agreement in the weak-probe limit (T-LIN-1, T-LIN-7) is the main correctness check, so a disagreement is a `finding`, never a reason to edit one engine until it matches the other.
- **Do not implement the QZ per-velocity trick** (TSD §14.5): measured to be slower in NumPy.

## 4. Milestones (do them in order; each ends green in CI)

| # | Scope | FR IDs | Done when |
|---|---|---|---|
| M0 | Repo scaffold, `pyproject.toml`, `Makefile`, CI (lint, type check, tests), `constants.py`, `thermal.py` | FR-PHY-1,2 | `make test lint` green; TSD §11 table reproduced |
| M1 | Gaussian core: `PIChannel`, composition, states, $\tilde\nu_-$, measures, separability conditions | FR-GAU-1..6 | All identity tests in TSD §10 pass |
| M2 | Stochastic channel: small-scale, log-gamma + baselines, compound, link budget, samplers | FR-CH-1..3,5,6 | Analytic vs MC within bands; $\mathcal Z_K$ and mass-above-1 references reproduced |
| M3 | Receivers + entanglement service + E3, E4 | FR-RX-1..6, FR-SVC-E1,2 | Threshold formula vs root-finding test; E3/E4 figures generated |
| M4 | Classical service + crossover + E5, E6, E8 | FR-SVC-C1..4 | Capacity ordering/monotonicity properties hold; E6 shows $S_{E,\rm BBR}\propto f$ slope |
| M5 | Diversity (ported from V0) + E7 | FR-DIV-1..4 | Rate function matches numeric Legendre transform; $N_{\rm req}$ table produced and compared to V0 values, discrepancies logged |
| M6 | Experiment runner, caching, metadata, LaTeX export, `make paper-figs`, E1, E2, optional obstacle generator | FR-EXP-1..6, FR-CH-4 | Fresh clone → `make paper-figs` succeeds; `docs/RESULTS.md` lists every reference value with its source |
| M7 | Receiver physics model: scheme, weak-probe engine, full Lindblad engine, Doppler averaging with `auto` method, response, readout, `LindbladRARModel`, E9 | FR-RX-6, FR-LIN-1..7 | T-LIN-1..11 green (T-LIN-9 may skip); E9 figures generated within NFR-3; per-class costs recorded in `docs/RESULTS.md` |

M7 can start any time after M1 (it needs only the Gaussian core for $N_{\rm amp}$).

After each milestone, append a short entry to `CHANGELOG.md`: what was built, test count, open findings.

## 5. Commands

```
make install      # create env, install pinned deps (editable)
make test         # pytest -q (unit + validation + regression)
make lint         # ruff check + ruff format --check + mypy --strict src/
make figs         # run all experiments with cache
make paper-figs   # force-run E1–E9, export PDFs/CSV/LaTeX to results/paper/
make bench        # per-velocity-class cost of both receiver engines, writes to docs/RESULTS.md
qmw run configs/<exp>.yaml [--force]
```

## 6. Code standards

- Python ≥ 3.11, full type hints, `mypy --strict` clean on `src/`.
- Pure functions on NumPy arrays, vectorized over the transmissivity axis; frozen dataclasses for parameter bundles.
- Validate inputs at public boundaries (e.g. $0<\bar\eta\le1$, $N_{\rm amp}\ge\frac12$, physical $(\tau,y)$); raise `ValueError` with a message naming the parameter and the violated condition.
- Monte Carlo functions return estimate **and** standard error.
- No prints in library code; use `logging`.
- Figures: IEEE single-column width (3.5 in), vector PDF + PNG, serif font, colorblind-safe palette, no in-figure titles, axis labels with units.
- Tests: `tests/unit`, `tests/validation` (cross-method, limits), `tests/regression` (reference values). Use `hypothesis` for property tests. Fixed seeds.
- Coverage ≥ 90% on `src/`.

## 7. `QUESTIONS.md` format

```
## Q-<number> [blocking | non-blocking | finding] <short title>
- Context: <file/function, FR ID, TSD section>
- Observation: <what you saw, with numbers>
- Options: <A / B / ...>
- Recommendation: <your pick and why>
- Status: open
```

Raise a `blocking` question only when continuing would require guessing physics or a parameter value. Otherwise mark `non-blocking`, pick the most conservative option, state it, and continue.

## 8. Git

- Small commits, imperative subject, reference IDs: `feat(channel): truncated ncx2 sampler (FR-CH-1)`.
- One milestone per branch; do not rewrite history.
- Never commit `results/`, large data, or credentials.

## 9. When you finish a session

Report to Abdurrahman, briefly: milestone status, tests passing/failing, new `QUESTIONS.md` entries (blocking first), and anything that changed a number in `docs/RESULTS.md`.