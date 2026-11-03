# CHANGELOG

## M0 — scaffold, constants, thermal (FR-PHY-1, FR-PHY-2)
- Built: `pyproject.toml` (pinned deps), `Makefile`, GitHub Actions CI (lint + test, py3.11/3.12),
  `constants.py`, `thermal.py` (`n_th`, `n_th_rj`, `eta_star`, `one_minus_eta_star`), frozen `tests/tolerances.py`.
- Tests: 19 passing (unit, hypothesis property, TSD §11 regression), coverage 100% on `src/`; ruff + mypy --strict clean.
- TSD §11 table (n_th, eta*, 1-eta*) reproduced at 5/30/100/300 GHz.
- Open findings: Q-1 (non-blocking) — FR-PHY-1 RJ error bound is slightly too tight.

## M1 — Gaussian core (FR-GAU-1..6)
- Built: `gaussian/channels.py` (`PIChannel`, constructors, `@` composition, `apply_to_mode_b`,
  attenuator-cascade closed form), `gaussian/states.py` (TMSV/TMST, physicality),
  `gaussian/entanglement.py` (general and standard-form `nu_-`, negativity, log-negativity,
  teleportation fidelity, TMSV/TMST separability conditions).
- Tests: 65 passing total (46 new), 100% coverage on `src/`, ruff + mypy --strict clean.
  TSD §10 identity/limit checks done: associativity, §4.3 closed form, general vs closed form,
  §5.4 condition vs `nu_-` on 10^4-point grids (r = 0.1, 0.8, 2), TMST vs TMSV reduction.
- Resolved: Q-1 (FR-PHY-1 AC amended in docs/FSD.md per Aan, option B).
- Open findings: Q-2 (non-blocking) — general `nu_-` formula loses precision when `nu_- << 1`.
