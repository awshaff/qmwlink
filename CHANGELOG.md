# CHANGELOG

## M0 — scaffold, constants, thermal (FR-PHY-1, FR-PHY-2)
- Built: `pyproject.toml` (pinned deps), `Makefile`, GitHub Actions CI (lint + test, py3.11/3.12),
  `constants.py`, `thermal.py` (`n_th`, `n_th_rj`, `eta_star`, `one_minus_eta_star`), frozen `tests/tolerances.py`.
- Added after spec v0.6/0.4 review: FR-PHY-3 (`effective_aperture_m2`, `s_e_bbr` Planck/RJ,
  `s_e_bbr_energy_density_rj`); FR-PHY-1 tests tightened to `rtol=1e-9` with the 300 GHz spot value;
  tolerances centralised in `tests/tolerances.py`; Makefile `exp` stub.
- Tests (initial): 19 passing (unit, hypothesis property, TSD §11 regression), coverage 100% on `src/`; ruff + mypy --strict clean.
- TSD §11 table (n_th, eta*, 1-eta*) reproduced at 5/30/100/300 GHz.
- Open findings: none (Q-1, the RJ error bound, was fixed in the spec by FSD v0.6).

## M1 — Gaussian core (FR-GAU-1..6)
- Built: `gaussian/channels.py` (`PIChannel`, constructors, `@` composition, `apply_to_mode_b`,
  attenuator-cascade closed form), `gaussian/states.py` (TMSV/TMST, physicality),
  `gaussian/entanglement.py` (general and standard-form `nu_-`, negativity, log-negativity,
  teleportation fidelity, TMSV/TMST separability conditions).
- Tests: 65 passing total (46 new), 100% coverage on `src/`, ruff + mypy --strict clean.
  TSD §10 identity/limit checks done: associativity, §4.3 closed form, general vs closed form,
  §5.4 condition vs `nu_-` on 10^4-point grids (r = 0.1, 0.8, 2), TMST vs TMSV reduction.
- Open findings: none.

### M1 revision after spec v0.6 / TSD v0.4
- `nu_minus_general` now uses the mandated cancellation-free form (resolves Q-2); optional analytic `det` on the closed form.
- `teleportation_fidelity` renamed `fidelity_opt`; added `fidelity_bk`; `additive_noise(n_quanta)` -> `n_add`.
- Added `tmst_one_minus_eta_star` / `tmst_eta_star` (brentq on omega = 1 - eta; reference values reproduced).
- New tests: FR-GAU-1 TMST grid and c^2 identity; FR-GAU-2(c) combining-network threshold (V0 div_netthreshold, exact
  reduction to eta*); FR-GAU-4 (a)-(d); FR-GAU-5 exclusion rule |y-(1+tau)| < 1e-9; FR-GAU-6 (c).
- Tolerances centralised in `tests/tolerances.py`. 123 tests passing (incl. M0), 100% coverage; Q-1 and Q-2 closed.
