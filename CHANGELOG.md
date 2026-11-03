# CHANGELOG

## M0 — scaffold, constants, thermal (FR-PHY-1, FR-PHY-2)
- Built: `pyproject.toml` (pinned deps), `Makefile`, GitHub Actions CI (lint + test, py3.11/3.12),
  `constants.py`, `thermal.py` (`n_th`, `n_th_rj`, `eta_star`, `one_minus_eta_star`), frozen `tests/tolerances.py`.
- Added after spec v0.6/0.4 review: FR-PHY-3 (`effective_aperture_m2`, `s_e_bbr` Planck/RJ,
  `s_e_bbr_energy_density_rj`); FR-PHY-1 tests tightened to `rtol=1e-9` with the 300 GHz spot value;
  tolerances centralised in `tests/tolerances.py`; Makefile `exp` stub.
- Tests (initial): 19 passing (unit, hypothesis property, TSD §11 regression), coverage 100% on `src/`; ruff + mypy --strict clean.
- TSD §11 table (n_th, eta*, 1-eta*) reproduced at 5/30/100/300 GHz.
- Open findings: Q-1 (non-blocking) — FR-PHY-1 RJ error bound is slightly too tight.
