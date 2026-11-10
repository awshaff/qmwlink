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

## M2 — Stochastic channel (FR-CH-1,2,3,5,6)
- Built: `numerics/` (`integrate.checked_quad` with the TSD §6.4 mpmath fallback, `sampling.rejection_sample`,
  `mcagree` Bonferroni/KS rule of FSD §4.3), `channel/small_scale.py` (`SmallScale`: pdf/cdf/sf/ppf, truncated
  moments, `sample_sum`, `sample_elements`), `channel/large_scale.py` (`ShiftedLogGamma`, `DeterministicLoss`,
  `LogNormalV0` flagged non-physical, `ConditionedLogNormal`), `channel/compound.py` (`CompoundPDT`: cdf/sf/pdf/ppf/
  moments/sampler in Psi-space), `channel/link_budget.py` (Goubau default, flagged `friis_clipped`).
  FR-CH-4 (obstacle generator) is optional and assigned to M6, so not built here.
- References reproduced: Z_K = 0.63249 / 0.54404 / 0.51689; log-gamma mean 0.96098; V0 mass above 1 = 0.158655;
  tau^2 = 2.7816, eta_geo = 0.93806; absorption range 555.47 m; A = 1.82 m^2 for eta_bar = 0.9999 at 10 m.
- Validation: MC-agree on cdf at 50 points for 5 compound cases; quadrature moments rtol 1e-8; KS D < 2e-3 at 1e6
  samples; sample_sum vs sample_elements (two-sample KS, N <= 64); mpmath at N in {1e5, 1e7} down to ~2e-8 tails.
- Tests: 226 total, 99.8% coverage on `src/`, ruff + mypy --strict clean. The slow tier (mpmath, MC) takes ~90 s.
- Open findings: Q-3 (scipy central chi2 cdf inaccurate in far lower tail at large N; worked around),
  Q-4 (G_enh scope), Q-5 (ncx2.pdf nan far from the mean; worked around).

### M2 revision after spec TSD v0.4.1 / FSD v0.6.1 (closes Q-3, Q-4, Q-5)
- `numerics/special.py`: `ncx2` floor helper (`NC_FLOOR = 1e-300`, cdf/sf/pdf/ppf, regularized gamma wrappers); every
  `SmallScale` cdf/sf/pdf/ppf now goes through it. Pinned `test_ncx2_floor_tail` (four N, both tails, rtol 1e-6).
- `SmallScale`: adaptive moment window with neglected-mass check (`moment_window`, `neglected_mass`; cap 320 sd then raise),
  `surrogate_cdf` (RQ1, flagged non-physical). `ShiftedLogGamma` routes k > 1e5 through the helper.
- `link_budget`: single receive-side `g_enh` (>= 1); `g_enh_tx`/`g_enh_rx` removed.
- `large_scale`: `matched_laws` (RQ2 moment matching), `LogNormalV0.sample_eta_clipped` (`eta_gt1: clip`, returns the mass).
- Tests: 245 total (all tiers, ~200 s), coverage 99.9%; ruff + mypy --strict clean; mpmath reference extended to N = 1e8 (K = 0).
- Deferred to M3 (needs the service layer): FR-SVC-E2 RQ1 outage identity, FR-CH-3 (b) in terms of P_out / ergodic capacity.

## M3 — Receivers, entanglement service, E3/E4 (FR-RX-1..5, FR-SVC-E1,2)
- Built: `receivers/` (`MeasuringReceiver` with required sideband, `ideal_heterodyne`, `IdealHomodyne`, quantum-limited amplifier,
  `electronic_receiver` with required `tn_convention`, `RydbergReceiver` with required `field_convention` and no aperture argument
  in the `S_E -> N_amp` mapping, `coherent_transducer`), `services/entanglement.py` (closed-form `eta_star_rx` / `window_rx`,
  numeric TMSV/TMST threshold, `EntanglementService`: `P_out`, E[E_N], E[nu~_-], E[F_opt], E[F_BK], RQ1 `truncation_effect`),
  `CompoundPDT.cdf_surrogate`, `experiments/{schema,common,e3,e4}.py`, `plotting/style.py`, `configs/e3.yaml`, `configs/e4.yaml`
  (temporary entry point `python -m qmwlink.experiments {e3|e4} <config>`; the cached runner and `qmw` CLI stay in M6).
- References reproduced: N_amp = 3.438e5 / 1.719e5, T_n = 8.25e4 K; T_n conventions 0.9102 / 0.5000 / 1.0000; eta*_rx = 0.99960
  (N_add = 1/2) and 0.999219 (transducer); closed form vs direct root of nu~_- = 1 to 1e-12.
- Validation: V0 and log-gamma limiting cases 1-4 (clamp active), RQ1 identity P_out/P_out^surr = 1/Z to 1e-12, MC agreement
  (Bonferroni rule) for the outage and all four metrics.
- Tests: 304 total (all tiers, ~145 s), coverage 99.3%; ruff + mypy --strict clean. E3 full run 3 min 2 s (NFR-3 ok), E4 3 s.
  Figures: E3 `fig_p_out`, `fig_p_ent` (1 - P_out, log), `fig_e_n`; E4 `fig` (written to `results/`, gitignored).
- Open findings: Q-7 (placeholder inputs), Q-8 (E[nu~_-], E[F] definition), Q-9 (scipy ncx2.sf at nc >~ 1e10). Q-6 (E3 runtime,
  10.5 min -> 3 min by caching the frozen ncx2) closed.
- V0's closed-form outage expressions are not in the repo; Cases 1-3 under the V0 log-normal were derived from first principles
  (Phi/clamped-cdf quadrature), not copied from V0.
