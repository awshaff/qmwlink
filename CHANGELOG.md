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

## M4 — Classical service, crossover, E5/E6/E8 (FR-SVC-C1..C5)
- Built: `services/classical.py` (cancellation-free `g_entropy`, Holevo / heterodyne / homodyne / measuring-receiver capacities,
  `ClassicalService`: ergodic by nested quadrature, `C_eps`, rate outage via `brentq`, throughput, `N_s = P_t/(h f B)`),
  `services/crossover.py` (closed-form and numeric required `S_E`/`N_amp` for a dB penalty, DSB image floor with NaN + reason,
  capacity-based variant), `services/throughput_map.py` (log-log band interpolation with masking, `rho = 10 log10(R_RAR/R_LNA)`),
  `experiments/{e5,e6,e8,rx_cfg}.py`, schemas `E5Config`/`E6Config`/`E8Config`/`RxSpec`/`BandRx`, `load_literature`
  (FR-EXP-7), `configs/{e5,e6,e8,literature_rar}.yaml`; entry point extended to `{e3|e4|e5|e6|e8}`.
- References reproduced (5 GHz, 300 K, eta -> 0): ssb 3 dB ratio 0.99823 (N_amp 1245.27); dsb 4 dB 0.71659 (641.71), dsb 6 dB 1.40849;
  dsb 3 dB NaN naming the 3.009 dB floor; floors 3.009/3.000/2.976/2.907 dB at 5/30/100/300 GHz. Closed form vs brentq to 1e-6.
  E6 RJ slope of S_E,BBR = 0.9997 (fit f <= 30 GHz).
- Properties tested: C_chi >= C_het, C_hom; all increasing in eta and -> 0 as eta -> 0; C_rx non-increasing in N_amp, = C_het at
  ssb/N_amp = 1/2, dsb <= ssb; C_eps = C(F^-1(eps)) to 1e-6 and rate-outage round trip; ergodic vs MC; power-limited rho asymptote;
  equal-bandwidth rho <= 0; masking outside the band range; missing `source`/`field_convention` fails validation.
- Fix found while testing: `g(x)` as `(x+1)log2(x+1) - x log2 x` cancels at large x (Holevo < homodyne by 2e-12 at n_th = 1250);
  now `log2(1+x) + x log2(1+1/x)`.
- Tests: 361 total (all tiers, ~100 s), coverage 99%; ruff + mypy --strict clean.
- Runtime: E6 under 1 s, E8 34 s, E5 1 min 48 s (8 x 8 x 3 map).
- Open findings: Q-10 ("smallest N_amp" wording), Q-11 (E5/E6/E8 placeholder inputs); Q-7, Q-8, Q-9 unchanged.

### M4 revision after FSD v0.6.3 / TSD v0.4.3 (closes Q-7..Q-11)
- Q-8: `EntanglementMetrics` reports `log_neg` (unconditional) and conditional `nu_minus_cond`, `fidelity_opt_cond`,
  `fidelity_bk_cond`, `log_neg_cond`, each next to `one_minus_p_out` (from the survival function); NaN with reason
  "no entangled realizations" when `1 - P_out < 1e-12`.
- Q-7: `LargeScale` takes per-metre `stochastic_per_m` per band (fully correlated path, `mu_G = m1 d`, `sigma_G = s1 d`, `k` constant in `d`),
  with `stochastic_source`; legacy per-link form kept and flagged `per_link` in `meta.json`; `correlation: partial` rejected.
  E3/E5/E8 configs carry the sourced ITU O2 absorption and KHU Seoul weather values (annual). E4 uses the published
  transducer classes (`N_t = eta_t N_add_in` checked); 5 GHz thresholds match TSD §16.5.
- Q-11: `literature_rar.yaml` filled from TSD §16.3 (`f_hz`, optional `b_hz`, `includes_bbr`); N_amp / T_equiv of all six entries reproduce the
  TSD table. E5 uses the RAR points with `b_hz` (13.9-96 GHz) and the LNA datasheet table (5-243 GHz), log-log interpolated and masked
  outside, nearest-entry sideband, P_t = 0.2 W, and adds the equal-bandwidth case (`rho_equal_b_db`). E8 runs at 30 GHz.
  A unit slip (nV/cm vs V/m) in my first YAML transcription was caught by that cross-check and fixed.
- Q-9: `filterwarnings = error::RuntimeWarning` in pytest. Q-10: no code change (already the largest tolerable N_amp); AC (f) 1.01x check added.
- Not done: `scripts/derive_inputs.py` (TSD §16, needs optional `itur`, not installed; the exact integration ranges would need confirming).
- Tests: 368 total, ruff + mypy --strict clean. E5 now takes several minutes (k < 1 weather laws).
- Preview (placeholders replaced by sourced inputs, E5, P_t = 0.2 W): rho between -36.6 and -12.0 dB over 13.9-96 GHz; equal-bandwidth
  rho between -1.17 and -0.28 dB, i.e. the RAR does not win anywhere on this grid, mainly through bandwidth and then noise.

## M5 — Diversity and E7 (FR-DIV-1..4)
- Built: `services/diversity.py` (`array_outage`, exact SC tail `sc_exact_tail` with domain error for x < 1/2, `sc_chernoff_bound`,
  `sc_sample`, `cramer_rate` + numeric Legendre oracle, `exponent_bound`, `design_dimension` / `design_law_holds`, `n_req` by exponential
  search + integer bisection), `experiments/e7.py`, `E7Config`, `configs/e7.yaml`; entry point now `{e3|e4|e5|e6|e7|e8}`.
  MRC array outage reuses `SmallScale(n_elements=N)` and the compound PDT (ncx2 floor helper), so large N needs no new code.
- Reference: all 12 `N_req` integers of FSD FR-DIV-3 (5/30/100/300 GHz x K = 0/10/100, eps = 1e-3) reproduced exactly (5 GHz K = 0: 20 017 832);
  N and N-1 bracket the target. No discrepancy with the table, so nothing new in QUESTIONS.
- Validation: N = 1 reduces to the single-element cdf/sf; Chernoff >= exact SC tail on x in [1/2, 1); Cramer rate R_K(1) = 0,
  R_0 = a - 1 - ln a, equal to the numeric Legendre transform (K up to 100); exponent bound >= exact cdf; design law implies outage <= eps;
  MRC sum-sampling vs array cdf (MC-agree, N = 1e6); sum vs per-element two-sample KS (N = 8); SC exact tail vs per-element MC; N > 64 error.
- Found while testing: the Legendre oracle needed a wide search interval (optimal s is far below 0 for K = 100, a = 0.2); oracle fixed, closed form unchanged.
- Tests: 392 total (all tiers), coverage 99%; ruff + mypy --strict clean. E7 runs in 3 s.

## M6 — Runner, cache, CLI, LaTeX export, E1, E2, obstacle generator (FR-EXP-1..6, FR-CH-4)
- Built: `experiments/registry.py` (E1-E8; E9 joins in M7), `experiments/runner.py` (schema validation with a clear
  message naming file and field, results cache keyed by config hash plus a hash of external input files, `meta.json`
  completed with seed, wall time, package versions), `cli.py` (`qmw run|list|figs|paper-figs|tables`, also
  `python -m qmwlink.cli`), `io/latex.py` (booktabs parameter / thermal / N_req / literature tables), `io/export.py`
  (`paper_figs`: force-run, copy PDF/PNG/CSV/meta to `results/paper/<experiment>/`, tables to `results/paper/tables/`),
  `io/literature.py`, `channel/obstacles.py` (compound-Poisson generator, exact moment match, diagnostic histogram),
  `experiments/e1.py` (pdf vs untruncated surrogate, removed mass, Delta P_out and ratio vs eta_s_bar and vs distance),
  `experiments/e2.py` (three laws matched on Psi moments, Delta P_out, Delta C/C for Holevo / heterodyne / RAR, V0
  clip with both masses), `configs/{e1,e2,defaults}.yaml`, `docs/RESULTS.md`. Makefile: `exp ID=E<n> [FORCE=1]`, `figs`,
  `paper-figs`.
- Fixed: `test_property_fidelity_opt_ge_bk` was failing on `main` at r ~ 5e-9 (round-off of the general nu_- form,
  Q-12); the property now uses the closed form at the frozen slack and the general form with a documented allowance.
- Validation: second run is a cache hit, `--force` and a changed literature file recompute; unknown keys rejected;
  tables compile in IEEEtran (pdflatex, skipped when absent); E1 identity P_out/P_out^surr = 1/Z to 1e-12; E2 laws agree
  as sigma_Psi -> 0 (rtol 1e-6), V0 mass above 1 = Phi(-1), clipped mass vs MC; obstacle moments and atom vs MC.
- Tests: 422 total (all tiers, ~200 s), coverage 99.0%; ruff + mypy --strict clean. E1 runs in 92 s, E2 in 49 s.
- Open: Q-12 (finding), Q-13..Q-16 (non-blocking). `scripts/derive_inputs.py` not written (Q-16).
