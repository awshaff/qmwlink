# QUESTIONS

## Q-1 [non-blocking] FR-PHY-1 RJ error bound is slightly too tight
- Context: `thermal.n_th_rj`, FR-PHY-1 AC ("RJ relative error < hf/2k_BT"), TSD §3.1.
- Observation: with x = hf/k_BT, (n_RJ - n)/n = (e^x - 1)/x - 1 = x/2 + x^2/6 + ... exactly, which is
  always > x/2. E.g. 3.013 GHz, 1 K (x = 0.1446): relative error 0.0759 vs bound 0.0723.
  The bound holds only to leading order in x.
- Options: A) keep the AC as an asymptotic statement (error -> x/2); B) amend the AC to
  "error < (e^x - 1)/x - 1" or "< x/2 (1 + x/3)" or similar.
- Recommendation: B. Tests currently assert the exact identity and the x -> 0 limit; no tolerance was loosened.
- Status: resolved — superseded by FSD v0.6, which adopts the exact identity (rtol 1e-9, 300 GHz spot value 2.438e-2). Tests follow it.

## Q-2 [non-blocking] General PT-eigenvalue formula loses precision when nu_- << 1
- Context: `entanglement.nu_minus_general`, FR-GAU-4 AC ("general and closed forms agree to 1e-12"), TSD §5.2.
- Observation: for pure TMSV, nu_- = exp(-2r). The general formula (D - sqrt(D^2 - 4 det V))/2 gives
  relative error 6.8e-9 (abs 4.5e-11) at r = 2.5 (nu = 6.7e-3), since D^2 and 4 det V nearly cancel.
  The closed form with analytic det(ab - c'^2) = 1 is exact (< 1e-12 relative). The 1e-12 absolute
  agreement holds for nu_- >~ 0.1 (checked on random valid CMs with r <= 2).
- Options: A) keep AC, restrict to nu_- not tiny; B) amend AC to a relative tolerance scaled by 1/nu_-;
  C) compute nu_-^2 as det V / nu_+^2 in the general routine.
- Recommendation: A for now (general formula is test-only per TSD §5.2); closed form gets an optional
  analytic `det` argument, used in production paths. No tolerance was changed.
- Status: resolved — TSD v0.4 §5.2 / FSD FR-GAU-4 now mandate the cancellation-free form (option C). Implemented in `nu_minus_general`; r = 3 TMSV now matches exp(-6) to ~2e-12 and 10^4 random CMs (b up to 5e3) agree with the closed form to rtol 1e-10. My earlier recommendation (option A) was wrong.

## Q-3 [finding] scipy central chi-square cdf is inaccurate in the far lower tail at large N
- Context: `channel/small_scale.py`, FR-CH-1, TSD §6.1 ("K = 0: use scipy.stats.chi2"), NFR-2 (tail probabilities
  to 1e-8 within 1%), scipy 1.18.1.
- Observation: against a 40-digit mpmath reference (series and direct quadrature agree), the lower-tail cdf at
  z = -5.5 sd of chi2(df = 2N) (value ~1.8e-8) has relative error 8e-15 (N = 1e5), 1.6e-6 (N = 1e6), 2.3e-2 (N = 1e7)
  and 3.2e-1 (N = 1e8). `special.gammainc` and `chi2.cdf` are identical; cause is exponent cancellation in
  `a ln y - y - lgamma(a+1)` (~1e9 magnitude). `ncx2(nu, nc=0)` delegates to the same code; `ncx2(nu, nc=1e-300)`
  is accurate to 7.6e-14 (N = 1e7) and 2.5e-13 (N = 1e8) and agrees with chi2 to 1e-15 for N <= 100.
- Options: A) follow TSD literally (chi2) and accept the error; B) K = 0 uses `ncx2(nu, nc=1e-300)`;
  C) own Stirling-corrected gamma tail.
- Recommendation: B (implemented). `K == 0` closed forms (truncated exponential, Erlang) are tested at small N.
  The TSD §6.1 K = 0 wording should be amended to say so. Also note `ncx2.pdf` returns nan ~27 sd below the
  mean for N = 1e7, K = 10 (true value < 1e-100); moments integrate from mean - 20 sd instead of masking it.
- Status: closed (resolved by TSD v0.4.1 / FSD v0.6.1; implemented in M2 revision)

## Q-4 [non-blocking] Scope of the enhancement factor G_enh in the link budget
- Context: `channel/link_budget.py::link_budget`, FR-CH-6, TSD §6.5 ("Either is multiplied by an optional
  enhancement factor G_enh >= 1").
- Observation: "either" is ambiguous (transmit aperture, receive aperture, or both with one factor).
  TSD §7.4 says G_enh enters "only through A_r".
- Options: A) one factor on A_r only; B) separate `g_enh_tx`, `g_enh_rx`.
- Recommendation: B (implemented, both default to 1), so A is `g_enh_tx=1`. Confirm intended use.
- Status: closed (TSD v0.4.1: single receive-side `G_enh`; `g_enh_tx` removed)

## Q-5 [non-blocking] `ncx2.pdf` returns nan far from the mean (scipy 1.18.1)
- Context: `SmallScale.moment` (TSD §6.1), N = 1e7, K = 10 (and K = 1e8, N = 1).
- Observation: `stats.ncx2(...).pdf` returns nan in bands ~27 sd from the mean where the true density is < 1e-100.
- Options: A) mask nan as 0; B) integrate only over mean +/- 20 sd and raise on any non-finite value.
- Recommendation: B (implemented); neglected mass < 1e-80.
- Status: closed (resolved by TSD v0.4.1 / FSD v0.6.1; implemented in M2 revision)

## Q-6 [finding] E3 exceeded the 5-minute budget (NFR-3); profile and fix
- Context: `experiments/e3.py` with `configs/e3.yaml` (4 bands x 2 coupling models x 60 distances x 3 K = 1440 points), FSD NFR-3, NFR-7.
- Observation: first full run took 10 min 30 s. cProfile on 48 points: 30.3 s, of which 19.6 s (65%) was
  `scipy.stats.ncx2(...)` frozen-distribution construction (0.6 ms x 33 366 calls inside the compound quadrature, via
  `numerics/special._frozen`); the actual cdf/sf evaluations were a minor share.
- Options: A) cache the frozen distribution per `(nu, lam)`; B) vectorize the compound quadrature; C) reduce the grid.
- Recommendation: A (implemented, `functools.lru_cache`): 6.5 s for the same 48 points (4.7x), full E3 now 3 min 2 s.
  No algorithm, tolerance or grid changed; no compiled code.
- Status: closed (A implemented; E3 now within budget, see CHANGELOG M3)

## Q-7 [non-blocking] E3/E4 placeholder inputs
- Context: `configs/e3.yaml`, `configs/e4.yaml`, TSD §12 ("placeholders carry [A]").
- Observation: only the 5 GHz clear-air absorption (1.44e-6 /m, Gonzalez-Raya et al. 2022) has a verified source. The 30/100/300 GHz
  entries are set to 0.0 (no absorption) and flagged `[A]` in the config and in `meta.json`. The large-scale stochastic loss
  (mean 1e-3, std 5e-4), the apertures (2 m^2) and the E4 transducer classes (eta_t 0.8/0.6, n_t 0.01) are also placeholders.
- Options: A) supply ITU-R P.676 values with sources; B) keep zeros and restrict claims to 5 GHz.
- Recommendation: A, by Abdurrahman. Until then E3 absolute numbers above 5 GHz are optimistic and must not be quoted.
- Status: closed 2026-11-10: sourced inputs in configs/literature_q7_q11.yaml and TSD §16 (FSD v0.6.3 / TSD v0.4.3); implemented in M4 revision. Still [A]: apertures, P_t, B_LNA at 100 GHz, 300 GHz LNA, RAR bandwidths for Jing/Cai/She, Borowka sideband, Yang journal volume, full-path weather correlation (TSD §16.7)

## Q-8 [non-blocking] Meaning of E[nu~_-], E[F_opt], E[F_BK] in FR-SVC-E2
- Context: `services/entanglement.py::EntanglementService.metrics`, TSD §8.1 ("E[nu~_-], E[F_opt], E[F_BK] likewise").
- Observation: E[E_N] = int_{eta*}^1 E_N f d eta is unambiguous (E_N = 0 below eta*). For nu~_- and the fidelities the
  integrand does not vanish below eta*, so "likewise" can mean the restricted integral or the full mean.
- Options: A) restricted to eta > eta* (entangled realizations), divide by 1 - P_out for the conditional mean;
  B) full mean over all eta.
- Recommendation: A (implemented, matches the literal formula and keeps the integration domain finite); the full mean is
  a one-line extension if wanted.
- Status: closed 2026-11-10: conditional averages (Abdurrahman's decision), TSD §8.1 / FSD v0.6.2; implemented as nu_minus_cond / fidelity_opt_cond / fidelity_bk_cond, NaN when 1-P_out < 1e-12

## Q-9 [finding] scipy `ncx2.sf` warns and may be unreliable for nc >~ 1e10
- Context: `numerics/special.ncx2_sf`, FR-CH-1, K -> infinity limit tests (FR-SVC-E2 Case 1).
- Observation: at N = 1, K = 1e10 (nc = 2e10) `ncx2.sf` 8 sd above the mean raises "Series did not converge" (a RuntimeWarning);
  the cdf in the bulk and lower tail matched Phi to 4 digits at K up to 1e10. Tests use K = 1e9, which is warning-free.
- Options: A) document the limit (K <= 1e9 for the sf upper tail); B) add a normal-approximation guard.
- Recommendation: A for now; no configured experiment uses K above 100.
- Status: closed 2026-11-10: accepted as a documented limit (K <= 1e9, TSD §6.1); RuntimeWarning promoted to an error in pytest (pyproject filterwarnings)

## Q-10 [non-blocking] "Smallest N_amp" in the capacity-based crossover (TSD §8.3)
- Context: `services/crossover.py::crossover_capacity`, FR-SVC-C4.
- Observation: TSD §8.3 says "smallest N_amp with C_rx >= (1-delta) C_het". C_rx is non-increasing in N_amp, so every
  N_amp down to 1/2 satisfies the constraint and "smallest" is trivially 1/2 (ssb) or infeasible (dsb). The meaningful
  quantity is the largest tolerable N_amp, which is what the dB-penalty closed form also returns.
- Options: A) largest N_amp with C_rx >= (1-delta) C_het (implemented); B) literal "smallest" (trivial).
- Recommendation: A; please confirm and fix the TSD wording.
- Status: closed 2026-11-10: option A (largest tolerable N_amp), TSD §8.3 / FSD FR-SVC-C4 (f)

## Q-11 [non-blocking] E5/E6/E8 placeholder inputs
- Context: `configs/e5.yaml`, `configs/e8.yaml`, `configs/literature_rar.yaml`, FR-SVC-C5 (d), FR-EXP-7.
- Observation: all per-band RAR (S_E, B) and LNA (T_n, B) values, P_t, N_s and the E8 receivers are `[A]` illustrative
  placeholders with `source: "[A] placeholder"`; `literature_rar.yaml` is empty (E6 renders without the overlay and warns).
  With the placeholder bandwidths (RAR 1 MHz, LNA 100 MHz) the E5 map is bandwidth-limited everywhere (rho about -19 dB,
  no rho = 0 contour); this reflects the inputs, not a result. Same absorption/aperture/stochastic-loss placeholders as Q-7.
- Options: A) supply sourced RAR/LNA values and literature points; B) keep placeholders and quote nothing absolute.
- Recommendation: A, by Abdurrahman, together with Q-7.
- Status: closed 2026-11-10: sourced RAR / LNA / bandwidth / P_t inputs in TSD §16.3-16.4 and configs/literature_q7_q11.yaml; E5 adds the equal-bandwidth case

## Q-12 [finding] General nu_- formula has ~sqrt(eps) absolute error when nu_+ ~ nu_- (r -> 0)
- Context: `gaussian/entanglement.py::nu_minus_general`, FR-GAU-4 (c), TSD §5.2; `tests/validation/test_entanglement.py::test_property_fidelity_opt_ge_bk`.
- Observation: `hypothesis` found r = 5.3e-9, eta = 0.875, n_th = 0, n_JPA = 0, where the general form gives
  F_opt = 0.5000000018626 < F_BK = 0.5000000024779 (relative 1.2e-9, above the 1e-9 slack of the test). Against a
  40-digit reference the general form is off by +2.5e-9 in nu_-; the standard-form closed form is exact to 1e-16.
  Cause: `Delta^2 - 4 det V = ((a-b)^2 + 4c'^2)(a+b)^2` is formed by subtraction of two numbers near 4 whose
  difference (1.6e-15) is at round-off, so the square root carries an absolute error of about sqrt(eps) ~ 1e-8.
  The cancellation-free numerator of TSD §5.2 does not remove this second cancellation. The failure was on `main`
  (stored in the local hypothesis database), so `make test` was red before M6.
- Options: A) keep the general form for tests only (as the TSD already says) and check the property with the
  closed form at the frozen 1e-9 plus the general form with an absolute allowance; B) restrict the property to
  r >= 1e-3; C) amend TSD §5.2 with a discriminant computed from the blocks.
- Recommendation: A (implemented). New frozen tolerance `ABS_F_GENERAL_DEGENERATE = 3e-8` in `tests/tolerances.py`;
  the 1e-9 slack is unchanged and now applies to the production (closed-form) path. No production code changed:
  every service uses the closed form. FR-GAU-4 (a) (rtol 1e-10 on random CMs) still passes.
- Status: open

## Q-13 [non-blocking] Meaning of "clip" for the V0 log-normal (eta_l vs total eta) and which mass to report
- Context: `experiments/e2.py`, FR-CH-3, TSD §6.2 ("evaluate the service at min(eta, 1) ... report the affected mass");
  `LogNormalV0.sample_eta_clipped` (M2).
- Observation: TSD/FSD clip the total eta = eta_l eta_s; the M2 helper clips eta_l and reports P(eta_l > 1). These differ:
  clipping the total leaves P_out untouched (min(eta,1) <= eta* < 1 iff eta <= eta*) and affects only the mass
  P(eta_l eta_s > 1), which is far smaller than P(eta_l > 1) (E2 at sigma/Psi_bar = 1: 2.4e-4 vs 0.159).
- Options: A) total-eta clip per the TSD, report both masses; B) clip eta_l (changes P_out and inflates the clip effect).
- Recommendation: A (implemented in E2: columns `clipped_mass` = P(eta > 1) and `mass_eta_l_above_one`).
  `sample_eta_clipped` is left as is (used only by its own M2 test); say if it should be removed or redefined.
- Status: open

## Q-14 [non-blocking] `configs/defaults.yaml` is a parameter table, not a merge layer
- Context: FSD §6 ("Default parameter sets live in configs/defaults.yaml"), FR-EXP-4, FR-EXP-5.
- Observation: M3-M5 configs are self-contained, and the cache is keyed by the hash of the experiment file. Merging a
  defaults file into every config would make results depend on a second file that the hash does not cover.
- Options: A) `defaults.yaml` lists the shared parameters with sources, feeds the LaTeX parameter table, and a test
  checks it against `e3.yaml`/`e5.yaml`; B) implement merging and hash the merged config.
- Recommendation: A (implemented). The runner also hashes external input files (`literature_rar.yaml`) into the
  cache check (`inputs_hash` in `meta.json`), which FR-EXP-4 does not ask for but a stale overlay would otherwise survive.
- Status: open

## Q-15 [non-blocking] E1/E2 design choices not fixed by the spec
- Context: `configs/e1.yaml`, `configs/e2.yaml`, FSD §6 (E1, E2), TSD §12.
- Observation: the spec gives the sweeps but not: E1's band for eta* in the eta_s_bar sweep (5 GHz used), the compound
  bands (30 and 100 GHz; 5 GHz is in outage beyond 12.5 m), the pdf point (eta_s_bar = 0.999); E2's operating point
  (30 GHz, 50 m, K = 10, N_s = 1e4, 2 m^2 apertures) and sigma_Psi/Psi_bar grid (0.05-3; the sourced Seoul value is 2.06).
  E2's RAR is the log-log interpolated literature point at 30 GHz, as in E8.
- Options: A) keep; B) other operating points.
- Recommendation: A until Abdurrahman picks the paper's points; all are config values, no code change needed.
- Status: open

## Q-16 [non-blocking] `scripts/derive_inputs.py` (TSD §1, §16.2) is still missing
- Context: TSD §16.2 ("must reproduce both tables to 3 significant figures"), optional dependency `itur` 0.4.0.
- Observation: `itur` is not installed and is not a pinned dependency; the quantile integration (0.1-99 % vapour,
  0.001 %-P0 rain) leaves the quadrature rule and the P.836 quantile set open, so a script written now could not be
  checked against the tables. The §16.1-16.2 values in the configs are the ones compiled by Abdurrahman.
- Options: A) Abdurrahman supplies the script used for §16; B) I write it once `itur` may be installed and the
  integration rule is stated.
- Recommendation: A. Not needed for `make paper-figs` (the configs carry the numbers with `source:`).
- Status: open
