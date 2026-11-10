# RESULTS — reference values and benchmarks

Every reference value asserted in `tests/`, with where it comes from (FSD §8 DoD 4). Source tags:
**TSD §x** = pre-verified value in the spec; **lit.** = literature value as published; **derived** =
closed form or independent method evaluated in the test itself. Tolerances are in `tests/tolerances.py`.
All values at $T=300$ K unless stated. The code reproduces every entry; none was tuned to pass.

## 1. Physics basics (M0)

| Quantity | Value | Source | Test |
|---|---|---|---|
| $\bar n$(5 GHz) | 1249.70 ± 0.01 | FSD FR-PHY-1, TSD §11 | `unit/test_thermal.py` |
| $\bar n$ at 5/30/100/300 GHz | 1249.697 / 207.867 / 62.011 / 20.341 | TSD §11 | `regression/test_reference_values.py` |
| $\eta^*$ at 5/30/100/300 GHz | 0.99920 / 0.99521 / 0.98413 / 0.95314 | TSD §11 | same |
| $1-\eta^*$ | 7.996e-4 / 4.788e-3 / 1.587e-2 / 4.686e-2 | TSD §11 | same |
| RJ relative error, 300 GHz | 2.438e-2 (= expm1(x)/x − 1) | TSD §3.1; derived identity | `unit/test_thermal.py` |
| $S_{E,\rm BBR}$ (nV/cm/√Hz) | 0.603 / 3.614 / 12.012 / 35.748 | TSD §7.4, §11 | `unit/test_bbr.py` |
| Planck vs RJ field amplitude, 300 GHz | 1.21 % (< 1.5 %) | FSD FR-PHY-3 | same |

## 2. Gaussian core (M1)

| Quantity | Value | Source | Test |
|---|---|---|---|
| TMSV $r=3$: $\tilde\nu_-$ | $e^{-6}$ | derived (exact) | `validation/test_entanglement.py` |
| $F_{\rm opt}$, $F_{\rm BK}$ (TMSV $r=1$, 5 GHz, $\eta=0.9995$) | 0.5858, 0.5681 | TSD §5.3 | same |
| $\eta^*_{\rm TMST}$ at 5 GHz, $(\bar n_{\rm JPA}, r)$ = (0, any) / (0.002, 0.5) / (0.05, 0.5) | 0.99920045 / 0.99920628 / 0.99932075 | TSD §5.4 | same |
| Combining-network threshold | V0 eq. (div_netthreshold) | TSD §4.3, derived by direct root | `validation/test_combining.py` |

## 3. Stochastic channel (M2)

| Quantity | Value | Source | Test |
|---|---|---|---|
| $\mathcal Z_K$ at $\bar\eta=0.999$, $K=0/10/100$ | 0.63249 / 0.54404 / 0.51689 | TSD §6.1 | `unit/test_small_scale.py` |
| Central $\chi^2_{2N}$ tails at ∓5.5 sd, $N=10^5..10^8$ | cdf 1.590617e-8 / 1.796389e-8 / 1.865997e-8 / 1.888480e-8; sf 2.256714e-8 / 2.006462e-8 / 1.932410e-8 / 1.909481e-8 | TSD §11 (series / continued fraction with `mpmath` prefactor) | `regression/test_ncx2_floor_tail.py` |
| Large-$N$ cdf tails ($K=0$: $N=10^5,10^7,10^8$; $K=10$: $N=10^5,10^7$) | 1 % relative down to 1e-8 | derived (`mpmath`, `tests/mp_ref.py`) | `validation/test_small_scale_mpmath.py` |
| Truncated-exponential moments, $\bar\eta=0.01$ | closed form | derived | `unit/test_small_scale.py` |
| Log-gamma mean ($k=4$, $\theta=0.01$) | 0.96098 | TSD §6.2 | `unit/test_large_scale.py` |
| V0 log-normal mass above 1 ($\sigma_\Psi=\bar\Psi$) | 0.158655 = Φ(−1) | TSD §6.2 | `unit/test_large_scale.py`, `unit/test_experiments_m6.py` |
| Fresnel product, Goubau coupling (5 GHz, 1 m², 10 m) | 2.7816, 0.93806 | TSD §6.5 | `unit/test_link_budget.py` |
| Absorption-only range at $\eta^*$ (5 GHz) | 555.47 m with $\gamma_{\rm abs}=1.44\times10^{-6}$ m⁻¹ | TSD §6.5; lit. Gonzalez-Raya et al., PRApplied 18, 044002 (2022) | same |
| Aperture for $\bar\eta=0.9999$ at 10 m | 1.82 m² | TSD §6.5 | same |

## 4. Receivers and entanglement service (M3)

| Quantity | Value | Source | Test |
|---|---|---|---|
| RAR $N_{\rm amp}$, 10 nV/cm/√Hz at 5 GHz | 3.438e5 (rms), 1.719e5 (peak); $T_n$ = 8.25e4 K | TSD §7.4 | `unit/test_receivers.py` |
| $T_n$ conventions at 5 GHz, $T_n = hf/(k_B\ln3)$ = 218.42 mK | 0.9102 (rj) / 0.5000 (planck) / 1.0000 (callen_welton) | TSD §7.3 | same |
| $\eta^*_{\rm rx}(N_{\rm add}=\tfrac12)$, 5 GHz | 0.99960 | TSD §8.1 | `unit/test_entanglement_service.py` |
| Transducer threshold ($\eta_t=0.6$, $\bar n_t=N_t=0.01$) | 0.999219 | TSD §8.1 | same |
| Log-gamma Case 1 ($K\to\infty$) | $Q(k,[\ln(\bar\eta_s/\eta^*)-\Psi_0]/\theta)$ | TSD §8.1, derived | `validation/test_entanglement_outage.py` |
| RQ1 identity | $P_{\rm out}/P_{\rm out}^{\rm surr}=1/\mathcal Z$ to 1e-12 | derived | same, `unit/test_experiments_m6.py` |
| E4 front-end thresholds at 5 GHz | TSD §16.5 table | lit. sources in `configs/e4.yaml` | `unit/test_experiments.py` |

**Aperture/field convention cross-check (TSD §7.4).** Santamaría Botello et al. (arXiv:2209.00908) use
$P=G\frac{\lambda^2}{4\pi}\frac{|E|^2}{2\eta_0}$ with $G=3/2$, i.e. a peak field and $A_e=3\lambda^2/8\pi$.
Their 1.25 µV/m/√Hz at 10 GHz read as peak maps to 16 114 K with the §7.4 formula (16 100 K published); an rms
reading would give 32 228 K. Literature sensitivities with no stated convention are therefore entered as `peak`.

## 5. Classical service and crossover (M4)

| Quantity | Value | Source | Test |
|---|---|---|---|
| SSB, Δ = 3 dB, 5 GHz, $\eta\to0$ | $N_{\rm amp}^{\rm req}$ = 1245.27, $S_E^{\rm req}/S_{E,\rm BBR}$ = 0.99823 | TSD §8.3 | `unit/test_classical.py` |
| DSB, Δ = 4 / 6 dB | 641.71 → 0.71659; 2479.22 → 1.40849 | TSD §8.3 | same |
| DSB image floor (dB) | 3.009 / 3.000 / 2.976 / 2.907 | TSD §8.3, §11 | same |
| Published RAR points → $N_{\rm amp}$, $T_{\rm equiv}$ | TSD §16.3 table | lit., sources in `configs/literature_rar.yaml` | `unit/test_experiments_m4.py` |
| E6 RJ slope of $S_{E,\rm BBR}$ | 0.9997 (expected 1) | derived | same |

## 6. Diversity (M5)

| $f$ (GHz) | $N_{\rm req}$, $K=0$ | $K=10$ | $K=100$ |
|---|---|---|---|
| 5 | 20 017 832 | 3 474 510 | 394 473 |
| 30 | 556 629 | 96 662 | 10 975 |
| 100 | 50 243 | 8 737 | 993 |
| 300 | 5 630 | 983 | 112 |

Source: FSD FR-DIV-3 / TSD §11 (consistent with the V0 table at its displayed precision). All 12 integers are
reproduced exactly (`unit/test_diversity.py`); no discrepancy with V0 to log. The Cramér rate is checked against a
numerical Legendre transform (derived).

## 7. Runner and experiments E1, E2 (M6)

| Quantity | Value | Source | Test |
|---|---|---|---|
| LaTeX thermal table entries | the §1 values, recomputed from code | derived | `unit/test_runner.py` |
| $N_{\rm req}$ LaTeX cell (5 GHz, $K=0$) | 20 017 832 | §6 above | same |
| E2 sourced $\sigma_\Psi/\bar\Psi$ (30 GHz) | 6.760e-5 / (4.394e-6 + 2.839e-5) = 2.062 | TSD §16.1–16.2 | `unit/test_experiments_m6.py` |
| E2 limit $\sigma_\Psi\to0$ | three laws agree to 1e-6 | FSD FR-CH-3 (b), derived | same |
| Obstacle generator moments | $\mu_G=\lambda_o d k_o\theta_o$, $\sigma_G^2=\lambda_o d k_o(k_o+1)\theta_o^2$ | TSD §6.3 | same |

Numbers produced by the default configs (not gated; inputs in TSD §16, design choices tagged [A]):

- **E1** (5 GHz sweep, $\eta_\ell=1$): removed mass $1-\mathcal Z_K$ between 0.21 ($K=100$, $\bar\eta_s=0.9$) and 0.49
  ($K=100$, $\bar\eta_s\to1$) for $\bar\eta_s\in[0.9,1]$; the untruncated surrogate understates the outage by
  $\Delta P_{\rm out}$ = 0.21–0.48 there. Over the E3 link (30 and 100 GHz) the ratio $P_{\rm out}/P_{\rm out}^{\rm surr}$ is
  1.2–1.9 while $\eta_{\rm geo}>0.99$ and tends to 1 well beyond the geometric reach.
- **E2** (30 GHz, 50 m, $K=10$): at the sourced ratio $\sigma_\Psi/\bar\Psi\approx2$, $\Delta P_{\rm out}$ = +3.1e-4 (V0
  log-normal, 31 % of its mass at $\eta_\ell>1$, 0.11 % clipped) and +3.0e-3 (conditioned log-normal);
  $|\Delta\bar C/\bar C|\le0.3$ % for $C_\chi$, $C_{\rm het}$ and the RAR $C_{\rm rx}$.

## 8. Rydberg receiver physics model (M7)

| Quantity | Value | Source | Test |
|---|---|---|---|
| $\Gamma_2$ (⁸⁷Rb D2) | $2\pi\times6.0666$ MHz | lit. Steck, *Rubidium 87 D Line Data*, rev. 2.3 (TSD §14.3) | `unit/test_rydberg_physics.py` |
| ⁸⁷Rb mass | 86.909180520 u | lit. Steck, rev. 2.3 | same (via $\sigma_v$) |
| $\sigma_v$ (⁸⁷Rb, 300 K) | 169.4 m/s (code: 169.41) | TSD §14.6 | same |
| Cycling-transition dipole $d_{12}$ | $\sqrt{3\pi\epsilon_0\hbar\Gamma_2/k_p^3}$ = 2.989 $ea_0$ | TSD §14.7, derived | same |
| Two-level resonant cross-section | $3\lambda_p^2/2\pi$ (weak probe: 1e-14; full: 1e-4) | TSD §14.7, T-LIN-10 | `validation/test_lindblad.py` |
| Cold 4-level AT splitting, $\Omega_m/2\pi=4$ MHz | 3.93 MHz (within 3 % of 4) | TSD §14.11, T-LIN-4 | same |
| Doppler-averaged AT splitting, $\Omega_m/2\pi=20$ MHz | 12.45 MHz vs $(\lambda_c/\lambda_p)\Omega_m$ = 12.30 (1.2 %, tol 2 %; TSD measured 12.4) | TSD §14.11, T-LIN-5 | same |
| Weak-probe vs full, cold 3-level, $\Omega_p/\Gamma_2=2\times10^{-3}$ | ≤ 1.6e-5 (tol 1e-4) | T-LIN-1, derived | same |
| Weak-probe vs full $H(i\omega)$, $\Omega_p/2\pi=0.01$ MHz | ≤ 1.5e-4 at $v=0$ (tol 2e-4; TSD states ≤ 8e-5, see QUESTIONS Q-18), ≤ 1e-8 at 100 m/s | TSD §14.4, T-LIN-7 | same |
| $H(0)$ vs central finite difference | < 1e-6, both engines | T-LIN-6, derived | same |
| Full engine vs QuTiP 5.3.1 `steadystate` (20 random sets) | rtol 1e-8 | T-LIN-9; run once in a separate environment, skipped when QuTiP is absent | same |
| Synthetic linear range, $P=kx\mp bx^3$ | $c=1\mp(3b/4k)s^2$, 1 dB point exact to 1e-8, cases `compression` / `expansion` | derived | `unit/test_rydberg_physics.py` |
| $B_{3\rm dB}$ refinement vs dense scan (60 001 points) | 1e-3 relative | FSD FR-LIN-5 | same |
| $\kappa$ vs finite difference of $P_{\rm out}(E_{\rm LO}\pm\delta E)$ | 1e-6 relative | derived | same |

**E9 with the placeholder inputs of `configs/e9.yaml`** (all [A], QUESTIONS Q-17; not results): method `auto` stays on
the weak-probe engine (max weak-probe/full discrepancy 0.53 % at 5 checkpoints, tolerance 1 %); baseline
($\Omega_p,\Omega_c,\Omega_{\rm LO}=2\pi\times(0.084, 5, 20)$ MHz) gives $S_E=1.2\times10^{-4}$ V m⁻¹ Hz⁻¹ᐟ² rms,
$B_{3\rm dB}=19.2$ MHz with peaking 3.9, $E_{1\rm dB,pk}=0.85$ V/m (compression) against $E_{\rm LO,pk}=1.33$ V/m; the best
sweep point is $3.2\times10^{-6}$ V m⁻¹ Hz⁻¹ᐟ² at $\Omega_c/2\pi=10$ MHz, $\Omega_{\rm LO}/2\pi=5$ MHz. The §14.8
cross-check against a published photon-shot-noise-limited sensitivity is not done (needs sourced experiment parameters).

### Receiver engine benchmarks (FR-LIN-2, FR-LIN-3; recorded, not CI-gated)

<!-- BENCH:START -->
| Engine | $N_v$ | Cost per velocity class | Target (not gated) | Met |
|---|---|---|---|---|
| Weak-probe closed form (`weak_probe.rho21`) | $10^4$ | 12.7 ns | ≤ 50 ns | yes |
| Full Lindblad, batched `solve` (`lindblad.steady_state`) | $10^3$ | 4.75 µs | ≤ 20 µs | yes |

Measured with `make bench` on x86_64 / Linux, Python 3.12.13, NumPy 2.5.4; best of 5 repetitions.
<!-- BENCH:END -->

### Experiment wall times (`make paper-figs`, one process, idle laptop-class machine)

| E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | Total |
|---|---|---|---|---|---|---|---|---|---|
| 93 s | 50 s | **325 s** | 0.7 s | 207 s | 0.8 s | 1.7 s | 53 s | 12.5 s | 12 min 25 s |

E3 exceeds the 5-minute budget of NFR-3 (profile and options in QUESTIONS Q-22). E9 uses the weak-probe engine; a
strong-probe configuration that `auto` escalates to the full engine is much slower (QUESTIONS Q-19).
