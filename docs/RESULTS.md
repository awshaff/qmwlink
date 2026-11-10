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

- **E1** (5 GHz sweep, $\eta_\ell=1$): removed mass $1-\mathcal Z_K$ between 0.32 and 0.49 for $\bar\eta_s\in[0.9,1]$;
  the untruncated surrogate understates the outage by $\Delta P_{\rm out}$ = 0.32–0.46 there. Over the E3 link the
  ratio $P_{\rm out}/P_{\rm out}^{\rm surr}$ is 1.5–1.9 inside the geometric reach and tends to 1 beyond it.
- **E2** (30 GHz, 50 m, $K=10$): at the sourced ratio $\sigma_\Psi/\bar\Psi\approx2$, $\Delta P_{\rm out}$ = +3.1e-4 (V0
  log-normal, 31 % of its mass at $\eta_\ell>1$, 0.11 % clipped) and +3.0e-3 (conditioned log-normal);
  $|\Delta\bar C/\bar C|\le0.3$ % for $C_\chi$, $C_{\rm het}$ and the RAR $C_{\rm rx}$.
