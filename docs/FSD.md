# FSD — Functional Specification

**Project:** `qmwlink` — simulation and analysis toolkit for the paper described in `docs/IDEA.md`
**Version:** 0.3 (2026-11-03) · **Owner:** Abdurrahman · **Implementer:** coding agent (see `agent.md`); pure Python, no compiled code
**Companion:** `docs/TSD.md` holds every formula, algorithm and numerical rule. This FSD says *what* the software must do; the TSD says *how*.

---

## 1. Purpose

Produce every number, table and figure of the paper from version-controlled code and configs, with each analytic result cross-checked by an independent method (closed form vs. numerical integration vs. Monte Carlo).

## 2. Users and use cases

| User | Use case |
|---|---|
| Abdurrahman (researcher) | Run an experiment config, get figures + CSV data; change a parameter and rerun; inspect intermediate quantities. |
| Coding agent | Implement modules against this spec and the TSD; prove correctness with tests. |
| Advisor / reviewers | Regenerate all paper figures with one command; read the parameter table. |

## 3. Scope

**In scope:** physical constants and thermal occupation; Gaussian state/channel core in $(\tau,y)$ form; entanglement measures; small-scale, large-scale and compound PDTs (pdf, cdf, sampler, moments); receiver models (ideal, quantum-limited amplifier, electronic LNA, Rydberg atomic receiver, coherent transducer); entanglement-service and classical-service metrics; N-element passive MRC and selection combining; experiment runner, figure and LaTeX-table export; validation suite; **Rydberg receiver physics model** (weak-probe closed form as the primary engine, full Lindblad model as the strong-probe check, both Doppler averaged) that derives $S_E$, $N_{\rm amp}$, bandwidth and linear range from atomic parameters (TSD §14).

**Out of scope (v1):** frequency-selective channels, inhomogeneous temperature, imperfect CSI, multi-hop/relays, any GUI, compiled extensions; inside the receiver model: hyperfine/Zeeman structure, optical propagation effects, Rydberg–Rydberg interactions, atom projection noise, time-domain waveform simulation (TSD §14.1).

## 4. Conventions (binding)

- Quadratures $\hat x=\hat a+\hat a^\dagger$, $\hat p=-i(\hat a-\hat a^\dagger)$; vacuum variance **1** per quadrature; covariance ordering $(x_1,p_1,x_2,p_2)$.
- SI units at all public interfaces. Frequencies in Hz, temperatures in K, distances in m, field sensitivity $S_E$ in V m⁻¹ Hz⁻¹ᐟ². Display units (GHz, nV/cm/√Hz, dB) only in plotting/IO layers.
- Capacities in **bits per mode**; throughput in bit/s = bandwidth × bits/mode.
- Random-number generators passed explicitly (`numpy.random.Generator`); no global seeding.

## 5. Functional requirements

Each requirement has an ID, a statement, and acceptance criteria (AC). Formulas referenced as `TSD §x`.

### 5.1 Physics basics — `FR-PHY`

| ID | Requirement | AC |
|---|---|---|
| FR-PHY-1 | Bose–Einstein occupation $\bar n(f,T)$, exact and Rayleigh–Jeans approximation (TSD §3.1). | $\bar n(5\text{ GHz},300\text{ K}) = 1249.70\pm0.01$; RJ relative error < $hf/2k_BT$. |
| FR-PHY-2 | Critical transmissivity $\eta^\*=\bar n/(1+\bar n)$. | Matches reference table TSD §11 to 5 decimals. |
| FR-PHY-3 | Short-dipole effective aperture $A_e=3\lambda^2/(8\pi)$ and BBR-equivalent field $S_{E,\rm BBR}(f,T)$ (TSD §7.4). | Two independent computations (mode-occupation and RJ energy density) agree within 1.5% for $f\le 300$ GHz at 300 K. |

### 5.2 Gaussian core — `FR-GAU`

| ID | Requirement | AC |
|---|---|---|
| FR-GAU-1 | Two-mode covariance matrices for TMSV and TMST; physicality check $V+i\Omega\ge0$. | TMSV passes for $r\in[0,3]$; a deliberately unphysical CM fails. |
| FR-GAU-2 | Phase-insensitive channel object `PIChannel(tau, y)` with constructors: thermal attenuator, additive noise, amplifier, identity; composition operator; physicality check $y\ge|1-\tau|$ (TSD §4). | Composition is associative; attenuator∘attenuator matches the closed form TSD §4.3. |
| FR-GAU-3 | Apply a channel to mode B of a two-mode CM. | Matches V0's $V_{\rm out}(\eta)$ block form for attenuator inputs. |
| FR-GAU-4 | Minimum PT symplectic eigenvalue $\tilde\nu_-$ (general 2-mode formula and standard-form closed form), negativity, log-negativity, teleportation fidelity $1/(1+\tilde\nu_-)$ (TSD §5). | General and closed forms agree to 1e-12 on random valid CMs. |
| FR-GAU-5 | Entanglement-breaking test $y\ge 1+\tau$ and TMSV separability equivalence. | For TMSV with $r\in\{0.1,0.8,2\}$: $\tilde\nu_-<1 \iff y<1+\tau$ on a 10⁴-point grid. |
| FR-GAU-6 | TMST entanglement condition $(a-1)(\tau a+y-1)<\tau c^2$. | Agrees with direct $\tilde\nu_-$ evaluation on a grid; reduces to FR-GAU-5 at $\bar n_{\rm JPA}=0$. |

### 5.3 Stochastic channel — `FR-CH`

| ID | Requirement | AC |
|---|---|---|
| FR-CH-1 | Small-scale PDT: truncated noncentral $\chi^2$ (one element and $N$ elements): pdf, cdf, survival, inverse cdf, sampler, moments, $\mathcal Z_{K,N}$ (TSD §6.1). | pdf integrates to 1 ± 1e-8; MC (10⁶) vs cdf KS distance < 2e-3; limits $K=0$ (truncated exponential/Erlang) and large $K$ (concentration) pass. |
| FR-CH-2 | Large-scale PDT, **log-gamma** (primary): pdf, cdf, survival, sampler, moments; parameterization from $(\bar\Psi,\sigma_\Psi^2)$ or $(k,\theta)$ (TSD §6.2). | $\mathbb E[\eta_\ell^m]=(1+m\theta)^{-k}$ vs MC within 3σ; support strictly $(0,1]$. |
| FR-CH-3 | Large-scale baselines: V0 log-normal (untruncated, flagged as non-physical) and log-normal conditioned on $\Psi\ge0$. | Reports mass above 1 for the untruncated version (≈0.1587 for $\sigma_\Psi=\bar\Psi$). |
| FR-CH-4 | Optional obstacle generator: compound-Poisson log-loss moment-matched to Gamma (TSD §6.3). | Moment match exact; MC histogram vs fitted Gamma reported (no pass/fail threshold, diagnostic only). |
| FR-CH-5 | Compound PDT $\eta=\eta_\ell\eta_s$: pdf, cdf, survival, quantile, sampler, first two moments (TSD §6.4). | Numerical cdf vs MC (10⁶) within 3σ binomial band at every evaluated point; moments match product formula. |
| FR-CH-6 | Link-budget helper giving mean collection efficiency $\bar\eta_\Sigma$ from distance, frequency, apertures, absorption (TSD §6.5). | $\bar\eta_\Sigma\le1$ always; absorption-only range at 5 GHz for $\eta^\*$ = 555.5 m ± 0.5 m. |

### 5.4 Receivers — `FR-RX`

| ID | Requirement | AC |
|---|---|---|
| FR-RX-1 | Receiver base type with two classes: **measuring** (parameter $N_{\rm amp}\ge\tfrac12$) and **quantum front end** (a `PIChannel`) (TSD §7). | Constructing a measuring receiver with $N_{\rm amp}<\tfrac12$ raises an error. |
| FR-RX-2 | Ideal heterodyne ($N_{\rm amp}=\tfrac12$), ideal homodyne (special case flag), quantum-limited amplifier. | Ideal heterodyne capacity equals $C_{\rm het}$ of TSD §8.2 exactly. |
| FR-RX-3 | Electronic receiver from noise temperature $T_n$, with selectable convention (`RJ` or `planck`) (TSD §7.3). | $T_n = hf/k_B$ gives $N_{\rm amp}=1$ under RJ. |
| FR-RX-4 | Rydberg atomic receiver from field sensitivity $S_E$ (and optional bandwidth $B$, saturation field $E_{\rm sat}$ for validity flag) via the short-dipole mapping; also direct $T_n$ or $N_{\rm amp}$ input (TSD §7.4). | 10 nV/cm/√Hz at 5 GHz → $N_{\rm amp}=3.438\times10^5$ (±0.1%), $T_n=8.25\times10^4$ K. |
| FR-RX-5 | Coherent transducer $(\eta_t,\bar n_t)$ as quantum front end for entanglement service. | Composition with channel reproduces TSD §7.5 threshold. |
| FR-RX-6 | `LindbladRARModel`: SI atomic/optical inputs → $\kappa$, $S_E$, $N_{\rm amp}$, $B_{3\rm dB}$, $E_{1\rm dB}$ (TSD §14.10), with `method` = `auto` \| `weak_probe` \| `full` (TSD §14.6). `RydbergReceiver.from_physics(...)` builds an FR-RX-4 receiver from it. | Method used and max weak-probe/full discrepancy recorded in run metadata; T-LIN-11 passes. |

### 5.5 Services — `FR-SVC`

| ID | Requirement | AC |
|---|---|---|
| FR-SVC-E1 | Receiver-referred threshold $\eta^\*_{\rm rx}$ for any quantum front end (closed form for additive noise, root-find otherwise) (TSD §8.1). | $N=0\Rightarrow\eta^\*$; $N=\tfrac12\Rightarrow0.99960$ at 5 GHz; $N\ge1\Rightarrow$ "no window". |
| FR-SVC-E2 | Entanglement outage $P_{\rm out}$, average log-negativity, average $\tilde\nu_-$, average fidelity over the compound PDT. | Analytic (quadrature) vs MC within 3σ; V0 limiting cases 1–4 reproduced. |
| FR-SVC-C1 | Per-realization capacities: Holevo, heterodyne, homodyne, measuring receiver (TSD §8.2). | Ordering $C_\chi\ge C_{\rm het}$, $C_\chi\ge C_{\rm hom}$ holds on a grid; all → 0 as $\eta\to0$. |
| FR-SVC-C2 | Ergodic capacity, rate-outage probability $P(C<R)$, and $\epsilon$-outage capacity $C_\epsilon$ via the compound quantile. | $C_\epsilon$ equals $C(F^{-1}(\epsilon))$ to 1e-6 relative; MC cross-check. |
| FR-SVC-C3 | Throughput in bit/s using receiver bandwidth $B$. | Unit test on a hand case. |
| FR-SVC-C4 | Crossover solver: required $S_E$ for a receiver noise penalty of $\Delta$ dB relative to ideal heterodyne (closed form, TSD §8.3), plus a capacity-based variant (numeric). | Closed form vs numeric root agree to 1e-6; at 5 GHz, 300 K, $\Delta=3$ dB and $\bar\eta\to0$ the required $S_E$ equals $S_{E,\rm BBR}$ within 0.5%; unattainable targets return `NaN` with a reason string. |

### 5.6 Diversity — `FR-DIV` (keep in code even if moved out of the paper)

| ID | Requirement | AC |
|---|---|---|
| FR-DIV-1 | $N$-element passive-MRC PDT and array outage (TSD §6.1, §9). | $N=1$ reduces to single-element results exactly. |
| FR-DIV-2 | Selection-combining tail and Chernoff bound. | Bound ≥ exact tail on a grid. |
| FR-DIV-3 | Cramér rate $\mathcal R_K(a)$, design law, and $N_{\rm req}$ solver (integer bisection). | $\mathcal R_K(1)=0$; reproduces V0 table (TSD §11) **or** reports discrepancies to `QUESTIONS.md` (V0 table is not ground truth). |
| FR-DIV-4 | Monte Carlo with **joint** truncation $\sum_i\eta_{s,i}\le1$ by rejection. | Agreement with FR-DIV-1 within 3σ. |

### 5.7 Rydberg receiver physics model — `FR-LIN`

| ID | Requirement | AC |
|---|---|---|
| FR-LIN-1 | Level scheme, decay channels, transit reset, dephasing and derived coherence rates for $N\in\{3,4\}$ ladder schemes (TSD §14.3). | $N\notin\{3,4\}$ rejected; coherence rates match TSD §14.3 on a hand case. |
| FR-LIN-2 | Weak-probe engine: tridiagonal steady state, dynamics, transfer function $H(i\omega)$, vectorized over velocity classes (TSD §14.4). | T-LIN-1, T-LIN-3, T-LIN-7 pass; ≤ 50 ns per velocity class on a laptop. |
| FR-LIN-3 | Full Lindblad engine: Liouvillian as $\mathcal A+v\mathcal B$, batched bordered steady state and transfer function (TSD §14.5). | T-LIN-2, T-LIN-4, T-LIN-6 pass; ≤ 20 µs per velocity class on a laptop. |
| FR-LIN-4 | Doppler averaging with automatic grid refinement and recorded final $N_v$; method selection `auto` with checkpoint discrepancy (TSD §14.6). | T-LIN-5, T-LIN-8, T-LIN-11 pass. |
| FR-LIN-5 | Doppler-averaged response: $H(i\omega)$, $B_{3\rm dB}$ and peaking value (TSD §14.7). | Bandwidth refinement agrees with a dense-grid scan to 1e-3 relative. |
| FR-LIN-6 | Readout, sensitivity and linear range in SI: $\alpha$, $P_{\rm out}$, $\kappa$, $S_E$, $E_{1\rm dB}$ (TSD §14.7–14.9). | T-LIN-10 passes; unit tests on the SI conversions. |
| FR-LIN-7 | Optional QuTiP cross-check of the full engine (tests only). | T-LIN-9 passes or skips cleanly when QuTiP is absent. |

### 5.8 Experiments, outputs, reproducibility — `FR-EXP`

| ID | Requirement | AC |
|---|---|---|
| FR-EXP-1 | Experiments defined by YAML configs validated by a schema; unknown keys rejected. | Invalid config fails with a clear message. |
| FR-EXP-2 | One command per experiment and one command for all paper figures. | Fresh clone → `make paper-figs` succeeds. |
| FR-EXP-3 | Each run writes: figure (PDF + PNG), raw data (CSV), metadata (JSON: git hash, config hash, seed, package versions, wall time). | Files present and metadata complete. |
| FR-EXP-4 | Results cache keyed by config hash; `--force` to recompute. | Second run is a cache hit. |
| FR-EXP-5 | LaTeX export of the parameter table and reference tables (booktabs). | Compiles in IEEEtran. |
| FR-EXP-6 | Figure style: single-column IEEE width (3.5 in), vector PDF, serif font, colorblind-safe palette, no titles inside figures (captions live in LaTeX). | Visual check by Abdurrahman. |

## 6. Experiments (map to paper figures)

| ID | Question | Sweep | Output |
|---|---|---|---|
| E1 | Small-scale PDT shape and truncation mass | $K\in\{0,1,10,100\}$, $\bar\eta_s\in[0.9,1]$ | pdf curves; $\mathcal Z_K$ vs $\bar\eta_s$ |
| E2 | Large-scale law comparison (RQ2) | $\bar\Psi,\sigma_\Psi$ grid | pdfs; mass above 1 for V0 log-normal; outage difference |
| E3 | Entanglement outage vs distance | $f\in\{5,30,100,300\}$ GHz, $K\in\{0,10,100\}$, $d\in[10,10^3]$ m | $P_{\rm out}$, $\mathbb E[E_N]$ |
| E4 | Receiver-referred window (RQ3) | $N\in[0,1.2]$; receiver classes overlaid | $\eta^\*_{\rm rx}$ vs $N$; window width $1-\eta^\*_{\rm rx}$ |
| E5 | Classical ergodic capacity (RQ1, R1) | $N_s$, $d$, bands; receivers: Holevo, het, hom, LNA, RAR | bits/mode and bit/s |
| E6 | **BBR crossover map** (RQ4) | $f\in[1,300]$ GHz, $T=300$ K | $S_{E,\rm BBR}(f)$ and required $S_E$ for 1/3 dB gap; literature RAR points overlaid (values supplied by Abdurrahman) |
| E7 | Diversity hardening and $N_{\rm req}$ | $N$, $K$, bands | $N_{\rm req}$ table, outage vs $N$ |
| E8 | Outage capacity | $\epsilon\in[10^{-4},10^{-1}]$ | $C_\epsilon$ curves per receiver |
| E9 | **Validity of the Gaussian RAR model** (IDEA R4) and model-derived RAR points | $\Omega_c$, $E_{\rm LO}$ grid (TSD §12); link distance for received field | (a) $E_{1\rm dB}$ vs received signal field over distance → region where the linear model holds; (b) $S_E$, $N_{\rm amp}$, $B_{3\rm dB}$ vs $\Omega_c$, $E_{\rm LO}$; (c) model-derived points overlaid on the E6 crossover map |

Default parameter sets live in `configs/defaults.yaml`; any number taken from literature carries a `source:` field.

## 7. Non-functional requirements

- **NFR-1 Correctness first.** Every formula has ≥1 limiting-case test and ≥1 independent cross-check (TSD §10).
- **NFR-2 Numerical robustness.** No catastrophic cancellation near $\eta\to1$ (use survival functions, log-domain, scaled Bessel). Tail probabilities down to 1e-8 must be accurate to 1% relative (validated with `mpmath`).
- **NFR-3 Performance.** Any single figure regenerates in < 5 min on a laptop CPU; MC defaults to 10⁶ samples, vectorized.
- **NFR-4 Code quality.** Python ≥ 3.11, type hints, `ruff` + `mypy --strict` on `src/`, docstrings with the TSD section reference, test coverage ≥ 90% on `src/`.
- **NFR-5 Reproducibility.** Deterministic given seed and config; pinned dependencies.
- **NFR-6 Traceability.** Each public function cites its FR ID and TSD section in its docstring.
- **NFR-7 No compiled code.** NumPy/SciPy vectorization only. Adding Numba, Cython or C/C++ requires a `QUESTIONS.md` entry with a profile showing a figure that breaks NFR-3.

## 8. Definition of done (v1)

1. All FR acceptance criteria pass in CI.
2. E1–E9 regenerate from a clean clone with one command.
3. `QUESTIONS.md` has no unresolved item marked `blocking`.
4. `docs/RESULTS.md` lists every reference value used in tests with its source (TSD §11, literature, or derived).