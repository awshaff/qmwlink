# FSD — Functional Specification

**Project:** `qmwlink` — simulation and analysis toolkit for the paper described in `docs/IDEA.md`
**Version:** 0.6.1 (2026-11-10) · **Owner:** Abdurrahman · **Implementer:** coding agent (see `agent.md`); pure Python, no compiled code
**Companion:** `docs/TSD.md` **v0.4.1** holds every formula, algorithm, numerical rule and code identifier. This FSD says *what* the software must do; the TSD says *how*. FSD and TSD versions are bumped together.

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

**Out of scope (v1):** frequency-selective channels, inhomogeneous temperature along the propagation path, imperfect CSI, multi-hop/relays, any GUI, compiled extensions; inside the receiver model: hyperfine/Zeeman structure, optical propagation effects, Rydberg–Rydberg interactions, atom projection noise, time-domain waveform simulation (TSD §14.1). *(A cold/warm combining network at the receiver is in scope via channel composition, FR-GAU-2.)*

## 4. Conventions (binding)

- Quadratures $\hat x=\hat a+\hat a^\dagger$, $\hat p=-i(\hat a-\hat a^\dagger)$; vacuum variance **1** per quadrature; covariance ordering $(x_1,p_1,x_2,p_2)$.
- SI units at all public interfaces. Frequencies in Hz, temperatures in K, distances in m, field sensitivity $S_E$ in V m⁻¹ Hz⁻¹ᐟ². Display units (GHz, nV/cm/√Hz, dB) only in plotting/IO layers.
- **Field convention:** $S_E$ is always the **rms, single-sided** field spectral density, so the noise-equivalent power per Hz is $S_E^2A_e/Z_0$ with $Z_0=1/(\varepsilon_0c)$. A value quoted as a peak amplitude is converted at the IO boundary by $S_E^{\rm rms}=S_E^{\rm peak}/\sqrt2$ (a factor 2, i.e. 3 dB, in $N_{\rm amp}$). Rabi frequencies and linear-range limits use **peak** fields; every field variable carries a `_rms` or `_pk` suffix (TSD §2).
- **Sideband convention:** every measuring receiver declares `sideband: ssb | dsb`. A DSB receiver also collects the image band's thermal photons $N_{\rm img}$ (TSD §7.1). The reference "ideal heterodyne" is SSB.
- **Antenna-equivalence convention:** a Rydberg vapor cell is mapped to an ideal short dipole, $A_e=3\lambda^2/(8\pi)$. This is a modeling convention for comparing receivers, not derived physics; it is stated as such wherever it is used.
- Capacities in **bits per mode**; throughput in bit/s = bandwidth × bits/mode.
- Random-number generators passed explicitly (`numpy.random.Generator`); no global seeding.

### 4.1 Reserved symbols

Each symbol below has one meaning across code, configs and this document. **TSD §2 is the binding table of symbols and code identifiers**; this excerpt mirrors it, and on any conflict TSD §2 wins.

| Symbol | Meaning | Identifier |
|---|---|---|
| $N$ | number of receive elements (diversity) | `n_elements` |
| $N_{\rm add}$ | receiver-referred added noise of a quantum front end, in photons; added quadrature variance $y_{\rm add}=2N_{\rm add}$ | `n_add` |
| $N_{\rm amp}$ | added noise of a measuring receiver, in photons referred to its input (same units as $N_{\rm add}$; ideal heterodyne $=\tfrac12$) | `n_amp` |
| $N_{\rm img}$ | thermal occupation of the image-band mode at a DSB receiver | `n_img` |
| $N_s$ | mean signal photon number per mode | `n_s` |
| $N_{\rm lev}$ | number of atomic levels in the ladder scheme | `n_levels` |
| $N_v$ | number of Doppler velocity classes | `n_v` |
| $\bar n_{\rm th}$, $\bar n_{\rm JPA}$ | thermal occupation of environment / source | `n_th`, `n_jpa` |
| $\Psi=\Psi_0+G$ | large-scale log-loss: deterministic absorption plus Gamma part | `psi`, `psi0`, `g` |

### 4.2 Two-mode source states

- TMSV: $a=\cosh 2r$, $c=\sinh 2r$.
- TMST (binding definition): $a=(2\bar n_{\rm JPA}+1)\cosh 2r$, $c=(2\bar n_{\rm JPA}+1)\sinh 2r$, so $c^2=a^2-(2\bar n_{\rm JPA}+1)^2$.
  *Note:* V0 §IV-F writes $c=\sqrt{a^2-1}$ for the TMST, which is a pure state and not a TMST; V0's claim that $\eta^\*$ is independent of $\bar n_{\rm JPA}$ therefore holds only for TMSV. The code follows the definition above; the paper text must be corrected (tracked in `QUESTIONS.md`).

### 4.3 Monte Carlo agreement rule ("MC-agree")

Every AC below that compares an analytic/numerical quantity with Monte Carlo uses this rule, never a bare "within 3σ":

- Fixed seed, recorded in test metadata; default $10^6$ samples.
- At each of the $m$ evaluated points, $|\hat p - p|\le z_m\,\hat\sigma$, with $z_m=\Phi^{-1}\!\big(1-\alpha/(2m)\big)$ (Bonferroni) and family-wise $\alpha=10^{-3}$. For $m=100$ this gives $z_m\approx4.4$.
- For whole-distribution comparisons, a one-sample KS test against the analytic cdf at $\alpha=10^{-3}$ may replace the pointwise check.
- $\hat\sigma$ is the binomial standard error for probabilities and the sample standard error for moments.

## 5. Functional requirements

Each requirement has an ID, a statement, and acceptance criteria (AC). Formulas referenced as `TSD §x`.

### 5.1 Physics basics — `FR-PHY`

| ID | Requirement | AC |
|---|---|---|
| FR-PHY-1 | Bose–Einstein occupation $\bar n(f,T)$, exact and Rayleigh–Jeans approximation (TSD §3.1). | $\bar n(5\text{ GHz},300\text{ K}) = 1249.70\pm0.01$. RJ relative error, defined as $(\bar n_{\rm RJ}-\bar n)/\bar n$, matches $\mathrm{expm1}(x)/x-1$ with $x=hf/k_BT$ to `rtol=1e-9`; spot value at 300 GHz, 300 K: $2.438\times10^{-2}$ (vs $x/2=2.400\times10^{-2}$). |
| FR-PHY-2 | Critical transmissivity $\eta^\*=\bar n/(1+\bar n)$. | Matches reference table TSD §11 to 5 decimals (0.99920, 0.99521, 0.98413, 0.95314 at 5/30/100/300 GHz, 300 K). |
| FR-PHY-3 | Short-dipole effective aperture $A_e=3\lambda^2/(8\pi)$ (antenna-equivalence convention, §4) and BBR-equivalent rms field $S_{E,\rm BBR}(f,T)$, both Planck (exact) and RJ variants (TSD §7.4). | (a) The mode-occupation route and the energy-density route agree to `rtol=1e-12` under RJ (algebraic identity). (b) Planck vs RJ **field amplitude** differs by < 1.5% for $f\le300$ GHz at 300 K (1.21% at 300 GHz); the corresponding power ratio (2.44%) is reported, not gated. |

### 5.2 Gaussian core — `FR-GAU`

| ID | Requirement | AC |
|---|---|---|
| FR-GAU-1 | Two-mode covariance matrices for TMSV and TMST per §4.2; physicality check $V+i\Omega\ge0$. | TMSV and TMST pass for $r\in[0,3]$, $\bar n_{\rm JPA}\in[0,1]$; a deliberately unphysical CM fails; TMST satisfies $c^2=a^2-(2\bar n_{\rm JPA}+1)^2$. |
| FR-GAU-2 | Phase-insensitive channel object `PIChannel(tau, y)` with constructors: thermal attenuator, additive noise ($y=2N_{\rm add}$), amplifier, identity; composition operator; physicality check $y\ge|1-\tau|$ (TSD §4). | (a) Composition is associative. (b) attenuator∘attenuator matches the closed form TSD §4.3 and V0 eq. (div_neff). (c) Entanglement threshold of [combining network $(\eta_{\rm cmb},\bar n_{\rm cmb})$]∘[channel] matches V0 eq. (div_netthreshold), and reduces to $\eta^\*$ exactly when $\bar n_{\rm cmb}=0$. |
| FR-GAU-3 | Apply a channel to mode B of a two-mode CM. | Matches V0's $V_{\rm out}(\eta)$ block form for attenuator inputs. |
| FR-GAU-4 | Minimum PT symplectic eigenvalue $\tilde\nu_-$ (general 2-mode formula and standard-form closed form), negativity, log-negativity (TSD §5). Both forms **must** use the cancellation-free expression $\tilde\nu_-^2 = 2\det V/\big(\tilde\Delta+\sqrt{\tilde\Delta^2-4\det V}\big)$, not V0 eq. (symp_eig) as written. Teleportation fidelity for coherent-state inputs: **optimal** fidelity $F_{\rm opt}=1/(1+\tilde\nu_-)$ (Mari & Vitali, PRA 78, 062340, 2008; assumes optimized local Gaussian operations) as the primary metric, and Braunstein–Kimble unit-gain fidelity $F_{\rm BK}=\big[(1+V_{x-}/2)(1+V_{p+}/2)\big]^{-1/2}$ as a secondary metric (TSD §5.3). Names in code and figures are `fidelity_opt` and `fidelity_bk`; a bare "fidelity" is not allowed. | (a) General and closed forms agree to **relative** `rtol=1e-10` on $10^4$ random valid CMs, including $b$ up to $10^4$ (microwave thermal regime). (b) TMSV at $r=3$: $\tilde\nu_-=e^{-6}$ to `rtol=1e-10`. (c) $F_{\rm opt}\ge F_{\rm BK}$ on a grid, with equality for symmetric states ($a=b$). (d) TMSV $r=1$ through a 5 GHz, 300 K attenuator at $\eta=0.9995$: $F_{\rm opt}=0.5858$, $F_{\rm BK}=0.5681$ (±1e-4). |
| FR-GAU-5 | Entanglement-breaking test $y\ge 1+\tau$ and TMSV separability equivalence. | For TMSV with $r\in\{0.1,0.8,2\}$: $\tilde\nu_-<1 \iff y<1+\tau$ on a 10⁴-point grid, excluding points with $|y-(1+\tau)|<10^{-9}$. |
| FR-GAU-6 | TMST entanglement condition $(a-1)(\tau a+y-1)<\tau c^2$, and TMST threshold $\eta^\*_{\rm TMST}(r,\bar n_{\rm JPA})$ by root-finding. | (a) Agrees with direct $\tilde\nu_-$ evaluation on a grid. (b) Reduces to FR-GAU-5 at $\bar n_{\rm JPA}=0$. (c) At 5 GHz, 300 K: $\eta^\*_{\rm TMST}=0.99920045$ for $\bar n_{\rm JPA}=0$ at any $r$; $=0.99932075$ for $(\bar n_{\rm JPA},r)=(0.05,0.5)$; $=0.99920628$ for $(0.002,0.5)$ (all ±1e-8). |

### 5.3 Stochastic channel — `FR-CH`

| ID | Requirement | AC |
|---|---|---|
| FR-CH-1 | Small-scale PDT: truncated noncentral $\chi^2$ (one element and $N$ elements): pdf, cdf, survival, inverse cdf, sampler, moments, $\mathcal Z_{K,N}$ (TSD §6.1). The cdf clamps its argument, $F(x)=F(\min\{x,1\})$, so $F(x)=1$ for $x\ge1$. All surrogate cdf/sf evaluations, **including $K=0$**, use the `ncx2` floor helper ($\lambda\ge10^{-300}$), not `chi2`/`gamma`/`gammainc`. Moments use the adaptive integration window with a neglected-mass check; non-finite pdf values raise. Also exposes the untruncated `surrogate_cdf` (`physical=False`) for RQ1. | (a) pdf integrates to 1 ± 1e-8; MC (10⁶) vs cdf KS distance < 2e-3; $F(x)=1$ for $x\in\{1,1.5,10\}$; limits $K=0$ (truncated exponential/Erlang) and large $K$ (concentration) pass. (b) **Pinned regression** `test_ncx2_floor_tail`: lower-tail cdf and upper-tail sf at $z=\mp5.5$ sd for $N\in\{10^5,10^6,10^7,10^8\}$ match the TSD §11 reference to `rtol=1e-6`; a failure blocks any scipy upgrade. (c) $K=0$, $N=1$, $\bar\eta=0.01$: moments $m\in\{1,2\}$ match the truncated-exponential closed form to `rtol=1e-10`. (d) A `nan` injected into the pdf inside the window raises. (e) `surrogate_cdf`$/\mathcal Z_{K,N}$ equals the truncated cdf for $x<1$ to `rtol=1e-12`. |
| FR-CH-2 | Large-scale PDT, **shifted log-gamma** (primary): $\Psi=\Psi_0+G$ with deterministic absorption $\Psi_0=\gamma_{\rm abs}d\ge0$ and $G\sim\mathrm{Gamma}(k,\theta)$, $\eta_\ell=e^{-\Psi}\in(0,e^{-\Psi_0}]$; pdf, cdf, survival, sampler, moments; parameterization from $(\mu_G,\sigma_G^2)$ or $(k,\theta)$ (TSD §6.2). | $\mathbb E[\eta_\ell^m]=e^{-m\Psi_0}(1+m\theta)^{-k}$ for $m\in\{1,2\}$, MC-agree (§4.3); support strictly $(0,e^{-\Psi_0}]$; $\Psi_0=0$, $k=4$, $\theta=0.01$ gives mean 0.96098. |
| FR-CH-3 | Large-scale baselines: V0 log-normal (untruncated, flagged as non-physical) and log-normal conditioned on $\Psi\ge0$. For the RQ2 comparison all laws are matched on the mean and variance of $\Psi$ with the same $\Psi_0$ **[A]**; V0 log-normal realizations with $\eta>1$ are evaluated at $\min(\eta,1)$ (`eta_gt1: clip`), flagged `physical=False`, and the affected mass is reported with every number (TSD §6.2). | (a) Reports mass above 1 for the untruncated version ($\Phi(-1)=0.15866$ for $\sigma_\Psi=\bar\Psi$). (b) As $\sigma_\Psi\to0$, all three laws give the same $P_{\rm out}$ and ergodic capacity to `rtol=1e-6`. (c) Every V0 log-normal service result carries `physical=False` and the clipped mass. |
| FR-CH-4 | Optional obstacle generator: compound-Poisson log-loss moment-matched to Gamma (TSD §6.3). | Moment match exact; MC histogram vs fitted Gamma reported (no pass/fail threshold, diagnostic only). |
| FR-CH-5 | Compound PDT $\eta=\eta_\ell\eta_s$: pdf, cdf, survival, quantile, sampler, first two moments (TSD §6.4). Moments use the **truncated** small-scale moments $\mathbb E^{\rm QC}[\eta_s^m]$, not the scale parameter $\bar\eta_s$ (which overstates the mean after truncation; cf. V0 eq. (mean_eta)). | Numerical cdf vs MC: MC-agree (§4.3) over ≥ 50 points; $\mathbb E[\eta^m]=\mathbb E[\eta_\ell^m]\,\mathbb E^{\rm QC}[\eta_s^m]$ for $m\in\{1,2\}$ to `rtol=1e-8` (quadrature) and MC-agree. |
| FR-CH-6 | Link-budget helper giving mean collection efficiency $\bar\eta_\Sigma$ from distance, frequency, apertures, absorption (TSD §6.5). Geometric coupling model (config `coupling_model`):<br>• `goubau` (**default**): $\eta_{\rm geo}=1-e^{-\tau^2}$ with $\tau^2=A_tA_r/(\lambda d)^2$. It is passivity-preserving and reduces to Friis for $\tau^2\ll1$. It is an approximation (optimally tapered apertures), so results with $\bar\eta\to1$ are model-dependent.<br>• `friis_clipped`: $\min(\tau^2,1)$, a flagged baseline with `physical=False`, analogous to the V0 log-normal.<br>Absorption enters only through $\Psi_0$ (FR-CH-2), never twice.<br>**Receive aperture:** $A_r$ is the collecting aperture of an antenna-fed receiver (default) or $A_e$ for a bare cell. One optional receive-side enhancement factor `G_enh` (default 1.0, must be ≥ 1) gives $A_{r,\rm eff}=G_{\rm enh}A_r$; there is no transmit-side factor (transmit gain is $G_t$). Feed coupling is assumed lossless **[A]**; loss is modeled as an attenuator, not as $G_{\rm enh}<1$. | (a) `goubau`: $\bar\eta_\Sigma\le1$ by construction, with no clipping in its code path. (b) Friis limit: relative difference from Friis $\le\tau^2/2$ for $\tau^2\le10^{-2}$. (c) 5 GHz, $A_t=A_r=1$ m², $d=10$ m: $\tau^2=2.7816$, $\eta_{\rm geo}=0.93806$ (±1e-5); `friis_clipped` returns 1 and `physical=False`. (d) With $\gamma_{\rm abs}=1.44\times10^{-6}$ m⁻¹ at 5 GHz (`source:` GCF:22), absorption-only range at $\eta^\*$ = 555.47 m ± 0.5 m. (e) `G_enh`: $\tau^2$ scales linearly in $G_{\rm enh}$; under `goubau`, $\eta_{\rm geo}$ increases with $G_{\rm enh}$ and stays $<1$; $G_{\rm enh}<1$ raises; the link-budget API has no transmit-side enhancement argument. |

### 5.4 Receivers — `FR-RX`

| ID | Requirement | AC |
|---|---|---|
| FR-RX-1 | Receiver base type with two classes: **measuring** (parameters $N_{\rm amp}\ge\tfrac12$ and required `sideband: ssb\|dsb`) and **quantum front end** (a `PIChannel`) (TSD §7.1). DSB receivers add the image-band occupation $N_{\rm img}=(1-\eta)\bar n_{\rm th}+\eta\bar n_{\rm tx,img}$ (default $\bar n_{\rm tx,img}=0$) **[A]**. Defaults: RAR `dsb`, electronic `ssb`. | (a) Constructing a measuring receiver with $N_{\rm amp}<\tfrac12$ raises an error. (b) A measuring receiver without `sideband` fails validation. (c) `ssb` gives $N_{\rm img}=0$; `dsb` at $\eta=0$ gives $N_{\rm img}=\bar n_{\rm th}$. |
| FR-RX-2 | Ideal heterodyne ($N_{\rm amp}=\tfrac12$), ideal homodyne (special case flag), quantum-limited amplifier. | Ideal heterodyne capacity equals $C_{\rm het}$ of TSD §8.2 exactly. |
| FR-RX-3 | Electronic receiver from noise temperature $T_n\ge0$, with a required convention matching the source of the $T_n$ value (TSD §7.3):<br>• `rj`: $N_{\rm amp}=k_BT_n/hf$;<br>• `planck` (Caves): $N_{\rm amp}=\bar n(f,T_n)$;<br>• `callen_welton`: $N_{\rm amp}=\bar n(f,T_n)+\tfrac12$. | (a) $T_n=hf/k_B$ gives $N_{\rm amp}=1$ under `rj`. (b) At 5 GHz, $T_n=hf/(k_B\ln3)=218.42$ mK gives 0.9102 / 0.5000 / 1.0000 under `rj` / `planck` / `callen_welton` (±1e-4). (c) $T_n$ below the quantum limit raises under `rj` and `planck`. |
| FR-RX-4 | Rydberg atomic receiver from field sensitivity $S_E$ (and optional bandwidth $B$, saturation field $E_{\rm sat}$ for validity flag) via the short-dipole mapping (§4 conventions); also direct $T_n$ or $N_{\rm amp}$ input (TSD §7.4). Constructor takes a required `field_convention: "rms" \| "peak"` argument; there is no default. A mapped $N_{\rm amp}<\tfrac12$ raises; it is never clamped. The linear-range flag compares the received **peak** field $E_{\rm rx,pk}=\sqrt{2\eta_0P_{\rm rx}/A_r}$ with the peak $E_{1\rm dB,pk}$. $N_{\rm amp}$ always uses the bare-cell $A_e$ and the $S_E$ at the atoms; $A_r$ and $G_{\rm enh}$ never enter it (a lossless feed or enhancing structure raises the atom field by $\sqrt G$ and the collecting area by $G$, which cancel; TSD §7.4). | (a) 10 nV/cm/√Hz rms at 5 GHz → $N_{\rm amp}=3.438\times10^5$ (±0.1%), $T_n=8.25\times10^4$ K (±0.1%). (b) The same number declared `peak` → $N_{\rm amp}=1.719\times10^5$ (±0.1%). (c) Omitting `field_convention` raises an error. (d) An $S_E$ mapping below $N_{\rm amp}=\tfrac12$ raises. (e) Linear-range flag on a hand case where $E_{\rm rx,rms}<E_{1\rm dB,pk}<E_{\rm rx,pk}$ returns `linear_regime=False`. (f) $N_{\rm amp}$ is identical for $G_{\rm enh}\in\{1,10,100\}$ and $A_r\in\{A_e,\,2\text{ m}^2\}$; the $S_E\to N_{\rm amp}$ mapping accepts no aperture argument. |
| FR-RX-5 | Coherent transducer $(\eta_t,\bar n_t)$ as quantum front end for entanglement service. | Composition with channel reproduces TSD §7.5 threshold. |
| FR-RX-6 | `LindbladRARModel`: SI atomic/optical inputs → $\kappa$, $S_E$, $N_{\rm amp}$, $B_{3\rm dB}$, $E_{1\rm dB}$ (TSD §14.10), with `method` = `auto` \| `weak_probe` \| `full` (TSD §14.6). `RydbergReceiver.from_physics(...)` builds an FR-RX-4 receiver from it. | Method used and max weak-probe/full discrepancy recorded in run metadata; T-LIN-11 passes. |

### 5.5 Services — `FR-SVC`

| ID | Requirement | AC |
|---|---|---|
| FR-SVC-E1 | Receiver-referred threshold $\eta^\*_{\rm rx}$ for any quantum front end (closed form for additive noise, $\eta^\*_{\rm rx}=(\bar n+N_{\rm add})/(\bar n+1)$; root-find otherwise) (TSD §8.1). | $N_{\rm add}=0\Rightarrow\eta^\*$; $N_{\rm add}=\tfrac12\Rightarrow0.99960$ at 5 GHz; $N_{\rm add}\ge1\Rightarrow$ "no window"; closed form vs root-find agree to 1e-12 for additive noise. |
| FR-SVC-E2 | Entanglement outage $P_{\rm out}$, average log-negativity, average $\tilde\nu_-$, average $F_{\rm opt}$ and $F_{\rm BK}$ (FR-GAU-4) over the compound PDT. All small-scale cdf evaluations at $\eta^\*/\eta_\ell$ use the clamp of FR-CH-1. **Truncation effect (RQ1):** also $P_{\rm out}^{\rm surr}$ with the untruncated `surrogate_cdf`, reported as $\Delta P_{\rm out}=P_{\rm out}-P_{\rm out}^{\rm surr}$ and the ratio, next to $1-\mathcal Z_{K,N}$ (TSD §8.1). | (a) Quadrature vs MC: MC-agree (§4.3). (b) **Under the V0 log-normal baseline:** V0 limiting cases 1–4 reproduced with V0's closed forms, Cases 2–3 evaluated with the clamp $\min\{x,1\}$. (c) **Under shifted log-gamma:** Case 1 ($K\to\infty$) equals $Q\big(k,\ [\ln(\bar\eta_s/\eta^\*)-\Psi_0]/\theta\big)$ (regularized upper incomplete gamma) when the argument is positive and 1 otherwise (full table in TSD §8.1); Case 4 (deterministic) is the same step function as V0; Cases 2–3 match quadrature of the clamped cdf to `rtol=1e-8`. (d) Tested at the region $\eta_\ell<\eta^\*$ (where the clamp is active). (e) RQ1 identity: deterministic large-scale factor with $\eta^\*/\eta_\ell<1$ gives $P_{\rm out}/P_{\rm out}^{\rm surr}=1/\mathcal Z_{K,N}$ to `rtol=1e-12`. |
| FR-SVC-C1 | Per-realization capacities: Holevo, heterodyne, homodyne, measuring receiver (TSD §8.2). The measuring receiver's input-referred noise, in photons, is $(1-\eta)\bar n_{\rm th}+\tfrac12+N_{\rm amp}+N_{\rm img}$: state vacuum, receiver-added noise, and the image band for DSB. With `ssb` and $N_{\rm amp}=\tfrac12$ this equals the textbook heterodyne $(1-\eta)\bar n_{\rm th}+1$ (TSD §8.2). | (a) Ordering $C_\chi\ge C_{\rm het}$, $C_\chi\ge C_{\rm hom}$ holds on a grid. (b) $C_{\rm rx}$ is non-increasing in $N_{\rm amp}$, equals $C_{\rm het}$ at `ssb`, $N_{\rm amp}=\tfrac12$, and satisfies $C_{\rm rx}^{\rm dsb}\le C_{\rm rx}^{\rm ssb}$. (c) All → 0 as $\eta\to0$. |
| FR-SVC-C2 | Ergodic capacity, rate-outage probability $P(C<R)$, and $\epsilon$-outage capacity $C_\epsilon$ via the compound quantile. | $C_\epsilon$ equals $C(F^{-1}(\epsilon))$ to 1e-6 relative; MC cross-check per §4.3. |
| FR-SVC-C3 | Throughput in bit/s using receiver bandwidth $B$. | Unit test on a hand case. |
| FR-SVC-C4 | Crossover solver: required rms $S_E$ for a receiver noise penalty of $\Delta$ dB relative to **ideal SSB heterodyne**, for both `ssb` and `dsb` receivers (closed form, TSD §8.3), plus a capacity-based variant (numeric). DSB receivers have an image floor of $10\log_{10}[(2\bar n_{\rm th}+1)/(\bar n_{\rm th}+1)]$ at $\eta\to0$. | (a) Closed form vs numeric root agree to 1e-6. (b) 5 GHz, 300 K, $\eta\to0$, `ssb`, $\Delta=3$ dB: $S_E^{\rm req}/S_{E,\rm BBR}=0.99823$ (±1e-4). (c) Same with `dsb`: returns `NaN` with a reason naming the image floor (3.009 dB). (d) `dsb`, $\Delta=4$ dB: ratio 0.71659; $\Delta=6$ dB: 1.40849 (±1e-4). (e) Unattainable targets return `NaN` with a reason string. |
| FR-SVC-C5 | **RAR-vs-LNA throughput map (RQ5):** $\rho(f,d;K)=10\log_{10}(\bar R_b^{\rm RAR}/\bar R_b^{\rm LNA})$ with ergodic throughput $\bar R_b=B\,\mathbb E[C_{\rm rx}]$ (TSD §8.4). Both receivers share the same antenna-fed aperture $A_r$ (same $\eta$ per realization) and the same TX power $P_t$, so $N_s=P_t/(hfB)$ differs with each receiver's bandwidth. Per-band RAR $(S_E,B)$ and LNA $(T_n,B)$ inputs come from config with `source:` **[A]** (IDEA Q2), log-log interpolated between configured bands, masked outside them. | (a) Power-limited asymptote: for deterministic $\eta$ and $\eta N_s/D<10^{-4}$, $\rho\to10\log_{10}(D_{\rm LNA}/D_{\rm RAR})$ to `rtol=1e-3`, with $D=(1-\eta)\bar n_{\rm th}+\tfrac12+N_{\rm amp}+N_{\rm img}$. (b) Equal bandwidths and $D_{\rm RAR}\ge D_{\rm LNA}$ ⇒ $\rho\le0$ on the whole grid. (c) Cells outside the configured band range are masked, not extrapolated. (d) A missing `source:` for any per-band value fails validation. |

### 5.6 Diversity — `FR-DIV` (keep in code even if moved out of the paper)

| ID | Requirement | AC |
|---|---|---|
| FR-DIV-1 | $N$-element passive-MRC PDT and array outage (TSD §6.1, §9). Must stay accurate up to $N=10^8$ (noncentral $\chi^2$ with $2N$ dof), through the `ncx2` floor helper for every $K$ (FR-CH-1). | $N=1$ reduces to single-element results exactly; cdf agrees to 1% relative down to 1e-8 (NFR-2) with a series reference with an `mpmath` prefactor for $K=0$, $N\in\{10^5,10^7,10^8\}$, and with an `mpmath` Poisson-mixture reference for $K=10$, $N\in\{10^5,10^7\}$. |
| FR-DIV-2 | Selection-combining tail and Chernoff bound. The exact tail (V0 eq. div_sc) relies on at most one element exceeding $x$ under joint passivity, so it is valid **only for $x\ge\tfrac12$**. For $x<\tfrac12$ the function raises a domain error; it must not silently return a value. The Chernoff bound is valid for all $x$. At 300 K, $\eta^\*>\tfrac12$ holds only for $f<4.33$ THz. | (a) Bound ≥ exact tail on the grid $x\in[\max(\eta^\*,\tfrac12),1)$. (b) Exact tail vs per-element MC (FR-DIV-4) MC-agree on that grid. (c) Calling the exact tail with $x=0.3$ raises a domain error. |
| FR-DIV-3 | Cramér rate $\mathcal R_K(a)$, design law, and $N_{\rm req}$ solver (integer bisection). | $\mathcal R_K(1)=0$; $\mathcal R_0(a)=a-1-\ln a$; $N_{\rm req}$ for $F_N(\eta^\*)\le10^{-3}$, $\eta_\ell=1$, $\bar\eta_\Sigma=\eta^\*+0.9(1-\eta^\*)$, 300 K, matches the table below to ±0.1% (pre-verified; consistent with V0 Table at its displayed precision; re-verified unchanged through the `ncx2` floor helper in v0.6.1). Any discrepancy goes to `QUESTIONS.md`. |
| FR-DIV-4 | Monte Carlo for the array. **MRC:** sample $\eta_{\rm MRC}$ directly as $\bar\eta_\Sigma/(2N(K+1))\cdot\chi'^2_{2N}(2NK)$ and apply the joint truncation $\eta_{\rm MRC}\le1$ by rejection. **SC:** per-element sampling with joint rejection $\sum_i\eta_{s,i}\le1$, limited to $N\le N_{\max}$ (config, default 64). | (a) MRC sum-sampling vs FR-DIV-1: MC-agree (§4.3) for $N$ up to $10^7$. (b) For $N\le64$, sum-sampling and per-element sampling agree (two-sample KS, $\alpha=10^{-3}$). (c) SC MC vs FR-DIV-2 exact tail: MC-agree. (d) Requesting per-element MC with $N>N_{\max}$ raises an error naming the sum-sampling alternative. |

Reference values for FR-DIV-3 ($N_{\rm req}$, exact integers):

| $f$ (GHz) | $K=0$ | $K=10$ | $K=100$ |
|---|---|---|---|
| 5 | 20 017 832 | 3 474 510 | 394 473 |
| 30 | 556 629 | 96 662 | 10 975 |
| 100 | 50 243 | 8 737 | 993 |
| 300 | 5 630 | 983 | 112 |

### 5.7 Rydberg receiver physics model — `FR-LIN`

| ID | Requirement | AC |
|---|---|---|
| FR-LIN-1 | Level scheme, decay channels, transit reset, dephasing and derived coherence rates for $N_{\rm lev}\in\{3,4\}$ ladder schemes (TSD §14.3). | $N_{\rm lev}\notin\{3,4\}$ rejected; coherence rates match TSD §14.3 on a hand case. |
| FR-LIN-2 | Weak-probe engine: tridiagonal steady state, dynamics, transfer function $H(i\omega)$, vectorized over velocity classes (TSD §14.4). | T-LIN-1, T-LIN-3, T-LIN-7 pass. **Benchmark (recorded, not CI-gated):** time per velocity class at $N_v=10^4$, target ≤ 50 ns; result logged to `docs/RESULTS.md`. |
| FR-LIN-3 | Full Lindblad engine: Liouvillian as $\mathcal A+v\mathcal B$, batched bordered steady state and transfer function (TSD §14.5). | T-LIN-2, T-LIN-4, T-LIN-6 pass. **Benchmark (recorded, not CI-gated):** time per velocity class at $N_v=10^3$, target ≤ 20 µs. |
| FR-LIN-4 | Doppler averaging with automatic grid refinement and recorded final $N_v$; method selection `auto` with checkpoint discrepancy (TSD §14.6). | T-LIN-5, T-LIN-8, T-LIN-11 pass. |
| FR-LIN-5 | Doppler-averaged response: $H(i\omega)$, $B_{3\rm dB}$ and peaking value (TSD §14.7). | Bandwidth refinement agrees with a dense-grid scan to 1e-3 relative. |
| FR-LIN-6 | Readout, sensitivity and linear range in SI: $\alpha$, $P_{\rm out}$, $\kappa$, $S_E$ (rms), $E_{1\rm dB}$ (peak) (TSD §14.7–14.9). $E_{1\rm dB}$ is where $\lvert20\log_{10}c\rvert=1$ dB, covering compression and expansion; the case that occurred is recorded. A receiver built from the model has `sideband="dsb"`. | (a) T-LIN-10 and T-LIN-12 pass. (b) Unit tests on the SI conversions. (c) A synthetic expansion case returns $E_{1\rm dB}$ flagged `expansion`. |
| FR-LIN-7 | Optional QuTiP cross-check of the full engine (tests only). | T-LIN-9 passes or skips cleanly when QuTiP is absent. |

### 5.8 Experiments, outputs, reproducibility — `FR-EXP`

| ID | Requirement | AC |
|---|---|---|
| FR-EXP-1 | Experiments defined by YAML configs validated by a schema; unknown keys rejected. | Invalid config fails with a clear message. |
| FR-EXP-2 | One command per experiment (`make exp ID=E<n>`) and one command for all paper figures (`make paper-figs`). | Fresh clone → `make paper-figs` succeeds. |
| FR-EXP-3 | Each run writes: figure (PDF + PNG), raw data (CSV), metadata (JSON: git hash, config hash, seed, package versions, wall time). | Files present and metadata complete. |
| FR-EXP-4 | Results cache keyed by config hash; `--force` to recompute. | Second run is a cache hit. |
| FR-EXP-5 | LaTeX export of the parameter table and reference tables (booktabs). | Compiles in IEEEtran. |
| FR-EXP-6 | Figure style: single-column IEEE width (3.5 in), vector PDF, serif font, colorblind-safe palette, no titles inside figures (captions live in LaTeX). | Visual check by Abdurrahman. |
| FR-EXP-7 | External inputs: literature RAR points live in `configs/literature_rar.yaml` (schema in TSD §12). Each entry requires `source:`, `field_convention: rms\|peak` and `sideband: ssb\|dsb`, plus optional `includes_bbr`. Peak values are converted to rms on load. | (a) An entry missing `field_convention` fails schema validation. (b) If the file is missing or empty, E6/E9(c) render without the overlay and emit a warning (run does not fail). |

## 6. Experiments (map to paper figures)

| ID | Question | Sweep | Output |
|---|---|---|---|
| E1 | Small-scale PDT shape and truncation effect (RQ1) | $K\in\{0,1,10,100\}$, $\bar\eta_s\in[0.9,1]$; compound panel over the E3 distance grid | pdf curves; removed mass $1-\mathcal Z_K$ vs $\bar\eta_s$; $\Delta P_{\rm out}$ and $P_{\rm out}/P_{\rm out}^{\rm surr}$ (truncated vs untruncated) vs $\bar\eta_s$ and vs distance (FR-SVC-E2) |
| E2 | Large-scale law comparison (RQ2) | $\sigma_\Psi/\bar\Psi$ grid, laws matched on $\Psi$ moments (FR-CH-3) | pdfs; mass above 1 for V0 log-normal; $\Delta P_{\rm out}$ and relative ergodic-capacity difference $\Delta\bar C/\bar C$ ($C_\chi$, $C_{\rm het}$, RAR $C_{\rm rx}$) of each baseline vs log-gamma, with the clipped mass shown |
| E3 | Entanglement outage vs distance | $f\in\{5,30,100,300\}$ GHz, $K\in\{0,10,100\}$, $d\in[10,10^3]$ m; near-field apertures (TSD §12 example; a 30 dBi horn to a bare dipole gives outage ≡ 1) | $P_{\rm out}$, $\mathbb E[E_N]$; sensitivity panel `goubau` vs `friis_clipped` |
| E4 | Receiver-referred window (RQ3) | $N_{\rm add}\in[0,1.2]$; receiver classes overlaid | $\eta^\*_{\rm rx}$ vs $N_{\rm add}$; window width $1-\eta^\*_{\rm rx}$ |
| E5 | Classical ergodic capacity and RAR-vs-LNA throughput (RQ5) | (a) $N_s$, $d$, bands; receivers: Holevo, het, hom, LNA, RAR. (b) $(f,d)$ grid, $K\in\{0,10,100\}$, equal $P_t$, shared aperture (FR-SVC-C5) | (a) bits/mode and bit/s. (b) map of $\rho=10\log_{10}(\bar R_b^{\rm RAR}/\bar R_b^{\rm LNA})$ per $K$ with the $\rho=0$ contour and power-/bandwidth-limited regions labeled |
| E6 | **BBR crossover map** (RQ4) | $f\in[1,300]$ GHz, $T=300$ K | $S_{E,\rm BBR}(f)$; required $S_E$ for 1/3 dB gap as **separate SSB and DSB curves**, with the DSB image floor marked (3 dB unattainable for DSB); literature RAR points overlaid with their declared sideband (FR-EXP-7) |
| E7 | Diversity hardening and $N_{\rm req}$ | $N$, $K$, bands | $N_{\rm req}$ table, outage vs $N$ |
| E8 | Outage capacity | $\epsilon\in[10^{-4},10^{-1}]$ | $C_\epsilon$ curves per receiver |
| E9 | **Validity of the Gaussian RAR model** (IDEA R4) and model-derived RAR points | $\Omega_c$, $E_{\rm LO}$ grid (TSD §12); link distance for received field | (a) $E_{1\rm dB}$ vs received signal field over distance → region where the linear model holds; (b) $S_E$, $N_{\rm amp}$, $B_{3\rm dB}$ vs $\Omega_c$, $E_{\rm LO}$; (c) model-derived points overlaid on the E6 crossover map |

Default parameter sets live in `configs/defaults.yaml`; any number taken from literature carries a `source:` field.

## 7. Non-functional requirements

- **NFR-1 Correctness first.** Every formula has ≥1 limiting-case test and ≥1 independent cross-check (TSD §10).
- **NFR-2 Numerical robustness.** No catastrophic cancellation near $\eta\to1$ (use survival functions, log-domain, scaled Bessel, `expm1`/`log1p`, and the stable $\tilde\nu_-$ form of FR-GAU-4). Tail probabilities down to 1e-8 must be accurate to 1% relative (validated with `mpmath`). Large-shape $\chi^2$/gamma cdfs use the `ncx2` floor helper (TSD §6.1, §13 rule 10), guarded by a pinned regression test; non-finite intermediate values raise and are never masked.
- **NFR-3 Performance.** Any single figure regenerates in < 5 min on a laptop CPU; MC defaults to 10⁶ samples, vectorized. This is the only CI-gated performance requirement; per-kernel targets (FR-LIN-2/3) are benchmarks.
- **NFR-4 Code quality.** Python ≥ 3.11, type hints, `ruff` + `mypy --strict` on `src/`, docstrings with the TSD section reference, test coverage ≥ 90% on `src/`.
- **NFR-5 Reproducibility.** Deterministic given seed and config; pinned dependencies.
- **NFR-6 Traceability.** Each public function cites its FR ID and TSD section in its docstring.
- **NFR-7 No compiled code.** NumPy/SciPy vectorization only. Adding Numba, Cython or C/C++ requires a `QUESTIONS.md` entry with a profile showing a figure that breaks NFR-3.

## 8. Definition of done (v1)

1. All FR acceptance criteria pass in CI.
2. E1–E9 regenerate from a clean clone with one command (with or without `configs/literature_rar.yaml`, per FR-EXP-7).
3. `QUESTIONS.md` has no unresolved item marked `blocking`.
4. `docs/RESULTS.md` lists every reference value used in tests with its source (TSD §11, literature, or derived), plus the FR-LIN benchmark results.

## 9. Changelog

**0.6.1 (2026-11-10)**, paired with TSD v0.4.1. Resolves QUESTIONS Q-3, Q-4, Q-5; closes the IDEA RQ1/RQ2/RQ5 output gaps.
- FR-CH-1 (Q-3, Q-5): every surrogate cdf, including $K=0$, uses the `ncx2` floor helper, with a pinned scipy regression test (`chi2` is off by 2.3% at $N=10^7$ and 32% at $N=10^8$ in the lower tail, verified independently). Moments use an adaptive window with a neglected-mass check; `nan` raises. `surrogate_cdf` added for RQ1.
- FR-CH-3 (RQ2): moment-matched law comparison; V0 log-normal $\eta>1$ realizations are clipped, flagged and their mass reported.
- FR-CH-6 (Q-4): single receive-side `G_enh` (default 1, ≥ 1); no transmit factor; $A_r$ is an antenna-fed aperture by default, lossless feed **[A]**.
- FR-RX-4: $N_{\rm amp}$ is invariant to $A_r$ and $G_{\rm enh}$ (AC f).
- FR-SVC-E2 (RQ1): truncated-vs-untruncated outage with the $1/\mathcal Z$ identity check.
- FR-SVC-C5 added (RQ5): RAR-vs-LNA throughput map at equal TX power and shared aperture, with asymptote checks.
- FR-DIV-1/3: $N=10^8$ reference added; $N_{\rm req}$ table re-verified unchanged through the new helper.
- E1/E2/E5 outputs extended and RQ labels corrected (E5 was mislabeled RQ1); NFR-2 extended.

**0.6 (2026-11-03)**, paired with TSD v0.4.
- §4 sideband convention added. FR-RX-1 now requires `sideband`; FR-SVC-C1 includes $N_{\rm img}$; FR-SVC-C4 has separate SSB and DSB crossovers with the DSB image floor (≈3 dB at $\eta\to0$, so a 3 dB target is unattainable for a DSB RAR); E6 plots both curves. RAR defaults to DSB **[A]**.
- FR-CH-2 corrected to the shifted log-gamma $\Psi=\Psi_0+G$ (the v0.5 form omitted $e^{-m\Psi_0}$). The FR-SVC-E2 Case 1 formula is shifted accordingly.
- FR-CH-6: Goubau is the default and is labeled an approximation; `friis_clipped` is reinstated as a flagged baseline rather than forbidden; E3 adds a sensitivity panel and near-field apertures.
- FR-GAU-4: $F_{\rm BK}$ corrected to the general $[(1+V_{x-}/2)(1+V_{p+}/2)]^{-1/2}$ form (v0.5's averaged form is exact only for symmetric variances).
- FR-RX-3: three named conventions (`rj`, `planck`, `callen_welton`) with quantum-limit reference values.
- FR-RX-4: sub-quantum-limit input raises; linear-range flag is peak vs peak.
- FR-LIN-6: $S_E$ rms and $E_{1\rm dB}$ peak made explicit; expansion case covered; T-LIN-12 added.
- FR-EXP-7: `sideband` added to literature entries.
- §4.1: now mirrors TSD §2, which is binding for identifiers (`n_th`, `n_jpa`, `n_img`, `n_s`, `n_v`).

**0.5 (2026-11-03)** — physics corrections
- FR-CH-6: aperture coupling now uses Goubau's passivity-preserving form $1-e^{-\tau^2}$. Plain Friis exceeds 1 inside the E3 range (2.78 at 10 m, 1 m² apertures, 5 GHz), and clipping is forbidden.
- FR-GAU-4 and FR-SVC-E2: $1/(1+\tilde\nu_-)$ is now labeled the *optimal* fidelity (Mari & Vitali 2008). Braunstein–Kimble unit-gain fidelity is added as a separate metric, with reference values.
- §4, FR-RX-4, FR-EXP-7: $S_E$ is pinned to rms single-sided. `field_convention` is now required for every RAR input; a peak value means a factor 2 in $N_{\rm amp}$.
- §4: the short-dipole mapping is declared a modeling convention.
- FR-DIV-2: the exact selection-combining tail is restricted to $x\ge\tfrac12$ and raises outside it; the test grid is restricted accordingly.
- FR-SVC-C1: measuring-receiver noise bookkeeping $(1-\eta)\bar n+\tfrac12+N_{\rm amp}$ made explicit, with an escalation rule if TSD §8.2 differs.

**0.4 (2026-11-03)**
- §4.1 added: reserved symbols. $N$ is now only the element count; noise uses $N_{\rm add}$, levels $N_{\rm lev}$. Made explicit that $y_{\rm add}=2N_{\rm add}$ and that $N_{\rm add}$ and $N_{\rm amp}$ share units.
- §4.2 added: binding TMST definition; V0's $c=\sqrt{a^2-1}$ inconsistency flagged.
- §4.3 added: MC-agree rule (Bonferroni, family-wise $\alpha=10^{-3}$) replaces every "within 3σ".
- FR-PHY-1: denominator, tolerance and 300 GHz spot value fixed.
- FR-PHY-2: reference values listed inline.
- FR-PHY-3: split into an RJ algebraic identity and a Planck-vs-RJ field-amplitude bound.
- FR-GAU-1: now also checks TMST.
- FR-GAU-2: combining-network AC added (V0 eq. div_neff, div_netthreshold).
- FR-GAU-4: stable $\tilde\nu_-$ form mandated; tolerance now relative 1e-10; $r=3$ spot check added.
- FR-GAU-5: excludes boundary points.
- FR-GAU-6: TMST threshold reference values added.
- FR-CH-1: cdf clamp $\min\{x,1\}$ made explicit and tested.
- FR-CH-2: $\Psi\sim$ Gamma stated.
- FR-CH-3: exact value 0.15866.
- FR-CH-5: uses truncated moments.
- FR-CH-6: $\gamma_{\rm abs}$ pinned with source; value 555.47 m.
- FR-SVC-E1: closed form stated; $N\to N_{\rm add}$.
- FR-SVC-E2: limiting cases split by large-scale law; log-gamma Case 1 given; clamp required.
- FR-SVC-C1: monotonicity in $N_{\rm amp}$ added.
- FR-DIV-1: large-$N$ accuracy requirement added.
- FR-DIV-3: exact $N_{\rm req}$ reference table added (pre-verified against V0).
- FR-DIV-4: MRC uses sum-sampling; per-element MC only for SC with $N\le N_{\max}$.
- FR-LIN-2/3: per-class timings are now recorded benchmarks, not CI gates.
- FR-EXP-2: commands named.
- FR-EXP-7 added: literature RAR input file, with graceful fallback.
- E4: $N\to N_{\rm add}$.
- DoD 2 and DoD 4 updated.
- Out-of-scope clarified (path temperature vs receiver network).
- NFR-2/3 updated.

**0.3 (2026-11-03)** — initial reviewed draft.