# TSD — Technical Specification

**Project:** `qmwlink` · **Version:** 0.3 (2026-11-03; §14 is a pure-Python receiver physics model, no compiled code) · **Owner:** Abdurrahman
**Reads with:** `docs/IDEA.md` (why), `docs/FSD.md` (what). This document is the single source of truth for formulas, algorithms, numerics and code layout. If code and TSD disagree, the code is wrong or the TSD needs a reviewed change; never silently diverge.

Status tags used below: **[V]** verified numerically while writing this spec · **[D]** derived in this spec, needs a proof in the paper · **[L]** taken from literature, implementer must not alter · **[A]** assumption that Abdurrahman must confirm.

---

## 1. Architecture

```
qmwlink/
├── agent.md                 # instructions for the coding agent
├── QUESTIONS.md             # open questions raised during implementation
├── pyproject.toml           # deps pinned, ruff, mypy, pytest config
├── Makefile                 # test, lint, figs, paper-figs, clean
├── docs/  IDEA.md  FSD.md  TSD.md  RESULTS.md
├── configs/  defaults.yaml  e1.yaml … e9.yaml
├── src/qmwlink/
│   ├── constants.py         # FR-PHY      §3
│   ├── thermal.py           # FR-PHY      §3
│   ├── gaussian/
│   │   ├── states.py        # FR-GAU-1    §5.1
│   │   ├── channels.py      # FR-GAU-2,3  §4
│   │   └── entanglement.py  # FR-GAU-4..6 §5
│   ├── channel/
│   │   ├── small_scale.py   # FR-CH-1     §6.1
│   │   ├── large_scale.py   # FR-CH-2,3   §6.2
│   │   ├── obstacles.py     # FR-CH-4     §6.3
│   │   ├── compound.py      # FR-CH-5     §6.4
│   │   ├── link_budget.py   # FR-CH-6     §6.5
│   │   └── diversity.py     # FR-DIV      §9
│   ├── receivers/
│   │   ├── base.py          # FR-RX-1,2   §7.1–7.2
│   │   ├── electronic.py    # FR-RX-3     §7.3
│   │   ├── rydberg.py       # FR-RX-4     §7.4
│   │   ├── transducer.py    # FR-RX-5     §7.5
│   │   └── rydberg_physics/ # FR-RX-6, FR-LIN  §14 (layout in §14.10)
│   ├── services/
│   │   ├── entanglement.py  # FR-SVC-E    §8.1
│   │   ├── classical.py     # FR-SVC-C1..3 §8.2
│   │   └── crossover.py     # FR-SVC-C4   §8.3
│   ├── numerics/  special.py  integrate.py  sampling.py   §13
│   ├── experiments/  schema.py  runner.py  registry.py  e1.py … e9.py   §12
│   ├── io/  export.py  latex.py
│   └── plotting/  style.py
├── tests/  unit/  validation/  regression/
└── results/                 # gitignored; figures, CSV, metadata
```
The whole project is pure Python and is built by the coding agent.

**Dependencies:** Python ≥ 3.11; `numpy`, `scipy`, `mpmath` (reference values in tests only), `matplotlib`, `pydantic` (config schema), `pyyaml`, `pytest`, `hypothesis`, `ruff`, `mypy`. Optional test oracle: `qutip`. No compiled extensions (Numba, Cython, C/C++); §14.12 shows they are not needed. No other dependencies without an entry in `QUESTIONS.md`.

**Style:** pure functions on NumPy arrays where possible, vectorized over the transmissivity axis; small frozen dataclasses for parameter bundles; no hidden global state.

## 2. Notation and conventions

| Symbol | Meaning | Code name |
|---|---|---|
| $f$, $\lambda=c/f$ | carrier frequency, wavelength | `f_hz`, `lam_m` |
| $T$ | environment temperature | `T_env_k` |
| $\bar n_{\rm th}$ | environment occupation $\bar n(f,T)$ | `n_th` |
| $\eta,\ \eta_s,\ \eta_\ell$ | total, small-scale, large-scale transmissivity | `eta`, `eta_s`, `eta_l` |
| $\bar\eta_s$ | small-scale scale parameter (link-budget collection efficiency) | `eta_s_bar` |
| $\bar\eta_\Sigma$ | array total collection efficiency, $N\bar\eta_s$ | `eta_sigma` |
| $K$ | Rician factor | `K` |
| $\Psi$ | path-integrated log-loss, $\eta_\ell=e^{-\Psi}$ | `psi` |
| $(\tau,y)$ | phase-insensitive channel $V\mapsto\tau V+yI$ | `PIChannel(tau, y)` |
| $N_s$ | mean signal photons per mode at TX | `n_s` |
| $N_{\rm amp}$ | input-referred added noise quanta of a measuring receiver, $\ge\frac12$ | `n_amp` |
| $S_E$ | intrinsic field sensitivity, V m⁻¹ Hz⁻¹ᐟ² | `s_e` |
| $\eta_0$ | free-space impedance $\sqrt{\mu_0/\epsilon_0}$ | `ETA0` |

Quadratures: vacuum variance 1, $[\hat x,\hat p]=2i$. Covariance ordering $(x_A,p_A,x_B,p_B)$. Symplectic form $\Omega=\bigoplus\begin{psmallmatrix}0&1\\-1&0\end{psmallmatrix}$.

## 3. Physics basics

### 3.1 Thermal occupation (FR-PHY-1)
$$\bar n(f,T)=\frac{1}{\exp(hf/k_BT)-1}\quad\text{(use `numpy.expm1`)},\qquad \bar n_{\rm RJ}=\frac{k_BT}{hf}.$$
Constants from `scipy.constants` (CODATA). $T=0\Rightarrow\bar n=0$ exactly (guard the division).

### 3.2 Critical transmissivity (FR-PHY-2)
$\eta^\*=\bar n/(1+\bar n)$; compute the window as $1-\eta^\*=1/(1+\bar n)$ directly (no subtraction).

## 4. Phase-insensitive Gaussian channels (FR-GAU-2,3)

A single-mode phase-insensitive Gaussian channel acts on a mode's $2\times2$ block as $V\mapsto\tau V+yI$ with displacement $d\mapsto\sqrt\tau\,d$. Physicality: $y\ge|1-\tau|$ **[L]**.

### 4.1 Constructors
| Channel | $\tau$ | $y$ |
|---|---|---|
| Identity | 1 | 0 |
| Thermal attenuator $(\eta,\bar n)$ | $\eta$ | $(1-\eta)(2\bar n+1)$ |
| Additive classical noise $N$ (quanta) | 1 | $2N$ |
| Amplifier gain $G\ge1$, idler occupation $\bar n$ | $G$ | $(G-1)(2\bar n+1)$ |

### 4.2 Composition (apply $\mathcal A$ then $\mathcal B$)
$$(\tau,y)=(\tau_B\tau_A,\ \tau_By_A+y_B).$$
Implement as `B @ A` meaning "A first". Unit test associativity on random physical channels.

### 4.3 Attenuator∘attenuator closed form **[L]**
$(\eta_1,\bar n_1)$ then $(\eta_2,\bar n_2)$ equals an attenuator with $\eta=\eta_1\eta_2$ and
$$\bar n_{\rm eff}=\frac{\eta_2(1-\eta_1)\bar n_1+(1-\eta_2)\bar n_2}{1-\eta_1\eta_2}.$$

### 4.4 Output noise photons
For vacuum input the output is thermal with $N_{\rm out}=(\tau+y-1)/2$. For a thermal attenuator, $N_{\rm out}=(1-\eta)\bar n$.

### 4.5 Two-mode application
Applying `PIChannel` to mode B of $V=\begin{psmallmatrix}A&C\\C^\top&B\end{psmallmatrix}$ gives $\begin{psmallmatrix}A&\sqrt\tau C\\\sqrt\tau C^\top&\tau B+yI\end{psmallmatrix}$.

## 5. States and entanglement (FR-GAU-1,4,5,6)

### 5.1 States
- TMSV($r$): $a=\cosh2r$, $c=\sinh2r$, $V=\begin{psmallmatrix}aI&cZ\\cZ&aI\end{psmallmatrix}$, $Z=\mathrm{diag}(1,-1)$.
- TMST($r,\bar n_J$): $a=(2\bar n_J+1)\cosh2r$, $c=(2\bar n_J+1)\sinh2r$ (same block structure). **Note:** $a^2-c^2=(2\bar n_J+1)^2$, not 1. (This corrects V0, IDEA issue D1.)
- Physicality: smallest eigenvalue of the Hermitian matrix $V+i\Omega$ ≥ −1e-12.

### 5.2 PT symplectic eigenvalue
General two-mode CM with blocks $A,B,C$: $\tilde\Delta=\det A+\det B-2\det C$,
$$\tilde\nu_-^2=\frac{\tilde\Delta-\sqrt{\tilde\Delta^2-4\det V}}{2}.$$
Standard form $\begin{psmallmatrix}aI&c'Z\\c'Z&bI\end{psmallmatrix}$ closed form **[V]**:
$$\tilde\nu_-=\frac{(a+b)-\sqrt{(a-b)^2+4c'^2}}{2}.$$
Use the closed form when the CM is in standard form (the case for every service in this project); keep the general one for tests. Evaluate $\tilde\nu_-$ in the stable form $\tilde\nu_-=\dfrac{2(ab-c'^2)}{(a+b)+\sqrt{(a-b)^2+4c'^2}}$ when $(a+b)\gg1$ to avoid cancellation **[D]** (algebraically identical since $\tilde\nu_-\tilde\nu_+=ab-c'^2$).

### 5.3 Measures
$\mathcal N=\max\{0,(1-\tilde\nu_-)/(2\tilde\nu_-)\}$, $E_N=\max\{0,-\log_2\tilde\nu_-\}$, teleportation fidelity (coherent-state input, optimized local operations) $F=1/(1+\tilde\nu_-)$ **[L]**.

### 5.4 Separability conditions
- Standard-form state with $a,b\ge1$: entangled iff $(a-1)(b-1)<c'^2$ **[D][V]**.
- TMSV through $(\tau,y)$ on mode B: $b=\tau a+y$, $c'^2=\tau(a^2-1)$, hence **entangled iff $y<1+\tau$, independent of $r>0$** **[D][V]**. This coincides with the entanglement-breaking criterion of the channel **[L]** (Holevo 2008).
- TMST through $(\tau,y)$: entangled iff $(a-1)(\tau a+y-1)<\tau c^2$ with TMST $a,c$ **[D]**.

## 6. Stochastic channel

### 6.1 Small-scale PDT (FR-CH-1)
Single element ($N=1$) and $N$-element passive MRC share one implementation.

Parameters: $K\ge0$, scale $\bar\eta$ ($=\bar\eta_s$ for $N=1$, $=\bar\eta_\Sigma$ for arrays), $N\ge1$, with $0<\bar\eta\le1$ enforced.

- $\sigma_N^2=\bar\eta/\big(2N(K+1)\big)$; $X=\eta/\sigma_N^2\sim\chi'^2(\nu=2N,\ \lambda=2NK)$ (untruncated surrogate) **[L]** (V0).
- Truncation point $x_{\max}=1/\sigma_N^2=2N(K+1)/\bar\eta$; normalizer $\mathcal Z_{K,N}=F_{\chi'^2}(x_{\max})$.
- pdf on $[0,1]$: $f(\eta)=f_{\chi'^2}(\eta/\sigma_N^2)/(\sigma_N^2\,\mathcal Z)$.
- cdf: $F(x)=F_{\chi'^2}(\min(x,1)/\sigma_N^2)/\mathcal Z$.
- survival (use this near 1): $S(x)=\big[S_{\chi'^2}(x/\sigma_N^2)-S_{\chi'^2}(x_{\max})\big]/\mathcal Z$ for $x<1$; compute in the tail that is smaller and switch accordingly.
- $K=0$: use `scipy.stats.chi2` (or `gamma(N, scale=2σ²)`), not `ncx2` with `nc=0`.
- Sampler: draw $h_i\sim\mathcal{CN}\big(\sqrt{K\bar\eta/(N(K+1))}\,e^{i\phi_{0,i}},\ \bar\eta/(N(K+1))\big)$ for $i=1..N$, set $\eta=\sum|h_i|^2$, **reject** if $\eta>1$ (joint truncation, FR-DIV-4). Acceptance rate ≥ ~0.5 because $x_{\max}$ ≥ the surrogate mean for $\bar\eta\le1$ **[V]**. Phases $\phi_{0,i}=0$ by default (result is phase-independent).
- Moments: numerical integration of the truncated pdf (no closed form required).
- Reference: at $\bar\eta=0.999$, $N=1$: $\mathcal Z_K=0.63249$ ($K=0$), $0.54404$ ($K=10$), $0.51689$ ($K=100$) **[V]**.

### 6.2 Large-scale PDT (FR-CH-2,3)
**Primary: shifted log-gamma [D].** $\Psi=\Psi_0+G$, $\Psi_0\ge0$ deterministic (absorption, $\gamma_{\rm abs}d$), $G\sim\mathrm{Gamma}(k,\theta)$ (shape–scale), $\eta_\ell=e^{-\Psi}\in(0,e^{-\Psi_0}]$.
- From stochastic mean/variance $(\mu_G,\sigma_G^2)$: $k=\mu_G^2/\sigma_G^2$, $\theta=\sigma_G^2/\mu_G$.
- pdf (log-domain): with $u=-\ln\eta-\Psi_0>0$, $\ln f_\ell(\eta)=(k-1)\ln u-u/\theta-\ln\Gamma(k)-k\ln\theta-\ln\eta$; zero for $u\le0$.
- cdf: $P(\eta_\ell\le x)=Q(k,\,u_x/\theta)$ with $u_x=-\ln x-\Psi_0$ (`gammaincc`); equals 1 when $u_x\le0$. Survival: `gammainc`.
- Moments: $\mathbb E[\eta_\ell^m]=e^{-m\Psi_0}(1+m\theta)^{-k}$ **[V]** ($k=4,\theta=0.01$: 0.96098 analytic vs 0.96095 MC).
- Degenerate $\sigma_G\to0$: point mass at $e^{-\Psi_0-\mu_G}$ (handle explicitly).

**Baselines.**
- V0 log-normal: $\Psi\sim\mathcal N(\bar\Psi,\sigma_\Psi^2)$, untruncated; must expose `mass_above_one()` $=\Phi(-\bar\Psi/\sigma_\Psi)$. Reference: 0.158655 for $\sigma_\Psi=\bar\Psi$ **[V]**. Flag as `physical=False`.
- Conditioned log-normal: same, conditioned on $\Psi\ge0$; normalizer $\Phi(\bar\Psi/\sigma_\Psi)$.

### 6.3 Obstacle generator (FR-CH-4, optional) [A]
Number of obstacles $M\sim\mathrm{Poisson}(\lambda_od)$; each log-loss $\psi_j\sim\mathrm{Gamma}(k_o,\theta_o)$, i.i.d.; $G=\sum_j\psi_j$.
Moments: $\mu_G=\lambda_od\,k_o\theta_o$, $\sigma_G^2=\lambda_od\,k_o(k_o+1)\theta_o^2$. Moment-match to §6.2. Expose exact compound-Poisson sampler (note the atom $P(G=0)=e^{-\lambda_od}$, which the Gamma fit cannot represent) for diagnostics. All parameter values need a `source:`; none are defaulted from memory.

### 6.4 Compound PDT (FR-CH-5)
$\eta=\eta_\ell\eta_s$, independent. Work in $\Psi$-space, $\eta_\ell=e^{-\Psi}$:
- cdf: $F(x)=P(\eta_\ell\le x)+\displaystyle\int_{\Psi:\ \eta_\ell>x}F_s\!\big(xe^{\Psi}\big)\,f_\Psi(\Psi)\,d\Psi$.
- survival: $S(x)=\displaystyle\int_{\Psi:\ \eta_\ell>x}S_s\!\big(xe^{\Psi}\big)\,f_\Psi(\Psi)\,d\Psi$. Use whichever of $F,S$ is smaller as the primary computation.
- pdf: $f(x)=\displaystyle\int_{\Psi:\ \eta_\ell>x}e^{\Psi}f_s\!\big(xe^{\Psi}\big)f_\Psi(\Psi)\,d\Psi$.
- Integration: `scipy.integrate.quad`, `epsabs=0`, `epsrel=1e-10`, `limit=500`, breakpoints at $\Psi_0+\mu_G+j\sigma_G$, $j\in\{-6,-3,-1,0,1,3,6\}$ clipped to the domain, and at the kink $\Psi=-\ln x$. If `quad` reports an error estimate above 1e-8 absolute, fall back to `mpmath.quad` and log a warning.
- Quantile $F^{-1}(\epsilon)$: `brentq` on $[0,1]$, `xtol=1e-14`.
- Moments: $\mathbb E[\eta^m]=\mathbb E[\eta_\ell^m]\,\mathbb E[\eta_s^m]$.
- MC sampler: product of independent samples from §6.1 and §6.2.

### 6.5 Link budget (FR-CH-6)
- Apertures: $A_t=G_t\lambda^2/(4\pi)$; $A_r$ given, or $A_r=A_e=3\lambda^2/(8\pi)$ for a bare RAR cell (§7.4), times an optional enhancement factor $G_{\rm enh}\ge1$ for field-confining structures.
- Fresnel product $D_f=A_tA_r/(\lambda d)^2$ (equals the Friis power ratio).
- Collection efficiency models (config `coupling_model`): `"friis"`: $\min(D_f,1)$ (default); `"smooth"`: $1-e^{-D_f}$ (heuristic, labeled as such). Either gives $\bar\eta_s\le1$.
- Absorption: $\Psi_0=\gamma_{\rm abs}d$ with $\gamma_{\rm abs}$ in m⁻¹ (power). Default at 5 GHz: $1.44\times10^{-6}$ m⁻¹ **[L]** (Gonzalez-Raya et al. 2022). Other bands: from ITU-R P.676 values supplied in config with `source:`.
- Reference: $e^{-\gamma d}=\eta^\*$ at $d=555.5$ m (5 GHz, 300 K) **[V]**.

## 7. Receivers (FR-RX)

### 7.1 Two classes
- **Measuring receiver**: produces classical samples. Parameter $N_{\rm amp}\ge\frac12$ (input-referred added quanta, Caves bound for phase-insensitive detection of both quadratures). Optional bandwidth $B$ (Hz).
- **Quantum front end**: a `PIChannel` $(\tau_r,y_r)$ inserted before quantum processing (entanglement service, Holevo bound of the chain).

### 7.2 Ideal receivers
Ideal heterodyne = measuring receiver with $N_{\rm amp}=\frac12$. Ideal homodyne is a separate formula (§8.2), not an $N_{\rm amp}$ value. Quantum-limited amplifier front end: `PIChannel(G, G-1)`.

### 7.3 Electronic receiver (FR-RX-3)
Input: noise temperature $T_n$. Convention flag:
- `"RJ"` (default, radio-engineering): $N_{\rm amp}=k_BT_n/(hf)$.
- `"planck"`: $N_{\rm amp}=\bar n(f,T_n)+\frac12$.
Raise `ValueError` if the result is below $\frac12$ (unphysical for a phase-insensitive receiver).

### 7.4 Rydberg atomic receiver (FR-RX-4) [D]
Short-dipole equivalence: a bare atomic sensor of one field component has $A_e=3\lambda^2/(8\pi)$. Consistency check **[V]**: $\sqrt{\bar n_{\rm th}hf\eta_0/A_e}$ equals the Rayleigh–Jeans single-component BBR field density $\sqrt{8\pi f^2k_BT/(3\epsilon_0c^3)}$ (0.603 nV/cm/√Hz at 5 GHz, 300 K).

Mapping (input $S_E$ in V m⁻¹ Hz⁻¹ᐟ², **intrinsic**, i.e. BBR excluded):
$$N_{\rm amp}=\max\!\Big\{\tfrac12,\ \frac{S_E^2A_e}{\eta_0hf}\Big\}.$$
- If the supplied $S_E$ was measured including in-band BBR, subtract in power first: $S_{E,\rm int}^2=S_{E,\rm meas}^2-S_{E,\rm BBR}^2$ (config flag `includes_bbr`). Error if negative.
- $G_{\rm enh}$ does **not** change $N_{\rm amp}$; it enters only through $A_r$ in §6.5. A lossy enhancing structure is modeled as an extra attenuator $(\eta_{\rm struct},\bar n_{\rm th})$ composed before the receiver.
- Equivalent noise temperature (RJ): $T_n=N_{\rm amp}hf/k_B$.
- Reference: $S_E=10$ nV/cm/√Hz $=10^{-6}$ V m⁻¹ Hz⁻¹ᐟ² at 5 GHz → $N_{\rm amp}=3.438\times10^5$, $T_n=8.25\times10^4$ K **[V]**.
- BBR-equivalent sensitivity $S_{E,\rm BBR}(f,T)=\sqrt{\bar n(f,T)hf\eta_0/A_e}$: 0.603 / 3.614 / 12.01 / 35.75 nV/cm/√Hz at 5 / 30 / 100 / 300 GHz, 300 K **[V]**.
- Optional fields: `bandwidth_hz`, `e_sat_v_per_m` (linear-range limit). If the mean received field $E_{\rm rx}=\sqrt{\eta_0P_{\rm rx}/A_r}$ exceeds `e_sat`, set a `linear_regime=False` flag on results (do not alter numbers).
- Model-derived alternative: `RydbergReceiver.from_physics(model_params)` takes $S_E$, $B$ and $E_{\rm sat}$ from §14 instead of literature inputs.
- **[A]** Cross-check this aperture convention against Santamaría Botello et al. (arXiv 2209.00908) before quoting absolute numbers in the paper (IDEA R5). Record the outcome in `docs/RESULTS.md`.

### 7.5 Coherent transducer front end (FR-RX-5)
`PIChannel` = attenuator $(\eta_t,\bar n_t)$ followed by optional additive noise $N_t$: $\tau_r=\eta_t$, $y_r=(1-\eta_t)(2\bar n_t+1)+2N_t$. Parameter values: config only, with `source:` **[A]**.

## 8. Services

### 8.1 Entanglement service (FR-SVC-E1,2)
Chain per realization: fading attenuator $(\eta,\bar n_{\rm th})$ then front end $(\tau_r,y_r)$:
$$\tau=\tau_r\eta,\qquad y=\tau_r(1-\eta)(2\bar n_{\rm th}+1)+y_r.$$
**Receiver-referred threshold [D][V]** (TMSV, any $r>0$): entangled iff $\eta>\eta^\*_{\rm rx}$ with
$$\eta^\*_{\rm rx}=\frac{\tau_r(2\bar n_{\rm th}+1)+y_r-1}{2\tau_r(\bar n_{\rm th}+1)}.$$
Special cases: identity front end → $\eta^\*$; additive noise $N$ → $(\bar n_{\rm th}+N)/(\bar n_{\rm th}+1)$; window exists iff $\eta^\*_{\rm rx}<1$, i.e. for additive noise iff $N<1$. Reference at 5 GHz: $N=\frac12\Rightarrow0.99960$, $N=1\Rightarrow1$ (no window); transducer $\eta_t=0.6,\ \bar n_t=0.01,\ N_t=0.01\Rightarrow0.999219$ **[V]** (formula matches a direct root of $\tilde\nu_-(\eta)=1$ to machine precision for all four cases tested). V0's warm-combining-network threshold is the attenuator special case (regression test).
For TMST inputs, solve §5.4 numerically per realization (`brentq` in $\eta$).

Metrics over the compound PDT $f(\eta)$:
- $P_{\rm out}=F(\eta^\*_{\rm rx})$ (§6.4).
- $\mathbb E[E_N]=\int_{\eta^\*_{\rm rx}}^1E_N(\eta)f(\eta)\,d\eta$, computed as a nested integral in $\Psi$-space, cross-checked by MC.
- $\mathbb E[\tilde\nu_-]$, $\mathbb E[F]$ likewise.
- V0 limiting cases (pure LOS, Rayleigh, no shadowing, deterministic) as regression tests, rewritten for the log-gamma large-scale law.

### 8.2 Classical service (FR-SVC-C1..3)
Coherent-state Gaussian encoding with mean photon number $N_s$ per mode. Per realization with total chain $(\tau,y)$ and $N_{\rm out}=(\tau+y-1)/2$:
- Holevo capacity **[L]** (Giovannetti, García-Patrón, Cerf, Holevo, Nat. Photon. 2014): $C_\chi=g(\tau N_s+N_{\rm out})-g(N_{\rm out})$, $g(x)=(x+1)\log_2(x+1)-x\log_2x$, $g(0)=0$ (use `scipy.special.xlogy`/log1p forms).
- Heterodyne: $C_{\rm het}=\log_2\!\big(1+\tau N_s/(N_{\rm out}+1)\big)$.
- Homodyne: $C_{\rm hom}=\tfrac12\log_2\!\big(1+4\tau N_s/(2N_{\rm out}+1)\big)$.
- Measuring receiver after the fading channel only ($\tau=\eta$, $N_{\rm out}=(1-\eta)\bar n_{\rm th}$): $C_{\rm rx}=\log_2\!\big(1+\eta N_s/(N_{\rm out}+\tfrac12+N_{\rm amp})\big)$. Check: $N_{\rm amp}=\frac12\Rightarrow C_{\rm het}$.
- All are increasing in $\eta$ (assert in tests); hence $\epsilon$-outage capacity $C_\epsilon=C\big(F^{-1}(\epsilon)\big)$ and rate outage $P(C<R)=F\big(C^{-1}(R)\big)$ with $C^{-1}$ by `brentq`.
- Ergodic capacity $\bar C=\mathbb E_\eta[C(\eta)]$ by nested quadrature, MC cross-check.
- Throughput $R_b=B\cdot C$ (bit/s). TX energy helper: $N_s=P_t/(hfB)$.

### 8.3 Crossover sensitivity (FR-SVC-C4) [D]
Noise-penalty definition: receiver noise relative to ideal heterodyne at a reference transmissivity $\eta_{\rm ref}$ (default: compound mean),
$$\frac{N_{\rm out}+\tfrac12+N_{\rm amp}}{N_{\rm out}+1}\le10^{\Delta/10}\ \Longrightarrow\ N_{\rm amp}^{\rm req}=(10^{\Delta/10}-1)(N_{\rm out}+1)+\tfrac12,$$
$$S_E^{\rm req}=\sqrt{N_{\rm amp}^{\rm req}\,\eta_0hf/A_e}.$$
If $N_{\rm amp}^{\rm req}<\frac12$ is required, return `NaN` with reason. Capacity-based variant: smallest $N_{\rm amp}$ with $\bar C_{\rm rx}\ge(1-\delta)\bar C_{\rm het}$, solved by `brentq` on $\log N_{\rm amp}$. In the RJ regime $S_{E,\rm BBR}\propto f$; E6 must show this slope as a sanity check.

## 9. Diversity (FR-DIV) — ported from V0, keep exact forms

With $\sigma_N^2=\bar\eta_\Sigma/(2N(K+1))$:
- Array PDT and cdf: §6.1 with $N>1$; array outage $P_{{\rm out},N}=\int F_N(\eta^\*/\eta_\ell)f_\ell\,d\eta_\ell$ (same nested scheme as §6.4).
- Selection combining (valid for $x\ge\eta^\*>\frac12$): $P(\eta_{\rm SC}>x)=\dfrac{N}{\mathcal Z_{K,N}}\displaystyle\int_x^1f_{s}(t)\,G_{N-1}(1-t)\,dt$, where $f_s$ is the **untruncated** single-element pdf with scale $\bar\eta_\Sigma/N$, and $G_{N-1}(y)=F_{\chi'^2}\big(y/\sigma_N^2;\ \nu=2(N-1),\ \lambda=2(N-1)K\big)$.
- Chernoff bound: $\dfrac{N}{\mathcal Z_{K,N}}\inf_{0<s<K+1}\dfrac{K+1}{K+1-s}\exp\!\Big(\dfrac{Ks}{K+1-s}-\dfrac{sNx}{\bar\eta_\Sigma}\Big)$ (`minimize_scalar`, bounded).
- Cramér rate: $u_a=\dfrac{1+\sqrt{1+4aK(K+1)}}{2a}$, $\mathcal R_K(a)=(K+1-u_a)\big(a-K/u_a\big)+\ln\dfrac{u_a}{K+1}$; $\mathcal R_0(a)=a-1-\ln a$. Test against a numerical Legendre transform of $\ln M_K(s)=\ln\frac{K+1}{K+1-s}+\frac{Ks}{K+1-s}$.
- Exponent bounds: $P\le e^{-N\mathcal R_K(a)}/\mathcal Z_{K,N}$, $a=\eta^\*/(\eta_\ell\bar\eta_\Sigma)$.
- Design law: $D_{N,K}=N(K+1)^2/(1+2K)\ge2\ln\!\big(1/(\epsilon\mathcal Z_{K,N})\big)/(1-a)^2$.
- $N_{\rm req}$: smallest integer $N$ with $F_N(\eta^\*)\le\epsilon$ (exponential search then integer bisection), using the exact cdf.

## 10. Validation plan (maps to FSD NFR-1)

| Layer | Tests |
|---|---|
| Unit | every function: shapes, dtypes, domain errors, edge values ($\eta\in\{0,1\}$, $K=0$, $N=1$, $T=0$) |
| Identity | §5.2 general vs closed form; §5.4 condition vs $\tilde\nu_-$ grid; §4.2 associativity; §4.3 closed form; §8.1 threshold vs direct root of $\tilde\nu_-(\eta)=1$ |
| Limits | small-scale $K=0$ and $K\to\infty$; large-scale $\sigma_G\to0$; compound reduces to each factor; V0 cases 1–4 |
| Cross-method | analytic/quadrature vs MC (10⁶, fixed seed) inside 3σ binomial or CLT bands; quadrature vs `mpmath` at 3 points per function |
| Property (`hypothesis`) | pdfs ≥ 0 and integrate to 1; cdfs monotone in $[0,1]$; capacities monotone in $\eta$ and $N_s$; $C_\chi\ge C_{\rm het},C_{\rm hom}$ |
| Regression | §11 reference values; figure CSVs hashed after Abdurrahman approves them |
| Receiver physics | T-LIN-1..11 (§14.11): weak-probe vs full Lindblad, analytic limits, optional QuTiP cross-check |

Tolerances live in `tests/tolerances.py`. Changing a tolerance requires an entry in `QUESTIONS.md`.

## 11. Reference values

Verified while writing this spec (Python, `scipy.constants`), $T=300$ K:

| $f$ (GHz) | $\bar n_{\rm th}$ | $\eta^\*$ | $1-\eta^\*$ | $S_{E,\rm BBR}$ (nV/cm/√Hz) |
|---|---|---|---|---|
| 5 | 1249.697 | 0.99920 | 7.996e-4 | 0.603 |
| 30 | 207.867 | 0.99521 | 4.788e-3 | 3.614 |
| 100 | 62.011 | 0.98413 | 1.587e-2 | 12.012 |
| 300 | 20.341 | 0.95314 | 4.686e-2 | 35.748 |

Other verified values: absorption-only range 555.5 m (5 GHz); $\eta^\*_{\rm rx}(N=\frac12)=0.99960$; RAR $N_{\rm amp}=3.438\times10^5$ for 10 nV/cm/√Hz at 5 GHz; V0 log-normal mass above 1 = 0.158655 ($\sigma_\Psi=\bar\Psi$); $\mathcal Z_K$ values in §6.1; log-gamma mean 0.96098 ($k=4$, $\theta=0.01$).

**Unverified (from V0, treat as targets to confirm, not ground truth):** elements $N$ for $F_N(\eta^\*)\le10^{-3}$ under passive MRC, $\eta_\ell=1$, $\bar\eta_\Sigma=\eta^\*+0.9(1-\eta^\*)$:

| $f$ (GHz) | $K=0$ | $K=10$ | $K=100$ |
|---|---|---|---|
| 5 | 2.0e7 | 3.5e6 | 3.9e5 |
| 30 | 5.6e5 | 9.7e4 | 1.1e4 |
| 100 | 50 243 | 8 737 | 993 |
| 300 | 5 630 | 983 | 112 |

(A rough Gaussian check of the 300 GHz, $K=0$ entry is consistent; the rest is unchecked.)

## 12. Experiments, config, CLI

Config schema (pydantic) example `configs/e3.yaml`:
```yaml
experiment: e3_entanglement_outage
seed: 20261028
environment: {T_env_k: 300.0}
bands_ghz: [5, 30, 100, 300]
distance_m: {start: 10, stop: 1000, num: 60, scale: log}
small_scale: {K: [0, 10, 100]}
large_scale:
  model: loggamma            # loggamma | lognormal_v0 | lognormal_conditioned
  psi0_from_absorption: true
  gamma_abs_per_m: {5: 1.44e-6}            # per band, source required
  gamma_abs_source: "Gonzalez-Raya et al., PRApplied 18, 044002 (2022)"
  stochastic: {mean: 1.0e-3, std: 5.0e-4}   # [A] placeholder, Abdurrahman to set
link_budget: {coupling_model: friis, G_t_dBi: 30, A_r: dipole, G_enh: 1.0}
source_state: {type: tmsv, r: 1.0}
front_end: {type: identity}
mc: {samples: 1000000}
outputs: {figure: true, csv: true, latex_table: false}
```
E9 adds a `lindblad:` block (all values `# [A]` until Abdurrahman fills them with sources):
```yaml
lindblad:
  method: auto               # auto | weak_probe | full   (§14.6)
  weak_probe_tol: 0.01
  species: Rb87
  n_levels: 4
  T_vapor_k: 300.0
  n0_per_m3: null            # [A] or from a vapor-pressure helper with source
  cell_length_m: null        # [A]
  probe: {wavelength_m: 780.241e-9, power_w: null, waist_m: null, d12_cm: null}
  coupling: {wavelength_m: null, power_w: null, waist_m: null, d23_cm: null}
  mw: {d34_cm: null, e_lo_v_per_m: null}
  decays: []                 # list of {from, to, rate_per_s, source}
  gamma_t_per_s: null        # [A] transit rate
  detector: {eta_d: null, nep_w_per_rthz: 0.0}
  doppler: {n_sigma: 5, n_v_start: 4097, rtol: 1.0e-5, n_v_max: 1048577}
  sweep: {omega_c_over_2pi_mhz: [], e_lo_v_per_m: []}
```
Every numeric literature value carries `source:`; placeholders carry `# [A]` and are listed in `QUESTIONS.md`.

CLI: `qmw run configs/e3.yaml [--force]`, `qmw list`, `make test`, `make lint`, `make figs`, `make paper-figs`.
Outputs: `results/<experiment>/<config_hash>/{fig.pdf, fig.png, data.csv, meta.json}`.

## 13. Numerical rules (binding)

1. Never compute $1-F$ when $F\approx1$; use survival functions (`sf`, `gammainc`/`gammaincc` complements).
2. Bessel terms: `scipy.special.ive` (scaled) in log-domain pdfs; prefer `scipy.stats.ncx2` methods over hand-written Marcum-Q.
3. Work with $1-\eta$ (`one_minus_eta`) as a primary variable wherever $\eta\to1$ matters (thresholds, windows of width ~1e-3).
4. `float64` everywhere; no `float32`.
5. Root finding: `brentq` with explicit brackets and `xtol≤1e-14`; assert sign change, never fall back silently.
6. Quadrature failures raise or warn; never return an unchecked result.
7. MC: vectorized, chunked (≤10⁶ per chunk), explicit `Generator`, report standard error with every MC estimate.

## 14. Rydberg receiver physics model (pure Python)

**Purpose.** Replace the phenomenological $S_E\to N_{\rm amp}$ input of §7.4 with a physics model, and map where the linear (Gaussian) receiver model is valid (IDEA risk R4, experiment E9). Everything here is NumPy/SciPy; no compiled extension is needed for this paper (cost budget in §14.12).

### 14.1 Physical model and scope
Room-temperature alkali vapor (default ⁸⁷Rb), ladder scheme, $N\in\{3,4\}$ levels:
$|1\rangle$ ground ($5S_{1/2}$), $|2\rangle$ intermediate ($5P_{3/2}$, probe ≈780.24 nm), $|3\rangle$ Rydberg (coupling ≈480 nm), $|4\rangle$ adjacent Rydberg (MW transition $|3\rangle\leftrightarrow|4\rangle$, $N=4$ only).

**In scope:** RWA Hamiltonian, Lindblad decay, transit relaxation, 1D Doppler averaging (counter-propagating probe/coupling), weak-probe Beer–Lambert readout, steady state, small-signal transfer function, photon-shot-noise-limited sensitivity, linear range.
**Out of scope (state in paper):** hyperfine/Zeeman substructure, optical propagation with $\Omega_p(z)$, Rydberg–Rydberg interactions, BBR-induced Rydberg transitions beyond a phenomenological decay rate, atom projection noise (optional extension, §14.8), time-domain waveform simulation.

### 14.2 Units (binding inside the physics module)
Time in µs, angular frequencies in rad/µs (i.e. $2\pi\times$MHz), wavevectors in rad/m, velocities in m/s; Doppler shift $kv\cdot10^{-6}$ rad/µs. SI conversions happen only in `model.py` (§14.10).

### 14.3 Hamiltonian and dissipators [V]
With $\hbar=1$ and level $|j\rangle$ indexed from 1:
$$H=-\Delta_p|2\rangle\langle2|-(\Delta_p+\Delta_c)|3\rangle\langle3|-(\Delta_p+\Delta_c+\Delta_m)|4\rangle\langle4|+\tfrac12\big(\Omega_p|2\rangle\langle1|+\Omega_c|3\rangle\langle2|+\Omega_m|4\rangle\langle3|+\text{h.c.}\big).$$
Rabi frequencies real and $\ge0$. Master equation $\dot\rho=-i[H,\rho]+\sum_kD[L_k]\rho$, $D[L]\rho=L\rho L^\dagger-\tfrac12\{L^\dagger L,\rho\}$, with default collapse operators:
- $\sqrt{\Gamma_2}\,|1\rangle\langle2|$, $\Gamma_2=2\pi\times6.065$ MHz **[L]** (Steck, *Rubidium 87 D Line Data*);
- $\sqrt{\Gamma_3}\,|2\rangle\langle3|$ and $\sqrt{\Gamma_4}\,|2\rangle\langle4|$, Rydberg rates from config (e.g. via ARC) **[A]**; the decay target of each level is configurable as a list of `(from, to, rate)`;
- transit relaxation $\sqrt{\gamma_t}\,|1\rangle\langle k|$ for all $k$ (exactly the reset map $\dot\rho=\gamma_t(|1\rangle\langle1|-\rho)$) **[D][V]**;
- optional laser dephasing $\sqrt{2\gamma_{\phi,k}}\,|k\rangle\langle k|$.

Coherence decay rates (used by the weak-probe model and tests): $\gamma_{j1}=\tfrac12\Gamma_j^{\rm out}+\gamma_t+\gamma_{\phi,j}$, where $\Gamma_j^{\rm out}$ is the total decay rate out of level $j$. Defaults: $\gamma_{21}=\Gamma_2/2+\gamma_t$, $\gamma_{31}=\Gamma_3/2+\gamma_t$, $\gamma_{41}=\Gamma_4/2+\gamma_t$.

### 14.4 Weak-probe model (primary engine) [D][V]
For $\Omega_p\ll\Gamma_2$, $\rho_{11}\approx1$ and only the coherences $x=(\rho_{21},\rho_{31},\rho_{41})^\top$ matter. With per-velocity detunings $d_2=\Delta_p-k_pv$, $d_3=d_2+\Delta_c+k_cv$, $d_4=d_3+\Delta_m$:
$$M=\begin{pmatrix}d_2+i\gamma_{21}&-\Omega_c/2&0\\-\Omega_c/2&d_3+i\gamma_{31}&-\Omega_m/2\\0&-\Omega_m/2&d_4+i\gamma_{41}\end{pmatrix},\qquad s=\begin{pmatrix}\Omega_p/2\\0\\0\end{pmatrix}.$$
- **Steady state:** $Mx_0=s$, i.e. the continued fraction
$$\rho_{21}=\cfrac{\Omega_p/2}{d_2+i\gamma_{21}-\cfrac{\Omega_c^2/4}{d_3+i\gamma_{31}-\cfrac{\Omega_m^2/4}{d_4+i\gamma_{41}}}}.$$
  $N=3$: drop the last row/column ($\Omega_m=0$).
- **Dynamics:** $\dot x=i(Mx-s)$ (stable: eigenvalues of $iM$ have negative real part).
- **Small-signal transfer function in $\Omega_m$:** with $M_1=\partial M/\partial\Omega_m$ (entries $(2,3),(3,2)=-\tfrac12$),
$$H(i\omega)=e_1^\top\,(i\omega I-iM)^{-1}\,(iM_1x_0),\qquad H(0)=-e_1^\top M^{-1}M_1x_0.$$
- Vectorize over velocity classes (closed-form tridiagonal solve or the continued fraction directly); no Python loops over $v$.
- **Verified:** against the full Lindblad model (§14.5) at $\Omega_p/2\pi=0.01$ MHz, $\Omega_c/2\pi=5$, $\Omega_m/2\pi=20$ MHz: steady state and $H(i\omega)$ for $\omega/2\pi\in\{0,0.3,1,3\}$ MHz agree to ≤ 8e-5 relative at $v=0$ and ≤ 1e-8 at $v=100$ m/s.
- **Validity (measured, Doppler-averaged, same $\Omega_c,\Omega_m$):** relative error in $\langle\rho_{21}\rangle$ of 4e-6 at $\Omega_p/2\pi=0.01$ MHz, 3.5% at 1 MHz, 65% at 5 MHz. The error grows as $\Omega_p^2$; the weak-probe model must not be used for sensitivity optimization at strong probe without the §14.5 check.

### 14.5 Full Lindblad model (strong-probe check)
Row-major vectorization $\mathrm{vec}(AXB)=(A\otimes B^\top)\mathrm{vec}(X)$:
$$\mathcal L=-i(H\otimes I-I\otimes H^\top)+\sum_k\Big(L_k\otimes L_k^\*-\tfrac12L_k^\dagger L_k\otimes I-\tfrac12I\otimes(L_k^\dagger L_k)^\top\Big),$$
size $N^2\times N^2$ (9 or 16).
- **Affine in velocity:** $\mathcal L(v)=\mathcal A+v\,\mathcal B$, where $\mathcal B$ comes from $H_v=10^{-6}\big(k_p|2\rangle\langle2|+(k_p-k_c)(|3\rangle\langle3|+|4\rangle\langle4|)\big)$. Build $\mathcal A,\mathcal B$ once per parameter point and form the batch by broadcasting.
- **Steady state:** replace row 0 of $\mathcal L$ (and of $\mathcal B$) by $\mathrm{vec}(I)^\top$ (resp. 0), right-hand side $e_0$, batched `numpy.linalg.solve` over $v$.
- **Transfer function:** $\mathcal L=\mathcal L_0+\delta\Omega_m\mathcal L_1$; $\delta\rho(s)=(sI-\mathcal L_0)^{-1}\mathcal L_1\rho_0$; at $s=0$ use the bordered system (trace row, zero right-hand side entry).
- **Do not implement** the generalized-Schur (QZ) per-velocity trick: it is accurate (max relative error 2e-14) but slower in NumPy (≈8 µs vs ≈5 µs per class) because the triangular solve runs as a Python loop **[V]**.
- Use: (i) the oracle for weak-probe tests; (ii) strong-probe points in E9 (sensitivity vs $\Omega_p$); (iii) the automatic validity check (§14.6).

### 14.6 Doppler averaging and method selection
- Velocity $v$ along the probe; counter-propagating coupling; MW Doppler neglected. Maxwell–Boltzmann weights, $\sigma_v=\sqrt{k_BT_{\rm vap}/m}$ (≈169 m/s for ⁸⁷Rb at 300 K).
- Grid: uniform on $[-5\sigma_v,5\sigma_v]$, normalized weights; start $N_v=2^{12}+1$, double until the relative change of $\langle\rho_{21}\rangle$ is below `rtol_doppler` (default 1e-5); record final $N_v$. Features are narrow in $v$ because of the residual shift $(k_c-k_p)v$: 4001 vs 8001 points still differ by 0.25% in a test case **[V]**.
- `method` per config: `weak_probe` | `full` | `auto` (default).
  - `auto`: use `weak_probe` for the sweep, then evaluate `full` at ≥ 5 checkpoints (the largest-$\Omega_p$ point plus 4 seeded random points). Record the max relative discrepancy in run metadata; if it exceeds `weak_probe_tol` (default 1%), recompute the affected sweep with `full` and log a warning.

### 14.7 Readout and small-signal response
- Probe intensity absorption coefficient **[D][V]**: $\alpha=\dfrac{2k_pN_0d_{12}^2}{\epsilon_0\hbar\,\Omega_p}\big(-\mathrm{Im}\langle\rho_{21}\rangle\big)$, $\rho_{21}=\langle2|\rho|1\rangle$. Check: with $\gamma_t=0$, the weak-probe two-level resonance gives $\alpha/N_0=3\lambda_p^2/2\pi$ when $d_{12}^2=3\pi\epsilon_0\hbar\Gamma_2/k_p^3$ (closed two-level cycling transition).
- Transmission (weak probe, thin uniform cell): $P_{\rm out}=P_{\rm in}e^{-\alpha L}$.
- Superheterodyne: LO field $E_{\rm LO}$ plus signal $E_{\rm sig}e^{i\delta t}$ gives $\Omega_m(t)\approx\Omega_{\rm LO}+\Omega_{\rm sig}\cos(\delta t+\phi)$, $\Omega=d_{34}E/\hbar$ (peak fields).
- Slope: $\kappa\equiv\partial P_{\rm out}/\partial E_{\rm sig}=-P_{\rm out}L\,\dfrac{\partial\alpha}{\partial\Omega_m}\dfrac{d_{34}}{\hbar}$, with $\partial\alpha/\partial\Omega_m$ from $H(0)$ (§14.4 or §14.5).
- Instantaneous bandwidth $B_{3\rm dB}$: highest $\omega/2\pi$ with $|H(i\omega)|\ge|H(0)|/\sqrt2$, on a log grid then refined by `brentq`. $|H|$ can peak above $|H(0)|$ near the EIT dynamics (seen at ~1 MHz in a test case **[V]**); report the peaking value too.

### 14.8 Sensitivity and $N_{\rm amp}$ [D]
Single-sided optical-power noise PSD at the detector: $S_P=2h\nu_pP_{\rm out}/\eta_d+\mathrm{NEP}^2$. RMS noise-equivalent field density, consistent with the §7.4 convention (input-referred noise power density $S_E^2A_e/\eta_0$):
$$S_E=\frac{\sqrt{S_P}}{|\kappa|}\quad(\text{signal referred to peak }E_{\rm sig},\ \text{noise to rms}).$$
Then $N_{\rm amp}$ from §7.4. **[A]** Cross-check the peak/rms bookkeeping against one published PSN-limited sensitivity (e.g. Jing et al., Nat. Phys. 2020, or Cox et al., PRL 2018) and record it in `docs/RESULTS.md`. Atom projection noise: optional term added in quadrature with a formula supplied by Abdurrahman; not implemented until then.

### 14.9 Linear range [D]
Quasi-static (valid only if $\delta\ll B_{3\rm dB}$; assert): first-harmonic amplitude $A_1(\Omega_{\rm sig})=\frac1\pi\int_0^{2\pi}P_{\rm ss}(\Omega_{\rm LO}+\Omega_{\rm sig}\cos\theta)\cos\theta\,d\theta$ (Gauss–Legendre, 64 nodes). Compression $c=A_1/(\kappa_\Omega\,\Omega_{\rm sig})$ with $\kappa_\Omega=\partial P_{\rm out}/\partial\Omega_m$ at $\Omega_{\rm LO}$. $E_{1\rm dB}$: signal field where $c=10^{-1/20}$ (`brentq` on $\log\Omega_{\rm sig}$). Used as `e_sat_v_per_m` in §7.4.

### 14.10 Module layout
```
src/qmwlink/receivers/rydberg_physics/
├── scheme.py      # level scheme, decay channels, coherence rates      §14.3
├── weak_probe.py  # M, s, steady state, dynamics, transfer function    §14.4
├── lindblad.py    # Liouvillian A + vB, batched steady state, transfer §14.5
├── doppler.py     # grids, weights, convergence loop                    §14.6
├── response.py    # Doppler-averaged H(iω), bandwidth, peaking          §14.7
├── readout.py     # alpha, P_out, kappa, S_E, E_1dB                     §14.7–14.9
└── model.py       # LindbladRARModel: SI in -> (kappa, S_E, N_amp, B_3dB, E_1dB, method, discrepancy)
```
- `model.py` converts SI inputs (probe/coupling power and waist → Rabi via $E=\sqrt{2I/c\epsilon_0}$, $I=2P/\pi w_0^2$; $N_0$; $d_{12}$, $d_{34}$; $L$; $T_{\rm vap}$; $E_{\rm LO}$) to core units and back.
- Parameter values (dipoles, Rydberg lifetimes, $N_0$) come from config with `source:`; ARC (Alkali Rydberg Calculator) may be used offline to produce them **[A]**.
- Optional independent cross-check with QuTiP (`qutip.steadystate`) in tests only; skipped if not installed.

### 14.11 Validation
| ID | Test | Reference |
|---|---|---|
| T-LIN-1 | Cold 3-level weak-probe: `full` vs the continued fraction ($\Omega_p/\Gamma_2\le10^{-2}$) | rtol 1e-4 **[V]** (agrees to ~1e-5 rel.) |
| T-LIN-2 | `full`: trace 1, Hermitian, eigenvalues ≥ −1e-12 for random valid params | — |
| T-LIN-3 | Two-level limit ($\Omega_c=0$): $\rho_{21}=(\Omega_p/2)/(\Delta_p+i\gamma_{21})$ in both methods | rtol 1e-4 |
| T-LIN-4 | Cold 4-level AT splitting in probe scan $\approx\Omega_m$ (for $\Omega_m\gg$ EIT width) | within 3% at $\Omega_m/2\pi=4$ MHz **[V]** (3.93 MHz) |
| T-LIN-5 | Doppler-averaged (300 K) probe-scan AT splitting $\approx(\lambda_c/\lambda_p)\Omega_m$, two most prominent absorption minima (`scipy.signal.find_peaks` with prominence) | Case $\Omega_p,\Omega_c,\Omega_m,\gamma_t=2\pi\times(0.05,5,20,0.1)$ MHz, $\lambda_c=480$ nm: 12.4 MHz vs 12.30 predicted (0.8%), tol 2% **[V]**. Asymptotic formula: 5.7% off at $\Omega_m/2\pi=10$ MHz, so do not test below $\Omega_m\approx4\Omega_c$. |
| T-LIN-6 | $H(0)$ equals central finite difference of the steady state in $\Omega_m$, both methods | rtol 1e-6 **[V]** |
| T-LIN-7 | Weak-probe vs full transfer function at $\Omega_p/\Gamma_2\le2\times10^{-3}$, $\omega/2\pi\in\{0,0.3,1,3\}$ MHz | rtol 2e-4 **[V]** (≤ 8e-5 measured) |
| T-LIN-8 | Doppler convergence loop terminates and is monotone in tolerance | — |
| T-LIN-9 | `full` vs QuTiP on 20 random parameter sets (optional) | rtol 1e-8 |
| T-LIN-10 | Two-level resonant cross-section $3\lambda^2/2\pi$ from §14.7 ($\gamma_t=0$, weak probe) | rtol 1e-4 |
| T-LIN-11 | `auto` method: discrepancy above `weak_probe_tol` triggers recomputation with `full` | synthetic strong-probe case |

### 14.12 Cost budget (measured on a 2-core cloud machine, single-threaded BLAS) [V]
| Method | Cost per velocity class |
|---|---|
| Weak-probe closed form, vectorized | ≈ 20–25 ns |
| Full Lindblad, batched `solve` with $\mathcal A+v\mathcal B$ | ≈ 5–10 µs |
| Full Lindblad, naive (Python loop building $\mathcal L$) | ≈ 45 µs (**do not do this**) |

Implications for E9 with $N_v\approx3\times10^4$: a weak-probe sweep of $10^4$ parameter points costs ≈ 10 s; a full-model curve of 20 $\Omega_p$ values (steady state + derivative) costs ≈ 20 s; the `auto` checkpoints add seconds. E9 therefore fits NFR-3 (< 5 min per figure). No compiled code (Numba, Cython, C++) is needed; adding any requires a `QUESTIONS.md` entry with a profile showing the hotspot.