# TSD — Technical Specification

**Project:** `qmwlink` · **Version:** 0.4.3 (2026-11-10; pairs with FSD v0.6.3; §14 is a pure-Python receiver physics model, no compiled code) · **Owner:** Abdurrahman
**Reads with:** `docs/IDEA.md` (why), `docs/FSD.md` (what). This document is the single source of truth for formulas, algorithms, numerics, code identifiers and code layout. If code and TSD disagree, the code is wrong or the TSD needs a reviewed change; never silently diverge. FSD and TSD versions are bumped together.

Status tags used below:
- **[V]** verified numerically while writing this spec.
- **[D]** derived in this spec, needs a proof in the paper.
- **[L]** taken from literature; the implementer must not alter it.
- **[A]** assumption that Abdurrahman must confirm.

---

## 1. Architecture

```
qmwlink/
├── agent.md                 # instructions for the coding agent
├── QUESTIONS.md             # open questions raised during implementation
├── pyproject.toml           # deps pinned, ruff, mypy, pytest config
├── Makefile                 # test, lint, figs, paper-figs, exp, clean
├── docs/  IDEA.md  FSD.md  TSD.md  RESULTS.md
├── configs/  defaults.yaml  literature_rar.yaml  e1.yaml … e9.yaml
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
│   ├── numerics/  special.py  integrate.py  sampling.py  mcagree.py   §13, §10.1
│   ├── experiments/  schema.py  runner.py  registry.py  e1.py … e9.py   §12
│   ├── io/  export.py  latex.py  literature.py
│   └── plotting/  style.py
├── scripts/  derive_inputs.py  # regenerates the §16.1–16.2 ITU-derived inputs (needs optional `itur`)
├── tests/  unit/  validation/  regression/  tolerances.py
└── results/                 # gitignored; figures, CSV, metadata
```
The whole project is pure Python and is built by the coding agent.

**Dependencies:** Python ≥ 3.11; `numpy`, `scipy`, `mpmath` (reference values in tests only), `matplotlib`, `pydantic` (config schema), `pyyaml`, `pytest`, `hypothesis`, `ruff`, `mypy`. Optional test oracle: `qutip`. Optional, scripts only (never imported by `src/`): `itur` 0.4.0 for `scripts/derive_inputs.py` (§16). No compiled extensions (Numba, Cython, C/C++); §14.12 shows they are not needed. No other dependencies without an entry in `QUESTIONS.md`.

**Style:** pure functions on NumPy arrays where possible, vectorized over the transmissivity axis; small frozen dataclasses for parameter bundles; no hidden global state.

## 2. Notation and conventions (binding; FSD §4.1 mirrors this table)

| Symbol | Meaning | Code name |
|---|---|---|
| $f$, $\lambda=c/f$ | carrier frequency, wavelength | `f_hz`, `lam_m` |
| $T$ | environment temperature | `T_env_k` |
| $\bar n_{\rm th}$ | environment occupation $\bar n(f,T)$ | `n_th` |
| $\bar n_{\rm JPA}$ | source (JPA) thermal occupation | `n_jpa` |
| $\eta,\ \eta_s,\ \eta_\ell$ | total, small-scale, large-scale transmissivity | `eta`, `eta_s`, `eta_l` |
| $\bar\eta_s$ | small-scale scale parameter (link-budget collection efficiency) | `eta_s_bar` |
| $\bar\eta_\Sigma$ | array total collection efficiency, $N\bar\eta_s$ | `eta_sigma` |
| $K$ | Rician factor | `K` |
| $N$ | number of receive elements | `n_elements` |
| $\Psi=\Psi_0+G$ | path-integrated log-loss, $\eta_\ell=e^{-\Psi}$ | `psi`, `psi0`, `g` |
| $(\tau,y)$ | phase-insensitive channel $V\mapsto\tau V+yI$ | `PIChannel(tau, y)` |
| $N_s$ | mean signal photons per mode at TX | `n_s` |
| $N_{\rm add}$ | additive noise quanta of a quantum front end, $y=2N_{\rm add}$ | `n_add` |
| $N_{\rm amp}$ | input-referred added noise quanta of a measuring receiver, $\ge\frac12$ | `n_amp` |
| $N_{\rm img}$ | thermal occupation of the image-band mode at a DSB receiver input (§7.1) | `n_img` |
| $N_{\rm lev}$ | number of atomic levels | `n_levels` |
| $N_v$ | number of Doppler velocity classes | `n_v` |
| $S_E$ | intrinsic field sensitivity, **rms single-sided**, V m⁻¹ Hz⁻¹ᐟ² | `s_e` |
| $\eta_0$ | free-space impedance $\sqrt{\mu_0/\epsilon_0}$ | `ETA0` |

Quadratures: vacuum variance 1, $[\hat x,\hat p]=2i$. Covariance ordering $(x_A,p_A,x_B,p_B)$. Symplectic form $\Omega=\bigoplus\begin{psmallmatrix}0&1\\-1&0\end{psmallmatrix}$.

**Field convention (binding).** Every field spectral density at a public interface is rms single-sided. Noise-equivalent power per Hz is $S_E^2A_e/\eta_0$. Rabi frequencies use **peak** fields ($\Omega=dE_{\rm pk}/\hbar$, $E_{\rm pk}=\sqrt2E_{\rm rms}$); conversions happen only in `rydberg_physics/model.py` and `io/literature.py`. Any variable holding a field carries a `_rms` or `_pk` suffix.

## 3. Physics basics

### 3.1 Thermal occupation (FR-PHY-1)
$$\bar n(f,T)=\frac{1}{\exp(hf/k_BT)-1}\quad\text{(use `numpy.expm1`)},\qquad \bar n_{\rm RJ}=\frac{k_BT}{hf}.$$
Constants from `scipy.constants` (CODATA). $T=0\Rightarrow\bar n=0$ exactly (guard the division). RJ relative error relative to the exact value: $(\bar n_{\rm RJ}-\bar n)/\bar n=\mathrm{expm1}(x)/x-1$, $x=hf/k_BT$; 2.438e-2 at 300 GHz, 300 K **[V]**.

### 3.2 Critical transmissivity (FR-PHY-2)
$\eta^\*=\bar n/(1+\bar n)$; compute the window as $1-\eta^\*=1/(1+\bar n)$ directly (no subtraction).

## 4. Phase-insensitive Gaussian channels (FR-GAU-2,3)

A single-mode phase-insensitive Gaussian channel acts on a mode's $2\times2$ block as $V\mapsto\tau V+yI$ with displacement $d\mapsto\sqrt\tau\,d$. Physicality: $y\ge|1-\tau|$ **[L]**.

### 4.1 Constructors
| Channel | $\tau$ | $y$ |
|---|---|---|
| Identity | 1 | 0 |
| Thermal attenuator $(\eta,\bar n)$ | $\eta$ | $(1-\eta)(2\bar n+1)$ |
| Additive classical noise $N_{\rm add}$ (quanta) | 1 | $2N_{\rm add}$ |
| Amplifier gain $G\ge1$, idler occupation $\bar n$ | $G$ | $(G-1)(2\bar n+1)$ |

### 4.2 Composition (apply $\mathcal A$ then $\mathcal B$)
$$(\tau,y)=(\tau_B\tau_A,\ \tau_By_A+y_B).$$
Implement as `B @ A` meaning "A first". Unit test associativity on random physical channels.

### 4.3 Attenuator∘attenuator closed form **[L]**
$(\eta_1,\bar n_1)$ then $(\eta_2,\bar n_2)$ equals an attenuator with $\eta=\eta_1\eta_2$ and
$$\bar n_{\rm eff}=\frac{\eta_2(1-\eta_1)\bar n_1+(1-\eta_2)\bar n_2}{1-\eta_1\eta_2}.$$
This is V0 eq. (div_neff) with $(\eta_1,\bar n_1)=(\eta_{\rm MRC},\bar n_{\rm th})$, $(\eta_2,\bar n_2)=(\eta_{\rm cmb},\bar n_{\rm cmb})$; the resulting threshold $\eta_{\rm MRC}>[\bar n_{\rm th}+(1-\eta_{\rm cmb})\bar n_{\rm cmb}/\eta_{\rm cmb}]/(1+\bar n_{\rm th})$ is V0 eq. (div_netthreshold) **[V]**.

### 4.4 Output noise photons
For vacuum input the output is thermal with $N_{\rm out}=(\tau+y-1)/2$. For a thermal attenuator, $N_{\rm out}=(1-\eta)\bar n$.

### 4.5 Two-mode application
Applying `PIChannel` to mode B of $V=\begin{psmallmatrix}A&C\\C^\top&B\end{psmallmatrix}$ gives $\begin{psmallmatrix}A&\sqrt\tau C\\\sqrt\tau C^\top&\tau B+yI\end{psmallmatrix}$.

## 5. States and entanglement (FR-GAU-1,4,5,6)

### 5.1 States
- TMSV($r$): $a=\cosh2r$, $c=\sinh2r$, $V=\begin{psmallmatrix}aI&cZ\\cZ&aI\end{psmallmatrix}$, $Z=\mathrm{diag}(1,-1)$.
- TMST($r,\bar n_{\rm JPA}$): $a=(2\bar n_{\rm JPA}+1)\cosh2r$, $c=(2\bar n_{\rm JPA}+1)\sinh2r$ (same block structure). **Note:** $a^2-c^2=(2\bar n_{\rm JPA}+1)^2$, not 1. This corrects V0 (IDEA issue D1): V0's $c=\sqrt{a^2-1}$ is a pure TMSV, so V0's universality claim holds only for TMSV.
- Physicality: smallest eigenvalue of the Hermitian matrix $V+i\Omega$ ≥ −1e-12.

### 5.2 PT symplectic eigenvalue
General two-mode CM with blocks $A,B,C$: $\tilde\Delta=\det A+\det B-2\det C$. **Always** evaluate in the cancellation-free form **[D][V]**
$$\tilde\nu_-^2=\frac{2\det V}{\tilde\Delta+\sqrt{\tilde\Delta^2-4\det V}}.$$
V0 eq. (symp_eig), $\tilde\nu_-^2=(\tilde\Delta-\sqrt{\tilde\Delta^2-4\det V})/2$, is algebraically identical but loses ~7e-8 relative accuracy at $r=3$; do not implement it.

Standard form $\begin{psmallmatrix}aI&c'Z\\c'Z&bI\end{psmallmatrix}$ closed form, stable version **[V]** (agrees with the general form to 6e-16 relative on 2×10⁴ random CMs):
$$\tilde\nu_-=\frac{2(ab-c'^2)}{(a+b)+\sqrt{(a-b)^2+4c'^2}}.$$
Use the closed form when the CM is in standard form (the case for every service in this project); keep the general one for tests. Tolerance between the two: relative 1e-10.

### 5.3 Measures
- $\mathcal N=\max\{0,(1-\tilde\nu_-)/(2\tilde\nu_-)\}$, $E_N=\max\{0,-\log_2\tilde\nu_-\}$.
- **Optimal** teleportation fidelity for coherent-state inputs (optimized local Gaussian operations): $F_{\rm opt}=1/(1+\tilde\nu_-)$ **[L]** (Mari & Vitali, PRA 78, 062340, 2008). Code name `fidelity_opt`.
- Braunstein–Kimble unit-gain fidelity: with $V_{x-}=\mathrm{Var}(x_A-x_B)=V_{11}+V_{33}-2V_{13}$ and $V_{p+}=\mathrm{Var}(p_A+p_B)=V_{22}+V_{44}+2V_{24}$,
$$F_{\rm BK}=\Big[\big(1+V_{x-}/2\big)\big(1+V_{p+}/2\big)\Big]^{-1/2}\ \textbf{[L]}.$$
For standard form, $V_{x-}=V_{p+}=a+b-2c'$. Code name `fidelity_bk`. $F_{\rm opt}\ge F_{\rm BK}$, with equality when $a=b$. Reference: TMSV $r=1$, 5 GHz, 300 K attenuator, $\eta=0.9995$: $F_{\rm opt}=0.5858$, $F_{\rm BK}=0.5681$ **[V]**. A bare "fidelity" identifier is not allowed.

### 5.4 Separability conditions
- Standard-form state with $a,b\ge1$: entangled iff $(a-1)(b-1)<c'^2$ **[D][V]**.
- TMSV through $(\tau,y)$ on mode B: $b=\tau a+y$, $c'^2=\tau(a^2-1)$, hence **entangled iff $y<1+\tau$, independent of $r>0$** **[D][V]**. This coincides with the entanglement-breaking criterion of the channel **[L]** (Holevo 2008). Tests exclude points with $|y-(1+\tau)|<10^{-9}$.
- TMST through $(\tau,y)$: entangled iff $(a-1)(\tau a+y-1)<\tau c^2$ with TMST $a,c$ **[D][V]** (0 mismatches against direct $\tilde\nu_-$ on 2×10⁴ random cases). TMST threshold $\eta^\*_{\rm TMST}(r,\bar n_{\rm JPA})$ by `brentq`; at 5 GHz, 300 K: 0.99920045 ($\bar n_{\rm JPA}=0$, any $r$), 0.99920628 $(0.002,0.5)$, 0.99932075 $(0.05,0.5)$ **[V]**.

## 6. Stochastic channel

### 6.1 Small-scale PDT (FR-CH-1)
Single element ($N=1$) and $N$-element passive MRC share one implementation.

Parameters: $K\ge0$, scale $\bar\eta$ ($=\bar\eta_s$ for $N=1$, $=\bar\eta_\Sigma$ for arrays), $N\ge1$, with $0<\bar\eta\le1$ enforced.

- $\sigma_N^2=\bar\eta/\big(2N(K+1)\big)$; $X=\eta/\sigma_N^2\sim\chi'^2(\nu=2N,\ \lambda=2NK)$ (untruncated surrogate) **[L]** (V0).
- Truncation point $x_{\max}=1/\sigma_N^2=2N(K+1)/\bar\eta$; normalizer $\mathcal Z_{K,N}=F_{\chi'^2}(x_{\max})$.
- pdf on $[0,1]$: $f(\eta)=f_{\chi'^2}(\eta/\sigma_N^2)/(\sigma_N^2\,\mathcal Z)$.
- cdf: $F(x)=F_{\chi'^2}(\min(x,1)/\sigma_N^2)/\mathcal Z$ (the clamp is mandatory; $F(x)=1$ for $x\ge1$).
- survival (use this near 1): $S(x)=\big[S_{\chi'^2}(x/\sigma_N^2)-S_{\chi'^2}(x_{\max})\big]/\mathcal Z$ for $x<1$; compute in the tail that is smaller and switch accordingly.
- **Surrogate cdf/sf, all $K$ (including $K=0$):** every evaluation goes through one helper, `numerics.special.ncx2_cdf(x, nu, lam)` / `ncx2_sf`, which calls `scipy.stats.ncx2` with $\lambda\leftarrow\max(\lambda,\lambda_{\rm floor})$, $\lambda_{\rm floor}=10^{-300}$ (`NC_FLOOR`). At double precision this is the central $\chi^2_{2N}$ exactly (the $j\ge1$ Poisson weights are below 1e-300).
  - **Why not `chi2`:** `scipy.stats.chi2`, `scipy.stats.gamma`, `special.gammainc` and `ncx2(nc=0)` share one code path that loses accuracy in the lower tail at large shape (cancellation in $a\ln y-y-\ln\Gamma(a+1)$, magnitude ~$10^9$). At $z=-5.5$ sd (cdf ≈ 1.8e-8), relative error is 3e-14 / 1.6e-6 / 2.3e-2 / 0.32 for $N=10^5/10^6/10^7/10^8$; the `ncx2` floor path gives 3e-14 / 2e-13 / 5e-13 / 1.2e-12 **[V]** (scipy 1.18.1; reference: series for $P(a,x)$ with an `mpmath` prefactor; QUESTIONS Q-3).
  - The fix relies on a scipy implementation detail, so it is guarded by the pinned regression test `test_ncx2_floor_tail` (§10): the four points above, both tails, `rtol=1e-6`. A scipy upgrade that fails it blocks the upgrade.
  - Closed forms (truncated exponential for $N=1$, truncated Erlang for small $N$) are test oracles only, at $N\le100$.
- **Very large $\lambda$ (QUESTIONS Q-9, accepted as documented limit):** `ncx2.sf` warns "Series did not converge" in the far upper tail for $\lambda\gtrsim2\times10^{10}$ (e.g. $N=1$, $K=10^{10}$, 8 sd above the mean); the bulk and lower tail still match the normal limit to 4 digits. Supported range for the upper-tail sf: $K\le10^9$. Limit tests for $K\to\infty$ use $K=10^9$; configured experiments use $K\le100$. A `RuntimeWarning` from this path is promoted to an error in tests, so a silent excursion past the limit fails.
- **Large $N$ (up to $10^8$):** covered by the same helper and the same regression test; NFR-2 (1% relative down to 1e-8) is the requirement, the regression gate is tighter on purpose.
- **Samplers.**
  - `sample_sum` (default, any $N$): draw $X\sim\chi'^2(2N,2NK)$ with `Generator.noncentral_chisquare` (or `chisquare` for $K=0$), set $\eta=\sigma_N^2X$, and reject if $\eta>1$. This is exact for passive MRC because the joint truncation constrains only the sum.
  - `sample_elements` (needed only for selection combining, $N\le N_{\max}$ with default 64): draw $h_i\sim\mathcal{CN}\big(\sqrt{K\bar\eta/(N(K+1))}\,e^{i\phi_{0,i}},\ \bar\eta/(N(K+1))\big)$ for $i=1..N$, reject the whole vector if $\sum|h_i|^2>1$. Calling it with $N>N_{\max}$ raises an error that names `sample_sum`.
  - Acceptance rate is $\mathcal Z_{K,N}$, which is ≥ ~0.5 because $x_{\max}$ ≥ the surrogate mean for $\bar\eta\le1$ **[V]**. Phases $\phi_{0,i}=0$ by default (result is phase-independent).
- Moments: numerical integration of the **truncated** pdf, $\mathbb E^{\rm QC}[\eta_s^m]$. $\bar\eta$ is a scale parameter, not the mean.
  - **Integration window (adaptive) [D]:** in $X$-space, with surrogate mean $\mu_X=2N(1+K)$ and sd $\sigma_X=\sqrt{4N(1+2K)}$, integrate over $[x_{\rm lo},x_{\rm hi}]=[\max(0,\mu_X-w\sigma_X),\ \min(x_{\max},\mu_X+w\sigma_X)]$, starting at $w=20$.
  - **Mass check:** the neglected mass $M_{\rm out}=F_{\chi'^2}(x_{\rm lo})+\big[S_{\chi'^2}(x_{\rm hi})-S_{\chi'^2}(x_{\max})\big]$ (second term only if $x_{\rm hi}<x_{\max}$) must satisfy $M_{\rm out}\le10^{-14}\mathcal Z_{K,N}$; otherwise double $w$ (cap $w=320$, then raise). Because $\eta\in[0,1]$, the moment error is at most $M_{\rm out}/\mathcal Z$ for every $m\ge0$.
  - **Why adaptive:** a fixed 20 sd window is safe only when the surrogate is near-Gaussian. For $K=0$, $N=1$ (exponential) it leaves $e^{-21}\approx7.6\times10^{-10}$ out whenever $x_{\max}>\mu_X+20\sigma_X$, i.e. $\bar\eta<0.048$ (a bare-cell or long link) **[V]**.
  - Any non-finite pdf value inside the window raises (`ncx2.pdf` returns `nan` ≳27 sd from the mean at $N=10^7$, $K=10$, and at $K=10^8$, $N=1$; QUESTIONS Q-5). `nan` is never masked to 0.
- **Untruncated surrogate (RQ1 only, `physical=False`):** `SmallScale.surrogate_cdf(x)` $=F_{\chi'^2}(x/\sigma_N^2)$, without $\mathcal Z$ and without the clamp. For a deterministic large-scale factor it satisfies $F(x)=F_{\rm surr}(x)/\mathcal Z_{K,N}$ for $x<1$ exactly; the removed mass is $1-\mathcal Z_{K,N}$.
- Reference: at $\bar\eta=0.999$, $N=1$: $\mathcal Z_K=0.63249$ ($K=0$), $0.54404$ ($K=10$), $0.51689$ ($K=100$) **[V]**.

### 6.2 Large-scale PDT (FR-CH-2,3)
**Primary: shifted log-gamma [D].** $\Psi=\Psi_0+G$, $\Psi_0\ge0$ deterministic (absorption, $\gamma_{\rm abs}d$), $G\sim\mathrm{Gamma}(k,\theta)$ (shape–scale), $\eta_\ell=e^{-\Psi}\in(0,e^{-\Psi_0}]$.
- From stochastic mean/variance $(\mu_G,\sigma_G^2)$: $k=\mu_G^2/\sigma_G^2$, $\theta=\sigma_G^2/\mu_G$.
- **Per-metre parameterization (config default) [D]:** weather-driven loss (water vapour, rain) is specified per metre as $(m_1,s_1)$ per band, `stochastic_per_m`. On links up to ~1 km one weather state covers the whole path, so the path is fully correlated **[A]** and both moments scale linearly: $\mu_G=m_1d$, $\sigma_G=s_1d$. Hence $k=m_1^2/s_1^2$ is distance-independent and $\theta=(s_1^2/m_1)\,d$.
  - Partially correlated paths ($\sigma_G\propto\sqrt d$, longer links) are out of scope; a config that needs them must say so and is rejected until implemented.
  - The legacy per-link form `stochastic: {mean, std}` remains accepted for unit tests and sensitivity sweeps only; it is flagged `per_link=True` in `meta.json`.
  - Reference values (KHU Seoul, ITU-R climatology) are in §16.2.
- pdf (log-domain): with $u=-\ln\eta-\Psi_0>0$, $\ln f_\ell(\eta)=(k-1)\ln u-u/\theta-\ln\Gamma(k)-k\ln\theta-\ln\eta$; zero for $u\le0$.
- cdf: $P(\eta_\ell\le x)=Q(k,\,u_x/\theta)$ with $u_x=-\ln x-\Psi_0$ (`gammaincc`); equals 1 when $u_x\le0$. Survival: `gammainc`.
- Moments: $\mathbb E[\eta_\ell^m]=e^{-m\Psi_0}(1+m\theta)^{-k}$ **[V]** ($\Psi_0=0$, $k=4,\theta=0.01$: 0.96098 analytic vs 0.96095 MC).
- Degenerate $\sigma_G\to0$: point mass at $e^{-\Psi_0-\mu_G}$ (handle explicitly).
- Large shape ($k>10^5$, i.e. $\sigma_G/\mu_G<3\times10^{-3}$): `gammainc`/`gammaincc` have the §6.1 lower-tail problem. Evaluate via $P(k,u)=F_{\chi^2_{2k}}(2u)$ with the `ncx2` floor helper (§13 rule 10).

**Baselines.**
- V0 log-normal: $\Psi\sim\mathcal N(\bar\Psi,\sigma_\Psi^2)$, untruncated; must expose `mass_above_one()` $=\Phi(-\bar\Psi/\sigma_\Psi)$. Reference: 0.158655 for $\sigma_\Psi=\bar\Psi$ **[V]**. Flag as `physical=False`.
- Conditioned log-normal: same, conditioned on $\Psi\ge0$; normalizer $\Phi(\bar\Psi/\sigma_\Psi)$.

**Law comparison (RQ2, E2).**
- **Matching rule [A]:** all three laws share the first two moments of $\Psi$: $\bar\Psi=\Psi_0+\mu_G$, $\sigma_\Psi=\sigma_G$, with the same $\Psi_0$. (The conditioned log-normal matches only before conditioning; its post-conditioning moments are reported.)
- **Realizations with $\eta>1$ under the V0 log-normal:** service formulas are undefined there ($N_{\rm out}=(1-\eta)\bar n_{\rm th}<0$). Policy `eta_gt1: clip` (the only one implemented): evaluate the service at $\min(\eta,1)$, set `physical=False`, and report the affected mass next to every number. Together with `friis_clipped` (§6.5), this is one of the two sanctioned exceptions to §13 rule 9, and both exist only inside flagged baselines.
- **Outputs:** $\Delta P_{\rm out}$ (absolute) and $\Delta\bar C/\bar C$ (relative, for $C_\chi$, $C_{\rm het}$ and the default RAR $C_{\rm rx}$) of each baseline against log-gamma, vs $\sigma_\Psi/\bar\Psi$.
- **Limit check:** as $\sigma_\Psi\to0$ all three laws give the same $P_{\rm out}$ and $\bar C$ to `rtol=1e-6`.

### 6.3 Obstacle generator (FR-CH-4, optional) [A]
Number of obstacles $M\sim\mathrm{Poisson}(\lambda_od)$; each log-loss $\psi_j\sim\mathrm{Gamma}(k_o,\theta_o)$, i.i.d.; $G=\sum_j\psi_j$.
Moments: $\mu_G=\lambda_od\,k_o\theta_o$, $\sigma_G^2=\lambda_od\,k_o(k_o+1)\theta_o^2$ **[D]**. Moment-match to §6.2. Expose the exact compound-Poisson sampler for diagnostics. Note the atom $P(G=0)=e^{-\lambda_od}$, which the Gamma fit cannot represent. All parameter values need a `source:`; none are defaulted from memory.

### 6.4 Compound PDT (FR-CH-5)
$\eta=\eta_\ell\eta_s$, independent. Work in $\Psi$-space, $\eta_\ell=e^{-\Psi}$:
- cdf: $F(x)=P(\eta_\ell\le x)+\displaystyle\int_{\Psi:\ \eta_\ell>x}F_s\!\big(xe^{\Psi}\big)\,f_\Psi(\Psi)\,d\Psi$.
- survival: $S(x)=\displaystyle\int_{\Psi:\ \eta_\ell>x}S_s\!\big(xe^{\Psi}\big)\,f_\Psi(\Psi)\,d\Psi$. Use whichever of $F,S$ is smaller as the primary computation.
- pdf: $f(x)=\displaystyle\int_{\Psi:\ \eta_\ell>x}e^{\Psi}f_s\!\big(xe^{\Psi}\big)f_\Psi(\Psi)\,d\Psi$.
- Integration settings: `scipy.integrate.quad`, `epsabs=0`, `epsrel=1e-10`, `limit=500`.
  - Breakpoints at $\Psi_0+\mu_G+j\sigma_G$, $j\in\{-6,-3,-1,0,1,3,6\}$, clipped to the domain, and at the kink $\Psi=-\ln x$.
  - If `quad` reports an error estimate above 1e-8 absolute, fall back to `mpmath.quad` and log a warning.
- Quantile $F^{-1}(\epsilon)$: `brentq` in the variable $1-\eta$ on $[0,1]$ (§13 rule 3), `xtol=1e-14`.
- Moments: $\mathbb E[\eta^m]=\mathbb E[\eta_\ell^m]\,\mathbb E^{\rm QC}[\eta_s^m]$ (truncated small-scale moments, §6.1).
- MC sampler: product of independent samples from §6.1 (`sample_sum`) and §6.2.

### 6.5 Link budget (FR-CH-6)
- Apertures:
  - $A_t=G_t\lambda^2/(4\pi)$.
  - $A_r$ is the receive collecting aperture: given in config for an **antenna-fed receiver** (horn, lens or dish feeding the RAR or the LNA; the default), or $A_r=A_e=3\lambda^2/(8\pi)$ for a bare RAR cell (§7.4).
  - **Enhancement factor (receive side only):** $A_{r,\rm eff}=G_{\rm enh}A_r$, with `G_enh: float = 1.0` and $G_{\rm enh}\ge1$ (raise otherwise). It represents a lossless, matched field-confining cell structure on top of $A_r$. There is no transmit-side factor; transmit gain is $G_t$ only (QUESTIONS Q-4).
  - In the antenna-fed case leave $G_{\rm enh}=1$, otherwise the collecting area is counted twice. Loss in a feed or enhancing structure is an attenuator, not $G_{\rm enh}<1$ (§7.4).
  - Antenna-fed coupling into the receiver is assumed lossless and matched **[A]**: an ideal-receiver bound, stated as such in the paper.
- Fresnel product $\tau^2=D_f=A_tA_{r,\rm eff}/(\lambda d)^2$ (equals the Friis power ratio).
- Geometric coupling models (config `coupling_model`):
  - `"goubau"` (**default**): $\eta_{\rm geo}=1-e^{-\tau^2}$ **[L]** (Goubau & Schwering 1961; optimally tapered apertures). It reduces to Friis for $\tau^2\ll1$, with relative difference $\le\tau^2/2$. It is an approximation, so results with $\bar\eta\to1$ are model-dependent; E3 reports the sensitivity to this choice.
  - `"friis_clipped"` (baseline, `physical=False`): $\min(\tau^2,1)$. Allowed only as a labeled comparison, analogous to the V0 log-normal.
- Absorption: $\Psi_0=\gamma_{\rm abs}d$ with $\gamma_{\rm abs}$ in m⁻¹ (power). Default at 5 GHz: $1.44\times10^{-6}$ m⁻¹ **[L]** (Gonzalez-Raya et al. 2022). Other bands: from ITU-R P.676 values supplied in config with `source:`.
- Mean collection efficiency: $\bar\eta_s=\eta_{\rm geo}$; absorption enters through $\Psi_0$ in §6.2, never twice.
- References **[V]**:
  - $e^{-\gamma d}=\eta^\*$ at $d=555.47$ m (5 GHz, 300 K).
  - 5 GHz, $A_t=A_r=1$ m², $d=10$ m: $\tau^2=2.7816$, $\eta_{\rm geo}=0.93806$ (Friis would give 2.78).
  - $\bar\eta=0.9999$ at 10 m needs $A_t=A_r\approx1.82$ m².
  - A 30 dBi horn to a bare dipole cell gives $\tau^2=3.4\times10^{-4}$ at 10 m, i.e. outage ≡ 1.

## 7. Receivers (FR-RX)

### 7.1 Two classes, and sidebands
- **Measuring receiver:** produces classical samples.
  - Parameter $N_{\rm amp}\ge\frac12$: input-referred added quanta, the Caves bound for phase-insensitive detection of both quadratures.
  - Optional bandwidth $B$ (Hz).
  - Required `sideband: "ssb" | "dsb"` **[A]**:
    - `ssb`: the image band is rejected (or sees vacuum), $N_{\rm img}=0$.
    - `dsb`: signals at $f_{\rm LO}\pm\delta$ both reach the output (e.g. a Rydberg superheterodyne, whose response depends on the field envelope). The image-band mode adds its thermal occupation $N_{\rm img}=(1-\eta)\bar n_{\rm th}+\eta\,\bar n_{\rm tx,img}$, with $\bar n_{\rm tx,img}$ the TX image-port occupation (config, default 0).
  - Defaults: RAR `dsb`, electronic `ssb`.
- **Quantum front end:** a `PIChannel` $(\tau_r,y_r)$ inserted before quantum processing (entanglement service, Holevo bound of the chain). The sideband setting does not apply.

### 7.2 Ideal receivers
Ideal heterodyne = measuring receiver with $N_{\rm amp}=\frac12$, `ssb`; its ½ is the image-band vacuum. Ideal homodyne is a separate formula (§8.2), not an $N_{\rm amp}$ value. Quantum-limited amplifier front end: `PIChannel(G, G-1)`.

### 7.3 Electronic receiver (FR-RX-3)
Input: noise temperature $T_n\ge0$. Convention flag (required in config; it must match the source of the $T_n$ value):

| Convention | $N_{\rm amp}$ | Quantum limit ($N_{\rm amp}=\tfrac12$) |
|---|---|---|
| `"rj"` (default, radio engineering) | $k_BT_n/(hf)$ | $T_n=hf/(2k_B)$ |
| `"planck"` (Caves) | $\bar n(f,T_n)$ | $T_n=hf/(k_B\ln3)$ |
| `"callen_welton"` (includes zero-point) | $\bar n(f,T_n)+\tfrac12$ | $T_n=0$ |

- Raise `ValueError` if $N_{\rm amp}<\frac12$; this guard can fire only for `rj` and `planck`.
- Reference at 5 GHz with $T_n=hf/(k_B\ln3)=218.42$ mK: $N_{\rm amp}$ = 0.9102 (rj), 0.5000 (planck), 1.0000 (callen_welton) **[V]**.
- At high $T_n$, `rj` and `callen_welton` agree to $O(x)$; `planck` is lower by ½.

### 7.4 Rydberg atomic receiver (FR-RX-4) [D]
**Short-dipole equivalence (modeling convention, not derived physics):** a bare atomic sensor of one field component is mapped to $A_e=3\lambda^2/(8\pi)$. Consistency check **[V]**: $\sqrt{\bar n_{\rm th}hf\eta_0/A_e}$ equals the RJ single-component BBR field density $\sqrt{8\pi f^2k_BT/(3\epsilon_0c^3)}$ to $O(x)$ (0.603 nV/cm/√Hz at 5 GHz, 300 K).

Mapping (input $S_E$ in V m⁻¹ Hz⁻¹ᐟ², **intrinsic**, i.e. BBR excluded, rms):
$$N_{\rm amp}=\frac{S_E^2A_e}{\eta_0hf}.$$
- **Field convention:** the constructor takes a required `field_convention: "rms" | "peak"` argument (no default); a peak value is divided by $\sqrt2$ before use.
- **Sub-quantum-limit input:** if the mapped $N_{\rm amp}<\frac12$, raise `ValueError`. Do not clamp silently.
- **BBR-inclusive input:** if the supplied $S_E$ was measured including in-band BBR, subtract in power first: $S_{E,\rm int}^2=S_{E,\rm meas}^2-S_{E,\rm BBR}^2$ (config flag `includes_bbr`). Error if negative.
- **Enhancement structures and antenna feeds:** neither $G_{\rm enh}$ nor $A_r$ changes $N_{\rm amp}$; they enter only through $A_{r,\rm eff}$ in §6.5. $N_{\rm amp}$ always uses the bare-cell $A_e$ and the $S_E$ at the atoms **[D]**: a lossless matched structure raises the field at the atoms by $\sqrt{G}$, so the incident-referred sensitivity is $S_E/\sqrt G$ while the collecting area is $GA_e$, and $(S_E^2/G)(GA_e)=S_E^2A_e$. Evaluating $N_{\rm amp}$ with $A_r$ or $A_{r,\rm eff}$ in place of $A_e$ overcounts the noise by $A_{r,\rm eff}/A_e$; the code must not accept an aperture argument in the $N_{\rm amp}$ mapping. A lossy feed or enhancing structure is an extra attenuator $(\eta_{\rm struct},\bar n_{\rm th})$ composed before the receiver.
- Sideband default `dsb` (§7.1).
- Equivalent noise temperature (RJ): $T_n=N_{\rm amp}hf/k_B$.
- References **[V]**:
  - $S_E=10$ nV/cm/√Hz rms $=10^{-6}$ V m⁻¹ Hz⁻¹ᐟ² at 5 GHz → $N_{\rm amp}=3.438\times10^5$, $T_n=8.25\times10^4$ K.
  - The same number declared `peak` → $N_{\rm amp}=1.719\times10^5$.
  - BBR-equivalent sensitivity $S_{E,\rm BBR}(f,T)=\sqrt{\bar n(f,T)hf\eta_0/A_e}$: 0.603 / 3.614 / 12.01 / 35.75 nV/cm/√Hz at 5 / 30 / 100 / 300 GHz, 300 K.
- **Optional fields:** `bandwidth_hz` and `e_1db_pk_v_per_m` (linear-range limit, a **peak** field, from §14.9 or literature).
  - The received peak field is $E_{\rm rx,pk}=\sqrt{2\eta_0P_{\rm rx}/A_r}$.
  - If $E_{\rm rx,pk}>E_{1\rm dB,pk}$, set `linear_regime=False` on results; do not alter numbers.
  - Never compare an rms field with a peak field.
- **Model-derived alternative:** `RydbergReceiver.from_physics(model_params)` takes $S_E$, $B$ and $E_{1\rm dB}$ from §14 instead of literature inputs.
- **Aperture/field convention cross-check (resolved 2026-11-10) [V]:** Santamaría Botello et al. (arXiv 2209.00908) relate field and power by $P=G\frac{\lambda^2}{4\pi}\frac{|E|^2}{2\eta_0}$ with $G=3/2$, i.e. a **peak** field and the same $A_e=3\lambda^2/8\pi$ as here. Reading their 1.25 µV/m/√Hz at 10 GHz as peak, the $N_{\rm amp}$ mapping above gives $T=16{,}114$ K, against their 16,100 K; an rms reading would give 32,228 K. So literature NEFs with no stated convention are entered as `peak` (§16.3). Record in `docs/RESULTS.md`.

### 7.5 Coherent transducer front end (FR-RX-5)
`PIChannel` = attenuator $(\eta_t,\bar n_t)$ followed by optional additive noise $N_t$: $\tau_r=\eta_t$, $y_r=(1-\eta_t)(2\bar n_t+1)+2N_t$. Parameter values: config only, with `source:` **[A]**.

## 8. Services

### 8.1 Entanglement service (FR-SVC-E1,2)
Chain per realization: fading attenuator $(\eta,\bar n_{\rm th})$ then front end $(\tau_r,y_r)$:
$$\tau=\tau_r\eta,\qquad y=\tau_r(1-\eta)(2\bar n_{\rm th}+1)+y_r.$$
**Receiver-referred threshold [D][V]** (TMSV, any $r>0$): entangled iff $\eta>\eta^\*_{\rm rx}$ with
$$\eta^\*_{\rm rx}=\frac{\tau_r(2\bar n_{\rm th}+1)+y_r-1}{2\tau_r(\bar n_{\rm th}+1)}.$$
Special cases:
- Identity front end → $\eta^\*$.
- Additive noise $N_{\rm add}$ → $(\bar n_{\rm th}+N_{\rm add})/(\bar n_{\rm th}+1)$.
- A window exists iff $\eta^\*_{\rm rx}<1$; for additive noise, iff $N_{\rm add}<1$.

References at 5 GHz:
- $N_{\rm add}=\frac12\Rightarrow0.99960$; $N_{\rm add}=1\Rightarrow1$ (no window).
- Transducer $\eta_t=0.6,\ \bar n_t=0.01,\ N_t=0.01\Rightarrow0.999219$ **[V]**. The formula matches a direct root of $\tilde\nu_-(\eta)=1$ to machine precision for all four cases tested.
- V0's warm-combining-network threshold is the attenuator special case (§4.3; regression test).

For TMST inputs, solve §5.4 numerically per realization (`brentq` in $1-\eta$).

Metrics over the compound PDT $f(\eta)$:
- $P_{\rm out}=F(\eta^\*_{\rm rx})$ (§6.4).
- $\mathbb E[E_N]=\int_{\eta^\*_{\rm rx}}^1E_N(\eta)f(\eta)\,d\eta$, computed as a nested integral in $\Psi$-space, cross-checked by MC (§10.1).
- **Conditional ("when the link works") averages [D]** for $\tilde\nu_-$, $F_{\rm opt}$ and $F_{\rm BK}$ (QUESTIONS Q-8; decided by Abdurrahman 2026-11-10). With $g\in\{\tilde\nu_-,F_{\rm opt},F_{\rm BK}\}$:
$$\mathbb E[g\mid\eta>\eta^\*_{\rm rx}]=\frac{1}{1-P_{\rm out}}\int_{\eta^\*_{\rm rx}}^1 g(\eta)f(\eta)\,d\eta.$$
  - Code names `nu_minus_cond`, `fidelity_opt_cond`, `fidelity_bk_cond`. The bare restricted integral (without the division) is an internal quantity, never reported.
  - If $1-P_{\rm out}<10^{-12}$ (link essentially always down), return `NaN` with reason `"no entangled realizations"`; never 0.
  - Every figure or table that shows a conditional average also shows $1-P_{\rm out}$ next to it, so a high fidelity on a rarely working link is not misread.
  - Sanity: $\mathbb E[F_{\rm opt}\mid\cdot]>1/2$ and $\mathbb E[\tilde\nu_-\mid\cdot]<1$ always, since $\tilde\nu_-<1\iff F_{\rm opt}>1/2$ for each realization. No such bound holds for $F_{\rm BK}$.
- $\mathbb E[E_N]$ stays the **unconditional** mean above (its integrand is 0 in outage, so it is already "per attempt"). The conditional version is $\mathbb E[E_N]/(1-P_{\rm out})$ and may be reported as `log_neg_cond`.
- **Truncation effect (RQ1, E1):** $\Delta P_{\rm out}=P_{\rm out}-P_{\rm out}^{\rm surr}$, where $P_{\rm out}^{\rm surr}$ uses `surrogate_cdf` (§6.1) in place of the truncated small-scale cdf in the §6.4 nested integral. Report $\Delta P_{\rm out}$ and $P_{\rm out}/P_{\rm out}^{\rm surr}$ next to the removed mass $1-\mathcal Z_{K,N}$. Identity check: with a deterministic large-scale factor and $\eta^\*/\eta_\ell<1$, $P_{\rm out}/P_{\rm out}^{\rm surr}=1/\mathcal Z_{K,N}$ to `rtol=1e-12`.

**Limiting cases (regression tests).** The small-scale cdf is always evaluated with the clamp $F_s(\min\{x,1\})$, and the tests include $\eta_\ell<\eta^\*$, where the clamp is active.

| Case | Under V0 log-normal (V0 closed forms) | Under shifted log-gamma |
|---|---|---|
| 1, $K\to\infty$ | V0 eq. (outage_LOS) | $Q\!\big(k,\ [\ln(\bar\eta_s/\eta^\*)-\Psi_0]/\theta\big)$ if positive argument, else 1 **[D][V]** (unshifted check: 0.19952 closed vs 0.19892 MC, $k=2$, $\theta=10^{-4}$, $\bar\eta_s=0.9995$) |
| 2, $K=0$ | V0 eq. (outage_Rayleigh) **with clamp** | quadrature of the clamped truncated-exponential cdf, `rtol=1e-8` |
| 3, $\sigma\to0$ | V0 eq. (outage_no_shadow) **with clamp** | same as V0 with $e^{-\bar\Psi}\to e^{-\Psi_0-\mu_G}$ |
| 4, deterministic | step function at $\eta^\*$ | identical |

### 8.2 Classical service (FR-SVC-C1..3)
Coherent-state Gaussian encoding with mean photon number $N_s$ per mode. Per realization with total chain $(\tau,y)$ and $N_{\rm out}=(\tau+y-1)/2$:
- Holevo capacity **[L]** (Giovannetti, García-Patrón, Cerf, Holevo, Nat. Photon. 2014): $C_\chi=g(\tau N_s+N_{\rm out})-g(N_{\rm out})$, $g(x)=(x+1)\log_2(x+1)-x\log_2x$, $g(0)=0$ (use `scipy.special.xlogy`/log1p forms).
- Heterodyne (ideal, SSB): $C_{\rm het}=\log_2\!\big(1+\tau N_s/(N_{\rm out}+1)\big)$.
- Homodyne: $C_{\rm hom}=\tfrac12\log_2\!\big(1+4\tau N_s/(2N_{\rm out}+1)\big)$.
- Measuring receiver after the fading channel only ($\tau=\eta$, $N_{\rm out}=(1-\eta)\bar n_{\rm th}$):
$$C_{\rm rx}=\log_2\!\Big(1+\frac{\eta N_s}{N_{\rm out}+\tfrac12+N_{\rm amp}+N_{\rm img}}\Big),$$
  with $N_{\rm img}$ from §7.1 (0 for `ssb`). Check: `ssb`, $N_{\rm amp}=\frac12\Rightarrow C_{\rm het}$.
- All are increasing in $\eta$ (assert in tests); $C_{\rm rx}$ is non-increasing in $N_{\rm amp}$. Hence $\epsilon$-outage capacity $C_\epsilon=C\big(F^{-1}(\epsilon)\big)$ and rate outage $P(C<R)=F\big(C^{-1}(R)\big)$, with $C^{-1}$ by `brentq`.
- Ergodic capacity $\bar C=\mathbb E_\eta[C(\eta)]$ by nested quadrature, MC cross-check.
- Throughput $R_b=B\cdot C$ (bit/s). TX energy helper: $N_s=P_t/(hfB)$.

### 8.3 Crossover sensitivity (FR-SVC-C4) [D]
Noise-penalty definition: receiver noise relative to **ideal SSB heterodyne** at a reference transmissivity $\eta_{\rm ref}$ (default: compound mean),
$$\frac{N_{\rm out}+\tfrac12+N_{\rm amp}+N_{\rm img}}{N_{\rm out}+1}\le10^{\Delta/10}\ \Longrightarrow\ N_{\rm amp}^{\rm req}=(10^{\Delta/10}-1)(N_{\rm out}+1)+\tfrac12-N_{\rm img},$$
$$S_E^{\rm req}=\sqrt{N_{\rm amp}^{\rm req}\,\eta_0hf/A_e}\quad(\text{rms}).$$
- If $N_{\rm amp}^{\rm req}<\frac12$, return `NaN` with a reason string. For `dsb`, the reason must name the image floor.
- **DSB image floor [D][V]:** at $\eta\to0$ a `dsb` receiver cannot beat $10\log_{10}\big[(2\bar n_{\rm th}+1)/(\bar n_{\rm th}+1)\big]$ = 3.009 / 3.000 / 2.976 / 2.907 dB at 5 / 30 / 100 / 300 GHz, 300 K. So a 3 dB target is unattainable for a DSB RAR.
- References at 5 GHz, 300 K, $\eta\to0$ **[V]**:
  - SSB: $\Delta=3$ dB → $N_{\rm amp}^{\rm req}=1245.27$, $S_E^{\rm req}/S_{E,\rm BBR}=0.99823$.
  - DSB: $\Delta=4$ dB → $N_{\rm amp}^{\rm req}=641.71$, ratio 0.71659; $\Delta=6$ dB → 2479.22, ratio 1.40849.
- Capacity-based variant: **largest tolerable** $N_{\rm amp}$ with $\bar C_{\rm rx}\ge(1-\delta)\bar C_{\rm het}$, solved by `brentq` on $\log N_{\rm amp}$ (QUESTIONS Q-10). $\bar C_{\rm rx}$ is non-increasing in $N_{\rm amp}$, so every smaller $N_{\rm amp}\ge\tfrac12$ also satisfies the constraint; the boundary is the meaningful quantity and matches what the dB-penalty closed form returns. If even $N_{\rm amp}=\tfrac12$ fails (e.g. `dsb` against a tight $\delta$), return `NaN` with a reason.
- In the RJ regime $S_{E,\rm BBR}\propto f$; E6 must show this slope as a sanity check, and must plot SSB and DSB required-$S_E$ curves separately.

### 8.4 RAR vs LNA throughput map (FR-SVC-C5, RQ5) [D]
Question: where in $(f,d,K)$ does an RAR deliver more bit/s than an electronic LNA, if anywhere?
- **Fair comparison (binding):** both receivers sit behind the **same** collecting aperture $A_r$ (antenna-fed, §6.5, $G_{\rm enh}=1$), so they see the same $\eta$ per realization. They differ only in $N_{\rm amp}$, $N_{\rm img}$ (RAR `dsb`, LNA `ssb` by default) and bandwidth $B$.
- **Equal transmit power, not equal $N_s$:** $N_s=P_t/(hfB)$ per receiver, so a wider band trades photons per mode for modes per second.
- Ergodic throughput $\bar R_b=B\cdot\mathbb E_\eta\big[C_{\rm rx}(\eta;N_s,N_{\rm amp},N_{\rm img})\big]$ (§8.2). Optional outage variant: $R_{b,\epsilon}=B\cdot C_{\rm rx}(F^{-1}(\epsilon))$.
- **Output:** $\rho(f,d;K)=10\log_{10}\big(\bar R_b^{\rm RAR}/\bar R_b^{\rm LNA}\big)$ on an $(f,d)$ grid, one panel per $K$, with the $\rho=0$ contour drawn and labeled.
- **Per-band receiver inputs [A]** (IDEA Q2): RAR $(S_E,B_{\rm RAR})$ from `literature_rar.yaml` or `from_physics`; LNA $(T_n,B_{\rm LNA},\texttt{tn\_convention})$ from config with `source:`. Between configured bands, interpolate $\log S_E$, $\log T_n$, $\log B$ linearly in $\log f$; outside the configured range, mask (no extrapolation).
- **Asymptotes (tests):** write $D(\eta)=(1-\eta)\bar n_{\rm th}+\tfrac12+N_{\rm amp}+N_{\rm img}$.
  - Power-limited ($\eta N_s\ll D$): $R_b\to\eta P_t/(hf\,D\ln2)$, independent of $B$, so for deterministic $\eta$, $\rho\to10\log_{10}(D_{\rm LNA}/D_{\rm RAR})$.
  - Bandwidth-limited ($\eta N_s\gg D$): $R_b\approx B\log_2(\eta N_s/D)$, so bandwidth dominates.
  - Hence an RAR can win only through bandwidth at high SNR, or through lower noise at low SNR. The figure caption states which regime each region is in.

## 9. Diversity (FR-DIV) — ported from V0, keep exact forms

With $\sigma_N^2=\bar\eta_\Sigma/(2N(K+1))$:
- **Array PDT and outage:** the array PDT and cdf are §6.1 with $N>1$. Array outage is $P_{{\rm out},N}=\int F_N(\eta^\*/\eta_\ell)f_\ell\,d\eta_\ell$ (same nested scheme as §6.4).
- **Selection combining exact tail:** valid **only for $x\ge\frac12$**; for $x<\frac12$ raise a domain error. At 300 K, $\eta^\*>\frac12$ for $f<4.33$ THz.
$$P(\eta_{\rm SC}>x)=\dfrac{N}{\mathcal Z_{K,N}}\displaystyle\int_x^1f_{s}(t)\,G_{N-1}(1-t)\,dt.$$
  - $f_s$ is the **untruncated** single-element pdf with scale $\bar\eta_\Sigma/N$.
  - $G_{N-1}(y)=F_{\chi'^2}\big(y/\sigma_N^2;\ \nu=2(N-1),\ \lambda=2(N-1)K\big)$.
  - Test grid: $x\in[\max(\eta^\*,\tfrac12),1)$.
- **Chernoff bound (all $x$):** $\dfrac{N}{\mathcal Z_{K,N}}\inf_{0<s<K+1}\dfrac{K+1}{K+1-s}\exp\!\Big(\dfrac{Ks}{K+1-s}-\dfrac{sNx}{\bar\eta_\Sigma}\Big)$, via `minimize_scalar` (bounded).
- **Cramér rate:** $u_a=\dfrac{1+\sqrt{1+4aK(K+1)}}{2a}$, $\mathcal R_K(a)=(K+1-u_a)\big(a-K/u_a\big)+\ln\dfrac{u_a}{K+1}$, with $\mathcal R_0(a)=a-1-\ln a$. Test against a numerical Legendre transform of $\ln M_K(s)=\ln\frac{K+1}{K+1-s}+\frac{Ks}{K+1-s}$.
- **Exponent bounds:** $P\le e^{-N\mathcal R_K(a)}/\mathcal Z_{K,N}$, $a=\eta^\*/(\eta_\ell\bar\eta_\Sigma)$.
- **Design law:** $D_{N,K}=N(K+1)^2/(1+2K)\ge2\ln\!\big(1/(\epsilon\mathcal Z_{K,N})\big)/(1-a)^2$.
- **$N_{\rm req}$:** smallest integer $N$ with $F_N(\eta^\*)\le\epsilon$, found by exponential search then integer bisection on the exact cdf. Reference table in §11 **[V]**.
- **MC:** MRC uses `sample_sum`; SC uses `sample_elements` (§6.1).

## 10. Validation plan (maps to FSD NFR-1)

| Layer | Tests |
|---|---|
| Unit | every function: shapes, dtypes, domain errors, edge values ($\eta\in\{0,1\}$, $K=0$, $N=1$, $T=0$) |
| Identity | §5.2 general vs closed form; §5.4 condition vs $\tilde\nu_-$ grid; §4.2 associativity; §4.3 closed form; §8.1 threshold vs direct root of $\tilde\nu_-(\eta)=1$; `sample_sum` vs `sample_elements` for $N\le64$ (two-sample KS, $\alpha=10^{-3}$) |
| Limits | small-scale $K=0$ and $K\to\infty$; large-scale $\sigma_G\to0$; compound reduces to each factor; §8.1 limiting-case table (both large-scale laws); Goubau → Friis for $\tau^2\to0$ |
| Cross-method | analytic/quadrature vs MC per §10.1; quadrature vs `mpmath` at 3 points per function |
| Property (`hypothesis`) | pdfs ≥ 0 and integrate to 1; cdfs monotone in $[0,1]$; capacities monotone in $\eta$ and $N_s$; $C_\chi\ge C_{\rm het},C_{\rm hom}$; $F_{\rm opt}\ge F_{\rm BK}$; $\bar\eta_\Sigma\le1$ without clipping for `goubau` |
| Regression | §11 reference values; figure CSVs hashed after Abdurrahman approves them; `test_ncx2_floor_tail` (§6.1, pinned scipy guard); §11 $N_{\rm req}$ table through the `ncx2` floor helper |
| Receiver aperture | $N_{\rm amp}$ identical for $G_{\rm enh}\in\{1,10,100\}$ and for any $A_r$ (§7.4); $\tau^2\propto G_{\rm enh}$ under `friis_clipped` below 1; `goubau` $\eta_{\rm geo}<1$ and increasing in $G_{\rm enh}$; $G_{\rm enh}<1$ raises |
| Moments window | $K=0$, $N=1$, $\bar\eta=0.01$: truncated-exponential moments $m\in\{1,2\}$ match the closed form to `rtol=1e-10` (window must widen past 20 sd); a forced `nan` in the window raises |
| Throughput map | §8.4 power-limited asymptote to `rtol=1e-3` at $\eta N_s/D<10^{-4}$ (deterministic $\eta$); equal $B$ and $D_{\rm RAR}\ge D_{\rm LNA}$ ⇒ $\rho\le0$ everywhere |
| Receiver physics | T-LIN-1..11 (§14.11): weak-probe vs full Lindblad, analytic limits, optional QuTiP cross-check |

Tolerances live in `tests/tolerances.py`. Changing a tolerance requires an entry in `QUESTIONS.md`.

### 10.1 MC agreement rule (`numerics/mcagree.py`)
Every MC comparison uses this rule and never a bare "within 3σ":
- **Seed and size:** fixed seed, recorded in test metadata; default $10^6$ samples, chunked (§13 rule 7).
- **Pointwise check:** at each of the $m$ evaluated points, $|\hat p-p|\le z_m\hat\sigma$ with $z_m=\Phi^{-1}(1-\alpha/(2m))$ (Bonferroni), family-wise $\alpha=10^{-3}$. For $m=100$, $z_m\approx4.4$.
- **Standard error:** $\hat\sigma$ is the binomial standard error for probabilities and the sample standard error for moments.
- **Whole-distribution alternative:** a one-sample KS test against the analytic cdf at $\alpha=10^{-3}$ may replace the pointwise check.

## 11. Reference values

Verified while writing this spec (Python, `scipy.constants`), $T=300$ K:

| $f$ (GHz) | $\bar n_{\rm th}$ | $\eta^\*$ | $1-\eta^\*$ | $S_{E,\rm BBR}$ (nV/cm/√Hz) | DSB floor (dB) |
|---|---|---|---|---|---|
| 5 | 1249.697 | 0.99920 | 7.996e-4 | 0.603 | 3.009 |
| 30 | 207.867 | 0.99521 | 4.788e-3 | 3.614 | 3.000 |
| 100 | 62.011 | 0.98413 | 1.587e-2 | 12.012 | 2.976 |
| 300 | 20.341 | 0.95314 | 4.686e-2 | 35.748 | 2.907 |

Other verified values:

| Quantity | Value |
|---|---|
| Absorption-only range (5 GHz) | 555.47 m |
| Goubau coupling, 1 m² apertures, 10 m, 5 GHz | 0.93806 |
| $\eta^\*_{\rm rx}(N_{\rm add}=\frac12)$ | 0.99960 |
| Transducer threshold | 0.999219 |
| RAR $N_{\rm amp}$, 10 nV/cm/√Hz at 5 GHz | 3.438e5 (rms), 1.719e5 (peak) |
| V0 log-normal mass above 1 ($\sigma_\Psi=\bar\Psi$) | 0.158655 |
| $\mathcal Z_K$ | values in §6.1 |
| Log-gamma mean ($k=4$, $\theta=0.01$) | 0.96098 |
| TMST thresholds | values in §5.4 |
| $F_{\rm opt}$, $F_{\rm BK}$ | values in §5.3 |
| $T_n$ conventions | values in §7.3 |
| Crossover values | values in §8.3 |

**Surrogate tails at $z=\mp5.5$ sd** (central $\chi^2_{2N}$, cdf at $2N-5.5\sqrt{4N}$ and sf at $2N+5.5\sqrt{4N}$; reference values for `test_ncx2_floor_tail`) **[V]**. Lower-tail reference: series for $P(a,x)$; upper-tail reference: Legendre continued fraction for $Q(a,x)$; both with an `mpmath` prefactor, and both match `mpmath.gammainc` at $N=10^5$. Only the lower tail is affected; `chi2.sf` is accurate (< 1e-14) at all four $N$.

| $N$ | cdf (lower) | `chi2` rel. error | `ncx2` floor rel. error | sf (upper) | `ncx2` floor rel. error |
|---|---|---|---|---|---|
| $10^5$ | 1.590617e-8 | 3e-14 | 3e-14 | 2.256714e-8 | 3e-15 |
| $10^6$ | 1.796389e-8 | 1.6e-6 | 2e-13 | 2.006462e-8 | 2e-14 |
| $10^7$ | 1.865997e-8 | 2.3e-2 | 5e-13 | 1.932410e-8 | 8e-14 |
| $10^8$ | 1.888480e-8 | 3.2e-1 | 1.2e-12 | 1.909481e-8 | 2.3e-13 |

**$N_{\rm req}$ (verified, exact integers; consistent with V0 Table at its displayed precision)** for $F_N(\eta^\*)\le10^{-3}$ under passive MRC, $\eta_\ell=1$, $\bar\eta_\Sigma=\eta^\*+0.9(1-\eta^\*)$. Re-verified in v0.4.1 through the `ncx2` floor helper: every integer unchanged, since the operating point ($10^{-3}$, about 3 sd) is far from where `chi2` degrades **[V]**. Tolerance ±0.1%:

| $f$ (GHz) | $K=0$ | $K=10$ | $K=100$ |
|---|---|---|---|
| 5 | 20 017 832 | 3 474 510 | 394 473 |
| 30 | 556 629 | 96 662 | 10 975 |
| 100 | 50 243 | 8 737 | 993 |
| 300 | 5 630 | 983 | 112 |

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
  gamma_abs_per_m: {5: 1.524e-6, 30: 4.394e-6, 100: 6.641e-6, 300: 5.088e-6}   # O2 only, §16.1
  gamma_abs_source: "ITU-R P.676-12 Annex 1, 1013.25 hPa, 300 K (itur 0.4.0)"
  stochastic_per_m:                         # water vapour + rain, KHU Seoul, §16.2
    5:   {mean: 4.160e-7, std: 9.028e-7}
    30:  {mean: 2.839e-5, std: 6.760e-5}
    100: {mean: 1.509e-4, std: 2.049e-4}
    300: {mean: 1.523e-3, std: 1.101e-3}
  stochastic_source: "ITU-R P.836 + P.676-12 (vapour), P.837-7 + P.838-3 (rain), annual, 37.597N 127.052E"
link_budget:
  coupling_model: goubau     # goubau | friis_clipped (baseline, physical=False)
  A_t_m2: 2.0                # [A] near-field apertures; a 30 dBi horn to a bare dipole gives outage = 1 everywhere
  A_r_m2: 2.0                # [A] antenna-fed receiver (about 38 dBi dish at 5 GHz feeding the RAR/LNA), not a bare cell
  G_enh: 1.0                 # receive side only; keep 1 when A_r is an antenna aperture (§6.5)
  sensitivity_models: [goubau, friis_clipped]   # E3 reports both
source_state: {type: tmsv, r: 1.0}
front_end: {type: identity}
mc: {samples: 1000000, alpha_familywise: 1.0e-3}
outputs: {figure: true, csv: true, latex_table: false}
```
Receivers in E5/E6/E8 declare `sideband` and, for the RAR, `field_convention`; electronic receivers declare `tn_convention`.

E2 adds `law_comparison: {match: psi_moments, eta_gt1: clip, sigma_ratio: [...]}` (§6.2). E5 adds the throughput-map block (§8.4):
```yaml
throughput_map:
  P_t_w: 0.2                 # [A] design choice (23 dBm, 3GPP power class 3); equal for both receivers
  f_ghz: {start: 5, stop: 300, num: 40, scale: log}
  distance_m: {start: 10, stop: 1000, num: 40, scale: log}
  K: [0, 10, 100]
  rar: {source: literature_rar, require_b_hz: true, sideband: from_entry}   # §16.3
  lna: {bands: [...], B_hz: [...], tn_convention: rj, sideband: ssb}        # §16.4
  interpolate: loglog        # no extrapolation; out-of-range cells are masked
  equal_bandwidth_case: true # also run with B_LNA := B_RAR per cell (separates noise gap from bandwidth gap)
```

`configs/literature_rar.yaml` entries (FR-EXP-7; full sourced list in §16.3):
```yaml
- label: "Borowka 2024 MW-to-optical converter (13.9 GHz)"
  f_hz: 13.9e9
  s_e: 3.98e-7                # V m^-1 Hz^-1/2 (= 3.98 nV/cm/rtHz)
  field_convention: rms       # rms | peak  (required); unstated in paper -> peak (§7.4)
  includes_bbr: true
  sideband: ssb               # ssb | dsb (required)
  b_hz: 16.0e6                # optional; instantaneous bandwidth, needed by E5 (§8.4)
  source: "S. Borowka et al., Nat. Photon. (2024), doi:10.1038/s41566-023-01295-w"   # required
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
  mw: {d34_cm: null, e_lo_pk_v_per_m: null}
  decays: []                 # list of {from, to, rate_per_s, source}; default 4->1, see §14.3
  gamma_t_per_s: null        # [A] transit rate
  detector: {eta_d: null, nep_w_per_rthz: 0.0}
  doppler: {n_sigma: 5, n_v_start: 4097, rtol: 1.0e-5, n_v_max: 1048577}
  sweep: {omega_c_over_2pi_mhz: [], e_lo_pk_v_per_m: []}
```
Every numeric literature value carries `source:`; placeholders carry `# [A]` and are listed in `QUESTIONS.md`.

CLI: `qmw run configs/e3.yaml [--force]`, `qmw list`, `make test`, `make lint`, `make exp ID=E3`, `make figs`, `make paper-figs`.
Outputs: `results/<experiment>/<config_hash>/{fig.pdf, fig.png, data.csv, meta.json}`.

## 13. Numerical rules (binding)

1. Never compute $1-F$ when $F\approx1$; use survival functions (`sf`, `gammainc`/`gammaincc` complements).
2. Bessel terms: `scipy.special.ive` (scaled) in log-domain pdfs; prefer `scipy.stats.ncx2` methods over hand-written Marcum-Q.
3. Work with $1-\eta$ (`one_minus_eta`) as a primary variable wherever $\eta\to1$ matters (thresholds, quantiles, windows of width ~1e-3).
4. `float64` everywhere; no `float32`.
5. Root finding: `brentq` with explicit brackets and `xtol≤1e-14`; assert sign change, never fall back silently.
6. Quadrature failures raise or warn; never return an unchecked result.
7. MC: vectorized, chunked (≤10⁶ per chunk), explicit `Generator`, report standard error with every MC estimate; compare per §10.1.
8. Stable algebra: `expm1`/`log1p` for thermal quantities; the §5.2 stable $\tilde\nu_-$ form only.
9. No silent clamping of physical quantities (transmissivities, $N_{\rm amp}$); out-of-domain inputs raise. Sanctioned exceptions exist only inside flagged baselines (`friis_clipped`, §6.5; V0 log-normal `eta_gt1: clip`, §6.2).
10. Regularized incomplete gamma / $\chi^2$ cdfs with shape $>10^5$ (dof $>2\times10^5$) go through the `ncx2` floor helper (§6.1), never through `chi2`, `gamma`, `gammainc` or `gammaincc` directly. Never mask a non-finite pdf value to 0; raise.

## 14. Rydberg receiver physics model (pure Python)

**Purpose.** Replace the phenomenological $S_E\to N_{\rm amp}$ input of §7.4 with a physics model, and map where the linear (Gaussian) receiver model is valid (IDEA risk R4, experiment E9). Everything here is NumPy/SciPy; no compiled extension is needed for this paper (cost budget in §14.12).

### 14.1 Physical model and scope
Room-temperature alkali vapor (default ⁸⁷Rb), ladder scheme, $N_{\rm lev}\in\{3,4\}$ levels:
- $|1\rangle$ ground ($5S_{1/2}$);
- $|2\rangle$ intermediate ($5P_{3/2}$, probe ≈780.24 nm);
- $|3\rangle$ Rydberg (coupling ≈480 nm, typically $nD_{5/2}$);
- $|4\rangle$ adjacent Rydberg (MW transition $|3\rangle\leftrightarrow|4\rangle$, typically $(n\pm1)P$; $N_{\rm lev}=4$ only).

**In scope:** RWA Hamiltonian, Lindblad decay, transit relaxation, 1D Doppler averaging (counter-propagating probe/coupling), weak-probe Beer–Lambert readout, steady state, small-signal transfer function, photon-shot-noise-limited sensitivity, linear range.

**Out of scope (state in paper):** hyperfine/Zeeman substructure, optical propagation with $\Omega_p(z)$, Rydberg–Rydberg interactions, BBR-induced Rydberg transitions beyond a phenomenological decay rate, atom projection noise (optional extension, §14.8), time-domain waveform simulation.

**Sideband note:** the superheterodyne readout of §14.7 responds to the field envelope, so it is double-sideband; the receiver built from this model has `sideband="dsb"` (§7.1).

### 14.2 Units (binding inside the physics module)
- Time in µs, angular frequencies in rad/µs (i.e. $2\pi\times$MHz), wavevectors in rad/m, velocities in m/s.
- Doppler shift $kv\cdot10^{-6}$ rad/µs.
- SI conversions happen only in `model.py` (§14.10).

### 14.3 Hamiltonian and dissipators [V]
With $\hbar=1$ and level $|j\rangle$ indexed from 1:
$$H=-\Delta_p|2\rangle\langle2|-(\Delta_p+\Delta_c)|3\rangle\langle3|-(\Delta_p+\Delta_c+\Delta_m)|4\rangle\langle4|+\tfrac12\big(\Omega_p|2\rangle\langle1|+\Omega_c|3\rangle\langle2|+\Omega_m|4\rangle\langle3|+\text{h.c.}\big).$$
Rabi frequencies real and $\ge0$, defined with **peak** fields. Master equation $\dot\rho=-i[H,\rho]+\sum_kD[L_k]\rho$, $D[L]\rho=L\rho L^\dagger-\tfrac12\{L^\dagger L,\rho\}$, with default collapse operators:
- $\sqrt{\Gamma_2}\,|1\rangle\langle2|$, with $\Gamma_2=2\pi\times6.0666$ MHz **[L]** (Steck, *Rubidium 87 D Line Data*, rev. 2.3; cite the revision used). The §14 [V] numbers were measured with 6.065 MHz; re-run them (expected change ≈0.03%, within all tolerances).
- $\sqrt{\Gamma_3}\,|2\rangle\langle3|$ ($nD\to5P$ is dipole-allowed) and $\sqrt{\Gamma_4}\,|1\rangle\langle4|$, with Rydberg rates from config (e.g. via ARC) **[A]**.
  - The $|4\rangle$ default is a phenomenological cascade to ground: a direct $nP\to5P_{3/2}$ decay is parity-forbidden, so $|2\rangle\langle4|$ must not be the default.
  - The decay target of each level is configurable as a list of `(from, to, rate)`.
  - In the weak-probe model only $\gamma_{41}$ enters, so the target matters only for the full model.
- Transit relaxation $\sqrt{\gamma_t}\,|1\rangle\langle k|$ for all $k$, which is exactly the reset map $\dot\rho=\gamma_t(|1\rangle\langle1|-\rho)$ **[D][V]**.
- Optional laser dephasing $\sqrt{2\gamma_{\phi,k}}\,|k\rangle\langle k|$ for $k\ge2$ (ground not dephased).

Coherence decay rates (used by the weak-probe model and tests): $\gamma_{j1}=\tfrac12\Gamma_j^{\rm out}+\gamma_t+\gamma_{\phi,j}$, where $\Gamma_j^{\rm out}$ is the total decay rate out of level $j$. Defaults: $\gamma_{21}=\Gamma_2/2+\gamma_t$, $\gamma_{31}=\Gamma_3/2+\gamma_t$, $\gamma_{41}=\Gamma_4/2+\gamma_t$.

### 14.4 Weak-probe model (primary engine) [D][V]
For $\Omega_p\ll\Gamma_2$, $\rho_{11}\approx1$ and only the coherences $x=(\rho_{21},\rho_{31},\rho_{41})^\top$ matter. With per-velocity detunings $d_2=\Delta_p-k_pv$, $d_3=d_2+\Delta_c+k_cv$, $d_4=d_3+\Delta_m$:
$$M=\begin{pmatrix}d_2+i\gamma_{21}&-\Omega_c/2&0\\-\Omega_c/2&d_3+i\gamma_{31}&-\Omega_m/2\\0&-\Omega_m/2&d_4+i\gamma_{41}\end{pmatrix},\qquad s=\begin{pmatrix}\Omega_p/2\\0\\0\end{pmatrix}.$$
- **Steady state:** $Mx_0=s$, i.e. the continued fraction
$$\rho_{21}=\cfrac{\Omega_p/2}{d_2+i\gamma_{21}-\cfrac{\Omega_c^2/4}{d_3+i\gamma_{31}-\cfrac{\Omega_m^2/4}{d_4+i\gamma_{41}}}}.$$
  $N_{\rm lev}=3$: drop the last row/column ($\Omega_m=0$).
- **Dynamics:** $\dot x=i(Mx-s)$ (stable: eigenvalues of $iM$ have negative real part).
- **Small-signal transfer function in $\Omega_m$:** with $M_1=\partial M/\partial\Omega_m$ (entries $(2,3),(3,2)=-\tfrac12$),
$$H(i\omega)=e_1^\top\,(i\omega I-iM)^{-1}\,(iM_1x_0),\qquad H(0)=-e_1^\top M^{-1}M_1x_0.$$
- Vectorize over velocity classes (closed-form tridiagonal solve or the continued fraction directly); no Python loops over $v$.
- **Verified:** against the full Lindblad model (§14.5) at $\Omega_p/2\pi=0.01$ MHz, $\Omega_c/2\pi=5$, $\Omega_m/2\pi=20$ MHz: steady state and $H(i\omega)$ for $\omega/2\pi\in\{0,0.3,1,3\}$ MHz agree to ≤ 8e-5 relative at $v=0$ and ≤ 1e-8 at $v=100$ m/s.
- **Validity (measured, Doppler-averaged, same $\Omega_c,\Omega_m$):** relative error in $\langle\rho_{21}\rangle$ of 4e-6 at $\Omega_p/2\pi=0.01$ MHz, 3.5% at 1 MHz, 65% at 5 MHz. The error grows as $\Omega_p^2$; the weak-probe model must not be used for sensitivity optimization at strong probe without the §14.5 check.

### 14.5 Full Lindblad model (strong-probe check)
Row-major vectorization $\mathrm{vec}(AXB)=(A\otimes B^\top)\mathrm{vec}(X)$:
$$\mathcal L=-i(H\otimes I-I\otimes H^\top)+\sum_k\Big(L_k\otimes L_k^\*-\tfrac12L_k^\dagger L_k\otimes I-\tfrac12I\otimes(L_k^\dagger L_k)^\top\Big),$$
size $N_{\rm lev}^2\times N_{\rm lev}^2$ (9 or 16).
- **Affine in velocity:** $\mathcal L(v)=\mathcal A+v\,\mathcal B$, where $\mathcal B$ comes from $H_v=10^{-6}\big(k_p|2\rangle\langle2|+(k_p-k_c)(|3\rangle\langle3|+|4\rangle\langle4|)\big)$. Build $\mathcal A,\mathcal B$ once per parameter point and form the batch by broadcasting.
- **Steady state:** replace row 0 of $\mathcal L$ (and of $\mathcal B$) by $\mathrm{vec}(I)^\top$ (resp. 0), right-hand side $e_0$, batched `numpy.linalg.solve` over $v$.
- **Transfer function:** $\mathcal L=\mathcal L_0+\delta\Omega_m\mathcal L_1$; $\delta\rho(s)=(sI-\mathcal L_0)^{-1}\mathcal L_1\rho_0$; at $s=0$ use the bordered system (trace row, zero right-hand side entry).
- **Do not implement** the generalized-Schur (QZ) per-velocity trick: it is accurate (max relative error 2e-14) but slower in NumPy (≈8 µs vs ≈5 µs per class) because the triangular solve runs as a Python loop **[V]**.
- Uses:
  - the oracle for weak-probe tests;
  - strong-probe points in E9 (sensitivity vs $\Omega_p$);
  - the automatic validity check (§14.6).

### 14.6 Doppler averaging and method selection
- Velocity $v$ along the probe; counter-propagating coupling; MW Doppler neglected. Maxwell–Boltzmann weights, $\sigma_v=\sqrt{k_BT_{\rm vap}/m}$ (169.4 m/s for ⁸⁷Rb at 300 K **[V]**).
- **Grid:**
  - Uniform on $[-5\sigma_v,5\sigma_v]$ with normalized weights.
  - Start at $N_v=2^{12}+1$ and double until the relative change of $\langle\rho_{21}\rangle$ is below `rtol_doppler` (default 1e-5); record the final $N_v$.
  - Features are narrow in $v$ because of the residual shift $(k_c-k_p)v$: 4001 vs 8001 points still differ by 0.25% in a test case **[V]**.
- `method` per config: `weak_probe` | `full` | `auto` (default).
  - `auto` runs `weak_probe` for the sweep, then evaluates `full` at ≥ 5 checkpoints (the largest-$\Omega_p$ point plus 4 seeded random points).
  - It records the max relative discrepancy in run metadata.
  - If that exceeds `weak_probe_tol` (default 1%), it recomputes the affected sweep with `full` and logs a warning.

### 14.7 Readout and small-signal response
- **Absorption [D][V]:** probe intensity absorption coefficient $\alpha=\dfrac{2k_pN_0d_{12}^2}{\epsilon_0\hbar\,\Omega_p}\big(-\mathrm{Im}\langle\rho_{21}\rangle\big)$, $\rho_{21}=\langle2|\rho|1\rangle$.
  - Check: with $\gamma_t=0$, the weak-probe two-level resonance gives $\alpha/N_0=3\lambda_p^2/2\pi$ when $d_{12}^2=3\pi\epsilon_0\hbar\Gamma_2/k_p^3$ (closed two-level cycling transition). Exact to 1e-16 **[V]**.
- **Transmission** (weak probe, thin uniform cell): $P_{\rm out}=P_{\rm in}e^{-\alpha L}$.
- **Superheterodyne:** LO field $E_{\rm LO}$ plus signal $E_{\rm sig}e^{i\delta t}$ gives $\Omega_m(t)\approx\Omega_{\rm LO}+\Omega_{\rm sig}\cos(\delta t+\phi)$ for $E_{\rm sig}\ll E_{\rm LO}$, with $\Omega=d_{34}E_{\rm pk}/\hbar$. Because the response is to the envelope, signals at $\pm\delta$ are indistinguishable (DSB, §7.1).
- **Slope:** $\kappa\equiv\partial P_{\rm out}/\partial E_{\rm sig}=-P_{\rm out}L\,\dfrac{\partial\alpha}{\partial\Omega_m}\dfrac{d_{34}}{\hbar}$, with $\partial\alpha/\partial\Omega_m$ from the Doppler-averaged $H(0)$ (§14.4 or §14.5). Since κ is linear, it is the same per peak or per rms field.
- **Instantaneous bandwidth $B_{3\rm dB}$:** the highest $\omega/2\pi$ with $|H(i\omega)|\ge|H(0)|/\sqrt2$, found on a log grid then refined by `brentq`. $|H|$ can peak above $|H(0)|$ near the EIT dynamics (seen at ~1 MHz in a test case **[V]**); report the peaking value too.

### 14.8 Sensitivity and $N_{\rm amp}$ [D]
Single-sided optical-power noise PSD at the detector: $S_P=2h\nu_pP_{\rm out}/\eta_d+\mathrm{NEP}^2$. The noise-equivalent field is
$$S_E=\frac{\sqrt{S_P}}{|\kappa|}\qquad(\textbf{rms}\text{, single-sided; consistent with §2 and §7.4}).$$
Derivation: the output at $\delta$ has rms amplitude $|\kappa|E_{\rm sig,rms}$; SNR = 1 in 1 Hz gives $\kappa^2E_{\rm rms}^2=S_P$. Do **not** insert a factor $\sqrt2$.

Then $N_{\rm amp}$ from §7.4 (with `field_convention="rms"`, `sideband="dsb"`). **[A]** Cross-check against one published PSN-limited sensitivity (e.g. Jing et al., Nat. Phys. 2020, or Cox et al., PRL 2018), noting whether that paper quotes rms or peak fields, and record it in `docs/RESULTS.md`. Atom projection noise is an optional term added in quadrature, with a formula supplied by Abdurrahman; not implemented until then.

### 14.9 Linear range [D]
- **Validity:** quasi-static, valid only if $\delta\ll B_{3\rm dB}$ (assert).
- **First-harmonic amplitude:** $A_1(\Omega_{\rm sig})=\frac1\pi\int_0^{2\pi}P_{\rm ss}(\Omega_{\rm LO}+\Omega_{\rm sig}\cos\theta)\cos\theta\,d\theta$ (Gauss–Legendre, 64 nodes).
- **Compression:** $c=A_1/(\kappa_\Omega\,\Omega_{\rm sig})$, with $\kappa_\Omega=\partial P_{\rm out}/\partial\Omega_m$ at $\Omega_{\rm LO}$.
- **$E_{1\rm dB}$:** the smallest signal field where $|20\log_{10}c|=1$ dB, covering both compression ($c<1$) and expansion ($c>1$, possible near EIT features). Found with `brentq` on $\log\Omega_{\rm sig}$; record which case occurred.
- **Output:** $E_{1\rm dB,pk}=\hbar\Omega_{\rm sig}/d_{34}$ is a **peak** field and is passed to §7.4 as `e_1db_pk_v_per_m`.

### 14.10 Module layout
```
src/qmwlink/receivers/rydberg_physics/
├── scheme.py      # level scheme, decay channels, coherence rates      §14.3
├── weak_probe.py  # M, s, steady state, dynamics, transfer function    §14.4
├── lindblad.py    # Liouvillian A + vB, batched steady state, transfer §14.5
├── doppler.py     # grids, weights, convergence loop                    §14.6
├── response.py    # Doppler-averaged H(iω), bandwidth, peaking          §14.7
├── readout.py     # alpha, P_out, kappa, S_E, E_1dB                     §14.7–14.9
└── model.py       # LindbladRARModel: SI in -> (kappa, S_E_rms, N_amp, B_3dB, E_1dB_pk, method, discrepancy)
```
- `model.py` converts SI inputs to core units and back:
  - probe/coupling power and waist → Rabi frequency, via the peak field $E_{\rm pk}=\sqrt{2I/c\epsilon_0}$ with peak intensity $I=2P/\pi w_0^2$;
  - $N_0$; $d_{12}$, $d_{34}$; $L$; $T_{\rm vap}$; $E_{\rm LO,pk}$.
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
| T-LIN-12 | Readout units: $S_E$ from §14.8 is rms; $E_{1\rm dB}$ from §14.9 is peak; the §7.4 linear-range flag compares peak with peak | unit test on a hand case |

### 14.12 Cost budget (measured on a 2-core cloud machine, single-threaded BLAS) [V]
| Method | Cost per velocity class |
|---|---|
| Weak-probe closed form, vectorized | ≈ 20–25 ns |
| Full Lindblad, batched `solve` with $\mathcal A+v\mathcal B$ | ≈ 5–10 µs |
| Full Lindblad, naive (Python loop building $\mathcal L$) | ≈ 45 µs (**do not do this**) |

Implications for E9 with $N_v\approx3\times10^4$:
- A weak-probe sweep of $10^4$ parameter points costs ≈ 10 s.
- A full-model curve of 20 $\Omega_p$ values (steady state + derivative) costs ≈ 20 s.
- The `auto` checkpoints add seconds.

E9 therefore fits NFR-3 (< 5 min per figure). These per-class costs are recorded benchmarks, not CI gates (FSD FR-LIN-2/3). No compiled code (Numba, Cython, C++) is needed; adding any requires a `QUESTIONS.md` entry with a profile showing the hotspot.

## 15. Changelog

**0.4.3 (2026-11-10)**, paired with FSD v0.6.3. Resolves QUESTIONS Q-7, Q-9, Q-10, Q-11.
- §6.1 (Q-9): `ncx2.sf` upper tail supported for $K\le10^9$; the warning is an error in tests.
- §6.2: per-metre, fully correlated parameterization `stochastic_per_m` ($k$ distance-independent, $\theta\propto d$); per-link form kept for tests and sweeps only.
- §7.4: the aperture/field convention cross-check against Santamaría Botello is resolved; unstated literature NEFs are peak.
- §8.3 (Q-10): capacity-based crossover returns the largest tolerable $N_{\rm amp}$.
- §12: config examples carry the sourced values; `b_hz` added to `literature_rar.yaml`; E5 adds `equal_bandwidth_case`.
- §16 (new; Q-7, Q-11): sourced reference inputs (ITU absorption, Seoul weather statistics, published RAR points, LNAs, channel bandwidths, transducers, aperture reach), with the items still [A]. `scripts/derive_inputs.py` and optional `itur` added in §1.

**0.4.2 (2026-11-10)**, paired with FSD v0.6.2. Resolves QUESTIONS Q-8.
- §8.1: averages of $\tilde\nu_-$, $F_{\rm opt}$, $F_{\rm BK}$ are conditional on the link working ($\eta>\eta^\*_{\rm rx}$), normalized by $1-P_{\rm out}$; `NaN` when no realization is entangled; always reported next to $1-P_{\rm out}$. $\mathbb E[E_N]$ stays unconditional, with an optional conditional variant.

**0.4.1 (2026-11-10)**, paired with FSD v0.6.1. Resolves QUESTIONS Q-3, Q-4, Q-5 and adds the RQ1/RQ2/RQ5 outputs.
- **Numerics:**
  - §6.1 (Q-3): every surrogate cdf/sf, including $K=0$, goes through the `ncx2` floor helper ($\lambda\ge10^{-300}$); `chi2`/`gamma`/`gammainc` lose up to 32% relative accuracy at $N=10^8$ in the lower tail **[V]**. Pinned regression `test_ncx2_floor_tail` added; the `gamma(N, scale=2σ²)` alternative removed. §6.2 and §13 rule 10 extend this to large-shape log-gamma.
  - §6.1 (Q-5): moment integration uses an adaptive window with a neglected-mass check ($\le10^{-14}\mathcal Z$); the fixed 20 sd window fails for $K=0$, $N=1$, $\bar\eta<0.048$. Non-finite pdf values raise.
  - §11: $N_{\rm req}$ re-verified through the new helper (unchanged); tail reference table added.
- **Link budget (Q-4):** §6.5 single receive-side `G_enh` (default 1, $\ge1$); no transmit factor. $A_r$ defaults to an antenna-fed receiver aperture, coupling assumed lossless **[A]**. §7.4 states the $N_{\rm amp}$ invariance with its derivation and forbids an aperture argument in the mapping.
- **New outputs:** §8.1 truncation effect on outage (RQ1) with `surrogate_cdf`; §6.2 law comparison with moment matching and the `eta_gt1: clip` baseline policy (RQ2); §8.4 RAR-vs-LNA throughput map at equal TX power and shared aperture (RQ5).
- **Validation/config:** §10 rows for the regression guard, aperture invariance, moments window and throughput asymptotes; §12 config comments and E2/E5 blocks; §13 rule 9 lists the sanctioned baseline exceptions.

**0.4 (2026-11-03)**, paired with FSD v0.6.
- **Physics:**
  - DSB/SSB sideband model added (§7.1); $N_{\rm img}$ enters $C_{\rm rx}$ (§8.2) and the crossover (§8.3), with the DSB image floor; the RAR defaults to DSB **[A]**.
  - §7.3 conventions split into `rj` / `planck` (Caves, limit $hf/k_B\ln3$) / `callen_welton`.
  - §7.4 linear-range check is now peak vs peak.
  - §14.8 states explicitly that $S_E$ is rms.
  - §14.9 $E_{1\rm dB}$ covers expansion as well as compression.
  - §14.3 default $|4\rangle$ decay changed to $|1\rangle$ (since $nP\to5P$ is forbidden); $\Gamma_2$ updated to 6.0666 MHz with a revision note.
  - §5.3 adds `fidelity_bk` and labels `fidelity_opt`.
- **Numerics:**
  - §5.2 cancellation-free form mandated for both $\tilde\nu_-$ forms.
  - §6.1 `sample_sum` (MRC, any $N$) vs `sample_elements` (SC, $N\le64$).
  - §6.4 quantile in $1-\eta$.
  - §10.1 MC-agree rule replaces 3σ bands.
  - §13 rules 8–9 added.
- **Link budget:** Goubau is the default, and `friis_clipped` is a flagged baseline (§6.5).
- **Notation:** §2 is now the binding symbol and code-name table ($N_{\rm add}$, $N_{\rm img}$, $N_{\rm lev}$, `_rms`/`_pk` suffixes).
- **Reference values:**
  - §8.1 limiting cases tabulated for both large-scale laws, with the clamp.
  - §11 $N_{\rm req}$ table verified and promoted; new references added (Goubau, DSB floor, conventions, fidelities, TMST thresholds); range 555.47 m.
- **Config:** §12 example now uses near-field apertures and adds `literature_rar.yaml`, `sideband`, `field_convention` and `tn_convention`.

**0.3 (2026-11-03)** — initial version.

## 16. Sourced reference inputs (Q-7, Q-11)

The real-world values behind the configs, compiled 2026-11-10. Tags follow §2: **[L]** literature value as published, **[V]** derived and checked here, **[A]** still a design choice or assumption. Every config value still carries its own `source:`; this section is the reference, not a substitute for it.

### 16.1 Gas absorption (E3, deterministic $\Psi_0$) [V]
ITU-R P.676-12 Annex 1 line-by-line (`itur` 0.4.0), $p=1013.25$ hPa, $T=300$ K (= $T_{\rm env}$). Conversion: $\gamma[1/{\rm m}]=\gamma[{\rm dB/km}]\cdot\ln10/10^4$.

| Band | O₂ only, $\Psi_0$ per m (config) | Standard atm., $\rho=7.5$ g/m³ (dB/km) | Humid, $\rho=20$ g/m³ (dB/km) |
|---|---|---|---|
| 5 GHz | 1.524e-6 | 0.0078 | 0.011 |
| 30 GHz | 4.394e-6 | 0.085 | 0.225 |
| 100 GHz | 6.641e-6 | 0.395 | 1.28 |
| 300 GHz | 5.088e-6 | 4.58 | 15.0 |

- Only O₂ enters $\Psi_0$. Water vapour is weather-dependent, so it goes into $G$ (§16.2).
- Cross-check: Gonzalez-Raya et al. (PRApplied 18, 044002, 2022) use 1.44e-6 /m at 5 GHz (O₂), within 6%. With 1.524e-6 the absorption-only range at $\eta^\*$ is 525 m, against 555.47 m with 1.44e-6 (§11 keeps the latter as a code-path reference).

### 16.2 Weather-driven stochastic loss $G$ (E3) [V]
Annual climatology at Kyung Hee Univ. Seoul campus (37.597 N, 127.052 E):
- **Water vapour:** P.836 surface density (median 7.96 g/m³, 1 % exceedance 22.5 g/m³), mapped through P.676-12 at 300 K. Quantiles integrated over 0.1–99 %.
- **Rain:** P.837-7 rain rate (P₀ = 6.10 %, R₀.₀₁ = 60.6, R₀.₁ = 20.1, R₁ = 3.85 mm/h), mapped through P.838-3 (horizontal path, circular polarization). Exceedance integrated from 0.001 % to P₀.
- Vapour and rain are treated as independent; moments are per metre (§6.2).

| Band | $m_1$ (per m) | $s_1$ (per m) | $k=m_1^2/s_1^2$ | Rain share of variance |
|---|---|---|---|---|
| 5 GHz | 4.160e-7 | 9.028e-7 | 0.21 | 90 % |
| 30 GHz | 2.839e-5 | 6.760e-5 | 0.18 | 95 % |
| 100 GHz | 1.509e-4 | 2.049e-4 | 0.54 | 80 % |
| 300 GHz | 1.523e-3 | 1.101e-3 | 1.91 | 3 % |

- $k<1$ below 300 GHz: the density piles up near $G=0$ (dry most of the time), with a long rain tail.
- These are annual statistics; a single season (monsoon) is worse. Every E3 caption states the location and "annual".
- `scripts/derive_inputs.py` must reproduce both tables to 3 significant figures from the stated ITU recommendations, location and integration ranges.

### 16.3 Published Rydberg receiver points (E5, E6, E8; `literature_rar.yaml`) [L]
$N_{\rm amp}$ and $T_{\rm equiv}=N_{\rm amp}hf/k_B$ are derived [V] with the §7.4 mapping; unstated conventions are read as peak (§7.4 cross-check). "Gap" is a preview against a room-temperature LNA (§16.4) at equal bandwidth, long link ($\eta\to0$), power-limited; E5 is authoritative.

| Entry | $f$ (GHz) | $S_E$ as published | Convention | Bandwidth | $N_{\rm amp}$ | $T_{\rm equiv}$ | Gap |
|---|---|---|---|---|---|---|---|
| Jing 2020, Nat. Phys. 16, 911 | 6.94 | 55 nV/cm/√Hz | peak (unstated) | n/r (EIT ≈ 7.9 MHz bound) | 1.9e6 | 6.5e5 K | 33 dB |
| Cai 2021, arXiv:2111.06582 | 10.68 | 12.5 nV/cm/√Hz | peak (unstated) | n/r | 2.8e4 | 1.4e4 K | 16 dB |
| Borówka 2024, Nat. Photon., doi:10.1038/s41566-023-01295-w | 13.9 | 3.98 nV/cm/√Hz | rms (stated) | 16 MHz | 2.1e3 (BBR removed) | 1.4e3 K | 7 dB |
| Yang 2024, PRApplied 21, L031003 (VERIFY) | 16.03 | 62 nV/cm/√Hz | peak (unstated) | 10.2 MHz | 2.0e5 | 1.5e5 K | 26 dB |
| Legaie 2023, arXiv:2306.17114 | 95.99 | 7.9 µV/m/√Hz | peak (unstated) | 127 kHz | 1.5e3 | 7.0e3 K | 11 dB |
| She 2024, Chin. Phys. Lett. 41, 084201 | 338.7 | 2.88 µV/cm/√Hz | peak (unstated) | n/r | 4.6e4 | 7.5e5 K | ~29 dB (LNA extrapolated) |

- Superheterodyne/LO-based entries are `dsb` (no image rejection). Borówka is a converter, not an LO mixer: `ssb` **[A]**, verify.
- Bandwidth-only references: Yan et al., arXiv:2506.10541 (54.6 MHz at 140.4 nV/cm/√Hz, carrier not stated); Holloway et al., arXiv:1903.10644 (19.6 GHz, atomic response ≈ 3–10 MHz).
- With `require_b_hz: true`, E5's RAR range is 13.9–96 GHz; the other cells are masked.

### 16.4 Room-temperature LNAs and channel bandwidths (E5, E8) [L]

| $f$ anchor | $T_n$ | Source | $N_{\rm amp}=k_BT_n/hf$ |
|---|---|---|---|
| 5, 15 GHz | 57 K (NF 0.78 dB) | Low Noise Factory LNF-LNR1_15B_SV (1–15 GHz, 296 K) | 238 at 5 GHz |
| 22 GHz | 129 K (NF 1.6 dB) | LNF-LNR16_28WA_SV (16–28 GHz) | 122 |
| 30, 60 GHz | 200 K (NF 2.3 dB) | LNF-LNR26_62A_SV (26–62 GHz) | 139 at 30 GHz |
| 96 GHz | 280 K (NF 2.86 dB) | LNF-LNR55_96WA_SV (55–96 GHz) | 61 |
| 243 GHz | 627 K (NF 5.0 dB) | Tessmann et al., Int. J. Microw. Wirel. Technol. 6, 215 (2014), 35 nm mHEMT | 54 |

- No sourced standalone LNA at 300 GHz, so LNA cells above 243 GHz are masked. Full-receiver reference only: Soylu et al., IEEE RFIC 2023, 280 GHz InP HBT receiver, 10.8–11.6 dB **DSB** NF (261–300 GHz).
- System channel bandwidth $B_{\rm LNA}$: 160 MHz at 5 GHz (IEEE 802.11ax); 400 MHz at 30 GHz (3GPP TS 38.101-2 FR2 maximum); 2.16 GHz at 300 GHz (smallest IEEE 802.15.3d channel, 252.72–321.84 GHz; Petrov et al., arXiv:2011.01683); 2.16 GHz at 100 GHz **[A]**.

### 16.5 Coherent front-end classes (E4) [L]
PIChannel (§7.5) with $\bar n_t=0$, $N_t=\eta_tN_{\rm add}$, so $N_r=N_{\rm add}$ exactly. The window depends only on $N_{\rm add}$; $\eta_t$ affects rate only. Thresholds [V] at 5 GHz, 300 K.

| Class | $\eta_t$ | $N_{\rm add}$ (input-referred) | $\eta^\*_{\rm rx}$ | $1-\eta^\*_{\rm rx}$ | Source |
|---|---|---|---|---|---|
| Ideal | 1 | 0 | 0.999200 | 8.0e-4 | — |
| Electro-optic, low noise (pulsed) | 0.087 | 0.16 | 0.999328 | 6.7e-4 | Sahu et al., Nat. Commun. (2022) |
| Electro-optic, high efficiency (pulsed) | 0.15 | 0.41 | 0.999528 | 4.7e-4 | Sahu et al. (2022) |
| Phase-insensitive amplifier bound | — | 0.5 | 0.999600 | 4.0e-4 | Caves, PRD 26, 1817 (1982) |
| Piezo-optomechanical, low noise (CW, 5.07 GHz) | 0.0047 | 0.58 | 0.999664 | 3.4e-4 | Zhao et al., Nat. Nanotechnol. (2025); arXiv:2406.02704 |
| Rydberg mm-wave→optical (99.4 GHz, 5 K, internal η) | 0.58 | 0.60 | 0.999680 | 3.2e-4 | Kumar et al., Nature 615, 614 (2023) |
| Piezo-optomechanical, high efficiency | 0.022 | 0.94 | 0.999952 | 4.8e-5 | Zhao et al. (2025) |
| Cryogenic HEMT 1.5 K (outside E4 axis) | — | 6.25 | — | entanglement-breaking | LNF-LNC4_8F_LG datasheet |

### 16.6 Apertures (E3) [A]
$A_t=A_r=2$ m² is a design choice: about a 2.0 m dish at aperture efficiency $e_a=0.65$, the parameterization of Gonzalez-Raya Eq. (66–67). The geometric entanglement reach is $d_{\max}=A/(\lambda\sqrt{\ln(1+\bar n_{\rm th})})$ before absorption:

| Band | 5 GHz | 30 GHz | 100 GHz | 300 GHz |
|---|---|---|---|---|
| $d_{\max}$ (2 m²) | 12.5 m | 86.6 m | 327.7 m | 1144 m |

E3's 10–1000 m grid is therefore in outage almost everywhere at 5 GHz. That is the aperture-limited result, not a defect; a swept-aperture panel is recommended.

### 16.7 Still [A] after this pass
- Apertures (§16.6) and $P_t=0.2$ W (design choices).
- $B_{\rm LNA}$ at 100 GHz.
- A standalone LNA at 300 GHz.
- RAR bandwidths for the Jing, Cai and She entries.
- Borówka's sideband.
- Yang 2024 journal volume (VERIFY).
- Full-path correlation of weather loss (§6.2).