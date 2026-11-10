# IDEA — Research Brief

**Working title:** *Quantum-Consistent Stochastic Microwave Channels with Rydberg Atomic Receivers: A Unified Gaussian Framework for Classical and Entanglement Services*

**Owner:** Abdurrahman Wachid Shaffar (CQILAB, Kyung Hee University)
**Status:** v0.1, 2026-10-28. Rewrite of the WMQC stochastic-channel draft (V0). The V0 draft is the author's own unpublished work and is not cited.
**Target venue:** IEEE JSAC Quantum Series: Quantum Communications and Networking (rolling submission; issues in March, July, November). Closest listed topics: *Theoretical Foundations for Quantum Sensing, Communication, Control, and Computing*; *Integration of Quantum Sensing, Communication, Control, and Computing*; *Design, Analysis, and Applications of Quantum Networks*. Page limits follow the general JSAC guidelines.

---

## 1. One-paragraph pitch

Open-air microwave links between superconducting nodes need a channel model that respects quantum mechanics: classical fading laws allow channel gains above one, which a passive bosonic channel forbids. The V0 draft built such a model (passivity-constrained Rician small-scale fading, path-integrated large-scale fading, compound PDT) and found a hard wall: entanglement survives only if the transmissivity exceeds $\eta^\* = \bar n_{\rm th}/(1+\bar n_{\rm th}) \approx 0.9992$ at 5 GHz and 300 K. This paper keeps that channel, fixes its large-scale component, and adds the missing half of the link: **the receiver**. Every receiver (ideal measurement, JPA, HEMT/LNA, Rydberg atomic receiver, coherent transducer) is written as a phase-insensitive Gaussian channel or an input-referred added noise $N_{\rm amp}$. With one language for channel and receiver, two services are analyzed on the same stochastic link: **(S1) classical communication** (Holevo, heterodyne, homodyne, real receivers; ergodic and outage capacity) and **(S2) entanglement distribution** (receiver-referred threshold, outage, negativity). The Rydberg atomic receiver (RAR), the lab's other main topic, becomes a first-class receiver model through an explicit mapping from its measured field sensitivity $S_E$ to photon-number noise.

## 2. Why this is the right next step

- The V0 result $\eta^\*\approx 0.9992$ is strong but bleak: entanglement at 5 GHz outdoors is almost impossible, and the window $1-\eta^\* \approx 8\times10^{-4}$ is narrower than the absorption loss of a 556 m clear-air link (verified: $e^{-\gamma d}=\eta^\*$ at $d\approx 555.5$ m, consistent with the ~550 m range of Gonzalez-Raya et al., PRApplied 2022).
- Classical information over the same quantum channel has **no wall**: capacity degrades smoothly. That makes the channel model useful beyond entanglement and connects it to the lab's RAR work, which is about receiving classical signals.
- The professor's lens is communication theory. The paper speaks that language: equivalent channel models, capacity, outage, design rules. The Gaussian formalism is the translation layer between atomic-physics parameters and channel parameters.

## 3. What carries over, what gets fixed, what is new

| From V0 | Decision |
|---|---|
| Per-realization passivity bound (row normalization of the scattering unitary) | **Keep.** Core of the quantum-consistent model. |
| Joint passivity bound (column normalization), passive MRC theorem, selection-combining failure, outage exponent | **Keep, compress.** Short section or move to a follow-up paper (open question Q3). |
| Small-scale PDT: truncated noncentral $\chi^2_2$ with $\mathcal Z_K$ | **Keep.** Note: at $\bar\eta_s=0.999$, $\mathcal Z_K\approx 0.52$–$0.63$ (verified), so truncation is not a small correction near unity. |
| Large-scale PDT: log-normal from CLT on $\Psi$ | **Fix** (see §5, issue D2). Replace with log-gamma. |
| $\eta^\*$ universality | **Fix wording** (issue D1), then **generalize** to receiver-referred thresholds. |
| Teleportation fidelity $F = 1/(1+\tilde\nu_-)$ | Keep as secondary metric. |
| Frequency table (5 to 300 GHz) | Keep; extend with receiver classes and RAR sensitivity. |

**New in this paper**

1. **Phase-insensitive $(\tau, y)$ language** for every block: $V \mapsto \tau V + y I$. Thermal attenuator $(\eta,(1-\eta)(2\bar n+1))$, additive noise $(1, 2N)$, amplifier $(G,(G-1)(2\bar n+1))$. Composition is $(\tau_2\tau_1,\ \tau_2 y_1 + y_2)$. The whole link is one pair $(\tau, y)$ per fading realization.
2. **Receiver-referred entanglement threshold.** For a TMSV source, the output is entangled iff $y < 1+\tau$ (any $r>0$; verified numerically). For fading $\eta$ followed by additive receiver noise $N$: $\eta^\*_{\rm rx} = (\bar n_{\rm th} + N)/(\bar n_{\rm th}+1)$. **Entanglement is impossible if $N \ge 1$.** V0's "cold combining network" remark becomes a special case. General front end $(\tau_r,y_r)$: $\eta^\*_{\rm rx} = [\tau_r(2\bar n_{\rm th}+1)+y_r-1]/[2\tau_r(\bar n_{\rm th}+1)]$ (verified against direct root-finding). Design insight: a cold transducer's *efficiency* barely moves the threshold (e.g. $\eta_t=0.6$, $\bar n_t=0.01$, $N_t=0.01$ gives 0.99922 vs 0.99920); its *added noise* is what counts.
3. **RAR as a bosonic receiver.** A bare atomic field sensor measuring one polarization component is equivalent to a short dipole with effective aperture $A_e = 3\lambda^2/(8\pi)$. Check: the BBR field density from the Rayleigh–Jeans energy density matches $\bar n_{\rm th}\,hf\,\eta_0/A_e$ (0.603 nV/cm/√Hz at 5 GHz both ways, verified). This gives
   $$N_{\rm amp} = \frac{S_E^2\, A_e}{\eta_0\, h f} = \frac{3 c^2 S_E^2}{8\pi \eta_0 h f^3}.$$
   Example: $S_E = 10$ nV/cm/√Hz at 5 GHz gives $N_{\rm amp}\approx 3.4\times10^5$ ($T_n\approx 8.3\times10^4$ K), versus $\bar n_{\rm th}\approx 1250$. Field-enhancing structures raise the collection efficiency $\eta$ (aperture) but leave this intrinsic $N_{\rm amp}$ unchanged; lossy structures enter as an extra thermal attenuator.
4. **BBR-limited crossover sensitivity.** The RAR becomes channel-limited instead of receiver-limited when $N_{\rm amp}\lesssim \bar n_{\rm th}$, i.e. $S_E \lesssim S_{E,\rm BBR}(f,T) = \sqrt{\bar n_{\rm th} h f \eta_0/A_e}$. Verified values at 300 K: 0.60 (5 GHz), 3.6 (30 GHz), 12.0 (100 GHz), 35.7 (300 GHz) nV/cm/√Hz. In the Rayleigh–Jeans regime $S_{E,\rm BBR} \propto f$, so the sensitivity target relaxes linearly with carrier frequency. This is the headline design map.
5. **Physics-grounded receiver parameters.** A small-signal Lindblad model (4-level ladder, Doppler averaged) turns lab-controllable parameters (laser powers, LO field, vapor temperature) into $S_E$, $N_{\rm amp}$, $B_{3\rm dB}$ and $E_{1\rm dB}$, closing the loop between the atomic-physics side of the lab and the channel model, and marking where the Gaussian receiver abstraction stops being valid.
6. **Two services, one channel.** Ergodic and $\epsilon$-outage capacity for classical signaling (Holevo bound via Giovannetti et al., Nat. Photon. 2014; heterodyne/homodyne; real receivers with bandwidth $B$) alongside entanglement outage and average log-negativity, all over the same compound PDT.

## 4. Key equations (headline set)

- Thermal occupation: $\bar n(f,T) = 1/(e^{hf/k_BT}-1)$.
- Link per realization: $\tau = \tau_r\eta$, $y = \tau_r(1-\eta)(2\bar n_{\rm th}+1) + y_r$.
- Entanglement (TMSV, any $r>0$): entangled iff $y < 1+\tau$.
- Output noise photons (vacuum input): $N_{\rm out} = (\tau + y - 1)/2$.
- Holevo capacity per mode: $C_\chi = g(\tau N_s + N_{\rm out}) - g(N_{\rm out})$, $g(x) = (x+1)\log_2(x+1) - x\log_2 x$.
- Heterodyne: $C_{\rm het} = \log_2\!\big(1 + \tau N_s/(N_{\rm out}+1)\big)$. Homodyne: $C_{\rm hom} = \tfrac12\log_2\!\big(1 + 4\tau N_s/(2N_{\rm out}+1)\big)$.
- Measuring receiver with added noise $N_{\rm amp}\ge \tfrac12$: $C_{\rm rx} = \log_2\!\big(1 + \eta N_s/(N_{\rm out} + \tfrac12 + N_{\rm amp})\big)$, where $N_{\rm out}$ is from the channel alone. $N_{\rm amp} = \tfrac12$ recovers heterodyne.
- Large-scale log-gamma PDT: $\Psi\sim\mathrm{Gamma}(k,\theta)$, $\eta_\ell = e^{-\Psi}\in(0,1]$,
  $$f_\ell(\eta) = \frac{(-\ln\eta)^{k-1}\,\eta^{1/\theta-1}}{\Gamma(k)\,\theta^{k}},\qquad \mathbb E[\eta_\ell^m] = (1+m\theta)^{-k}\le 1.$$

## 5. Issues found in the V0 draft (fix during the rewrite)

- **D1. TMST universality contradiction.** Sec. "Universal Critical Transmissivity" claims $\eta^\*$ is independent of $\bar n_{\rm JPA}$ using $c=\sqrt{a^2-1}$, but for a TMST state the correlation is $c=(2\bar n_{\rm JPA}+1)\sinh 2r$, so $a^2-c^2=(2\bar n_{\rm JPA}+1)^2\neq 1$. The later Remark (threshold depends on $r$ and $\bar n_{\rm JPA}$ for mixed input) is the correct one. Correct statement: the channel is entanglement-breaking iff $\eta\le\eta^\*$ (input-independent); for **pure** TMSV this is also the separability threshold; for TMST, $\eta^\*_{\rm TMST}\ge\eta^\*$ and depends on $(r,\bar n_{\rm JPA})$. General condition: entangled iff $(a-1)(\tau a + y - 1) < \tau c^2$.
- **D2. Log-normal large-scale PDT violates the passivity it argues for.** $\Psi\sim\mathcal N(\bar\Psi,\sigma_\Psi^2)$ puts mass on $\Psi<0$, i.e. $\eta_\ell>1$, and $\mathbb E[\eta_\ell]=e^{-\bar\Psi+\sigma_\Psi^2/2}$ exceeds 1 when $\sigma_\Psi^2>2\bar\Psi$. With $\bar\Psi = \gamma d = 1.44\times10^{-4}$ (100 m) and $\sigma_\Psi=\bar\Psi$, about 16% of the mass sits at $\eta_\ell>1$ (verified). Also, $\delta\gamma\ge-\gamma_{\rm abs}$ with zero mean forces heavy skew, which a Gaussian cannot represent. **Fix:** log-gamma (§4), moment-matched to $(\bar\Psi,\sigma_\Psi^2)$, with positive support by construction. Keep truncated log-normal only as a comparison baseline.
- **D3. Missing large-scale physics.** The intro mentions $\bar\Psi_{\rm obs}$ but the body sets $\bar\Psi=\gamma_{\rm abs}d$ only. Urban shadowing comes from obstacles (buildings, foliage, vehicles). Proposed generator: Poisson number of obstacles along the path (rate $\lambda_o$ per meter), each adding a positive log-loss; moment-match the compound-Poisson sum to the Gamma law. Parameter values need a measurement-based source.
- **D4. Dangling reference** `eq:delayed` (used in the flat-fading section, never defined).
- **D5. Novelty claims** ("first physically derived...") need a fresh literature check before submission.

## 6. Positioning

| Work | Fading | Passivity-consistent | Receiver noise model | Services |
|---|---|---|---|---|
| Vasylyev–Semenov–Vogel (PRL 2012/2016, PRA 2018) | Optical beam wandering | Yes | No thermal noise | Nonclassicality |
| Gonzalez-Raya et al. (PRApplied 2022) | None (deterministic) | n/a | Thermal attenuator | Entanglement, teleportation |
| Classical fading theory (Rappaport, Tse–Viswanath) | Rician / log-normal | **No** | Classical $T_{\rm sys}$ | Classical |
| Gong et al., RAQ-MIMO (arXiv 2506.01355) | Classical fading | No | SQL / photon-shot limits | Classical rate |
| Santamaría Botello et al. (arXiv 2209.00908) | None | n/a | RAR vs LNA noise temperature | None |
| Cox et al. (PRL 2018), Zhu & Dai (arXiv 2506.23455) | None | n/a | SQL capacity / dynamic model with in-band BBR | Classical |
| **This paper** | Rician + log-gamma compound | **Yes** | Unified $(\tau,y)$ / $N_{\rm amp}$, RAR mapping from $S_E$ | **Classical + entanglement** |

## 7. Honest risks (state them in the paper)

- **R1. Quantum vs. classical capacity gap is small at room temperature.** With $\bar n_{\rm th}\gg1$ (1250 at 5 GHz, 20 at 300 GHz), Holevo and heterodyne nearly coincide. Do not sell "quantum capacity advantage". The value is the consistent model, the receiver equivalence, and the design maps.
- **R2. Today's free-space RARs are receiver-limited by orders of magnitude.** Santamaría Botello et al. show room-temperature electronic receivers greatly outperform demonstrated free-space RARs. Our crossover map quantifies the gap; it does not hide it.
- **R3. Bandwidth.** RAR instantaneous bandwidth is MHz-scale vs GHz for electronics. Throughput (bit/s) $= B \times$ bits/mode must be reported, not just bits/mode.
- **R4. Gaussian validity for RAR.** The linear $N_{\rm amp}$ model holds in the weak-signal, linear-response regime. Large signals saturate EIT. A Doppler-averaged receiver physics model (weak-probe closed form plus a full Lindblad check, TSD §14) maps the validity region (E9) and supplies model-derived $S_E$, bandwidth and 1 dB compression field, so receiver numbers no longer rest only on literature inputs.
- **R5. Aperture convention.** The $A_e = 3\lambda^2/8\pi$ equivalence is standard antenna theory but must be stated and cross-checked against the convention in Santamaría Botello et al. before quoting absolute numbers.
- **R6. Detecting receivers cannot carry entanglement.** For S2, the RAR enters only as a coherent MW-to-optical transducer $(\eta_t, \bar n_t)$; parameter values from the transduction literature (e.g., Kumar et al., Nature 2023) must be filled in by the author.

## 8. Proposed paper outline

1. Introduction (problem, gaps, contributions)
2. Gaussian preliminaries in $(\tau,y)$ form (compact)
3. System model: link geometry, two services, receiver classes
4. Quantum-consistent stochastic channel: passivity bounds; small-scale truncated $\chi^2$; large-scale log-gamma; compound PDT
5. Entanglement service: receiver-referred threshold (theorem), outage, average log-negativity, (optional) passive MRC
6. Classical service and Rydberg receivers: $S_E \to N_{\rm amp}$ mapping, ergodic and outage capacity, BBR crossover map, throughput with bandwidth
7. Numerical results (E1–E9 in FSD)
8. Conclusion and outlook (atom projection noise and propagation effects in the receiver model, frequency-selective channels, inhomogeneous temperature)

## 9. Research questions to answer numerically

- RQ1. How much probability mass does the passivity truncation remove near $\bar\eta\to1$, and how much does it change outage?
- RQ2. How different are log-gamma and the (untruncated) V0 log-normal in outage and ergodic capacity?
- RQ3. Which receiver classes keep $\eta^\*_{\rm rx}<1$, and by how much does each shrink the operating window?
- RQ4. At which $S_E$ (per band) is RAR capacity within 1 dB / 3 dB of the BBR-limited (heterodyne) capacity under fading?
- RQ5. With realistic bandwidths, where (frequency, distance, $K$) does an RAR beat an LNA in bit/s, if anywhere?

## 10. Open questions for Abdurrahman

- Q1. ~~Target journal~~ **Resolved (2026-11-03):** IEEE JSAC Quantum Series: Quantum Communications and Networking. Follow-up: the series lists Hyundong Shin as Area Editor for Quantum Sensing and Control; confirm with him which area the manuscript should be routed to under the conflict-of-interest rules (Quantum Communications and Cryptography or Quantum Information Theory look like the natural alternatives).
- Q2. Which RAR sensitivity and bandwidth values does the lab trust per band (5, 30, 100, 300 GHz)?
- Q3. Keep diversity (passive MRC, outage exponent) in this paper or move it to a follow-up?
- Q4. Source for obstacle-shadowing parameters (D3): measurement campaign, ITU-R model, or literature fit?
- Q5. Transducer parameters $(\eta_t,\bar n_t)$ for the S2 analysis.