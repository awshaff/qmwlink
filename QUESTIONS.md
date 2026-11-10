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
