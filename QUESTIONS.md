# QUESTIONS

## Q-1 [non-blocking] FR-PHY-1 RJ error bound is slightly too tight
- Context: `thermal.n_th_rj`, FR-PHY-1 AC ("RJ relative error < hf/2k_BT"), TSD §3.1.
- Observation: with x = hf/k_BT, (n_RJ - n)/n = (e^x - 1)/x - 1 = x/2 + x^2/6 + ... exactly, which is
  always > x/2. E.g. 3.013 GHz, 1 K (x = 0.1446): relative error 0.0759 vs bound 0.0723.
  The bound holds only to leading order in x.
- Options: A) keep the AC as an asymptotic statement (error -> x/2); B) amend the AC to
  "error < (e^x - 1)/x - 1" or "< x/2 (1 + x/3)" or similar.
- Recommendation: B. Tests currently assert the exact identity and the x -> 0 limit; no tolerance was loosened.
- Status: resolved — option B chosen by Aan; FR-PHY-1 AC in docs/FSD.md amended to the exact identity (x/2 as x→0).

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
- Status: open
