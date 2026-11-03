# QUESTIONS

## Q-1 [non-blocking] FR-PHY-1 RJ error bound is slightly too tight
- Context: `thermal.n_th_rj`, FR-PHY-1 AC ("RJ relative error < hf/2k_BT"), TSD §3.1.
- Observation: with x = hf/k_BT, (n_RJ - n)/n = (e^x - 1)/x - 1 = x/2 + x^2/6 + ... exactly, which is
  always > x/2. E.g. 3.013 GHz, 1 K (x = 0.1446): relative error 0.0759 vs bound 0.0723.
  The bound holds only to leading order in x.
- Options: A) keep the AC as an asymptotic statement (error -> x/2); B) amend the AC to
  "error < (e^x - 1)/x - 1" or "< x/2 (1 + x/3)" or similar.
- Recommendation: B. Tests currently assert the exact identity and the x -> 0 limit; no tolerance was loosened.
- Status: open
