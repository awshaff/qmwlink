"""Frozen tolerances. Changing any value requires a QUESTIONS.md entry (agent.md rule 5)."""

# --- M0: thermal / BBR ---------------------------------------------------------------------
ABS_N_TH_5GHZ = 0.01  # FR-PHY-1 AC: 1249.70 +/- 0.01
RTOL_RJ_REL_ERR = 1e-9  # FR-PHY-1 AC: RJ relative error vs expm1(x)/x - 1
ABS_RJ_REL_ERR_300GHZ = 5e-6  # FR-PHY-1 spot value 2.438e-2 (4 s.f.)
ABS_ETA_STAR = 5e-6  # FR-PHY-2 AC: 5 decimals
REL_ONE_MINUS_ETA_STAR = 5e-4  # TSD §11 table gives 4 significant digits
RTOL_BBR_ROUTES = 1e-12  # FR-PHY-3 (a): mode-occupation vs energy-density, RJ
MAX_PLANCK_VS_RJ_FIELD = 0.015  # FR-PHY-3 (b): field amplitude, f <= 300 GHz, 300 K
REL_S_E_BBR_TABLE = 2e-3  # TSD §7.4/§11: 0.603/3.614/12.01/35.75 (3-4 s.f.)

# --- M1: Gaussian core ---------------------------------------------------------------------
RTOL_NU_FORMS = 1e-10  # FR-GAU-4 (a),(b): general vs closed form, TMSV r=3
ATOL_FIDELITY_REF = 1e-4  # FR-GAU-4 (d)
BOUNDARY_EXCLUDE_Y = 1e-9  # FR-GAU-5: exclude |y - (1 + tau)| < this
ATOL_TMST_THRESHOLD = 1e-8  # FR-GAU-6 (c)
PHYS_EIG_ATOL = 1e-12  # TSD §5.1 physicality
RTOL_ALGEBRA = 1e-12  # exact-algebra identities (associativity, §4.3, block form)
ATOL_NETWORK_THRESHOLD = 1e-11  # FR-GAU-2 (c): combining-network threshold, direct root vs formula
RTOL_COMPOSITION = 1e-12  # FR-GAU-2 (a),(b): associativity and §4.3 closed form
ATOL_COMPOSITION_Y = 1e-12  # absolute floor on y when y -> 0 (eta -> 1)
RTOL_STATE_BLOCK = 1e-14  # FR-GAU-3: block form after the channel (pure algebra)

# --- M2: stochastic channel ----------------------------------------------------------------
ATOL_PDF_NORM = 1e-8  # FR-CH-1: pdf integrates to 1 +/- 1e-8
MAX_KS_D = 2e-3  # FR-CH-1: MC (1e6) vs cdf KS distance
ATOL_Z_REF = 5e-6  # TSD §6.1 Z_K references (5 s.f.)
ATOL_MEAN_LOGGAMMA = 5e-6  # FR-CH-2: 0.96098
ATOL_MASS_ABOVE_ONE = 1e-6  # FR-CH-3: 0.158655
RTOL_MOMENT_QUAD = 1e-8  # FR-CH-5: E[eta^m] product vs quadrature
RTOL_MPMATH_TAIL = 1e-2  # NFR-2: 1% relative vs mpmath, down to 1e-8
ATOL_ETA_GEO = 1e-5  # FR-CH-6 (c): 0.93806
ATOL_RANGE_M = 0.5  # FR-CH-6 (d): 555.47 m +/- 0.5
MC_SEED = 20261103  # recorded seed for all MC tests (FSD §4.3)
MC_N = 10**6
