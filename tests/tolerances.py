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
