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

# --- M3: receivers and entanglement service ------------------------------------------------
ATOL_TN_REF = 1e-4  # FR-RX-3 (b): 0.9102 / 0.5000 / 1.0000
RTOL_N_AMP_REF = 1e-3  # FR-RX-4 (a),(b): 3.438e5 / 1.719e5, +/-0.1%
RTOL_T_N_REF = 1e-3  # FR-RX-4 (a): 8.25e4 K, +/-0.1%
ATOL_ETA_STAR_RX_REF = 5e-6  # FR-SVC-E1: 0.99960 (5 s.f.), 0.999219 (6 s.f.)
RTOL_ETA_STAR_ROOT = 1e-12  # FR-SVC-E1: closed form vs root-finding (additive noise)
RTOL_P_OUT_QUAD = 1e-8  # FR-SVC-E2 (c): Cases 2-3 vs quadrature of the clamped cdf
RTOL_RQ1_IDENTITY = 1e-12  # FR-SVC-E2 (e): P_out / P_out^surr = 1/Z

# --- M6: runner, E1/E2 -----------------------------------------------------------------------
ABS_F_GENERAL_DEGENERATE = 3e-8  # Q-12: general nu_- form has ~2 sqrt(eps) abs error at nu_+ ~ nu_-
RTOL_LAW_LIMIT = 1e-6  # FR-CH-3 (b): laws agree as sigma_Psi -> 0

# --- M7: Rydberg receiver physics (TSD §14.11) -------------------------------------------------
RTOL_TLIN1 = 1e-4  # T-LIN-1: cold 3-level full vs continued fraction
ATOL_TLIN2_EIG = 1e-12  # T-LIN-2: eigenvalues >= -1e-12, trace and Hermiticity
RTOL_TLIN3 = 1e-4  # T-LIN-3: two-level limit
RTOL_TLIN4_SPLIT = 0.03  # T-LIN-4: cold AT splitting ~ Omega_m
RTOL_TLIN5_SPLIT = 0.02  # T-LIN-5: Doppler-averaged AT splitting ~ (lambda_c/lambda_p) Omega_m
RTOL_TLIN6_FD = 1e-6  # T-LIN-6: H(0) vs central finite difference
RTOL_TLIN7 = 2e-4  # T-LIN-7: weak-probe vs full transfer function
RTOL_TLIN9_QUTIP = 1e-8  # T-LIN-9: full vs QuTiP
RTOL_TLIN10 = 1e-4  # T-LIN-10: resonant cross-section 3 lambda^2 / (2 pi)
RTOL_B3DB_DENSE = 1e-3  # FR-LIN-5 AC: refined bandwidth vs dense-grid scan
REL_SIGMA_V_RB87 = 5e-4  # TSD §14.6: 169.4 m/s (4 s.f.)
