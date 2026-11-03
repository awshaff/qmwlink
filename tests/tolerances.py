"""Frozen tolerances. Changing any value requires a QUESTIONS.md entry (agent.md rule 5)."""

# Reference-table comparisons (TSD §11), relative.
ABS_N_TH_5GHZ = 0.01  # FR-PHY-1 AC: 1249.70 +/- 0.01
ABS_ETA_STAR = 5e-6  # FR-PHY-2 AC: 5 decimals
REL_ONE_MINUS_ETA_STAR = 5e-4  # table gives 4 significant digits
