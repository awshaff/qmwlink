"""Literature RAR points (FR-EXP-7, TSD §12, §16.3).

The schema and loader live with the other config schemas; this module is the IO-layer entry point
named in TSD §1. Peak values are converted to rms on load via ``LiteratureEntry.s_e_rms``.
"""

from __future__ import annotations

from qmwlink.experiments.schema import LiteratureEntry, LiteratureFile, load_literature

__all__ = ["LiteratureEntry", "LiteratureFile", "load_literature"]
