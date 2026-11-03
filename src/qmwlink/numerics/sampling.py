"""Chunked rejection sampling with explicit RNG. (TSD §13 rule 7)"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

CHUNK = 1_000_000
"""Maximum number of raw draws per chunk (TSD §13 rule 7)."""
MAX_ROUNDS = 10_000

FloatArray = NDArray[np.float64]


def rejection_sample(
    draw: Callable[[int], FloatArray],
    accept: Callable[[FloatArray], NDArray[np.bool_]],
    size: int,
    accept_rate: float,
) -> FloatArray:
    """Collect ``size`` accepted draws (rows along axis 0) from ``draw`` filtered by ``accept``.

    ``accept_rate`` only sizes the chunks. Raises if acceptance is hopelessly low.
    """
    if size < 0:
        raise ValueError("size must be >= 0")
    kept: list[FloatArray] = []
    have = 0
    for _ in range(MAX_ROUNDS):
        if have >= size:
            break
        need = size - have
        n = int(min(CHUNK, max(16, np.ceil(1.1 * need / max(accept_rate, 1e-12)))))
        x = draw(n)
        x = x[accept(x)]
        kept.append(x)
        have += len(x)
    else:
        raise RuntimeError("rejection sampler did not converge: acceptance rate too low")
    out = np.concatenate(kept, axis=0) if kept else draw(0)
    return out[:size]
