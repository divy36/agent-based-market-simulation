"""Run-level uncertainty and multiple-comparison correction."""

import numpy as np


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(np.where(np.isfinite(values), values, 1.0))
    adjusted = np.full(len(values), np.nan)
    previous = 0.0
    for rank, index in enumerate(order):
        if np.isfinite(values[index]):
            previous = max(previous, min(1.0, values[index] * (len(values) - rank)))
            adjusted[index] = previous
    return adjusted
