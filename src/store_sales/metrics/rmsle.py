from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rmsle(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared logarithmic error (competition formula).

    RMSLE = sqrt(mean((log1p(y_pred) - log1p(y_true))**2))

    Does not clip negatives; clip predictions in the prediction path if needed.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")
    if yt.size == 0:
        raise ValueError("Empty arrays")
    return float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt)) ** 2)))
