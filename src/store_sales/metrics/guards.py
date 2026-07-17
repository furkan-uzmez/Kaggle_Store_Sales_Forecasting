"""Secondary/guardrail metrics (not used for official model selection)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mae_log1p(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error on log1p scale (RMSLE-adjacent guardrail).

    MAE_log1p = mean(|log1p(y_pred) - log1p(y_true)|)
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")
    if yt.size == 0:
        raise ValueError("Empty arrays")
    return float(np.mean(np.abs(np.log1p(yp) - np.log1p(yt))))
