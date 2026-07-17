"""Simple prediction blends (mean / non-negative OOF weights)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


def mean_blend(preds: Sequence[ArrayLike]) -> np.ndarray:
    """Element-wise mean of aligned prediction vectors.

    Parameters
    ----------
    preds:
        Sequence of 1-d arrays with identical shape (OOF or test predictions).

    Returns
    -------
    Averaged predictions as float64 ndarray.
    """
    if not preds:
        raise ValueError("mean_blend requires at least one prediction array")
    arrays = [np.asarray(p, dtype=float).reshape(-1) for p in preds]
    n = arrays[0].shape[0]
    for a in arrays[1:]:
        if a.shape[0] != n:
            raise ValueError(
                f"prediction length mismatch: expected {n}, got {a.shape[0]}"
            )
    stacked = np.vstack(arrays)
    return stacked.mean(axis=0)


def weighted_blend(preds: Sequence[ArrayLike], weights: ArrayLike) -> np.ndarray:
    """Element-wise weighted average; weights need not sum to 1 (normalized)."""
    arrays = [np.asarray(p, dtype=float).reshape(-1) for p in preds]
    w = np.asarray(weights, dtype=float).reshape(-1)
    if len(arrays) != len(w):
        raise ValueError(
            f"len(preds)={len(arrays)} != len(weights)={len(w)}"
        )
    if not arrays:
        raise ValueError("weighted_blend requires at least one prediction")
    n = arrays[0].shape[0]
    for a in arrays[1:]:
        if a.shape[0] != n:
            raise ValueError("prediction length mismatch")
    w_sum = float(w.sum())
    if w_sum <= 0:
        raise ValueError("weights must sum to a positive value")
    w = w / w_sum
    stacked = np.vstack(arrays)
    return (w[:, None] * stacked).sum(axis=0)


def fit_nonneg_blend_weights(
    oof_preds: ArrayLike,
    y_true: ArrayLike,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Fit non-negative blend weights on OOF predictions only (no test leak).

    Solves min ||P w - y||_2 with w >= 0 via ``scipy.optimize.nnls`` when
    available, else a projected gradient fallback with non-negativity.

    Parameters
    ----------
    oof_preds:
        Shape ``(n_samples, n_models)`` OOF predictions in physical units.
    y_true:
        Shape ``(n_samples,)`` targets aligned to OOF rows.
    normalize:
        If True, scale weights to sum to 1 (when sum > 0).

    Returns
    -------
    Weight vector of length ``n_models``.
    """
    P = np.asarray(oof_preds, dtype=float)
    y = np.asarray(y_true, dtype=float).reshape(-1)
    if P.ndim != 2:
        raise ValueError(f"oof_preds must be 2-d, got shape {P.shape}")
    if P.shape[0] != y.shape[0]:
        raise ValueError(
            f"row mismatch: oof_preds {P.shape[0]} vs y_true {y.shape[0]}"
        )
    if P.shape[0] == 0:
        raise ValueError("empty OOF matrix")

    try:
        from scipy.optimize import nnls

        w, _ = nnls(P, y)
    except Exception:
        # Fallback: non-negative least squares via constrained ridge-ish solve
        # Solve unconstrained least squares then clip + renorm.
        w, _, _, _ = np.linalg.lstsq(P, y, rcond=None)
        w = np.clip(w, a_min=0.0, a_max=None)
        if float(w.sum()) <= 0:
            w = np.ones(P.shape[1], dtype=float)

    w = np.asarray(w, dtype=float).reshape(-1)
    if normalize:
        s = float(w.sum())
        if s > 0:
            w = w / s
        else:
            w = np.full(P.shape[1], 1.0 / P.shape[1], dtype=float)
    return w
