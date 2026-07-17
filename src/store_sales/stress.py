"""Robustness / stress operators for locked GBDT finalists.

Perturbations act on feature matrices at prediction time (frozen model).
Temporal order is preserved — no row shuffling. Domain clips keep non-negative
sales-like lags and promotion counts when requested.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def relative_noise(
    X: pd.DataFrame,
    columns: Sequence[str],
    *,
    relative: float,
    seed: int = 0,
    non_negative: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Multiply selected columns by (1 + relative * N(0,1)).

    ``relative`` is the noise scale (e.g. 0.05 → 5% relative Gaussian).
    Columns absent from ``X`` are skipped. Returns a copy.
    """
    if relative < 0:
        raise ValueError(f"relative must be >= 0, got {relative}")
    out = X.copy()
    rng = np.random.default_rng(int(seed))
    nn = set(non_negative or ())
    for col in columns:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        noise = rng.normal(0.0, 1.0, size=len(out))
        values = series.to_numpy(dtype=float) * (1.0 + float(relative) * noise)
        if col in nn:
            values = np.clip(values, a_min=0.0, a_max=None)
        out[col] = values
    return out


def null_columns(X: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Simulate join failure: set listed columns to NaN (copy)."""
    out = X.copy()
    for col in columns:
        if col in out.columns:
            out[col] = np.nan
    return out


def clip_spike_columns(
    X: pd.DataFrame,
    columns: Sequence[str],
    *,
    upper_quantile: float = 0.99,
    lower_quantile: float = 0.0,
) -> pd.DataFrame:
    """Clip lag/history spikes to empirical quantiles computed on ``X`` (copy)."""
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError(
            f"need 0 <= lower < upper <= 1; got {lower_quantile}, {upper_quantile}"
        )
    out = X.copy()
    for col in columns:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        lo = float(series.quantile(lower_quantile))
        hi = float(series.quantile(upper_quantile))
        out[col] = series.clip(lower=lo, upper=hi)
    return out


def predictions_changed(
    y_clean: np.ndarray | Sequence[float],
    y_stressed: np.ndarray | Sequence[float],
    *,
    atol: float = 1e-12,
) -> bool:
    """Return True when any finite prediction differs under stress."""
    a = np.asarray(y_clean, dtype=float)
    b = np.asarray(y_stressed, dtype=float)
    if a.shape != b.shape:
        return True
    return bool(np.any(np.abs(a - b) > atol))
