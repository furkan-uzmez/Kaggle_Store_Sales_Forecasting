"""Naive temporal baselines: last-value and multi-step seasonal naive."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def last_value_predict(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    entity_cols: Sequence[str],
    date_col: str,
    target_col: str,
) -> pd.Series:
    """Per-entity constant forecast = last observed target in history.

    Missing entities fill with 0.0. Output is aligned to ``future`` index.
    """
    entity_cols = list(entity_cols)
    hist = history.sort_values(list(entity_cols) + [date_col])
    last = hist.groupby(entity_cols, observed=True, sort=False)[target_col].last()
    keys = pd.MultiIndex.from_frame(future[entity_cols])
    mapped = keys.map(last)
    return pd.Series(mapped, index=future.index, dtype=float).fillna(0.0)


def seasonal_naive_predict(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    entity_cols: Sequence[str],
    date_col: str,
    target_col: str,
    period: int,
    train_end: pd.Timestamp | None = None,
) -> pd.Series:
    """Origin-based multi-step seasonal naive.

    For each future date ``d`` and forecast origin ``train_end`` (default:
    max history date), set ``k = max(1, ceil((d - train_end).days / period))``
    and predict the value at ``d - k * period`` for the same entity.

    Fallback chain when lag is missing: last-value per entity, then 0.0.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    entity_cols = list(entity_cols)
    hist = history.copy()
    fut = future.copy()
    hist[date_col] = pd.to_datetime(hist[date_col])
    fut[date_col] = pd.to_datetime(fut[date_col])

    if train_end is None:
        origin = pd.Timestamp(hist[date_col].max())
    else:
        origin = pd.Timestamp(train_end)

    hist_idx = hist.set_index(list(entity_cols) + [date_col])[target_col]

    delta = (fut[date_col] - origin).dt.days.to_numpy()
    k = np.ceil(delta / period).astype(int)
    k = np.maximum(k, 1)
    lookup_dates = fut[date_col].to_numpy() - pd.to_timedelta(k * period, unit="D")

    arrays = [fut[c].to_numpy() for c in entity_cols]
    arrays.append(lookup_dates)
    keys = pd.MultiIndex.from_arrays(
        arrays, names=list(entity_cols) + [date_col]
    )
    pred = hist_idx.reindex(keys).to_numpy(dtype=float)

    # Fallback: last value then 0.0
    missing = np.isnan(pred)
    if missing.any():
        last = last_value_predict(
            history=hist,
            future=fut,
            entity_cols=entity_cols,
            date_col=date_col,
            target_col=target_col,
        ).to_numpy(dtype=float)
        pred = np.where(missing, last, pred)

    pred = np.nan_to_num(pred, nan=0.0)
    return pd.Series(pred, index=future.index, dtype=float)
