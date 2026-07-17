"""Entity-level lag features (past target only)."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

DEFAULT_LAGS: tuple[int, ...] = (1, 7, 14, 28)


def add_lag_features(
    df: pd.DataFrame,
    *,
    entity_cols: Sequence[str],
    date_col: str = "date",
    target_col: str = "sales",
    lags: Sequence[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Add ``{target}_lag_k`` columns via grouped shift (no current target).

    Rows are sorted by entity then date. Lag ``k`` is the target value from
    ``k`` steps earlier within the same entity; the first ``k`` rows per entity
    are NaN.

    Uses only target values present in ``df`` (NaNs propagate). For multi-step
    features from origin T0, mask sales after T0 before building (see
    ``mask_target_after``) or fill recursively with predictions — otherwise
    true post-origin sales leak into lag features on train∪horizon panels.
    """
    entity_cols = list(entity_cols)
    if not lags:
        return df.copy()

    out = df.sort_values(entity_cols + [date_col]).copy()
    grouped = out.groupby(entity_cols, sort=False, observed=True)[target_col]
    for lag in lags:
        if lag < 1:
            raise ValueError(f"lags must be >= 1 for PIT safety; got {lag}")
        out[f"{target_col}_lag_{lag}"] = grouped.shift(lag)
    return out.reset_index(drop=True)


def lag_feature_names(
    target_col: str = "sales",
    lags: Sequence[int] = DEFAULT_LAGS,
) -> list[str]:
    return [f"{target_col}_lag_{lag}" for lag in lags]
