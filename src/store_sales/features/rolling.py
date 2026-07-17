"""Shifted rolling window features (right-closed past history only)."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

DEFAULT_WINDOWS: tuple[int, ...] = (7, 14, 28)
DEFAULT_STATS: tuple[str, ...] = ("mean", "std")


def add_rolling_features(
    df: pd.DataFrame,
    *,
    entity_cols: Sequence[str],
    date_col: str = "date",
    target_col: str = "sales",
    windows: Sequence[int] = DEFAULT_WINDOWS,
    stats: Sequence[str] = DEFAULT_STATS,
    shift: int = 1,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Rolling stats on already-shifted target history.

    Pattern: ``groupby(entity)[target].shift(shift).rolling(window).stat()``.
    Current-row target is never included when ``shift >= 1``.
    """
    entity_cols = list(entity_cols)
    if shift < 1:
        raise ValueError(f"shift must be >= 1 for PIT safety; got {shift}")
    if not windows:
        return df.copy()

    out = df.sort_values(entity_cols + [date_col]).copy()
    history = out.groupby(entity_cols, sort=False, observed=True)[target_col].shift(
        shift
    )

    for window in windows:
        if window < 1:
            raise ValueError(f"window must be >= 1; got {window}")
        rolled = history.groupby(
            [out[c] for c in entity_cols], sort=False, observed=True
        )
        for stat in stats:
            col = f"{target_col}_roll_{stat}_{window}"
            if stat == "mean":
                out[col] = rolled.transform(
                    lambda s, w=window: s.rolling(w, min_periods=min_periods).mean()
                )
            elif stat == "std":
                out[col] = rolled.transform(
                    lambda s, w=window: s.rolling(w, min_periods=min_periods).std()
                )
            else:
                raise ValueError(f"unsupported rolling stat: {stat}")
    return out.reset_index(drop=True)


def rolling_feature_names(
    target_col: str = "sales",
    windows: Sequence[int] = DEFAULT_WINDOWS,
    stats: Sequence[str] = DEFAULT_STATS,
) -> list[str]:
    return [
        f"{target_col}_roll_{stat}_{window}"
        for window in windows
        for stat in stats
    ]
