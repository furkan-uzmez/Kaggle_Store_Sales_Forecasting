"""Store-level transaction features (history only; no future joins)."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

DEFAULT_TX_LAGS: tuple[int, ...] = (1,)
DEFAULT_TX_WINDOWS: tuple[int, ...] = (7,)

TRANSACTIONS_FEATURE_NAMES: list[str] = [
    "transactions_lag_1",
    "transactions_roll_mean_7",
]


def add_transaction_features(
    df: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    date_col: str = "date",
    store_col: str = "store_nbr",
    value_col: str = "transactions",
    lags: Sequence[int] = DEFAULT_TX_LAGS,
    windows: Sequence[int] = DEFAULT_TX_WINDOWS,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Lag and shifted-rolling store transactions merged onto the panel.

    Transactions end with the train window and are **not** known for the full
    test horizon; only past values via lag/rolling are used.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    tx = (
        transactions[[date_col, store_col, value_col]]
        .drop_duplicates(subset=[date_col, store_col], keep="last")
        .sort_values([store_col, date_col])
        .copy()
    )
    tx[date_col] = pd.to_datetime(tx[date_col])
    grouped = tx.groupby(store_col, sort=False, observed=True)[value_col]

    for lag in lags:
        if lag < 1:
            raise ValueError(f"transaction lag must be >= 1; got {lag}")
        tx[f"transactions_lag_{lag}"] = grouped.shift(lag)

    history = grouped.shift(1)
    for window in windows:
        if window < 1:
            raise ValueError(f"window must be >= 1; got {window}")
        tx[f"transactions_roll_mean_{window}"] = history.groupby(
            tx[store_col], sort=False, observed=True
        ).transform(lambda s, w=window: s.rolling(w, min_periods=min_periods).mean())

    feat_cols = (
        [date_col, store_col]
        + [f"transactions_lag_{lag}" for lag in lags]
        + [f"transactions_roll_mean_{w}" for w in windows]
    )
    # Drop any prior feature columns before merge
    drop_existing = [c for c in feat_cols if c in out.columns and c not in (date_col, store_col)]
    if drop_existing:
        out = out.drop(columns=drop_existing)
    out = out.merge(tx[feat_cols], on=[date_col, store_col], how="left")
    return out
