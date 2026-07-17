"""Oil price features with explicit lag (no same-day / future join)."""

from __future__ import annotations

import pandas as pd

OIL_FEATURE_NAMES: list[str] = ["oil_lag_1"]


def add_oil_features(
    df: pd.DataFrame,
    oil: pd.DataFrame,
    *,
    date_col: str = "date",
    oil_col: str = "dcoilwtico",
    lag: int = 1,
) -> pd.DataFrame:
    """Join lagged oil price onto the panel.

    Builds a calendar oil series, applies ``shift(lag)`` so day ``t`` receives
    oil from ``t - lag`` only (PIT-safe; no future oil).
    """
    if lag < 1:
        raise ValueError(f"oil lag must be >= 1; got {lag}")

    out = df.copy()
    oil_clean = (
        oil[[date_col, oil_col]]
        .drop_duplicates(subset=[date_col], keep="last")
        .sort_values(date_col)
        .copy()
    )
    oil_clean[date_col] = pd.to_datetime(oil_clean[date_col])
    # Causal ffill already applied in prepare; keep any residual leading NA.
    oil_clean[oil_col] = oil_clean[oil_col].ffill()
    oil_clean[f"oil_lag_{lag}"] = oil_clean[oil_col].shift(lag)
    oil_feat = oil_clean[[date_col, f"oil_lag_{lag}"]]

    out[date_col] = pd.to_datetime(out[date_col])
    out = out.merge(oil_feat, on=date_col, how="left")
    return out
