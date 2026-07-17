"""Static store metadata joins."""

from __future__ import annotations

import pandas as pd

STORE_META_FEATURE_NAMES: list[str] = [
    "city",
    "state",
    "store_type",
    "cluster",
]


def add_store_meta_features(
    df: pd.DataFrame,
    stores: pd.DataFrame,
    *,
    store_col: str = "store_nbr",
) -> pd.DataFrame:
    """Join store type/cluster/city/state (static; no temporal leakage)."""
    out = df.copy()
    meta = stores.copy()
    # Rename type → store_type to avoid clashing with holiday type if both present
    if "type" in meta.columns and "store_type" not in meta.columns:
        meta = meta.rename(columns={"type": "store_type"})
    keep = [store_col] + [c for c in STORE_META_FEATURE_NAMES if c in meta.columns]
    # Also accept original "type" column
    if "type" in stores.columns and "store_type" not in keep:
        meta = stores.rename(columns={"type": "store_type"})
        keep = [store_col] + [c for c in STORE_META_FEATURE_NAMES if c in meta.columns]
    meta = meta[keep].drop_duplicates(subset=[store_col], keep="last")
    # Drop existing meta cols to avoid _x/_y from re-joins
    drop_cols = [c for c in STORE_META_FEATURE_NAMES if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    out = out.merge(meta, on=store_col, how="left")
    return out
