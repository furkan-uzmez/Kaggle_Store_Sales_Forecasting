"""Base identity / known-at-horizon columns retained for modeling."""

from __future__ import annotations

import pandas as pd

# Columns always available on train and competition test for the panel spine.
BASE_FEATURE_NAMES: list[str] = [
    "store_nbr",
    "family",
    "onpromotion",
]


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure base panel columns exist; no target-derived transforms.

    ``onpromotion`` is a known-future covariate on the test horizon.
    Missing optional base columns are left absent (caller supplies panel).
    """
    out = df.copy()
    if "onpromotion" not in out.columns:
        out["onpromotion"] = 0
    return out
