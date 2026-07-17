"""Promotion features (known-future on competition test)."""

from __future__ import annotations

import numpy as np
import pandas as pd

PROMO_FEATURE_NAMES: list[str] = [
    "onpromotion",
    "onpromotion_log1p",
    "has_promotion",
]


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add promotion level, log1p transform, and binary flag.

    Uses ``onpromotion`` already present on train/test (known for the horizon).
    """
    out = df.copy()
    if "onpromotion" not in out.columns:
        out["onpromotion"] = 0
    promo = pd.to_numeric(out["onpromotion"], errors="coerce").fillna(0)
    out["onpromotion"] = promo
    out["onpromotion_log1p"] = np.log1p(promo.clip(lower=0))
    out["has_promotion"] = (promo > 0).astype("int8")
    return out
