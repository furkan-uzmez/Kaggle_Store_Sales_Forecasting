"""Schema contracts for competition raw tables (Layer 1 data contract)."""

from __future__ import annotations

import pandas as pd

TRAIN_REQUIRED = ("date", "store_nbr", "family", "sales", "onpromotion")
TEST_REQUIRED = ("id", "date", "store_nbr", "family", "onpromotion")


def validate_train_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if train frame is missing required columns."""
    missing = [c for c in TRAIN_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"train missing columns: {missing}")


def validate_test_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if test frame is missing required columns."""
    missing = [c for c in TEST_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"test missing columns: {missing}")


def parse_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Return a copy with ``date_col`` parsed as datetime (errors raise)."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="raise")
    return out
