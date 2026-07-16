"""Deterministic structural cleaning for competition tables (Layer 1).

Policy notes
------------
- Oil: sort by date, drop duplicate dates (keep last), causal ``ffill`` on
  ``dcoilwtico``. Leading NaNs remain for fold-local handling later.
- Train: entity-time sort, drop exact key duplicates; negative sales fail fast
  (investigate — do not silent-clip in prepare).
- Holidays: preserve the ``transferred`` flag; do not invent celebration dates.
- Outliers: no spike deletion or global winsorize here (see ``outliers`` helpers).
- No scalers / stateful transforms (those belong fold-local after Task 5 splits).
"""

from __future__ import annotations

import pandas as pd

from store_sales.io.logging import get_logger

logger = get_logger(__name__)


def clean_oil(oil: pd.DataFrame) -> pd.DataFrame:
    """Sort oil by date, resolve date duplicates, forward-fill price gaps."""
    out = oil.sort_values("date").drop_duplicates(subset=["date"], keep="last").copy()
    out["dcoilwtico"] = out["dcoilwtico"].ffill()
    n_remaining_na = int(out["dcoilwtico"].isna().sum())
    logger.info(
        "clean_oil rows=%d remaining_leading_or_all_na=%d",
        len(out),
        n_remaining_na,
    )
    return out.reset_index(drop=True)


def clean_train(train: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate panel keys and sort; refuse silent clipping of negatives."""
    out = train.copy()
    out = out.drop_duplicates(subset=["date", "store_nbr", "family"], keep="last")
    out = out.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    n_neg = int((out["sales"] < 0).sum())
    if n_neg:
        raise ValueError(
            f"Negative sales found ({n_neg} rows); investigate before clipping"
        )
    logger.info("clean_train rows=%d", len(out))
    return out


def clean_holidays(holidays: pd.DataFrame) -> pd.DataFrame:
    """Keep holiday calendar as-is including ``transferred``; sort only.

    Does not invent or move celebration dates. Downstream features should treat
    ``transferred=True`` rows as non-celebrated calendar markers per competition
    semantics.
    """
    out = holidays.copy()
    if "transferred" not in out.columns:
        raise ValueError("holidays_events missing required column: transferred")
    if "date" in out.columns:
        out = out.sort_values("date").reset_index(drop=True)
    logger.info(
        "clean_holidays rows=%d transferred_true=%d",
        len(out),
        int(out["transferred"].astype(bool).sum()) if len(out) else 0,
    )
    return out


def clean_structural(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply deterministic structural cleaning; no fold-local / fitted state."""
    cleaned = dict(tables)
    cleaned["oil"] = clean_oil(tables["oil"])
    cleaned["train"] = clean_train(tables["train"])
    cleaned["test"] = (
        tables["test"]
        .sort_values(["store_nbr", "family", "date"])
        .reset_index(drop=True)
    )
    if "holidays_events" in tables:
        cleaned["holidays_events"] = clean_holidays(tables["holidays_events"])
    if "transactions" in tables:
        cleaned["transactions"] = (
            tables["transactions"]
            .sort_values(["store_nbr", "date"])
            .reset_index(drop=True)
        )
    logger.info("clean_structural keys=%s", sorted(cleaned.keys()))
    return cleaned
