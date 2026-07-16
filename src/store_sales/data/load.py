"""Load competition CSVs from ``data/raw`` with date parse + schema checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from store_sales.data.validate import parse_dates, validate_test_schema, validate_train_schema
from store_sales.io.logging import get_logger

logger = get_logger(__name__)

RAW_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays_events": "holidays_events.csv",
    "transactions": "transactions.csv",
    "sample_submission": "sample_submission.csv",
}

_DATE_TABLES = ("train", "test", "oil", "holidays_events", "transactions")


def load_raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all competition raw CSVs, parse dates, validate train/test schema.

    Returns a dict keyed by: train, test, stores, oil, holidays_events,
    transactions, sample_submission.
    """
    raw_dir = Path(raw_dir)
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in RAW_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        tables[key] = pd.read_csv(path)
        logger.info("Loaded %s rows=%d cols=%d path=%s", key, len(tables[key]), tables[key].shape[1], path)

    for key in _DATE_TABLES:
        if key in tables and "date" in tables[key].columns:
            tables[key] = parse_dates(tables[key])

    validate_train_schema(tables["train"])
    validate_test_schema(tables["test"])
    logger.info(
        "Raw tables validated keys=%s",
        sorted(tables.keys()),
    )
    return tables
