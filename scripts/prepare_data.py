"""Load raw CSVs → structural clean → write interim parquet tables.

Fold manifests are Task 5 (``build_expanding_folds`` / ``save_fold_manifests``).
This entrypoint intentionally omits fold generation so Task 4 stays self-contained.
Does not fit scalers or apply global winsorization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from store_sales.config import ProjectPaths, load_default_config
from store_sales.data.clean import clean_structural
from store_sales.data.load import load_raw_tables
from store_sales.data.outliers import describe_sales_tails, flag_invalid_sales
from store_sales.io.logging import get_logger

logger = get_logger(__name__)

INTERIM_TABLES = (
    "train",
    "test",
    "stores",
    "oil",
    "holidays_events",
    "transactions",
    "sample_submission",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare interim cleaned tables (no fold manifests; Task 5)"
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--interim-dir", type=Path, default=None)
    args = parser.parse_args()

    # Config loaded for future fold args (Task 5); unused for interim write path.
    _cfg = load_default_config()
    paths = ProjectPaths()
    raw_dir = args.raw_dir or paths.data_raw
    interim = args.interim_dir or paths.data_interim

    logger.info("Loading raw tables from %s", raw_dir)
    tables = load_raw_tables(raw_dir)
    cleaned = clean_structural(tables)

    n_invalid = int(flag_invalid_sales(cleaned["train"]).sum())
    if n_invalid:
        raise ValueError(
            f"Invalid (negative) sales in cleaned train: {n_invalid}; abort prepare"
        )
    tails = describe_sales_tails(cleaned["train"])
    logger.info("Sales tail quantiles (EDA only):\n%s", tails.to_string())

    interim.mkdir(parents=True, exist_ok=True)
    for name in INTERIM_TABLES:
        if name not in cleaned:
            raise KeyError(f"Missing cleaned table: {name}")
        out_path = interim / f"{name}.parquet"
        cleaned[name].to_parquet(out_path, index=False)
        logger.info("Wrote %s rows=%d path=%s", name, len(cleaned[name]), out_path)

    # Task 5: walk-forward folds + manifests (build_expanding_folds / save_fold_manifests).
    # Omit until split module lands so Task 4 is self-contained.
    logger.info(
        "Skipping fold manifests (Task 5). Wrote interim tables only to %s",
        interim,
    )
    print(f"Wrote interim tables to {interim} (fold generation deferred to Task 5)")


if __name__ == "__main__":
    main()
