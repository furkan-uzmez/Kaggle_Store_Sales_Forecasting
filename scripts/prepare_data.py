"""Load raw CSVs → structural clean → interim parquet + walk-forward fold manifests.

Does not fit scalers or apply global winsorization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from store_sales.config import ProjectPaths, load_default_config
from store_sales.data.clean import clean_structural
from store_sales.data.load import load_raw_tables
from store_sales.data.outliers import describe_sales_tails, flag_invalid_sales
from store_sales.data.split import build_expanding_folds, save_fold_manifests
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
        description="Prepare interim cleaned tables and expanding walk-forward folds"
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--interim-dir", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_default_config()
    paths = ProjectPaths()
    raw_dir = args.raw_dir or paths.data_raw
    interim = args.interim_dir or paths.data_interim
    splits_dir = args.splits_dir or paths.data_splits

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

    cv = cfg.get("cv", {})
    date_col = cfg.get("date_col", "date")
    folds = build_expanding_folds(
        cleaned["train"],
        date_col=date_col,
        n_folds=int(cv.get("n_folds", 3)),
        val_days=int(cv.get("val_days", 15)),
        gap_days=int(cv.get("gap_days", 0)),
        min_train_days=int(cv.get("min_train_days", 365)),
    )
    save_fold_manifests(folds, splits_dir)
    logger.info(
        "Wrote %d fold manifests to %s (folds_meta.json)",
        len(folds),
        splits_dir,
    )
    print(
        f"Wrote interim tables to {interim} and {len(folds)} folds to {splits_dir}"
    )


if __name__ == "__main__":
    main()
