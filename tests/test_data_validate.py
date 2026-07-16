"""Schema and raw-load contract tests for competition tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from store_sales.data.validate import validate_test_schema, validate_train_schema


def test_validate_train_schema_accepts_minimal_frame():
    df = pd.DataFrame(
        {
            "id": [1],
            "date": ["2017-01-01"],
            "store_nbr": [1],
            "family": ["AUTOMOTIVE"],
            "sales": [0.0],
            "onpromotion": [0],
        }
    )
    validate_train_schema(df)  # should not raise after implementation


def test_validate_train_schema_missing_target():
    df = pd.DataFrame(
        {"date": ["2017-01-01"], "store_nbr": [1], "family": ["A"], "onpromotion": [0]}
    )
    with pytest.raises(ValueError, match="sales"):
        validate_train_schema(df)


def test_validate_test_schema_missing_column():
    df = pd.DataFrame(
        {"id": [1], "date": ["2017-08-16"], "store_nbr": [1], "family": ["A"]}
    )
    with pytest.raises(ValueError, match="onpromotion"):
        validate_test_schema(df)


def test_load_raw_tables_parses_dates_and_returns_all_keys(tmp_path: Path):
    from store_sales.data.load import load_raw_tables

    raw = tmp_path / "raw"
    raw.mkdir()

    pd.DataFrame(
        {
            "id": [0],
            "date": ["2017-01-01"],
            "store_nbr": [1],
            "family": ["AUTOMOTIVE"],
            "sales": [1.5],
            "onpromotion": [0],
        }
    ).to_csv(raw / "train.csv", index=False)
    pd.DataFrame(
        {
            "id": [1],
            "date": ["2017-08-16"],
            "store_nbr": [1],
            "family": ["AUTOMOTIVE"],
            "onpromotion": [1],
        }
    ).to_csv(raw / "test.csv", index=False)
    pd.DataFrame(
        {
            "store_nbr": [1],
            "city": ["Quito"],
            "state": ["Pichincha"],
            "type": ["A"],
            "cluster": [1],
        }
    ).to_csv(raw / "stores.csv", index=False)
    pd.DataFrame({"date": ["2017-01-01"], "dcoilwtico": [50.0]}).to_csv(
        raw / "oil.csv", index=False
    )
    pd.DataFrame(
        {
            "date": ["2017-01-01"],
            "type": ["Holiday"],
            "locale": ["National"],
            "locale_name": ["Ecuador"],
            "description": ["New Year"],
            "transferred": [False],
        }
    ).to_csv(raw / "holidays_events.csv", index=False)
    pd.DataFrame(
        {"date": ["2017-01-01"], "store_nbr": [1], "transactions": [100]}
    ).to_csv(raw / "transactions.csv", index=False)
    pd.DataFrame({"id": [1], "sales": [0.0]}).to_csv(
        raw / "sample_submission.csv", index=False
    )

    tables = load_raw_tables(raw)

    expected_keys = {
        "train",
        "test",
        "stores",
        "oil",
        "holidays_events",
        "transactions",
        "sample_submission",
    }
    assert set(tables) == expected_keys
    assert pd.api.types.is_datetime64_any_dtype(tables["train"]["date"])
    assert pd.api.types.is_datetime64_any_dtype(tables["test"]["date"])
    assert pd.api.types.is_datetime64_any_dtype(tables["oil"]["date"])
    assert pd.api.types.is_datetime64_any_dtype(tables["holidays_events"]["date"])
    assert pd.api.types.is_datetime64_any_dtype(tables["transactions"]["date"])


def test_load_raw_tables_missing_file_raises(tmp_path: Path):
    from store_sales.data.load import load_raw_tables

    with pytest.raises(FileNotFoundError, match="train.csv"):
        load_raw_tables(tmp_path)
