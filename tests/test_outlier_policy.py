"""Outlier policy helpers: flag invalids; describe tails without mutating."""

from __future__ import annotations

import pandas as pd
import pytest

from store_sales.data.outliers import describe_sales_tails, flag_invalid_sales


def test_flag_invalid_sales_marks_negative_only():
    df = pd.DataFrame({"sales": [0.0, 1.5, -0.1, 10.0]})
    flags = flag_invalid_sales(df)
    assert isinstance(flags, pd.Series)
    assert flags.dtype == bool or flags.dtype == "bool"
    assert list(flags) == [False, False, True, False]


def test_flag_invalid_sales_preserves_index_alignment():
    df = pd.DataFrame({"sales": [1.0, -2.0, 3.0]}, index=[10, 20, 30])
    flags = flag_invalid_sales(df)
    assert list(flags.index) == [10, 20, 30]
    assert flags.loc[20] is True or flags.loc[20] == True  # noqa: E712


def test_describe_sales_tails_returns_quantiles_for_eda():
    df = pd.DataFrame({"sales": [0.0, 1.0, 2.0, 3.0, 4.0, 100.0]})
    tails = describe_sales_tails(df)
    assert isinstance(tails, pd.DataFrame)
    # EDA-only descriptive quantiles; must not mutate input
    assert set(df["sales"].tolist()) == {0.0, 1.0, 2.0, 3.0, 4.0, 100.0}
    # Expect common tail quantiles present (column or index depending on layout)
    text = tails.to_string().lower()
    assert any(tok in text for tok in ("0.99", "0.995", "0.999", "99%", "max", "0.95"))
    assert "sales" in tails.to_string().lower() or "sales" in tails.columns or (
        tails.index.name == "sales" or "sales" in str(tails.index)
    )


def test_describe_sales_tails_does_not_delete_spikes():
    """Policy: rare spikes are described, never silently dropped here."""
    df = pd.DataFrame({"sales": [1.0, 2.0, 1e6]})
    before = len(df)
    _ = describe_sales_tails(df)
    assert len(df) == before
    assert df["sales"].iloc[-1] == 1e6
