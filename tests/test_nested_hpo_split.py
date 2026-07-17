"""Leakage-safe nested HPO inner split: last block of outer train only."""

from __future__ import annotations

import pandas as pd
import pytest

from store_sales.data.split import split_last_block


def test_inner_val_is_subset_of_outer_train(tiny_panel: pd.DataFrame):
    """Inner val rows must come only from the outer train frame."""
    outer_train = tiny_panel.copy()
    result = split_last_block(
        outer_train,
        date_col="date",
        val_days=7,
        gap_days=0,
    )
    inner_train = outer_train.loc[result["train_idx"]]
    inner_val = outer_train.loc[result["val_idx"]]

    assert set(result["train_idx"]).issubset(set(outer_train.index))
    assert set(result["val_idx"]).issubset(set(outer_train.index))
    assert set(result["train_idx"]).isdisjoint(set(result["val_idx"]))
    assert len(inner_val) > 0
    assert len(inner_train) > 0


def test_inner_train_strictly_before_inner_val(tiny_panel: pd.DataFrame):
    """Temporal order: max(inner_train date) < min(inner_val date)."""
    outer_train = tiny_panel.copy()
    result = split_last_block(
        outer_train,
        date_col="date",
        val_days=7,
        gap_days=0,
    )
    train_dates = pd.to_datetime(outer_train.loc[result["train_idx"], "date"])
    val_dates = pd.to_datetime(outer_train.loc[result["val_idx"], "date"])
    assert train_dates.max() < val_dates.min()
    assert result["train_end"] == train_dates.max()
    assert result["val_start"] == val_dates.min()
    assert result["val_end"] == val_dates.max()


def test_gap_days_between_inner_train_and_val(tiny_panel: pd.DataFrame):
    """gap_days removes calendar buffer dates from both sides."""
    outer_train = tiny_panel.copy()
    result = split_last_block(
        outer_train,
        date_col="date",
        val_days=5,
        gap_days=3,
    )
    train_max = pd.to_datetime(outer_train.loc[result["train_idx"], "date"]).max()
    val_min = pd.to_datetime(outer_train.loc[result["val_idx"], "date"]).min()
    assert (val_min - train_max).days >= 3
    # Gap dates must not appear in either partition
    all_used = set(result["train_idx"]) | set(result["val_idx"])
    assert all_used.issubset(set(outer_train.index))
    unused = outer_train.loc[~outer_train.index.isin(all_used)]
    if len(unused) > 0:
        gap_dates = pd.to_datetime(unused["date"])
        assert ((gap_dates > train_max) & (gap_dates < val_min)).all()



def test_val_days_uses_unique_dates(tiny_panel: pd.DataFrame):
    """Inner val spans exactly val_days unique dates when panel is dense."""
    result = split_last_block(
        tiny_panel,
        date_col="date",
        val_days=7,
        gap_days=0,
    )
    val_dates = pd.to_datetime(tiny_panel.loc[result["val_idx"], "date"]).unique()
    assert len(val_dates) == 7


def test_rejects_insufficient_history(tiny_panel: pd.DataFrame):
    short = tiny_panel[tiny_panel["date"] <= tiny_panel["date"].min() + pd.Timedelta(days=5)]
    with pytest.raises(ValueError, match="insufficient"):
        split_last_block(short, date_col="date", val_days=10, gap_days=0)
