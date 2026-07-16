"""Leakage-safety tests for expanding walk-forward splits."""

from __future__ import annotations

import pandas as pd
import pytest

from store_sales.data.split import build_expanding_folds


def test_expanding_folds_have_strict_time_order(tiny_panel: pd.DataFrame):
    folds = build_expanding_folds(
        tiny_panel,
        date_col="date",
        n_folds=2,
        val_days=7,
        gap_days=0,
        min_train_days=14,
    )
    assert len(folds) >= 1
    for fold in folds:
        train_dates = tiny_panel.loc[fold["train_idx"], "date"]
        val_dates = tiny_panel.loc[fold["val_idx"], "date"]
        assert train_dates.max() < val_dates.min()
        assert (val_dates.max() - val_dates.min()).days <= 7


def test_gap_days_respected(tiny_panel: pd.DataFrame):
    folds = build_expanding_folds(
        tiny_panel,
        date_col="date",
        n_folds=1,
        val_days=5,
        gap_days=3,
        min_train_days=10,
    )
    fold = folds[0]
    train_max = tiny_panel.loc[fold["train_idx"], "date"].max()
    val_min = tiny_panel.loc[fold["val_idx"], "date"].min()
    assert (val_min - train_max).days >= 3
