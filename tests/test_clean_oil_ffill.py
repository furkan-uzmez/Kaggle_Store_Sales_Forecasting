"""Deterministic oil cleaning: causal forward-fill of interior gaps."""

from __future__ import annotations

import pandas as pd

from store_sales.data.clean import clean_oil


def test_clean_oil_forward_fills_interior_gaps():
    oil = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03"]),
            "dcoilwtico": [50.0, None, 52.0],
        }
    )
    out = clean_oil(oil)
    assert out.loc[1, "dcoilwtico"] == 50.0


def test_clean_oil_sorts_and_deduplicates_by_date():
    oil = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-03", "2017-01-01", "2017-01-01"]),
            "dcoilwtico": [52.0, 49.0, 50.0],
        }
    )
    out = clean_oil(oil)
    assert list(out["date"]) == list(pd.to_datetime(["2017-01-01", "2017-01-03"]))
    # keep last duplicate for date
    assert out.loc[0, "dcoilwtico"] == 50.0


def test_clean_oil_leaves_leading_nan_for_fold_local_handling():
    oil = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03"]),
            "dcoilwtico": [None, 50.0, None],
        }
    )
    out = clean_oil(oil)
    assert pd.isna(out.loc[0, "dcoilwtico"])
    assert out.loc[1, "dcoilwtico"] == 50.0
    assert out.loc[2, "dcoilwtico"] == 50.0
