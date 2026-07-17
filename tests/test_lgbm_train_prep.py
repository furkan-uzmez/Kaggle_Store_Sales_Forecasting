"""PIT prep for LightGBM: mask before lag; feature column union."""

from __future__ import annotations

import numpy as np
import pandas as pd

from store_sales.features.registry import mask_target_after
from store_sales.models.gbdt import inverse_target, transform_target

# Import helpers from train script package path via importlib
import importlib.util
from pathlib import Path

_TRAIN = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("train_script", _TRAIN)
assert _spec and _spec.loader
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)


def test_feature_columns_for_groups_union_unique():
    cols = _train.feature_columns_for_groups(["base", "calendar", "promo"])
    assert "store_nbr" in cols
    assert "dayofweek" in cols
    assert "onpromotion_log1p" in cols
    # onpromotion appears in base and promo — only once
    assert cols.count("onpromotion") == 1


def test_prepare_lgbm_matrices_masks_val_target_for_lags():
    """Val lag_1 must not equal true previous-day val sales (masked origin)."""
    rows = []
    for day in range(20):
        rows.append(
            {
                "date": pd.Timestamp("2017-01-01") + pd.Timedelta(days=day),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 0,
                "sales": float(100 + day),
                "id": day,
            }
        )
    panel = pd.DataFrame(rows)
    train_end = pd.Timestamp("2017-01-10")
    val_start = pd.Timestamp("2017-01-11")
    val_end = pd.Timestamp("2017-01-15")
    train_idx = panel.index[panel["date"] <= train_end].to_numpy()
    val_idx = panel.index[
        (panel["date"] >= val_start) & (panel["date"] <= val_end)
    ].to_numpy()

    groups = ["base", "lag"]
    feature_cols = _train.feature_columns_for_groups(groups)
    X_tr, y_tr, X_va, y_va, meta = _train._prepare_lgbm_matrices(
        panel=panel,
        train_idx=train_idx,
        val_idx=val_idx,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        groups=groups,
        extras={},
        entity_cols=["store_nbr", "family"],
        date_col="date",
        target_col="sales",
        feature_cols=feature_cols,
    )
    assert len(X_tr) == len(train_idx)
    assert len(X_va) == len(val_idx)
    assert np.allclose(y_va, [110, 111, 112, 113, 114])

    # First val day: lag_1 = last train sales 109 (origin-safe)
    assert X_va.iloc[0]["sales_lag_1"] == 109.0
    # Second val day: previous val sales was masked → lag_1 is NaN (not 110)
    assert pd.isna(X_va.iloc[1]["sales_lag_1"])


def test_mask_then_inverse_for_metrics():
    y = np.array([0.0, 5.0, 20.0])
    z = transform_target(y, "log1p")
    pred = inverse_target(z, "log1p")
    assert np.allclose(pred, y)


def test_needs_recursive_when_lag_or_rolling():
    assert _train._needs_recursive_forecast(["base", "calendar"]) is False
    assert _train._needs_recursive_forecast(["base", "lag"]) is True
    assert _train._needs_recursive_forecast(["rolling"]) is True
