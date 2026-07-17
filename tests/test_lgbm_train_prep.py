"""PIT prep for LightGBM: mask before lag; feature column union."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


class _CaptureLagModel:
    """Stub booster: record each day's feature frame; predict constant log1p(pred)."""

    def __init__(self, pred_sales: float = 42.0):
        self.pred_sales = float(pred_sales)
        self.calls: list[pd.DataFrame] = []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.calls.append(X.copy())
        return np.full(len(X), np.log1p(self.pred_sales), dtype=float)


def _mini_panel(n_days: int = 16) -> pd.DataFrame:
    rows = []
    for day in range(n_days):
        rows.append(
            {
                "date": pd.Timestamp("2017-01-01") + pd.Timedelta(days=day),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 0,
                # Distinctive true sales so leaked mid-horizon values are obvious.
                "sales": float(1000 + day),
                "id": day,
            }
        )
    return pd.DataFrame(rows)


def test_recursive_day2_lag1_equals_written_prediction_not_true_sales():
    """Day-2 lag_1 must use day-1 prediction, never true val sales (behavioral)."""
    panel = _mini_panel(16)
    train_end = pd.Timestamp("2017-01-10")  # sales 1009
    val_start = pd.Timestamp("2017-01-11")  # true sales 1010
    val_end = pd.Timestamp("2017-01-13")  # three-step horizon

    groups = ["base", "lag"]
    feature_cols = _train.feature_columns_for_groups(groups)
    stub = _CaptureLagModel(pred_sales=42.0)

    y_true, y_pred, meta = _train._recursive_val_predict(
        model=stub,
        panel=panel,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        groups=groups,
        extras={},
        entity_cols=["store_nbr", "family"],
        date_col="date",
        target_col="sales",
        feature_cols=feature_cols,
        cat_maps={},
        target_transform="log1p",
        clip_negative_preds=True,
        lookback_days=56,
    )

    assert len(stub.calls) == 3
    assert np.allclose(y_pred, 42.0)
    # Day-1 lag_1 is last train sales (origin-safe history).
    assert stub.calls[0]["sales_lag_1"].iloc[0] == pytest.approx(1009.0)
    # Day-2 lag_1 must equal written day-1 prediction, not true mid-horizon sales 1010.
    day1_true = float(panel.loc[panel["date"] == val_start, "sales"].iloc[0])
    assert day1_true == pytest.approx(1010.0)
    assert stub.calls[1]["sales_lag_1"].iloc[0] == pytest.approx(42.0)
    assert stub.calls[1]["sales_lag_1"].iloc[0] != pytest.approx(day1_true)
    # Day-3 lag_1 also from recursive preds (still 42).
    assert stub.calls[2]["sales_lag_1"].iloc[0] == pytest.approx(42.0)
    assert len(y_true) == 3
    assert len(meta) == 3


def test_mask_outcome_extras_nulls_post_origin_transactions():
    """Post-origin transactions must be nulled so mid-horizon tx cannot leak."""
    train_end = pd.Timestamp("2017-01-10")
    tx = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2017-01-09", "2017-01-10", "2017-01-11", "2017-01-12"]
            ),
            "store_nbr": [1, 1, 1, 1],
            "transactions": [90.0, 100.0, 999.0, 888.0],
        }
    )
    masked = _train.mask_outcome_extras_after(
        {"transactions": tx},
        train_end,
        date_col="date",
    )
    out = masked["transactions"].sort_values("date").reset_index(drop=True)
    assert out.loc[0, "transactions"] == pytest.approx(90.0)
    assert out.loc[1, "transactions"] == pytest.approx(100.0)
    assert pd.isna(out.loc[2, "transactions"])
    assert pd.isna(out.loc[3, "transactions"])
    # Other keys / oil untouched when absent
    assert "oil" not in masked


def test_prepare_matrices_masks_post_origin_tx_features():
    """Multi-horizon tx lag on day-2 must not equal true mid-horizon transactions."""
    panel = _mini_panel(16)
    train_end = pd.Timestamp("2017-01-10")
    val_start = pd.Timestamp("2017-01-11")
    val_end = pd.Timestamp("2017-01-13")
    train_idx = panel.index[panel["date"] <= train_end].to_numpy()
    val_idx = panel.index[
        (panel["date"] >= val_start) & (panel["date"] <= val_end)
    ].to_numpy()

    # Distinctive post-origin transaction values (would leak if not masked).
    tx = pd.DataFrame(
        {
            "date": pd.date_range("2017-01-01", periods=16, freq="D"),
            "store_nbr": 1,
            "transactions": [float(500 + i) for i in range(16)],
        }
    )
    groups = ["base", "transactions"]
    feature_cols = _train.feature_columns_for_groups(groups)
    _X_tr, _y_tr, X_va, _y_va, _meta = _train._prepare_lgbm_matrices(
        panel=panel,
        train_idx=train_idx,
        val_idx=val_idx,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        groups=groups,
        extras={"transactions": tx},
        entity_cols=["store_nbr", "family"],
        date_col="date",
        target_col="sales",
        feature_cols=feature_cols,
    )
    # Val day 0 (2017-01-11): lag_1 uses train_end tx = 500+9 = 509
    assert X_va.iloc[0]["transactions_lag_1"] == pytest.approx(509.0)
    # Val day 1 (2017-01-12): mid-horizon true tx would be 510; must not leak.
    true_mid_tx = 510.0  # transactions on 2017-01-11
    assert true_mid_tx == pytest.approx(500.0 + 10)
    assert pd.isna(X_va.iloc[1]["transactions_lag_1"]) or (
        X_va.iloc[1]["transactions_lag_1"] != pytest.approx(true_mid_tx)
    )
    # Stricter: after nulling post-origin values, lag from nulled day is NaN.
    assert pd.isna(X_va.iloc[1]["transactions_lag_1"])
