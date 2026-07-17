"""Point-in-time (PIT) feature builders: no current/future target or exogenous."""

from __future__ import annotations

import pandas as pd
import pytest

from store_sales.features.lag import add_lag_features
from store_sales.features.oil import add_oil_features
from store_sales.features.registry import (
    FEATURE_GROUPS,
    build_feature_matrix,
    list_group_ablation_configs,
    mask_target_after,
)
from store_sales.features.rolling import add_rolling_features
from store_sales.features.transactions import add_transaction_features


def test_lag_features_do_not_use_current_target():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2017-01-01", periods=5, freq="D"),
            "store_nbr": 1,
            "family": "A",
            "sales": [10, 20, 30, 40, 50],
        }
    )
    out = add_lag_features(
        df,
        entity_cols=["store_nbr", "family"],
        date_col="date",
        target_col="sales",
        lags=[1],
    )
    assert pd.isna(out.loc[0, "sales_lag_1"])
    assert out.loc[1, "sales_lag_1"] == 10
    assert out.loc[2, "sales_lag_1"] == 20
    # Never equal current target on any row where lag is defined
    both = out["sales_lag_1"].notna()
    assert not (out.loc[both, "sales_lag_1"] == out.loc[both, "sales"]).any()


def test_rolling_features_are_shifted_no_current_target():
    """Rolling stats use past-only history (shift before rolling)."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2017-01-01", periods=6, freq="D"),
            "store_nbr": 1,
            "family": "A",
            "sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    out = add_rolling_features(
        df,
        entity_cols=["store_nbr", "family"],
        date_col="date",
        target_col="sales",
        windows=[3],
        stats=("mean",),
        min_periods=1,
    )
    # First row: no past history after shift → NaN
    assert pd.isna(out.loc[0, "sales_roll_mean_3"])
    # Row 1: only lag-1 value 10 in window → mean 10
    assert out.loc[1, "sales_roll_mean_3"] == pytest.approx(10.0)
    # Row 3: past values 10,20,30 → mean 20 (must not include current 40)
    assert out.loc[3, "sales_roll_mean_3"] == pytest.approx(20.0)
    # Current target never equals unshifted self-mean at t for t with history
    assert out.loc[5, "sales_roll_mean_3"] == pytest.approx((30 + 40 + 50) / 3)


def test_oil_features_use_lagged_price_not_same_day():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03"]),
            "store_nbr": 1,
            "family": "A",
            "sales": [1.0, 2.0, 3.0],
        }
    )
    oil = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03"]),
            "dcoilwtico": [50.0, 60.0, 70.0],
        }
    )
    out = add_oil_features(panel, oil, date_col="date", lag=1)
    assert pd.isna(out.loc[0, "oil_lag_1"])
    assert out.loc[1, "oil_lag_1"] == pytest.approx(50.0)
    assert out.loc[2, "oil_lag_1"] == pytest.approx(60.0)
    # Not same-day oil
    assert out.loc[1, "oil_lag_1"] != 60.0


def test_transaction_features_past_only_no_same_day():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03"]),
            "store_nbr": 1,
            "family": "A",
            "sales": [1.0, 2.0, 3.0],
        }
    )
    tx = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03"]),
            "store_nbr": 1,
            "transactions": [100, 200, 300],
        }
    )
    out = add_transaction_features(panel, tx, date_col="date", lags=(1,), windows=(2,))
    assert pd.isna(out.loc[0, "transactions_lag_1"])
    assert out.loc[1, "transactions_lag_1"] == pytest.approx(100.0)
    assert out.loc[2, "transactions_lag_1"] == pytest.approx(200.0)
    # Rolling mean of shifted series: row 2 uses lag history only
    assert out.loc[2, "transactions_roll_mean_2"] == pytest.approx((100.0 + 200.0) / 2)


def test_feature_groups_registry_includes_admitted_groups():
    expected = {
        "base",
        "calendar",
        "lag",
        "rolling",
        "promo",
        "oil",
        "holiday",
        "store_meta",
        "transactions",
    }
    assert expected.issubset(set(FEATURE_GROUPS.keys()))
    for name, cols in FEATURE_GROUPS.items():
        assert isinstance(cols, list)
        assert all(isinstance(c, str) for c in cols)


def test_list_group_ablation_configs_progressive_and_optionals():
    admitted = [
        "base",
        "calendar",
        "promo",
        "lag",
        "rolling",
        "oil",
        "holiday",
        "store_meta",
        "transactions",
    ]
    configs = list_group_ablation_configs(admitted)
    assert isinstance(configs, list)
    assert len(configs) >= 5
    groups_seq = [c["feature_groups"] for c in configs]
    # Progressive core
    assert groups_seq[0] == ["base"]
    assert groups_seq[1] == ["base", "calendar"]
    assert groups_seq[2] == ["base", "calendar", "promo"]
    assert groups_seq[3] == ["base", "calendar", "promo", "lag"]
    assert groups_seq[4] == ["base", "calendar", "promo", "lag", "rolling"]
    # Optionals appear only if admitted, as extensions of full core
    core_full = ["base", "calendar", "promo", "lag", "rolling"]
    optional_sets = [g for g in groups_seq[5:]]
    assert ["base", "calendar", "promo", "lag", "rolling", "oil"] in optional_sets
    assert any(g == core_full + ["holiday"] for g in optional_sets)
    assert any(g == core_full + ["store_meta"] for g in optional_sets)
    assert any(g == core_full + ["transactions"] for g in optional_sets)


def test_list_group_ablation_skips_unadmitted_optionals():
    configs = list_group_ablation_configs(
        ["base", "calendar", "promo", "lag", "rolling"]
    )
    all_groups = {g for c in configs for g in c["feature_groups"]}
    assert "oil" not in all_groups
    assert "transactions" not in all_groups


def test_build_feature_matrix_core_groups_pit_safe(tiny_panel: pd.DataFrame):
    out = build_feature_matrix(
        tiny_panel,
        groups=["base", "calendar", "promo", "lag", "rolling"],
    )
    assert "sales_lag_1" in out.columns
    assert "sales_roll_mean_7" in out.columns
    assert "dayofweek" in out.columns or "dow" in out.columns
    assert "onpromotion" in out.columns
    # lag_1 != current sales where both present
    m = out["sales_lag_1"].notna()
    assert not (out.loc[m, "sales_lag_1"] == out.loc[m, "sales"]).any()


def test_mask_target_after_blocks_multi_step_lag_leakage():
    """After origin mask, lag_1 on T0+2 must not equal true sales[T0+1].

    Multi-step H>1 from origin T0 on a train∪horizon panel with true post-origin
    sales would otherwise leak mid-horizon targets into lag/rolling features.
    """
    origin = pd.Timestamp("2017-01-03")
    df = pd.DataFrame(
        {
            "date": pd.date_range("2017-01-01", periods=6, freq="D"),
            "store_nbr": 1,
            "family": "A",
            "sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    t0_plus_1 = origin + pd.Timedelta(days=1)
    t0_plus_2 = origin + pd.Timedelta(days=2)
    true_sales_t0_plus_1 = df.loc[df["date"] == t0_plus_1, "sales"].iloc[0]
    assert true_sales_t0_plus_1 == pytest.approx(40.0)

    # Without mask, lag_1 on T0+2 equals true mid-horizon sales (leak).
    leaked = add_lag_features(
        df,
        entity_cols=["store_nbr", "family"],
        date_col="date",
        target_col="sales",
        lags=[1],
    )
    leaked_lag = leaked.loc[leaked["date"] == t0_plus_2, "sales_lag_1"].iloc[0]
    assert leaked_lag == pytest.approx(true_sales_t0_plus_1)

    masked = mask_target_after(df, origin, date_col="date", target_col="sales")
    # Origin day kept; post-origin sales nulled.
    assert masked.loc[masked["date"] == origin, "sales"].iloc[0] == pytest.approx(30.0)
    assert pd.isna(masked.loc[masked["date"] == t0_plus_1, "sales"].iloc[0])
    assert pd.isna(masked.loc[masked["date"] == t0_plus_2, "sales"].iloc[0])

    safe = add_lag_features(
        masked,
        entity_cols=["store_nbr", "family"],
        date_col="date",
        target_col="sales",
        lags=[1],
    )
    safe_lag = safe.loc[safe["date"] == t0_plus_2, "sales_lag_1"].iloc[0]
    # Must not equal true mid-horizon sales (NaN via mask chain is expected).
    assert not (
        pd.notna(safe_lag) and safe_lag == pytest.approx(true_sales_t0_plus_1)
    )
    assert pd.isna(safe_lag) or safe_lag != true_sales_t0_plus_1
