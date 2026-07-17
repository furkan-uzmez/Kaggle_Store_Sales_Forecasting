"""TDD: stress noise path must alter predictions under a frozen model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from store_sales.stress import (
    clip_spike_columns,
    null_columns,
    predictions_changed,
    relative_noise,
)


def test_relative_noise_alters_values_and_predictions():
    """Noise path changes features and therefore model predictions."""
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(
        {
            "sales_lag_1": rng.uniform(0, 50, size=n),
            "sales_roll_mean_7": rng.uniform(0, 40, size=n),
            "onpromotion": rng.integers(0, 5, size=n).astype(float),
            "store_nbr": rng.integers(1, 5, size=n),
        }
    )
    # Toy linear model: pred = 0.5*lag + 0.3*roll + 0.1*promo
    w = np.array([0.5, 0.3, 0.1, 0.0])
    y_clean = X.to_numpy(dtype=float) @ w

    X_noisy = relative_noise(
        X,
        ["sales_lag_1", "sales_roll_mean_7", "onpromotion"],
        relative=0.10,
        seed=42,
        non_negative=["sales_lag_1", "sales_roll_mean_7", "onpromotion"],
    )
    y_noisy = X_noisy.to_numpy(dtype=float) @ w

    assert not np.allclose(
        X[["sales_lag_1", "sales_roll_mean_7"]].to_numpy(),
        X_noisy[["sales_lag_1", "sales_roll_mean_7"]].to_numpy(),
    )
    assert predictions_changed(y_clean, y_noisy)
    assert (X_noisy["sales_lag_1"] >= 0).all()
    # Untouched column stays equal
    assert np.allclose(X["store_nbr"], X_noisy["store_nbr"])


def test_relative_noise_zero_is_identity():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = relative_noise(X, ["a"], relative=0.0, seed=1)
    assert np.allclose(out["a"], X["a"])


def test_null_columns_sets_nan():
    X = pd.DataFrame({"oil_lag_1": [1.0, 2.0], "keep": [3.0, 4.0]})
    out = null_columns(X, ["oil_lag_1", "missing_col"])
    assert out["oil_lag_1"].isna().all()
    assert np.allclose(out["keep"], X["keep"])


def test_clip_spike_columns_caps_outliers():
    X = pd.DataFrame({"sales_lag_1": [1.0, 2.0, 3.0, 1000.0]})
    out = clip_spike_columns(X, ["sales_lag_1"], upper_quantile=0.75)
    assert out["sales_lag_1"].max() < 1000.0


def test_relative_noise_rejects_negative_scale():
    with pytest.raises(ValueError):
        relative_noise(pd.DataFrame({"a": [1.0]}), ["a"], relative=-0.1)
