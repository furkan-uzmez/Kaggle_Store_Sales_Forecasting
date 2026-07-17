"""LightGBM wrapper: CPU path, GPU fallback, seed, early stopping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from store_sales.models.gbdt import fit_lgbm, inverse_target, transform_target


def _xy(n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "cat": rng.integers(0, 4, size=n),
        }
    )
    y = 0.5 * X["f0"] + 0.2 * X["f1"] + 0.1 * X["cat"] + rng.normal(0, 0.1, size=n)
    return X, y


def test_fit_lgbm_cpu_smoke_and_predict():
    X, y = _xy()
    X_train, X_val = X.iloc[:90], X.iloc[90:]
    y_train, y_val = y[:90], y[90:]
    model = fit_lgbm(
        X_train,
        y_train,
        X_val,
        y_val,
        params={"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 15},
        seed=42,
        use_gpu=False,
        early_stopping_rounds=10,
        categorical_feature=["cat"],
    )
    pred = model.predict(X_val)
    assert len(pred) == len(X_val)
    assert np.isfinite(pred).all()


def test_fit_lgbm_gpu_request_falls_back_or_succeeds():
    """When GPU is requested, return a working model even if GPU is unavailable."""
    X, y = _xy(n=80)
    model = fit_lgbm(
        X.iloc[:60],
        y[:60],
        X.iloc[60:],
        y[60:],
        params={"n_estimators": 20, "num_leaves": 7, "learning_rate": 0.1},
        seed=7,
        use_gpu=True,
        early_stopping_rounds=5,
    )
    pred = model.predict(X.iloc[60:])
    assert np.isfinite(pred).all()


def test_transform_target_log1p_roundtrip():
    y = np.array([0.0, 1.0, 10.0, 100.0])
    z = transform_target(y, "log1p")
    back = inverse_target(z, "log1p")
    assert np.allclose(back, y)
    with pytest.raises(ValueError):
        transform_target(y, "unknown")


def test_fit_lgbm_respects_seed_for_finite_output():
    X, y = _xy(n=100, seed=1)
    kwargs = dict(
        X_train=X.iloc[:70],
        y_train=y[:70],
        X_val=X.iloc[70:],
        y_val=y[70:],
        params={"n_estimators": 25, "num_leaves": 8, "learning_rate": 0.1},
        use_gpu=False,
        early_stopping_rounds=None,
    )
    m1 = fit_lgbm(**kwargs, seed=42)
    m2 = fit_lgbm(**kwargs, seed=42)
    p1 = m1.predict(X.iloc[70:])
    p2 = m2.predict(X.iloc[70:])
    assert np.allclose(p1, p2)
