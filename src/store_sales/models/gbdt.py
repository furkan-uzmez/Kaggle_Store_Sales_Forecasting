"""Gradient boosting wrappers (LightGBM first; CatBoost/XGBoost later)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from store_sales.io.logging import get_logger

logger = get_logger(__name__)

# Cache GPU probe so multi-fold training does not re-init OpenCL each fold.
_DEVICE_CACHE: dict[bool, str] = {}


def transform_target(y: ArrayLike, transform: str) -> np.ndarray:
    """Apply training-time target transform (e.g. log1p)."""
    arr = np.asarray(y, dtype=float)
    if transform == "log1p":
        return np.log1p(np.clip(arr, a_min=0.0, a_max=None))
    if transform in ("none", "identity", None):
        return arr
    raise ValueError(f"Unsupported target transform: {transform!r}")


def inverse_target(y_hat: ArrayLike, transform: str) -> np.ndarray:
    """Invert target transform for metric scale (physical units)."""
    arr = np.asarray(y_hat, dtype=float)
    if transform == "log1p":
        return np.expm1(arr)
    if transform in ("none", "identity", None):
        return arr
    raise ValueError(f"Unsupported target transform: {transform!r}")


def _resolve_device(use_gpu: bool) -> str:
    if not use_gpu:
        return "cpu"
    if use_gpu in _DEVICE_CACHE:
        return _DEVICE_CACHE[use_gpu]
    # Probe with a tiny Dataset; OpenCL builds fail loudly when no device.
    try:
        X = np.zeros((8, 2), dtype=np.float32)
        y = np.zeros(8, dtype=np.float32)
        ds = lgb.Dataset(X, label=y, free_raw_data=True)
        lgb.train(
            {
                "objective": "regression",
                "device": "gpu",
                "verbosity": -1,
                "num_leaves": 2,
                "min_data_in_leaf": 1,
            },
            ds,
            num_boost_round=1,
        )
        device = "gpu"
    except Exception as exc:  # noqa: BLE001 — any GPU init failure → CPU
        logger.warning("LightGBM GPU unavailable (%s); falling back to CPU", exc)
        device = "cpu"
    _DEVICE_CACHE[use_gpu] = device
    return device


def fit_lgbm(
    X_train: pd.DataFrame | np.ndarray,
    y_train: ArrayLike,
    X_val: pd.DataFrame | np.ndarray | None,
    y_val: ArrayLike | None,
    params: Mapping[str, Any] | None,
    seed: int,
    use_gpu: bool = True,
    early_stopping_rounds: int | None = 50,
    categorical_feature: Sequence[str] | str | None = None,
) -> lgb.LGBMRegressor:
    """Fit a LightGBM regressor with optional val early-stopping and GPU fallback.

    Parameters
    ----------
    X_train, y_train:
        Training features and labels (already transformed if using log1p).
    X_val, y_val:
        Optional validation set for early stopping / eval.
    params:
        LGBMRegressor kwargs (n_estimators, learning_rate, num_leaves, ...).
        ``device`` / ``random_state`` are set by this function.
    seed:
        Passed as ``random_state``.
    use_gpu:
        Prefer GPU when available; fall back to CPU on failure.
    early_stopping_rounds:
        If set and val is provided, stop when val loss stalls.
    categorical_feature:
        Column names for LightGBM native categoricals, or None.

    Returns
    -------
    Fitted ``lgb.LGBMRegressor``.
    """
    device = _resolve_device(bool(use_gpu))
    raw = dict(params or {})
    # Map common aliases from experiment YAML
    n_estimators = int(raw.pop("n_estimators", raw.pop("num_boost_round", 500)))
    learning_rate = float(raw.pop("learning_rate", raw.pop("eta", 0.05)))
    num_leaves = int(raw.pop("num_leaves", 63))
    subsample = float(raw.pop("subsample", raw.pop("bagging_fraction", 0.8)))
    colsample_bytree = float(
        raw.pop("colsample_bytree", raw.pop("feature_fraction", 0.8))
    )
    # Remaining raw keys pass through (min_child_samples, reg_lambda, ...)
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=int(seed),
        device=device,
        verbosity=-1,
        **raw,
    )

    fit_kwargs: dict[str, Any] = {}
    if categorical_feature is not None:
        fit_kwargs["categorical_feature"] = list(categorical_feature)

    callbacks: list[Any] = []
    if (
        X_val is not None
        and y_val is not None
        and early_stopping_rounds is not None
        and int(early_stopping_rounds) > 0
    ):
        fit_kwargs["eval_set"] = [(X_val, np.asarray(y_val, dtype=float))]
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=int(early_stopping_rounds),
                verbose=False,
            )
        )
        callbacks.append(lgb.log_evaluation(period=0))
        fit_kwargs["callbacks"] = callbacks

    logger.info(
        "Fitting LightGBM device=%s n_estimators=%s num_leaves=%s lr=%s",
        device,
        n_estimators,
        num_leaves,
        learning_rate,
    )
    try:
        model.fit(
            X_train,
            np.asarray(y_train, dtype=float),
            **fit_kwargs,
        )
    except Exception as exc:
        if device != "gpu":
            raise
        logger.warning(
            "LightGBM fit failed on GPU (%s); retrying on CPU",
            exc,
        )
        model.set_params(device="cpu")
        model.fit(
            X_train,
            np.asarray(y_train, dtype=float),
            **fit_kwargs,
        )

    best = getattr(model, "best_iteration_", None)
    logger.info(
        "LightGBM fit complete best_iteration=%s n_features=%s",
        best,
        getattr(model, "n_features_in_", None),
    )
    return model
