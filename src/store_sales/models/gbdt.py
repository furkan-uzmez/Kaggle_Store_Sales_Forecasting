"""Gradient boosting wrappers: LightGBM, CatBoost, XGBoost."""

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
_CB_DEVICE_CACHE: dict[bool, str] = {}
_XGB_DEVICE_CACHE: dict[bool, str] = {}


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


def _resolve_catboost_device(use_gpu: bool) -> str:
    if not use_gpu:
        return "CPU"
    if use_gpu in _CB_DEVICE_CACHE:
        return _CB_DEVICE_CACHE[use_gpu]
    try:
        from catboost import CatBoostRegressor

        X = np.zeros((8, 2), dtype=np.float32)
        y = np.zeros(8, dtype=np.float32)
        m = CatBoostRegressor(
            iterations=1,
            depth=2,
            task_type="GPU",
            verbose=False,
            allow_writing_files=False,
        )
        m.fit(X, y)
        device = "GPU"
    except Exception as exc:  # noqa: BLE001
        logger.warning("CatBoost GPU unavailable (%s); falling back to CPU", exc)
        device = "CPU"
    _CB_DEVICE_CACHE[use_gpu] = device
    return device


def fit_catboost(
    X_train: pd.DataFrame | np.ndarray,
    y_train: ArrayLike,
    X_val: pd.DataFrame | np.ndarray | None,
    y_val: ArrayLike | None,
    params: Mapping[str, Any] | None,
    seed: int,
    use_gpu: bool = True,
    early_stopping_rounds: int | None = 50,
    categorical_feature: Sequence[str] | str | None = None,
) -> Any:
    """Fit a CatBoost regressor with optional val early-stopping and GPU fallback."""
    from catboost import CatBoostRegressor

    task_type = _resolve_catboost_device(bool(use_gpu))
    raw = dict(params or {})
    iterations = int(
        raw.pop("n_estimators", raw.pop("iterations", raw.pop("num_boost_round", 500)))
    )
    learning_rate = float(raw.pop("learning_rate", raw.pop("eta", 0.05)))
    depth = int(raw.pop("depth", raw.pop("max_depth", 8)))
    # Drop LGBM-only / alias keys that would confuse CatBoost
    for drop_key in (
        "num_leaves",
        "min_child_samples",
        "colsample_bytree",
        "feature_fraction",
        "bagging_fraction",
        "subsample",
        "device",
        "random_state",
        "random_seed",
        "task_type",
        "verbose",
        "allow_writing_files",
    ):
        raw.pop(drop_key, None)
    # CatBoost uses rsm / subsample differently; keep if user passed catboost names
    subsample = raw.pop("subsample", None)
    if subsample is not None and "subsample" not in raw:
        # bagging temperature / subsample — map bagging_fraction-style to subsample
        raw.setdefault("subsample", float(subsample))

    cat_cols: list[str] | None = None
    if categorical_feature is not None:
        cat_cols = (
            [categorical_feature]
            if isinstance(categorical_feature, str)
            else list(categorical_feature)
        )

    # CatBoost is faster/more reliable with string cats than pandas Categorical.
    X_tr: pd.DataFrame | np.ndarray = X_train
    X_va: pd.DataFrame | np.ndarray | None = X_val
    present_cats: list[str] = []
    if cat_cols is not None and isinstance(X_train, pd.DataFrame):
        present_cats = [c for c in cat_cols if c in X_train.columns]
        X_tr = X_train.copy()
        for c in present_cats:
            X_tr[c] = X_tr[c].astype(str)
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_va = X_val.copy()
            for c in present_cats:
                if c in X_va.columns:
                    X_va[c] = X_va[c].astype(str)

    # Default thread budget if not provided (avoid oversubscription on shared boxes).
    raw.setdefault("thread_count", 8)

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=int(seed),
        task_type=task_type,
        verbose=False,
        allow_writing_files=False,
        **raw,
    )

    y_tr = np.asarray(y_train, dtype=float)
    fit_kwargs: dict[str, Any] = {}
    if present_cats:
        fit_kwargs["cat_features"] = present_cats

    use_es = (
        X_va is not None
        and y_val is not None
        and early_stopping_rounds is not None
        and int(early_stopping_rounds) > 0
    )
    if use_es:
        fit_kwargs["eval_set"] = (X_va, np.asarray(y_val, dtype=float))
        fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
        fit_kwargs["use_best_model"] = True

    logger.info(
        "Fitting CatBoost task_type=%s iterations=%s depth=%s lr=%s",
        task_type,
        iterations,
        depth,
        learning_rate,
    )
    try:
        model.fit(X_tr, y_tr, **fit_kwargs)
    except Exception as exc:
        if task_type != "GPU":
            raise
        logger.warning("CatBoost fit failed on GPU (%s); retrying on CPU", exc)
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=int(seed),
            task_type="CPU",
            verbose=False,
            allow_writing_files=False,
            **raw,
        )
        model.fit(X_tr, y_tr, **fit_kwargs)

    best = getattr(model, "best_iteration_", None)
    if best is None and hasattr(model, "get_best_iteration"):
        try:
            best = model.get_best_iteration()
        except Exception:  # noqa: BLE001
            best = None
    logger.info(
        "CatBoost fit complete best_iteration=%s",
        best,
    )
    if present_cats:
        return _CatBoostStringCatWrapper(model, present_cats)
    return model


class _CatBoostStringCatWrapper:
    """Ensure predict/path paths cast cat columns to str (matches fit)."""

    def __init__(self, model: Any, cat_cols: Sequence[str]) -> None:
        self._model = model
        self._cat_cols = list(cat_cols)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def predict(self, X: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(X, pd.DataFrame) and self._cat_cols:
            X = X.copy()
            for c in self._cat_cols:
                if c in X.columns:
                    X[c] = X[c].astype(str)
        return self._model.predict(X, *args, **kwargs)

    def save_model(self, path: str, *args: Any, **kwargs: Any) -> Any:
        return self._model.save_model(path, *args, **kwargs)

    @property
    def best_iteration_(self) -> Any:
        return getattr(self._model, "best_iteration_", None)

    def get_best_iteration(self) -> Any:
        return self._model.get_best_iteration()


def _resolve_xgboost_device(use_gpu: bool) -> str:
    """Return tree_method / device hint: 'hist' (CPU) or 'gpu_hist'."""
    if not use_gpu:
        return "hist"
    if use_gpu in _XGB_DEVICE_CACHE:
        return _XGB_DEVICE_CACHE[use_gpu]
    try:
        import xgboost as xgb

        X = np.zeros((8, 2), dtype=np.float32)
        y = np.zeros(8, dtype=np.float32)
        m = xgb.XGBRegressor(
            n_estimators=1,
            max_depth=2,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        m.fit(X, y)
        device = "cuda"
    except Exception as exc:  # noqa: BLE001
        logger.warning("XGBoost GPU unavailable (%s); falling back to CPU", exc)
        device = "cpu"
    _XGB_DEVICE_CACHE[use_gpu] = device
    return device


def fit_xgboost(
    X_train: pd.DataFrame | np.ndarray,
    y_train: ArrayLike,
    X_val: pd.DataFrame | np.ndarray | None,
    y_val: ArrayLike | None,
    params: Mapping[str, Any] | None,
    seed: int,
    use_gpu: bool = True,
    early_stopping_rounds: int | None = 50,
    categorical_feature: Sequence[str] | str | None = None,
) -> Any:
    """Fit an XGBoost regressor with optional val early-stopping and GPU fallback."""
    import xgboost as xgb

    device = _resolve_xgboost_device(bool(use_gpu))
    raw = dict(params or {})
    n_estimators = int(raw.pop("n_estimators", raw.pop("num_boost_round", 500)))
    learning_rate = float(raw.pop("learning_rate", raw.pop("eta", 0.05)))
    max_depth = int(raw.pop("max_depth", raw.pop("depth", 8)))
    subsample = float(raw.pop("subsample", raw.pop("bagging_fraction", 0.8)))
    colsample_bytree = float(
        raw.pop("colsample_bytree", raw.pop("feature_fraction", 0.8))
    )
    # Drop LGBM-only keys
    for drop_key in (
        "num_leaves",
        "min_child_samples",
        "device",
        "random_state",
        "tree_method",
        "verbosity",
        "enable_categorical",
    ):
        raw.pop(drop_key, None)

    enable_cat = False
    X_tr = X_train
    X_va = X_val
    if categorical_feature is not None and isinstance(X_train, pd.DataFrame):
        cat_cols = (
            [categorical_feature]
            if isinstance(categorical_feature, str)
            else list(categorical_feature)
        )
        X_tr = X_train.copy()
        for col in cat_cols:
            if col in X_tr.columns and not isinstance(
                X_tr[col].dtype, pd.CategoricalDtype
            ):
                X_tr[col] = X_tr[col].astype("category")
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_va = X_val.copy()
            for col in cat_cols:
                if col in X_va.columns:
                    cats = (
                        X_tr[col].cat.categories
                        if isinstance(X_tr[col].dtype, pd.CategoricalDtype)
                        else None
                    )
                    if cats is not None:
                        X_va[col] = pd.Categorical(X_va[col], categories=cats)
                    else:
                        X_va[col] = X_va[col].astype("category")
        enable_cat = True

    model_kwargs: dict[str, Any] = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=int(seed),
        tree_method="hist",
        device=device if device == "cuda" else "cpu",
        verbosity=0,
        enable_categorical=enable_cat,
        **raw,
    )

    use_es = (
        X_va is not None
        and y_val is not None
        and early_stopping_rounds is not None
        and int(early_stopping_rounds) > 0
    )
    if use_es:
        # XGBoost 2+/3+: early_stopping_rounds on constructor
        model_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)

    model = xgb.XGBRegressor(**model_kwargs)

    y_tr = np.asarray(y_train, dtype=float)
    fit_kwargs: dict[str, Any] = {}
    if use_es:
        fit_kwargs["eval_set"] = [(X_va, np.asarray(y_val, dtype=float))]
        fit_kwargs["verbose"] = False

    logger.info(
        "Fitting XGBoost device=%s n_estimators=%s max_depth=%s lr=%s",
        device,
        n_estimators,
        max_depth,
        learning_rate,
    )
    try:
        model.fit(X_tr, y_tr, **fit_kwargs)
    except Exception as exc:
        if device != "cuda":
            raise
        logger.warning("XGBoost fit failed on GPU (%s); retrying on CPU", exc)
        model_kwargs["device"] = "cpu"
        model = xgb.XGBRegressor(**model_kwargs)
        model.fit(X_tr, y_tr, **fit_kwargs)

    best = getattr(model, "best_iteration", None)
    logger.info("XGBoost fit complete best_iteration=%s", best)
    return model
