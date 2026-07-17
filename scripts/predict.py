"""Generate Kaggle submission from a GBDT experiment config.

Loads fold models from a run directory (or retrains on full train for LightGBM),
runs recursive multi-step inference over the test horizon with train history,
clips negatives when configured, validates against sample_submission, and writes
a submission CSV.

Usage:
  uv run python scripts/predict.py --config configs/final.yaml
  uv run python scripts/predict.py --config configs/experiments/030_catboost_locked_groups.yaml \\
      --output outputs/submissions/submission_catboost_030.csv
  uv run python scripts/predict.py --config configs/experiments/031_xgboost_locked_groups.yaml \\
      --output outputs/submissions/submission_xgboost_031.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from store_sales.config import ProjectPaths, load_default_config, load_yaml
from store_sales.features.registry import FEATURE_GROUPS, build_feature_matrix
from store_sales.io.logging import get_logger
from store_sales.io.submission import (
    align_submission_to_sample,
    clip_negative_sales,
    validate_submission,
    write_submission,
)
from store_sales.models.gbdt import transform_target

# Reuse train helpers (mask extras, recursive multi-step, fitters).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))
import train as train_mod  # noqa: E402

logger = get_logger(__name__)


class _BoosterModel:
    """Predict adapter for a saved LightGBM booster (matches stress_test)."""

    def __init__(self, path: Path) -> None:
        self._booster = lgb.Booster(model_file=str(path))
        self._names = list(self._booster.feature_name())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        frame = X.reindex(columns=self._names).copy()
        cols: list[np.ndarray] = []
        for col in self._names:
            s = frame[col]
            if isinstance(s.dtype, pd.CategoricalDtype) or str(s.dtype) == "category":
                codes = s.cat.codes.to_numpy(dtype=float)
                codes[codes < 0] = np.nan
                cols.append(codes)
            else:
                cols.append(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))
        mat = np.column_stack(cols) if cols else np.empty((len(frame), 0))
        return np.asarray(self._booster.predict(mat), dtype=float)


class _CatBoostFoldModel:
    """Load a saved CatBoost .cbm and cast categorical columns to str at predict."""

    def __init__(self, path: Path, cat_cols: list[str] | None = None) -> None:
        from catboost import CatBoostRegressor

        self._model = CatBoostRegressor()
        self._model.load_model(str(path))
        names = list(getattr(self._model, "feature_names_", None) or [])
        if not names and hasattr(self._model, "feature_names_"):
            names = list(self._model.feature_names_ or [])
        self._names = names
        # Default: treat non-numeric training cats as string cats (family).
        default_cats = [c for c in ("family", "city", "state", "type") if c in names]
        self._cat_cols = list(cat_cols) if cat_cols is not None else default_cats

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._names:
            frame = X.reindex(columns=self._names).copy()
        else:
            frame = X.copy()
        for col in self._cat_cols:
            if col in frame.columns:
                # Training used str cats (wrapper); categorical codes must not leak.
                if isinstance(frame[col].dtype, pd.CategoricalDtype):
                    frame[col] = frame[col].astype(str)
                else:
                    frame[col] = frame[col].astype(str)
        return np.asarray(self._model.predict(frame), dtype=float)


class _XGBoostFoldModel:
    """Load a saved XGBoost model JSON and align feature columns.

    Training used pandas Categorical for cat columns (store_nbr, family, …).
    XGBoost 2.x records that feature type; at predict time those columns must
    remain categorical (not cast to float codes).
    """

    def __init__(self, path: Path) -> None:
        import xgboost as xgb

        self._model = xgb.XGBRegressor()
        self._model.load_model(str(path))
        names = getattr(self._model, "feature_names_in_", None)
        if names is not None:
            self._names = list(names)
        else:
            booster_names = self._model.get_booster().feature_names
            self._names = list(booster_names) if booster_names else []
        # Columns treated as categorical during train (_CAT_CANDIDATES).
        self._cat_like = {
            c for c in train_mod._CAT_CANDIDATES if c in (self._names or [])
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._names:
            frame = X.reindex(columns=self._names).copy()
        else:
            frame = X.copy()
        for col in frame.columns:
            s = frame[col]
            if col in self._cat_like:
                # Preserve / restore categorical dtype for XGB cat features.
                if not (
                    isinstance(s.dtype, pd.CategoricalDtype) or str(s.dtype) == "category"
                ):
                    frame[col] = pd.Categorical(s.astype(str) if s.dtype == object else s)
            elif not pd.api.types.is_numeric_dtype(s):
                if isinstance(s.dtype, pd.CategoricalDtype) or str(s.dtype) == "category":
                    # Non-registered cats: numeric codes as fallback.
                    codes = s.cat.codes.to_numpy(dtype=float)
                    codes[codes < 0] = np.nan
                    frame[col] = codes
                else:
                    frame[col] = pd.to_numeric(s, errors="coerce")
        return np.asarray(self._model.predict(frame), dtype=float)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_path(root: Path, maybe_rel: str | Path) -> Path:
    path = Path(maybe_rel)
    if path.is_absolute():
        return path
    return root / path


def _feature_columns(groups: list[str]) -> list[str]:
    cols: list[str] = []
    for group in groups:
        if group not in FEATURE_GROUPS:
            raise ValueError(f"unknown feature group: {group!r}")
        cols.extend(FEATURE_GROUPS[group])
    return list(dict.fromkeys(cols))


def _build_panel(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    entity_cols: list[str],
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Concatenate train∪test; null test targets so lag/rolling stay PIT-safe."""
    keep = list(
        dict.fromkeys(
            [date_col, *entity_cols, "onpromotion", "id", target_col]
        )
    )
    tr = train.copy()
    te = test.copy()
    tr[date_col] = pd.to_datetime(tr[date_col])
    te[date_col] = pd.to_datetime(te[date_col])
    if target_col not in te.columns:
        te[target_col] = np.nan
    else:
        te[target_col] = np.nan
    for frame in (tr, te):
        for col in keep:
            if col not in frame.columns and col != target_col:
                raise KeyError(f"panel missing column {col!r}")
    panel = pd.concat([tr[keep], te[keep]], ignore_index=True)
    panel[date_col] = pd.to_datetime(panel[date_col])
    return panel


def _fit_cat_maps_from_train(
    train: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, pd.Index]:
    """Category levels from train only (matches fold-local encode in train.py)."""
    cat_cols = [c for c in train_mod._CAT_CANDIDATES if c in feature_cols and c in train.columns]
    maps: dict[str, pd.Index] = {}
    for col in cat_cols:
        if pd.api.types.is_numeric_dtype(train[col]):
            maps[col] = pd.Index(sorted(train[col].dropna().unique()))
        else:
            maps[col] = pd.Index(sorted(train[col].dropna().astype(str).unique()))
    return maps


def _load_fold_models(run_dir: Path, model_name: str) -> list[Any]:
    """Load fold models for lightgbm / catboost / xgboost from a run dir."""
    models_dir = run_dir / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"missing models dir: {models_dir}")

    name = model_name.lower().strip()
    if name == "lightgbm":
        paths = sorted(models_dir.glob("fold_*.txt"))
        if not paths:
            raise FileNotFoundError(f"no fold_*.txt models under {models_dir}")
        models: list[Any] = [_BoosterModel(p) for p in paths]
    elif name == "catboost":
        paths = sorted(models_dir.glob("fold_*.cbm"))
        if not paths:
            raise FileNotFoundError(f"no fold_*.cbm models under {models_dir}")
        models = [_CatBoostFoldModel(p) for p in paths]
    elif name == "xgboost":
        paths = sorted(models_dir.glob("fold_*.json"))
        if not paths:
            raise FileNotFoundError(f"no fold_*.json models under {models_dir}")
        models = [_XGBoostFoldModel(p) for p in paths]
    else:
        raise ValueError(
            f"predict load mode supports lightgbm/catboost/xgboost; got model={model_name!r}"
        )

    logger.info(
        "Loaded %d fold models (%s) from %s",
        len(models),
        name,
        models_dir,
    )
    return models


def _retrain_full_model(
    *,
    panel_train: pd.DataFrame,
    groups: list[str],
    extras: dict[str, pd.DataFrame],
    entity_cols: list[str],
    date_col: str,
    target_col: str,
    feature_cols: list[str],
    model_params: dict[str, Any],
    target_transform: str,
    early_stopping_rounds: int | None,
    seed: int,
    use_gpu: bool,
    es_holdout_days: int = 15,
) -> tuple[Any, dict[str, pd.Index]]:
    """Fit one LightGBM on full train with last-N-day ES holdout."""
    train_end = pd.Timestamp(panel_train[date_col].max())
    es_start = train_end - pd.Timedelta(days=int(es_holdout_days) - 1)

    # Build features with no post-origin mask (all labels are train).
    featured = build_feature_matrix(
        panel_train,
        groups=groups,
        extras=extras,
        entity_cols=entity_cols,
        date_col=date_col,
        target_col=target_col,
    )
    featured[date_col] = pd.to_datetime(featured[date_col])
    missing = [c for c in feature_cols if c not in featured.columns]
    if missing:
        raise KeyError(f"feature columns missing after build: {missing}")

    X_all = featured[feature_cols].copy()
    y_all = featured[target_col].to_numpy(dtype=float)
    dates = featured[date_col]
    es_mask = (dates >= es_start).to_numpy()
    fit_mask = ~es_mask
    use_es = int(fit_mask.sum()) >= 1000 and int(es_mask.sum()) >= 100
    if not use_es:
        logger.warning("ES holdout too small; fitting without early stopping")
        X_fit = X_all.reset_index(drop=True)
        y_fit = y_all
        X_es = X_all.iloc[:0].copy()
        y_es_t = np.asarray([], dtype=float)
        early_stopping_rounds = None
    else:
        X_fit = X_all.loc[fit_mask].reset_index(drop=True)
        y_fit = y_all[fit_mask]
        X_es = X_all.loc[es_mask].reset_index(drop=True)
        y_es_t = transform_target(y_all[es_mask], target_transform)

    X_fit, X_es, cat_cols, cat_maps = train_mod._encode_categoricals(X_fit, X_es)
    y_fit_t = transform_target(y_fit, target_transform)

    model = train_mod._fit_gbdt_model(
        model_name="lightgbm",
        X_fit=X_fit,
        y_fit_t=y_fit_t,
        X_es=X_es,
        y_es_t=y_es_t,
        model_params=model_params,
        seed=seed,
        use_gpu=use_gpu,
        early_stopping_rounds=early_stopping_rounds,
        cat_cols=cat_cols,
    )
    logger.info(
        "Retrained full-train LightGBM best_iteration=%s",
        train_mod._best_iteration(model),
    )
    return model, cat_maps


def _recursive_test_predict(
    *,
    model: Any,
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    groups: list[str],
    extras: dict[str, pd.DataFrame],
    entity_cols: list[str],
    date_col: str,
    target_col: str,
    feature_cols: list[str],
    cat_maps: dict[str, pd.Index],
    target_transform: str,
    clip_negative_preds: bool,
) -> pd.DataFrame:
    """Day-by-day recursive forecast on test; return id/entity/date/y_pred."""
    # y_true is unused for test; recursive helper still returns it.
    _y_true, y_pred, meta = train_mod._recursive_val_predict(
        model=model,
        panel=panel,
        train_end=train_end,
        val_start=test_start,
        val_end=test_end,
        groups=groups,
        extras=extras,
        entity_cols=entity_cols,
        date_col=date_col,
        target_col=target_col,
        feature_cols=feature_cols,
        cat_maps=cat_maps,
        target_transform=target_transform,
        clip_negative_preds=clip_negative_preds,
    )
    out = meta.copy()
    out["y_pred"] = y_pred
    return out


def predict_test(
    *,
    cfg: dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
    interim_dir: Path,
    run_dir: Path,
    mode: str = "load",
) -> pd.DataFrame:
    """Return test predictions with columns id, store_nbr, family, date, y_pred."""
    model_cfg = cfg.get("model") or {}
    model_name = str(model_cfg.get("name", "lightgbm"))
    groups = list(cfg.get("feature_groups") or ["base"])
    feature_cols = _feature_columns(groups)
    entity_cols = list(cfg.get("entity_cols", ["store_nbr", "family"]))
    date_col = str(cfg.get("date_col", "date"))
    target_col = str(cfg.get("target_col", "sales"))
    target_transform = str(cfg.get("target_transform", "log1p"))
    clip_negative_preds = bool(cfg.get("clip_negative_preds", True))
    seed = int(cfg.get("primary_seed", cfg.get("seed", 42)))
    use_gpu = bool(cfg.get("gpu", True))
    model_params = dict(model_cfg.get("params") or {})
    early_stopping_rounds = model_cfg.get("early_stopping_rounds", 50)
    if early_stopping_rounds is not None:
        early_stopping_rounds = int(early_stopping_rounds)

    extras = train_mod._load_feature_extras(groups, interim_dir)
    panel = _build_panel(
        train,
        test,
        entity_cols=entity_cols,
        date_col=date_col,
        target_col=target_col,
    )
    train_end = pd.Timestamp(train[date_col].max())
    test_start = pd.Timestamp(test[date_col].min())
    test_end = pd.Timestamp(test[date_col].max())
    recursive = train_mod._needs_recursive_forecast(groups)

    logger.info(
        "Predict mode=%s groups=%s recursive=%s train_end=%s test=%s..%s n_test=%d",
        mode,
        groups,
        recursive,
        train_end.date(),
        test_start.date(),
        test_end.date(),
        len(test),
    )

    cat_maps = _fit_cat_maps_from_train(train, feature_cols)
    pred_parts: list[pd.DataFrame] = []

    if mode == "load":
        models = _load_fold_models(run_dir, model_name)
        for i, model in enumerate(models):
            logger.info("Recursive forecast with fold model %d/%d", i + 1, len(models))
            part = _recursive_test_predict(
                model=model,
                panel=panel,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                groups=groups,
                extras=extras,
                entity_cols=entity_cols,
                date_col=date_col,
                target_col=target_col,
                feature_cols=feature_cols,
                cat_maps=cat_maps,
                target_transform=target_transform,
                clip_negative_preds=clip_negative_preds,
            )
            part = part.rename(columns={"y_pred": f"y_pred_{i}"})
            pred_parts.append(part)
        # Mean ensemble across fold models
        base = pred_parts[0][list(entity_cols) + [date_col] + (
            ["id"] if "id" in pred_parts[0].columns else []
        )].copy()
        stack = np.column_stack(
            [p[f"y_pred_{i}"].to_numpy(dtype=float) for i, p in enumerate(pred_parts)]
        )
        base["y_pred"] = stack.mean(axis=1)
        preds = base
    elif mode == "retrain":
        keep_tr = list(
            dict.fromkeys([date_col, *entity_cols, target_col, "onpromotion", "id"])
        )
        panel_train = train.copy()
        panel_train[date_col] = pd.to_datetime(panel_train[date_col])
        panel_train = panel_train[
            [c for c in keep_tr if c in panel_train.columns]
        ].copy()
        model, cat_maps = _retrain_full_model(
            panel_train=panel_train,
            groups=groups,
            extras=extras,
            entity_cols=entity_cols,
            date_col=date_col,
            target_col=target_col,
            feature_cols=feature_cols,
            model_params=model_params,
            target_transform=target_transform,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
            use_gpu=use_gpu,
        )
        preds = _recursive_test_predict(
            model=model,
            panel=panel,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            groups=groups,
            extras=extras,
            entity_cols=entity_cols,
            date_col=date_col,
            target_col=target_col,
            feature_cols=feature_cols,
            cat_maps=cat_maps,
            target_transform=target_transform,
            clip_negative_preds=clip_negative_preds,
        )
    else:
        raise ValueError(f"Unknown predict mode: {mode!r} (use 'load' or 'retrain')")

    if not recursive and mode == "load":
        # Non-recursive path is handled inside recursive helper only when needed;
        # for completeness, single-shot path could be added later.
        pass

    preds["y_pred"] = clip_negative_sales(
        preds["y_pred"].to_numpy(dtype=float),
        enabled=clip_negative_preds,
    )
    if "id" not in preds.columns:
        # Recover ids from test
        te = test[[*entity_cols, date_col, "id"]].copy()
        te[date_col] = pd.to_datetime(te[date_col])
        preds = preds.merge(te, on=list(entity_cols) + [date_col], how="left")
    if preds["id"].isna().any():
        raise RuntimeError("Failed to align prediction ids with test rows")

    logger.info(
        "Predictions ready n=%d sales[min,max,mean]=[%.4f, %.4f, %.4f]",
        len(preds),
        float(preds["y_pred"].min()),
        float(preds["y_pred"].max()),
        float(preds["y_pred"].mean()),
    )
    return preds


def run_predict(
    *,
    config_path: Path,
    mode: str = "load",
    output_path: Path | None = None,
    run_dir_override: Path | None = None,
) -> Path:
    paths = ProjectPaths()
    default_cfg = load_default_config()
    exp_cfg = load_yaml(config_path)
    cfg = _deep_merge(default_cfg, exp_cfg)

    path_cfg = cfg.get("paths") or {}
    interim_dir = paths.root / path_cfg.get("interim_dir", "data/interim")
    outputs_dir = paths.root / path_cfg.get("outputs_dir", "outputs")

    artifacts = cfg.get("artifacts") or {}
    if run_dir_override is not None:
        run_dir = _resolve_path(paths.root, run_dir_override)
    else:
        run_dir_rel = artifacts.get("run_dir") or f"outputs/runs/{cfg.get('run_id', 'final')}"
        run_dir = _resolve_path(paths.root, run_dir_rel)

    train = pd.read_parquet(interim_dir / "train.parquet")
    test = pd.read_parquet(interim_dir / "test.parquet")
    sample = pd.read_parquet(interim_dir / "sample_submission.parquet")

    preds = predict_test(
        cfg=cfg,
        train=train,
        test=test,
        interim_dir=interim_dir,
        run_dir=run_dir,
        mode=mode,
    )
    sub = align_submission_to_sample(
        preds.rename(columns={"y_pred": "sales"}),
        sample,
    )
    validate_submission(sub, sample)

    model_name = str((cfg.get("model") or {}).get("name", "model"))
    run_id = str(cfg.get("run_id", "run"))
    # Keep locked finalist path stable for configs/final.yaml.
    if output_path is not None:
        out = Path(output_path)
    elif Path(config_path).name == "final.yaml":
        out = outputs_dir / "submissions" / "submission.csv"
    else:
        out = outputs_dir / "submissions" / f"submission_{model_name}_{run_id}.csv"
    out = Path(out)
    write_submission(sub, out, sample=sample)
    logger.info("Wrote submission %s rows=%d", out, len(sub))
    print(f"submission={out} rows={len(sub)} mode={mode} run_dir={run_dir} model={model_name}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict test sales and write Kaggle submission.csv"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final.yaml"),
        help="Experiment or final config (default: configs/final.yaml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("load", "retrain"),
        default="load",
        help="load=average fold models from run_dir; retrain=full-train refit (LGBM path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override submission path (default outputs/submissions/submission.csv)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override run dir with fold models (default: outputs/runs/<run_id>)",
    )
    args = parser.parse_args()
    run_predict(
        config_path=args.config,
        mode=args.mode,
        output_path=args.output,
        run_dir_override=args.run_dir,
    )


if __name__ == "__main__":
    main()
