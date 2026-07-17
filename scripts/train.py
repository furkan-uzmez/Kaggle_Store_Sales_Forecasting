"""Score baseline (and GBDT) configs under fixed walk-forward folds.

Writes ``outputs/runs/<run_id>/`` with config, metrics, environment, metadata.
LightGBM runs also persist fold metrics, OOF predictions, and model artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from store_sales.config import ProjectPaths, load_default_config, load_yaml
from store_sales.features.registry import (
    FEATURE_GROUPS,
    build_feature_matrix,
    mask_target_after,
)
from store_sales.io.artifacts import save_run_dir
from store_sales.io.logging import get_logger
from store_sales.metrics.guards import mae_log1p
from store_sales.metrics.rmsle import rmsle
from store_sales.models.baseline import last_value_predict, seasonal_naive_predict
from store_sales.models.gbdt import fit_lgbm, inverse_target, transform_target

logger = get_logger(__name__)

# Columns treated as LightGBM categoricals when present in the feature matrix.
_LGBM_CAT_CANDIDATES: tuple[str, ...] = (
    "store_nbr",
    "family",
    "city",
    "state",
    "store_type",
    "cluster",
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_folds_meta(splits_dir: Path) -> list[dict[str, Any]]:
    path = splits_dir / "folds_meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def feature_columns_for_groups(groups: Sequence[str]) -> list[str]:
    """Union feature names for the requested groups (order-preserving unique)."""
    cols: list[str] = []
    for group in groups:
        if group not in FEATURE_GROUPS:
            raise ValueError(f"unknown feature group: {group!r}")
        cols.extend(FEATURE_GROUPS[group])
    return list(dict.fromkeys(cols))


def _predict_baseline(
    *,
    model_name: str,
    period: int | None,
    history: pd.DataFrame,
    future: pd.DataFrame,
    entity_cols: list[str],
    date_col: str,
    target_col: str,
    train_end: pd.Timestamp,
) -> pd.Series:
    if model_name == "last_value":
        return last_value_predict(
            history=history,
            future=future,
            entity_cols=entity_cols,
            date_col=date_col,
            target_col=target_col,
        )
    if model_name == "seasonal_naive":
        if period is None:
            raise ValueError("seasonal_naive requires model.period")
        return seasonal_naive_predict(
            history=history,
            future=future,
            entity_cols=entity_cols,
            date_col=date_col,
            target_col=target_col,
            period=int(period),
            train_end=train_end,
        )
    raise ValueError(f"Unsupported model.name for train stub: {model_name!r}")


def score_baselines(
    *,
    train: pd.DataFrame,
    folds_meta: list[dict[str, Any]],
    splits_dir: Path,
    model_name: str,
    period: int | None,
    entity_cols: list[str],
    date_col: str,
    target_col: str,
) -> dict[str, Any]:
    """Score a naive baseline on each fold; return metrics dict."""
    fold_rmsle: list[float] = []
    fold_mae_log1p: list[float] = []
    fold_rows: list[dict[str, Any]] = []

    # Keep only columns needed for baselines (memory).
    cols = list(dict.fromkeys([date_col, *entity_cols, target_col]))
    panel = train[cols].copy()
    panel[date_col] = pd.to_datetime(panel[date_col])

    for meta in folds_meta:
        fold = int(meta["fold"])
        train_end = pd.Timestamp(meta["train_end"])
        val_start = pd.Timestamp(meta["val_start"])
        val_end = pd.Timestamp(meta["val_end"])

        train_idx = pd.read_parquet(splits_dir / f"fold_{fold}_train_idx.parquet")[
            "idx"
        ].to_numpy()
        val_idx = pd.read_parquet(splits_dir / f"fold_{fold}_val_idx.parquet")[
            "idx"
        ].to_numpy()

        hist = panel.loc[train_idx]
        fut = panel.loc[val_idx]
        y_true = fut[target_col].to_numpy(dtype=float)

        pred = _predict_baseline(
            model_name=model_name,
            period=period,
            history=hist,
            future=fut,
            entity_cols=entity_cols,
            date_col=date_col,
            target_col=target_col,
            train_end=train_end,
        )
        y_pred = pred.to_numpy(dtype=float)
        # Clip negatives for log metrics (naives should be non-negative).
        y_pred = y_pred.clip(min=0.0)

        fold_score = rmsle(y_true, y_pred)
        fold_guard = mae_log1p(y_true, y_pred)
        fold_rmsle.append(fold_score)
        fold_mae_log1p.append(fold_guard)
        fold_rows.append(
            {
                "fold": fold,
                "rmsle": fold_score,
                "mae_log1p": fold_guard,
                "n_val": int(len(fut)),
                "val_start": str(val_start.date()),
                "val_end": str(val_end.date()),
                "train_end": str(train_end.date()),
            }
        )
        logger.info(
            "fold=%s model=%s rmsle=%.6f mae_log1p=%.6f n_val=%d",
            fold,
            model_name,
            fold_score,
            fold_guard,
            len(fut),
        )

    return {
        "mean_rmsle": float(sum(fold_rmsle) / len(fold_rmsle)),
        "std_rmsle": float(pd.Series(fold_rmsle).std(ddof=1)),
        "fold_rmsle": fold_rmsle,
        "mean_mae_log1p": float(sum(fold_mae_log1p) / len(fold_mae_log1p)),
        "fold_mae_log1p": fold_mae_log1p,
        "folds": fold_rows,
        "model_name": model_name,
        "period": period,
    }


def _load_feature_extras(
    groups: Sequence[str],
    interim_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load side tables only for groups that need them."""
    extras: dict[str, pd.DataFrame] = {}
    needed = set(groups)
    if "oil" in needed:
        extras["oil"] = pd.read_parquet(interim_dir / "oil.parquet")
    if "holiday" in needed:
        extras["holidays_events"] = pd.read_parquet(
            interim_dir / "holidays_events.parquet"
        )
        extras["stores"] = pd.read_parquet(interim_dir / "stores.parquet")
    if "store_meta" in needed:
        extras["stores"] = pd.read_parquet(interim_dir / "stores.parquet")
    if "transactions" in needed:
        extras["transactions"] = pd.read_parquet(
            interim_dir / "transactions.parquet"
        )
    return extras


def _prepare_lgbm_matrices(
    *,
    panel: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    groups: Sequence[str],
    extras: dict[str, pd.DataFrame],
    entity_cols: list[str],
    date_col: str,
    target_col: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Build PIT-safe fold matrices.

    Critical leakage rule: mask target after ``train_end`` **before** lag/rolling
    so true val sales cannot feed target-derived features. Calendar/promo/known
    future still use val-row covariates.
    """
    idx = np.unique(np.concatenate([train_idx, val_idx]))
    fold_panel = panel.loc[idx].copy()
    fold_panel[date_col] = pd.to_datetime(fold_panel[date_col])

    # Preserve original targets for labels before masking.
    y_lookup = fold_panel[list(entity_cols) + [date_col, target_col]].copy()

    masked = mask_target_after(
        fold_panel,
        train_end,
        date_col=date_col,
        target_col=target_col,
    )
    featured = build_feature_matrix(
        masked,
        groups=groups,
        extras=extras,
        entity_cols=entity_cols,
        date_col=date_col,
        target_col=target_col,
    )
    featured[date_col] = pd.to_datetime(featured[date_col])

    # Re-attach true sales (mask may have nulled val targets).
    drop_sales = [c for c in featured.columns if c == target_col]
    if drop_sales:
        featured = featured.drop(columns=drop_sales)
    featured = featured.merge(
        y_lookup,
        on=list(entity_cols) + [date_col],
        how="left",
        validate="one_to_one",
    )

    missing = [c for c in feature_cols if c not in featured.columns]
    if missing:
        raise KeyError(f"feature columns missing after build: {missing}")

    dates = featured[date_col]
    train_mask = dates <= train_end
    val_mask = (dates >= val_start) & (dates <= val_end)

    train_df = featured.loc[train_mask]
    val_df = featured.loc[val_mask]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].to_numpy(dtype=float)

    meta_val = val_df[list(entity_cols) + [date_col]].copy()
    if "id" in val_df.columns:
        meta_val["id"] = val_df["id"].to_numpy()

    return X_train, y_train, X_val, y_val, meta_val


def _fit_categorical_maps(
    X_train: pd.DataFrame,
) -> tuple[list[str], dict[str, pd.Index]]:
    """Fold-local category levels fit on train only."""
    cat_cols = [c for c in _LGBM_CAT_CANDIDATES if c in X_train.columns]
    maps: dict[str, pd.Index] = {}
    for col in cat_cols:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            maps[col] = pd.Index(sorted(X_train[col].dropna().unique()))
        else:
            maps[col] = pd.Index(sorted(X_train[col].dropna().astype(str).unique()))
    return cat_cols, maps


def _apply_categorical_maps(
    X: pd.DataFrame,
    cat_maps: dict[str, pd.Index],
) -> pd.DataFrame:
    out = X.copy()
    for col, cats in cat_maps.items():
        if col not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[col]) and pd.api.types.is_numeric_dtype(cats):
            out[col] = pd.Categorical(out[col], categories=cats)
        else:
            out[col] = pd.Categorical(out[col].astype(str), categories=cats.astype(str))
    return out


def _encode_categoricals(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, pd.Index]]:
    """Fold-local categorical handling: category dtype for known cat columns."""
    cat_cols, cat_maps = _fit_categorical_maps(X_train)
    X_tr = _apply_categorical_maps(X_train, cat_maps)
    X_va = _apply_categorical_maps(X_val, cat_maps)
    return X_tr, X_va, cat_cols, cat_maps


def _needs_recursive_forecast(groups: Sequence[str]) -> bool:
    """Target-derived groups require recursive multi-step fill under origin mask."""
    return any(g in groups for g in ("lag", "rolling"))


def _recursive_val_predict(
    *,
    model: Any,
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    groups: Sequence[str],
    extras: dict[str, pd.DataFrame],
    entity_cols: list[str],
    date_col: str,
    target_col: str,
    feature_cols: list[str],
    cat_maps: dict[str, pd.Index],
    target_transform: str,
    clip_negative_preds: bool,
    lookback_days: int = 56,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Predict val horizon day-by-day, writing preds into sales for next lags.

    Uses only train sales plus previously predicted val sales (no true val target).
    """
    work = panel.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    val_mask = (work[date_col] >= val_start) & (work[date_col] <= val_end)
    # Labels from original panel (before nulling post-origin sales).
    meta_cols = list(entity_cols) + [date_col] + (
        ["id"] if "id" in panel.columns else []
    )
    ordered_val = (
        panel.loc[val_mask]
        .assign(**{date_col: lambda d: pd.to_datetime(d[date_col])})
        .sort_values(list(entity_cols) + [date_col])
        .reset_index(drop=True)
    )
    y_true = ordered_val[target_col].to_numpy(dtype=float)
    meta_val = ordered_val[meta_cols].copy()
    # Drop true post-origin sales so lag/rolling cannot see them.
    work.loc[work[date_col] > train_end, target_col] = np.nan

    val_dates = sorted(work.loc[val_mask, date_col].unique())
    pred_parts: list[pd.DataFrame] = []
    key_cols = list(entity_cols) + [date_col]

    for day in val_dates:
        window_start = pd.Timestamp(day) - pd.Timedelta(days=lookback_days)
        sub = work[
            (work[date_col] >= window_start) & (work[date_col] <= day)
        ].copy()
        featured = build_feature_matrix(
            sub,
            groups=groups,
            extras=extras,
            entity_cols=entity_cols,
            date_col=date_col,
            target_col=target_col,
        )
        featured[date_col] = pd.to_datetime(featured[date_col])
        day_rows = featured.loc[featured[date_col] == day].copy()
        if day_rows.empty:
            raise RuntimeError(f"No feature rows for recursive day {day}")

        X_day = _apply_categorical_maps(day_rows[feature_cols], cat_maps)
        pred_t = model.predict(X_day)
        pred = inverse_target(pred_t, target_transform)
        if clip_negative_preds:
            pred = np.clip(pred, a_min=0.0, a_max=None)

        # Write predictions into working panel for subsequent lag/rolling steps.
        day_keys = day_rows[key_cols].copy()
        day_keys["_pred"] = pred
        work = work.merge(day_keys, on=key_cols, how="left")
        fill = work["_pred"].notna()
        work.loc[fill, target_col] = work.loc[fill, "_pred"]
        work = work.drop(columns=["_pred"])

        part = day_keys[key_cols].copy()
        part["y_pred"] = pred
        pred_parts.append(part)

    pred_df = pd.concat(pred_parts, ignore_index=True)
    # Align to meta_val order
    aligned = meta_val.merge(pred_df, on=key_cols, how="left", validate="one_to_one")
    if aligned["y_pred"].isna().any():
        raise RuntimeError("Recursive predict failed to cover all val rows")
    y_pred = aligned["y_pred"].to_numpy(dtype=float)
    return y_true, y_pred, meta_val


def score_lightgbm(
    *,
    train: pd.DataFrame,
    folds_meta: list[dict[str, Any]],
    splits_dir: Path,
    interim_dir: Path,
    model_params: dict[str, Any],
    early_stopping_rounds: int | None,
    feature_groups: Sequence[str],
    entity_cols: list[str],
    date_col: str,
    target_col: str,
    target_transform: str,
    clip_negative_preds: bool,
    seed: int,
    use_gpu: bool,
    run_dir: Path | None = None,
    es_holdout_days: int = 15,
) -> dict[str, Any]:
    """Walk-forward LightGBM training; primary RMSLE on inverse-transformed preds.

    When ``lag``/``rolling`` groups are active, validation forecasts are produced
    recursively (predicted sales feed the next day's lags) so multi-step origin
    masking does not leave short lags as NaN for the whole horizon.
    """
    groups = list(feature_groups)
    feature_cols = feature_columns_for_groups(groups)
    extras = _load_feature_extras(groups, interim_dir)
    recursive = _needs_recursive_forecast(groups)

    # Slim panel columns for FE (plus id if present).
    keep = list(
        dict.fromkeys(
            [date_col, *entity_cols, target_col, "onpromotion"]
            + (["id"] if "id" in train.columns else [])
        )
    )
    panel = train[keep].copy()
    panel[date_col] = pd.to_datetime(panel[date_col])

    fold_rmsle: list[float] = []
    fold_mae_log1p: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    oof_parts: list[pd.DataFrame] = []

    models_dir: Path | None = None
    if run_dir is not None:
        models_dir = Path(run_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

    for meta in folds_meta:
        fold = int(meta["fold"])
        train_end = pd.Timestamp(meta["train_end"])
        val_start = pd.Timestamp(meta["val_start"])
        val_end = pd.Timestamp(meta["val_end"])

        train_idx = pd.read_parquet(splits_dir / f"fold_{fold}_train_idx.parquet")[
            "idx"
        ].to_numpy()
        val_idx = pd.read_parquet(splits_dir / f"fold_{fold}_val_idx.parquet")[
            "idx"
        ].to_numpy()

        logger.info(
            "fold=%s building features groups=%s recursive=%s train_end=%s "
            "n_train_idx=%d n_val_idx=%d",
            fold,
            groups,
            recursive,
            train_end.date(),
            len(train_idx),
            len(val_idx),
        )
        X_train, y_train, X_val, y_val, meta_val = _prepare_lgbm_matrices(
            panel=panel,
            train_idx=train_idx,
            val_idx=val_idx,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            groups=groups,
            extras=extras,
            entity_cols=entity_cols,
            date_col=date_col,
            target_col=target_col,
            feature_cols=feature_cols,
        )
        X_train, X_val, cat_cols, cat_maps = _encode_categoricals(X_train, X_val)

        # Early-stopping set: for recursive target-derived features, avoid using
        # masked multi-horizon val (short lags become NaN). Hold out last train days.
        y_train_t = transform_target(y_train, target_transform)
        X_fit, y_fit_t = X_train, y_train_t
        X_es, y_es_t = X_val, transform_target(y_val, target_transform)
        if recursive and early_stopping_rounds:
            tr_sorted = (
                panel.loc[train_idx]
                .assign(**{date_col: lambda d: pd.to_datetime(d[date_col])})
                .sort_values(list(entity_cols) + [date_col])
                .reset_index(drop=True)
            )
            if len(tr_sorted) == len(X_train):
                es_start = train_end - pd.Timedelta(days=int(es_holdout_days) - 1)
                es_mask = (tr_sorted[date_col] >= es_start).to_numpy()
                fit_mask = ~es_mask
                if int(fit_mask.sum()) >= 1000 and int(es_mask.sum()) >= 100:
                    X_fit = X_train.iloc[fit_mask].reset_index(drop=True)
                    y_fit_t = y_train_t[fit_mask]
                    X_es = X_train.iloc[es_mask].reset_index(drop=True)
                    y_es_t = y_train_t[es_mask]
                else:
                    logger.warning(
                        "fold=%s ES holdout too small; using masked val for ES",
                        fold,
                    )
            else:
                logger.warning(
                    "fold=%s train date alignment mismatch (%d vs %d); "
                    "using masked val for early stopping",
                    fold,
                    len(tr_sorted),
                    len(X_train),
                )

        model = fit_lgbm(
            X_fit,
            y_fit_t,
            X_es,
            y_es_t,
            params=model_params,
            seed=seed,
            use_gpu=use_gpu,
            early_stopping_rounds=early_stopping_rounds,
            categorical_feature=cat_cols or None,
        )

        if recursive:
            # Full panel for recursive: train history + val spine (sales masked inside).
            fold_idx = np.unique(np.concatenate([train_idx, val_idx]))
            fold_panel = panel.loc[fold_idx].copy()
            # Restore true train sales; val will be masked inside recursive helper.
            y_val, y_pred, meta_val = _recursive_val_predict(
                model=model,
                panel=fold_panel,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
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
            pred_t = model.predict(X_val)
            y_pred = inverse_target(pred_t, target_transform)
            if clip_negative_preds:
                y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

        fold_score = rmsle(y_val, y_pred)
        fold_guard = mae_log1p(y_val, y_pred)
        fold_rmsle.append(float(fold_score))
        fold_mae_log1p.append(float(fold_guard))

        best_iter = getattr(model, "best_iteration_", None)
        fold_rows.append(
            {
                "fold": fold,
                "rmsle": float(fold_score),
                "mae_log1p": float(fold_guard),
                "n_train": int(len(X_fit)),
                "n_val": int(len(y_val)),
                "best_iteration": int(best_iter) if best_iter is not None else None,
                "val_start": str(val_start.date()),
                "val_end": str(val_end.date()),
                "train_end": str(train_end.date()),
                "n_features": int(len(feature_cols)),
                "categorical_features": cat_cols,
                "recursive_forecast": recursive,
            }
        )
        logger.info(
            "fold=%s model=lightgbm rmsle=%.6f mae_log1p=%.6f n_val=%d best_iter=%s",
            fold,
            fold_score,
            fold_guard,
            len(y_val),
            best_iter,
        )

        oof = meta_val.copy()
        oof["fold"] = fold
        oof["y_true"] = y_val
        oof["y_pred"] = y_pred
        oof_parts.append(oof)

        if models_dir is not None:
            model_path = models_dir / f"fold_{fold}.txt"
            model.booster_.save_model(str(model_path))
            logger.info("Saved model fold=%s path=%s", fold, model_path)

    oof_df = pd.concat(oof_parts, axis=0, ignore_index=True) if oof_parts else pd.DataFrame()

    return {
        "mean_rmsle": float(sum(fold_rmsle) / len(fold_rmsle)),
        "std_rmsle": float(pd.Series(fold_rmsle).std(ddof=1)) if len(fold_rmsle) > 1 else 0.0,
        "fold_rmsle": fold_rmsle,
        "mean_mae_log1p": float(sum(fold_mae_log1p) / len(fold_mae_log1p)),
        "fold_mae_log1p": fold_mae_log1p,
        "folds": fold_rows,
        "model_name": "lightgbm",
        "feature_groups": groups,
        "feature_columns": feature_cols,
        "target_transform": target_transform,
        "recursive_forecast": recursive,
        "oof": oof_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/score experiment config under walk-forward folds"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment YAML (e.g. configs/experiments/000_last_value.yaml)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="Override outputs root (default from config/paths)",
    )
    args = parser.parse_args()

    paths = ProjectPaths()
    default_cfg = load_default_config()
    exp_cfg = load_yaml(args.config)
    cfg = _deep_merge(default_cfg, exp_cfg)

    seed = int(cfg.get("seed", 42))
    run_id = str(cfg.get("run_id", args.config.stem))
    model_cfg = cfg.get("model") or {}
    model_name = str(model_cfg.get("name", ""))
    period = model_cfg.get("period")
    if period is not None:
        period = int(period)

    entity_cols = list(cfg.get("entity_cols", ["store_nbr", "family"]))
    date_col = str(cfg.get("date_col", "date"))
    target_col = str(cfg.get("target_col", "sales"))

    path_cfg = cfg.get("paths") or {}
    interim_dir = paths.root / path_cfg.get("interim_dir", "data/interim")
    splits_dir = paths.root / path_cfg.get("splits_dir", "data/splits")
    outputs_dir = (
        args.outputs_dir
        if args.outputs_dir is not None
        else paths.root / path_cfg.get("outputs_dir", "outputs")
    )

    train_path = interim_dir / "train.parquet"
    logger.info(
        "Scoring run_id=%s model=%s period=%s seed=%s train=%s",
        run_id,
        model_name,
        period,
        seed,
        train_path,
    )
    train = pd.read_parquet(train_path)
    folds_meta = _load_folds_meta(splits_dir)

    if model_name in ("last_value", "seasonal_naive"):
        metrics = score_baselines(
            train=train,
            folds_meta=folds_meta,
            splits_dir=splits_dir,
            model_name=model_name,
            period=period,
            entity_cols=entity_cols,
            date_col=date_col,
            target_col=target_col,
        )
        run_dir = save_run_dir(
            outputs_root=outputs_dir,
            run_id=run_id,
            config=cfg,
            metrics={
                "mean_rmsle": metrics["mean_rmsle"],
                "std_rmsle": metrics["std_rmsle"],
                "fold_rmsle": metrics["fold_rmsle"],
                "mean_mae_log1p": metrics["mean_mae_log1p"],
                "fold_mae_log1p": metrics["fold_mae_log1p"],
                "folds": metrics["folds"],
                "model_name": metrics["model_name"],
                "period": metrics["period"],
            },
            seed=seed,
        )
    elif model_name == "lightgbm":
        feature_groups = list(cfg.get("feature_groups") or ["base"])
        target_transform = str(cfg.get("target_transform", "log1p"))
        clip_negative_preds = bool(cfg.get("clip_negative_preds", True))
        model_params = dict(model_cfg.get("params") or {})
        early_stopping_rounds = model_cfg.get("early_stopping_rounds", 50)
        if early_stopping_rounds is not None:
            early_stopping_rounds = int(early_stopping_rounds)
        use_gpu = bool(cfg.get("gpu", True))

        # Pre-create run dir so fold models land under it; save_run_dir reuses path.
        run_dir = Path(outputs_dir) / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics = score_lightgbm(
            train=train,
            folds_meta=folds_meta,
            splits_dir=splits_dir,
            interim_dir=interim_dir,
            model_params=model_params,
            early_stopping_rounds=early_stopping_rounds,
            feature_groups=feature_groups,
            entity_cols=entity_cols,
            date_col=date_col,
            target_col=target_col,
            target_transform=target_transform,
            clip_negative_preds=clip_negative_preds,
            seed=seed,
            use_gpu=use_gpu,
            run_dir=run_dir,
        )
        oof = metrics.pop("oof")
        metrics_payload = {
            "mean_rmsle": metrics["mean_rmsle"],
            "std_rmsle": metrics["std_rmsle"],
            "fold_rmsle": metrics["fold_rmsle"],
            "mean_mae_log1p": metrics["mean_mae_log1p"],
            "fold_mae_log1p": metrics["fold_mae_log1p"],
            "folds": metrics["folds"],
            "model_name": metrics["model_name"],
            "feature_groups": metrics["feature_groups"],
            "feature_columns": metrics["feature_columns"],
            "target_transform": metrics["target_transform"],
            "naive_floor_sn7_mean_rmsle": 0.5513,
        }
        run_dir = save_run_dir(
            outputs_root=outputs_dir,
            run_id=run_id,
            config=cfg,
            metrics=metrics_payload,
            seed=seed,
            extra={"fold_metrics": metrics["folds"]},
        )
        if oof is not None and len(oof) > 0:
            oof_path = run_dir / "oof_predictions.parquet"
            oof.to_parquet(oof_path, index=False)
            logger.info("Wrote OOF predictions %s rows=%d", oof_path, len(oof))
    else:
        raise ValueError(f"Unsupported model.name: {model_name!r}")

    logger.info(
        "Done run_id=%s mean_rmsle=%.6f ± %.6f artifacts=%s",
        run_id,
        metrics["mean_rmsle"],
        metrics["std_rmsle"],
        run_dir,
    )
    print(
        f"run_id={run_id} mean_rmsle={metrics['mean_rmsle']:.6f} "
        f"std_rmsle={metrics['std_rmsle']:.6f} dir={run_dir}"
    )


if __name__ == "__main__":
    main()
