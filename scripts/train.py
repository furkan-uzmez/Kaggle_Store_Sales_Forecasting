"""Score baseline (and later model) configs under fixed walk-forward folds.

Writes ``outputs/runs/<run_id>/`` with config, metrics, environment, metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from store_sales.config import ProjectPaths, load_default_config, load_yaml
from store_sales.io.artifacts import save_run_dir
from store_sales.io.logging import get_logger
from store_sales.metrics.guards import mae_log1p
from store_sales.metrics.rmsle import rmsle
from store_sales.models.baseline import last_value_predict, seasonal_naive_predict

logger = get_logger(__name__)


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
