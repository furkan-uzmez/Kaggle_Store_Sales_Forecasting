"""Evaluate the locked finalist from configs/final.yaml (no retune).

Re-scores primary walk-forward OOF RMSLE plus horizon/segment guardrails from
existing OOF artifacts. Writes ``outputs/final_evaluation/``.

Usage:
  uv run python scripts/evaluate.py --config configs/final.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from store_sales.config import ProjectPaths, load_yaml
from store_sales.io.artifacts import collect_environment
from store_sales.io.logging import get_logger
from store_sales.metrics.final_eval import (
    NAIVE_FLOOR_SN7_RMSLE,
    annotate_horizon,
    annotate_payday,
    annotate_zero_target,
    build_segment_table,
    fold_metrics_from_oof,
    gate_vs_naive,
    horizon_metrics,
)

logger = get_logger(__name__)

_DEFAULT_SEGMENTS = ("family", "cluster", "is_zero", "is_payday")


def _resolve_path(root: Path, maybe_rel: str | Path) -> Path:
    path = Path(maybe_rel)
    if path.is_absolute():
        return path
    return root / path


def load_locked_config(config_path: Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    if not cfg:
        raise ValueError(f"empty config: {config_path}")
    return cfg


def load_oof(oof_path: Path) -> pd.DataFrame:
    if not oof_path.exists():
        raise FileNotFoundError(f"missing OOF predictions: {oof_path}")
    oof = pd.read_parquet(oof_path)
    need = {"store_nbr", "family", "date", "fold", "y_true", "y_pred"}
    missing = need - set(oof.columns)
    if missing:
        raise KeyError(f"OOF missing columns: {sorted(missing)}")
    oof = oof.copy()
    oof["date"] = pd.to_datetime(oof["date"])
    return oof


def load_fold_val_starts(splits_dir: Path) -> dict[int, pd.Timestamp]:
    meta_path = splits_dir / "folds_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {int(m["fold"]): pd.Timestamp(m["val_start"]) for m in meta}


def prepare_oof_for_slices(
    oof: pd.DataFrame,
    *,
    fold_val_start: dict[int, pd.Timestamp],
    stores: pd.DataFrame | None,
) -> pd.DataFrame:
    out = annotate_horizon(oof, fold_val_start)
    out = annotate_payday(out)
    out = annotate_zero_target(out)
    if stores is not None and "store_nbr" in stores.columns:
        keep = [c for c in ("store_nbr", "cluster", "type", "city") if c in stores.columns]
        if len(keep) > 1:
            out = out.merge(stores[keep], on="store_nbr", how="left")
    return out


def evaluate_locked(
    *,
    cfg: dict[str, Any],
    oof: pd.DataFrame,
    fold_val_start: dict[int, pd.Timestamp],
    stores: pd.DataFrame | None,
    config_path: Path,
    oof_path: Path,
    naive_floor: float = NAIVE_FLOOR_SN7_RMSLE,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Return (metrics.json payload, horizon_metrics, segment_metrics)."""
    prepared = prepare_oof_for_slices(
        oof, fold_val_start=fold_val_start, stores=stores
    )
    fold_block = fold_metrics_from_oof(prepared)
    gate = gate_vs_naive(
        mean_rmsle=float(fold_block["mean_rmsle"]),
        naive_floor=naive_floor,
    )
    if not gate["beats_naive_floor"]:
        logger.error(
            "LOCKED EVAL FAILED vs naive floor sn7=%.4f: mean_rmsle=%.6f "
            "(do not silently edit final.yaml; roll back to Layer 2 with new id)",
            naive_floor,
            fold_block["mean_rmsle"],
        )
    else:
        logger.info(
            "Beats naive floor sn7=%.4f: mean_rmsle=%.6f (delta=%+.6f)",
            naive_floor,
            fold_block["mean_rmsle"],
            gate["delta_vs_naive_rmsle"],
        )

    horiz = horizon_metrics(prepared)
    segment_cols = [c for c in _DEFAULT_SEGMENTS if c in prepared.columns]
    segs = build_segment_table(prepared, segment_cols, min_n=50)

    model_cfg = cfg.get("model") or {}
    multi_seed = cfg.get("multi_seed") or {}
    artifacts = cfg.get("artifacts") or {}

    metrics: dict[str, Any] = {
        "run_id": cfg.get("run_id"),
        "config_path": str(config_path),
        "primary_seed": cfg.get("primary_seed"),
        "seeds": cfg.get("seeds"),
        "source": "oof_artifacts",
        "oof_path": str(oof_path),
        "metric": cfg.get("metric", "rmsle"),
        "model_name": model_cfg.get("name"),
        "feature_groups": list(cfg.get("feature_groups") or []),
        "target_transform": cfg.get("target_transform"),
        "clip_negative_preds": cfg.get("clip_negative_preds"),
        "horizon_days": cfg.get("horizon_days"),
        "entity_cols": cfg.get("entity_cols"),
        "mean_rmsle": fold_block["mean_rmsle"],
        "std_rmsle": fold_block["std_rmsle"],
        "fold_rmsle": fold_block["fold_rmsle"],
        "mean_mae_log1p": fold_block["mean_mae_log1p"],
        "fold_mae_log1p": fold_block["fold_mae_log1p"],
        "mean_bias_log1p": fold_block["mean_bias_log1p"],
        "fold_bias_log1p": fold_block["fold_bias_log1p"],
        "pooled_oof_rmsle": fold_block["pooled_oof_rmsle"],
        "pooled_oof_mae_log1p": fold_block["pooled_oof_mae_log1p"],
        "pooled_oof_bias_log1p": fold_block["pooled_oof_bias_log1p"],
        "folds": fold_block["folds"],
        "n_oof": fold_block["n_oof"],
        "multi_seed": multi_seed,
        "artifacts_ref": artifacts,
        **gate,
        "note": (
            "Primary selection metric is fold-mean walk-forward RMSLE from locked "
            "OOF; horizon/segment tables are guardrails only. No HPO after lock."
        ),
    }
    return metrics, horiz, segs


def write_final_evaluation(
    *,
    out_dir: Path,
    metrics: dict[str, Any],
    horizon_df: pd.DataFrame,
    segment_df: pd.DataFrame,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    horizon_df.to_csv(out_dir / "horizon_metrics.csv", index=False)
    segment_df.to_csv(out_dir / "segment_metrics.csv", index=False)
    env = collect_environment()
    (out_dir / "environment.json").write_text(
        json.dumps(env, indent=2), encoding="utf-8"
    )
    logger.info("Wrote final evaluation artifacts to %s", out_dir)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate locked finalist (final.yaml) from OOF; no retune"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final.yaml"),
        help="Locked config path (default configs/final.yaml)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="Project outputs root (default from paths / outputs)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Evaluation output dir (default <outputs>/final_evaluation)",
    )
    parser.add_argument(
        "--naive-floor",
        type=float,
        default=NAIVE_FLOOR_SN7_RMSLE,
        help=f"Seasonal-naive-7 RMSLE floor (default {NAIVE_FLOOR_SN7_RMSLE})",
    )
    args = parser.parse_args(argv)

    paths = ProjectPaths()
    root = paths.root
    cfg = load_locked_config(args.config if args.config.is_absolute() else root / args.config)

    artifacts = cfg.get("artifacts") or {}
    oof_rel = artifacts.get("oof")
    if not oof_rel:
        run_dir = artifacts.get("run_dir") or f"outputs/runs/{cfg.get('run_id')}"
        oof_rel = str(Path(run_dir) / "oof_predictions.parquet")
    oof_path = _resolve_path(root, oof_rel)

    path_cfg = cfg.get("paths") or {}
    splits_dir = _resolve_path(
        root, path_cfg.get("splits_dir", "data/splits")
    )
    interim_dir = _resolve_path(
        root, path_cfg.get("interim_dir", "data/interim")
    )

    outputs_root = args.outputs_dir or paths.outputs
    if not outputs_root.is_absolute():
        outputs_root = root / outputs_root
    out_dir = args.out_dir or (outputs_root / "final_evaluation")
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    logger.info(
        "Evaluating locked run_id=%s from OOF %s (no retune)",
        cfg.get("run_id"),
        oof_path,
    )

    oof = load_oof(oof_path)
    fold_starts = load_fold_val_starts(splits_dir)
    stores_path = interim_dir / "stores.parquet"
    stores = pd.read_parquet(stores_path) if stores_path.exists() else None

    metrics, horiz, segs = evaluate_locked(
        cfg=cfg,
        oof=oof,
        fold_val_start=fold_starts,
        stores=stores,
        config_path=Path(args.config),
        oof_path=oof_path,
        naive_floor=float(args.naive_floor),
    )
    write_final_evaluation(
        out_dir=out_dir,
        metrics=metrics,
        horizon_df=horiz,
        segment_df=segs,
    )

    logger.info(
        "Primary mean_rmsle=%.6f pooled_oof=%.6f go_no_go=%s",
        metrics["mean_rmsle"],
        metrics["pooled_oof_rmsle"],
        metrics["go_no_go"],
    )
    return 0 if metrics["beats_naive_floor"] else 1


if __name__ == "__main__":
    sys.exit(main())
