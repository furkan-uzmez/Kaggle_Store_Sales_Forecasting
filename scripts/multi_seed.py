"""Run multi-seed matrix for finalist configs and write summary CSV.

Usage:
  uv run python scripts/multi_seed.py \\
    --configs configs/experiments/020_lgbm_hpo_best.yaml \\
              configs/experiments/030_catboost_locked_groups.yaml \\
              configs/experiments/031_xgboost_locked_groups.yaml \\
    --seeds 42,43,44

Reuses existing run dirs when metrics.json already exists for run_id_s{seed}
(unless --force). Writes outputs/reports/multi_seed_summary.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow importing train helpers when launched as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from train import (  # noqa: E402
    _deep_merge,
    _load_folds_meta,
    _parse_seeds,
    run_single_experiment,
)

from store_sales.config import ProjectPaths, load_default_config, load_yaml
from store_sales.io.logging import get_logger

logger = get_logger(__name__)


def _metrics_path(outputs_dir: Path, run_id: str) -> Path:
    return outputs_dir / "runs" / run_id / "metrics.json"


def _load_existing_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed finalist matrix")
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        required=True,
        help="Finalist experiment YAML paths",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds (default 42,43,44)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-train even if metrics.json already exists",
    )
    parser.add_argument(
        "--alias-seed42",
        action="store_true",
        help="If run_id (no _s42) has metrics, treat as seed 42 for that config",
    )
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds) or [42, 43, 44]
    paths = ProjectPaths()
    default_cfg = load_default_config()

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for config_path in args.configs:
        exp_cfg = load_yaml(config_path)
        cfg = _deep_merge(default_cfg, exp_cfg)
        base_run_id = str(cfg.get("run_id", config_path.stem))
        model_name = str((cfg.get("model") or {}).get("name", ""))

        path_cfg = cfg.get("paths") or {}
        interim_dir = paths.root / path_cfg.get("interim_dir", "data/interim")
        splits_dir = paths.root / path_cfg.get("splits_dir", "data/splits")
        outputs_dir = (
            args.outputs_dir
            if args.outputs_dir is not None
            else paths.root / path_cfg.get("outputs_dir", "outputs")
        )

        train = pd.read_parquet(interim_dir / "train.parquet")
        folds_meta = _load_folds_meta(splits_dir)

        seed_means: list[float] = []
        seed_stds: list[float] = []
        for seed in seeds:
            run_id = f"{base_run_id}_s{seed}"
            metrics: dict[str, Any] | None = None

            if not args.force:
                metrics = _load_existing_metrics(_metrics_path(outputs_dir, run_id))
                if (
                    metrics is None
                    and args.alias_seed42
                    and seed == 42
                ):
                    # Allow pre-task single-seed run (no _s42 suffix) as seed 42.
                    metrics = _load_existing_metrics(
                        _metrics_path(outputs_dir, base_run_id)
                    )
                    if metrics is not None:
                        logger.info(
                            "Reusing unsuffixed run_id=%s as seed=42",
                            base_run_id,
                        )

            if metrics is None:
                logger.info(
                    "Training config=%s seed=%s run_id=%s",
                    config_path,
                    seed,
                    run_id,
                )
                _, metrics, _ = run_single_experiment(
                    cfg=cfg,
                    config_path=config_path,
                    outputs_dir=outputs_dir,
                    interim_dir=interim_dir,
                    splits_dir=splits_dir,
                    train=train,
                    folds_meta=folds_meta,
                    seed=seed,
                    run_id=run_id,
                )
            else:
                logger.info(
                    "Skip existing run_id=%s mean_rmsle=%.6f",
                    run_id if _metrics_path(outputs_dir, run_id).exists() else base_run_id,
                    float(metrics["mean_rmsle"]),
                )

            mean_r = float(metrics["mean_rmsle"])
            std_r = float(metrics.get("std_rmsle") or 0.0)
            seed_means.append(mean_r)
            seed_stds.append(std_r)
            detail_rows.append(
                {
                    "base_run_id": base_run_id,
                    "model_name": metrics.get("model_name", model_name),
                    "seed": seed,
                    "mean_rmsle": mean_r,
                    "std_rmsle": std_r,
                    "mean_mae_log1p": metrics.get("mean_mae_log1p"),
                }
            )

        row: dict[str, Any] = {
            "base_run_id": base_run_id,
            "model_name": model_name,
            "seeds": ",".join(str(s) for s in seeds),
            "mean_across_seeds": float(np.mean(seed_means)),
            "std_across_seeds": float(np.std(seed_means, ddof=1))
            if len(seed_means) > 1
            else 0.0,
            "worst_seed_rmsle": float(np.max(seed_means)),
            "best_seed_rmsle": float(np.min(seed_means)),
        }
        for s, m in zip(seeds, seed_means):
            row[f"seed_{s}_mean_rmsle"] = m
        summary_rows.append(row)
        logger.info(
            "config=%s mean_across_seeds=%.6f std=%.6f worst=%.6f",
            base_run_id,
            row["mean_across_seeds"],
            row["std_across_seeds"],
            row["worst_seed_rmsle"],
        )

    report_dir = (
        (args.outputs_dir if args.outputs_dir is not None else paths.root / "outputs")
        / "reports"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "multi_seed_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    detail_path = report_dir / "multi_seed_detail.csv"
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)
    logger.info("Wrote %s and %s", summary_path, detail_path)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
