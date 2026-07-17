"""Optional OOF mean / non-negative weighted blend of finalist runs.

Accepts run directories that already have oof_predictions.parquet with aligned
keys (store_nbr, family, date, fold) and y_true/y_pred. Fits blend weights
on OOF only; reports RMSLE vs members. Writes outputs/runs/<blend_id>/.

Usage:
  uv run python scripts/blend_oof.py \\
    --runs outputs/runs/020_lgbm_hpo_best_s42 \\
           outputs/runs/030_catboost_locked_groups_s42 \\
           outputs/runs/031_xgboost_locked_groups_s42 \\
    --blend-id 040_mean_blend_finalists
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from store_sales.io.artifacts import save_run_dir
from store_sales.io.logging import get_logger
from store_sales.metrics.guards import mae_log1p
from store_sales.metrics.rmsle import rmsle
from store_sales.models.blend import (
    fit_nonneg_blend_weights,
    mean_blend,
    weighted_blend,
)

logger = get_logger(__name__)

_KEY = ["store_nbr", "family", "date", "fold"]


def _load_oof(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "oof_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing OOF: {path}")
    df = pd.read_parquet(path)
    need = set(_KEY + ["y_true", "y_pred"])
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{run_dir}: OOF missing columns {missing}")
    out = df[_KEY + ["y_true", "y_pred"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="OOF mean/weighted blend")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--blend-id", type=str, default="040_oof_blend")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument(
        "--method",
        choices=["mean", "nonneg", "both"],
        default="both",
        help="mean average, non-negative OOF weights, or both (default both)",
    )
    args = parser.parse_args()

    oofs = [_load_oof(p) for p in args.runs]
    labels = [p.name for p in args.runs]

    # Align on keys from first run
    base = oofs[0][_KEY + ["y_true"]].rename(columns={"y_true": "y_true"})
    merged = base
    pred_cols: list[str] = []
    for lab, oof in zip(labels, oofs):
        col = f"y_pred__{lab}"
        piece = oof[_KEY + ["y_pred"]].rename(columns={"y_pred": col})
        merged = merged.merge(piece, on=_KEY, how="inner", validate="one_to_one")
        pred_cols.append(col)

    if len(merged) == 0:
        raise RuntimeError("No overlapping OOF rows after merge")

    y_true = merged["y_true"].to_numpy(dtype=float)
    preds = [merged[c].to_numpy(dtype=float) for c in pred_cols]

    member_scores = {
        lab: float(rmsle(y_true, p)) for lab, p in zip(labels, preds)
    }
    logger.info("Member OOF RMSLE: %s", member_scores)

    results: dict[str, Any] = {
        "members": labels,
        "member_oof_rmsle": member_scores,
        "n_oof": int(len(merged)),
    }

    chosen_method = None
    chosen_pred = None
    chosen_rmsle = None
    weights = None

    if args.method in ("mean", "both"):
        mean_pred = mean_blend(preds)
        mean_score = float(rmsle(y_true, mean_pred))
        mean_std_proxy = float(
            np.std(
                [float(rmsle(y_true[merged["fold"] == f], mean_pred[merged["fold"] == f]))
                 for f in sorted(merged["fold"].unique())],
                ddof=1,
            )
        ) if merged["fold"].nunique() > 1 else 0.0
        results["mean_blend"] = {
            "mean_rmsle": mean_score,
            "std_rmsle_across_folds": mean_std_proxy,
            "mean_mae_log1p": float(mae_log1p(y_true, mean_pred)),
        }
        logger.info("mean_blend OOF rmsle=%.6f", mean_score)
        chosen_method = "mean"
        chosen_pred = mean_pred
        chosen_rmsle = mean_score

    if args.method in ("nonneg", "both"):
        P = np.column_stack(preds)
        weights = fit_nonneg_blend_weights(P, y_true)
        w_pred = weighted_blend(preds, weights)
        w_score = float(rmsle(y_true, w_pred))
        results["nonneg_blend"] = {
            "weights": {lab: float(w) for lab, w in zip(labels, weights)},
            "mean_rmsle": w_score,
            "mean_mae_log1p": float(mae_log1p(y_true, w_pred)),
        }
        logger.info(
            "nonneg_blend OOF rmsle=%.6f weights=%s",
            w_score,
            results["nonneg_blend"]["weights"],
        )
        if chosen_rmsle is None or w_score < chosen_rmsle:
            chosen_method = "nonneg"
            chosen_pred = w_pred
            chosen_rmsle = w_score

    best_member = min(member_scores.values())
    accept = chosen_rmsle is not None and chosen_rmsle < best_member - 1e-6
    results["chosen_method"] = chosen_method
    results["chosen_mean_rmsle"] = chosen_rmsle
    results["best_member_rmsle"] = best_member
    results["accepted"] = bool(accept)
    results["accept_rule"] = "mean RMSLE improves vs best member (no large std check on single OOF)"

    # Fold-level stats for chosen
    fold_rmsle = []
    for f in sorted(merged["fold"].unique()):
        mask = merged["fold"].to_numpy() == f
        fold_rmsle.append(float(rmsle(y_true[mask], chosen_pred[mask])))
    results["fold_rmsle"] = fold_rmsle
    results["std_rmsle"] = (
        float(np.std(fold_rmsle, ddof=1)) if len(fold_rmsle) > 1 else 0.0
    )

    run_dir = save_run_dir(
        outputs_root=args.outputs_dir,
        run_id=args.blend_id,
        config={
            "run_id": args.blend_id,
            "members": labels,
            "method": chosen_method,
            "weights": (
                {lab: float(w) for lab, w in zip(labels, weights)}
                if weights is not None
                else None
            ),
            "accepted": accept,
        },
        metrics={
            "mean_rmsle": chosen_rmsle,
            "std_rmsle": results["std_rmsle"],
            "fold_rmsle": fold_rmsle,
            "mean_mae_log1p": float(mae_log1p(y_true, chosen_pred)),
            "model_name": f"blend_{chosen_method}",
            "members": labels,
            "member_oof_rmsle": member_scores,
            "accepted": accept,
            "blend_detail": results,
        },
        seed=42,
        extra={"blend_detail": results},
    )

    oof_out = merged[_KEY + ["y_true"]].copy()
    oof_out["y_pred"] = chosen_pred
    oof_out.to_parquet(run_dir / "oof_predictions.parquet", index=False)

    # Also write decision JSON next to reports
    report_dir = Path(args.outputs_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{args.blend_id}.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    print(
        f"blend_id={args.blend_id} method={chosen_method} "
        f"mean_rmsle={chosen_rmsle:.6f} best_member={best_member:.6f} "
        f"accepted={accept} dir={run_dir}"
    )


if __name__ == "__main__":
    main()
