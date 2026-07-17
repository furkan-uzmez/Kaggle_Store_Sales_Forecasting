"""Robustness battery on a locked finalist run (frozen fold models).

Rebuilds validation features using train sales + clean OOF predictions as the
multi-step history (matches recursive fill), then re-predicts under feature
perturbations. Writes ``outputs/stress/<name>/summary.json`` with ΔRMSLE vs clean.

Usage:
  uv run python scripts/stress_test.py --config configs/stress/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

from store_sales.config import ProjectPaths, load_yaml
from store_sales.features.registry import FEATURE_GROUPS, build_feature_matrix
from store_sales.io.logging import get_logger
from store_sales.metrics.rmsle import rmsle
from store_sales.models.gbdt import inverse_target
from store_sales.stress import clip_spike_columns, null_columns, relative_noise

# Reuse train helpers without re-exporting a public API.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))
import train as train_mod  # noqa: E402

logger = get_logger(__name__)

_CAT_CANDIDATES = train_mod._CAT_CANDIDATES


class _BoosterModel:
    """Thin predict adapter for a saved LightGBM booster text file."""

    def __init__(self, path: Path) -> None:
        self._booster = lgb.Booster(model_file=str(path))
        self._names = list(self._booster.feature_name())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with numpy float matrix (avoid pandas categorical mismatch)."""
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


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing run config: {path}")
    return load_yaml(path)


def _load_oof(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "oof_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing OOF: {path}")
    oof = pd.read_parquet(path)
    oof["date"] = pd.to_datetime(oof["date"])
    return oof


def _feature_columns(groups: list[str]) -> list[str]:
    cols: list[str] = []
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise ValueError(f"unknown feature group: {g!r}")
        cols.extend(FEATURE_GROUPS[g])
    return list(dict.fromkeys(cols))


def _build_val_matrix_with_oof_fill(
    *,
    panel: pd.DataFrame,
    oof_fold: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    groups: list[str],
    extras: dict[str, pd.DataFrame],
    entity_cols: list[str],
    date_col: str,
    target_col: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, list[str], dict[str, pd.Index]]:
    """Feature matrix on val using true train sales + OOF preds as history.

    This reconstructs the recursive multi-step feature state without re-running
    day-by-day prediction, then freezes that matrix for stress re-scoring.
    """
    fold_idx = np.unique(np.concatenate([train_idx, val_idx]))
    work = panel.loc[fold_idx].copy()
    work[date_col] = pd.to_datetime(work[date_col])

    # Labels before overwrite
    y_lookup = work[list(entity_cols) + [date_col, target_col]].copy()

    # Null post-origin sales then fill with OOF predictions (clean recursive path).
    work.loc[work[date_col] > train_end, target_col] = np.nan
    fill = oof_fold[list(entity_cols) + [date_col, "y_pred"]].copy()
    fill[date_col] = pd.to_datetime(fill[date_col])
    work = work.merge(fill, on=list(entity_cols) + [date_col], how="left")
    mask = work["y_pred"].notna()
    work.loc[mask, target_col] = work.loc[mask, "y_pred"]
    work = work.drop(columns=["y_pred"])

    safe_extras = train_mod.mask_outcome_extras_after(
        extras, train_end, date_col=date_col
    )
    featured = build_feature_matrix(
        work,
        groups=groups,
        extras=safe_extras,
        entity_cols=entity_cols,
        date_col=date_col,
        target_col=target_col,
    )
    featured[date_col] = pd.to_datetime(featured[date_col])

    if target_col in featured.columns:
        featured = featured.drop(columns=[target_col])
    featured = featured.merge(
        y_lookup,
        on=list(entity_cols) + [date_col],
        how="left",
        validate="one_to_one",
    )

    # Train category maps from train window features
    train_mask = featured[date_col] <= train_end
    val_mask = (featured[date_col] >= val_start) & (featured[date_col] <= val_end)
    train_df = featured.loc[train_mask]
    val_df = featured.loc[val_mask].copy()

    missing = [c for c in feature_cols if c not in val_df.columns]
    if missing:
        raise KeyError(f"feature columns missing after build: {missing}")

    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    cat_cols, cat_maps = train_mod._fit_categorical_maps(X_train)
    X_val = train_mod._apply_categorical_maps(X_val, cat_maps)

    y_true = val_df[target_col].to_numpy(dtype=float)
    meta = val_df[list(entity_cols) + [date_col]].copy()
    if "id" in val_df.columns:
        meta["id"] = val_df["id"].to_numpy()
    return X_val, y_true, meta, cat_cols, cat_maps


def _predict_sales(
    model: _BoosterModel,
    X: pd.DataFrame,
    *,
    target_transform: str,
    clip_negative_preds: bool,
) -> np.ndarray:
    pred_t = model.predict(X)
    pred = inverse_target(pred_t, target_transform)
    if clip_negative_preds:
        pred = np.clip(pred, a_min=0.0, a_max=None)
    return pred


def _mean_rmsle_over_folds(
    fold_true: list[np.ndarray],
    fold_pred: list[np.ndarray],
) -> float:
    scores = [rmsle(yt, yp) for yt, yp in zip(fold_true, fold_pred)]
    return float(np.mean(scores))


def _scenario_row(
    *,
    name: str,
    clean_rmsle: float,
    stressed_rmsle: float | None,
    detail: dict[str, Any] | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    if stressed_rmsle is None:
        delta = None
        rel = None
    else:
        delta = float(stressed_rmsle - clean_rmsle)
        rel = float(delta / clean_rmsle) if clean_rmsle else None
    return {
        "scenario": name,
        "clean_mean_rmsle": float(clean_rmsle),
        "stressed_mean_rmsle": None if stressed_rmsle is None else float(stressed_rmsle),
        "delta_rmsle": delta,
        "relative_delta": rel,
        "status": status,
        "detail": detail or {},
    }


def run_stress(
    *,
    stress_cfg: dict[str, Any],
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    """Execute the full battery; return summary dict and write artifacts."""
    paths = paths or ProjectPaths()
    run_id = str(stress_cfg.get("run_id", "020_lgbm_hpo_best"))
    run_dir = Path(stress_cfg.get("run_dir", paths.outputs / "runs" / run_id))
    if not run_dir.is_absolute():
        run_dir = paths.root / run_dir
    out_dir = Path(stress_cfg.get("output_dir", paths.outputs / "stress" / "default"))
    if not out_dir.is_absolute():
        out_dir = paths.root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(stress_cfg.get("seed", 42))
    run_cfg = _load_run_config(run_dir)
    groups = list(run_cfg.get("feature_groups") or ["base"])
    feature_cols = _feature_columns(groups)
    model_cfg = run_cfg.get("model") or {}
    target_transform = str(run_cfg.get("target_transform", "log1p"))
    clip_negative = bool(run_cfg.get("clip_negative_preds", True))
    entity_cols = list(run_cfg.get("entity_cols", ["store_nbr", "family"]))
    date_col = str(run_cfg.get("date_col", "date"))
    target_col = str(run_cfg.get("target_col", "sales"))

    train = pd.read_parquet(paths.data_interim / "train.parquet")
    train[date_col] = pd.to_datetime(train[date_col])
    folds_meta = json.loads(
        (paths.data_splits / "folds_meta.json").read_text(encoding="utf-8")
    )
    oof = _load_oof(run_dir)
    extras = train_mod._load_feature_extras(groups, paths.data_interim)

    keep = list(
        dict.fromkeys(
            [date_col, *entity_cols, target_col, "onpromotion"]
            + (["id"] if "id" in train.columns else [])
        )
    )
    panel = train[keep].copy()
    panel[date_col] = pd.to_datetime(panel[date_col])

    # Per-fold frozen matrices + clean predictions
    fold_payloads: list[dict[str, Any]] = []
    clean_true: list[np.ndarray] = []
    clean_pred: list[np.ndarray] = []

    for meta in folds_meta:
        fold = int(meta["fold"])
        train_end = pd.Timestamp(meta["train_end"])
        val_start = pd.Timestamp(meta["val_start"])
        val_end = pd.Timestamp(meta["val_end"])
        train_idx = pd.read_parquet(
            paths.data_splits / f"fold_{fold}_train_idx.parquet"
        )["idx"].to_numpy()
        val_idx = pd.read_parquet(
            paths.data_splits / f"fold_{fold}_val_idx.parquet"
        )["idx"].to_numpy()
        oof_fold = oof.loc[oof["fold"] == fold].copy()
        model_path = run_dir / "models" / f"fold_{fold}.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"missing model: {model_path}")

        logger.info(
            "stress fold=%s building OOF-filled val matrix train_end=%s",
            fold,
            train_end.date(),
        )
        X_val, y_true, meta_val, _cat_cols, _cat_maps = _build_val_matrix_with_oof_fill(
            panel=panel,
            oof_fold=oof_fold,
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
        model = _BoosterModel(model_path)
        y_pred = _predict_sales(
            model,
            X_val,
            target_transform=target_transform,
            clip_negative_preds=clip_negative,
        )
        fold_rmsle = rmsle(y_true, y_pred)
        logger.info("stress fold=%s clean_repredict_rmsle=%.6f", fold, fold_rmsle)

        fold_payloads.append(
            {
                "fold": fold,
                "X_val": X_val,
                "y_true": y_true,
                "y_pred_clean": y_pred,
                "meta": meta_val,
                "model": model,
                "val_start": val_start,
                "val_end": val_end,
            }
        )
        clean_true.append(y_true)
        clean_pred.append(y_pred)

    clean_mean = _mean_rmsle_over_folds(clean_true, clean_pred)
    # Also record artifact OOF mean for reference
    oof_mean = float(rmsle(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy()))
    metrics_path = run_dir / "metrics.json"
    artifact_mean = None
    if metrics_path.exists():
        artifact_mean = float(json.loads(metrics_path.read_text())["mean_rmsle"])

    scenarios: list[dict[str, Any]] = []

    def score_transform(
        name: str,
        transform: Callable[[pd.DataFrame], pd.DataFrame],
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stressed_true: list[np.ndarray] = []
        stressed_pred: list[np.ndarray] = []
        for fp in fold_payloads:
            Xs = transform(fp["X_val"])
            yp = _predict_sales(
                fp["model"],
                Xs,
                target_transform=target_transform,
                clip_negative_preds=clip_negative,
            )
            stressed_true.append(fp["y_true"])
            stressed_pred.append(yp)
        stressed_mean = _mean_rmsle_over_folds(stressed_true, stressed_pred)
        logger.info(
            "scenario=%s clean=%.6f stressed=%.6f delta=%+.6f",
            name,
            clean_mean,
            stressed_mean,
            stressed_mean - clean_mean,
        )
        return _scenario_row(
            name=name,
            clean_rmsle=clean_mean,
            stressed_rmsle=stressed_mean,
            detail=detail,
        )

    # 1) Relative noise 5% / 10%
    noise_cfg = stress_cfg.get("noise") or {}
    noise_cols = list(noise_cfg.get("columns") or [])
    nn_cols = list(noise_cfg.get("non_negative") or [])
    levels = list(noise_cfg.get("relative_levels") or [0.05, 0.10])
    present_noise = [c for c in noise_cols if c in feature_cols]
    for rel in levels:
        pct = int(round(float(rel) * 100))
        scenarios.append(
            score_transform(
                f"noise_rel_{pct}pct",
                lambda X, r=float(rel): relative_noise(
                    X,
                    present_noise,
                    relative=r,
                    seed=seed,
                    non_negative=nn_cols,
                ),
                detail={"relative": float(rel), "columns": present_noise},
            )
        )

    # 2) Missing oil/holiday join failure
    miss_cfg = stress_cfg.get("missing_join") or {}
    miss_cols = list(miss_cfg.get("columns") or [])
    present_miss = [c for c in miss_cols if c in feature_cols]
    if not present_miss:
        scenarios.append(
            _scenario_row(
                name="missing_oil_holiday_join",
                clean_rmsle=clean_mean,
                stressed_rmsle=clean_mean,
                status="not_applicable",
                detail={
                    "reason": "finalist feature_groups exclude oil/holiday",
                    "requested_columns": miss_cols,
                    "present_columns": present_miss,
                    "feature_groups": groups,
                },
            )
        )
        logger.info(
            "scenario=missing_oil_holiday_join status=not_applicable "
            "(oil/holiday not in locked groups)"
        )
    else:
        scenarios.append(
            score_transform(
                "missing_oil_holiday_join",
                lambda X: null_columns(X, present_miss),
                detail={"nulled_columns": present_miss},
            )
        )

    # 3) Payday / month-end subset vs complement (clean OOF path metrics)
    # Prefer is_payday from rebuilt matrix; fallback day==15 or month-end.
    payday_rows_true: list[np.ndarray] = []
    payday_rows_pred: list[np.ndarray] = []
    complement_true: list[np.ndarray] = []
    complement_pred: list[np.ndarray] = []
    n_payday = 0
    n_comp = 0
    for fp in fold_payloads:
        X = fp["X_val"]
        yt, yp = fp["y_true"], fp["y_pred_clean"]
        if "is_payday" in X.columns:
            mask = X["is_payday"].to_numpy(dtype=float) > 0.5
        else:
            dates = pd.to_datetime(fp["meta"][date_col])
            is_me = (dates + pd.Timedelta(days=1)).dt.day.eq(1)
            mask = (dates.dt.day.eq(15) | is_me).to_numpy()
        if mask.any():
            payday_rows_true.append(yt[mask])
            payday_rows_pred.append(yp[mask])
            n_payday += int(mask.sum())
        if (~mask).any():
            complement_true.append(yt[~mask])
            complement_pred.append(yp[~mask])
            n_comp += int((~mask).sum())
    if payday_rows_true:
        pay_rmsle = float(
            rmsle(np.concatenate(payday_rows_true), np.concatenate(payday_rows_pred))
        )
    else:
        pay_rmsle = float("nan")
    if complement_true:
        comp_rmsle = float(
            rmsle(np.concatenate(complement_true), np.concatenate(complement_pred))
        )
    else:
        comp_rmsle = float("nan")
    scenarios.append(
        _scenario_row(
            name="payday_subset_vs_complement",
            clean_rmsle=clean_mean,
            stressed_rmsle=pay_rmsle if np.isfinite(pay_rmsle) else None,
            status="ok" if np.isfinite(pay_rmsle) else "empty_subset",
            detail={
                "payday_rmsle": pay_rmsle if np.isfinite(pay_rmsle) else None,
                "complement_rmsle": comp_rmsle if np.isfinite(comp_rmsle) else None,
                "n_payday": n_payday,
                "n_complement": n_comp,
                "delta_payday_minus_clean": (
                    float(pay_rmsle - clean_mean) if np.isfinite(pay_rmsle) else None
                ),
                "delta_complement_minus_clean": (
                    float(comp_rmsle - clean_mean) if np.isfinite(comp_rmsle) else None
                ),
            },
        )
    )
    logger.info(
        "scenario=payday payday_rmsle=%s complement_rmsle=%s n_pay=%d n_comp=%d",
        pay_rmsle,
        comp_rmsle,
        n_payday,
        n_comp,
    )

    # 4) Lag-history spike clipping
    clip_cfg = stress_cfg.get("lag_spike_clip") or {}
    clip_cols = [c for c in (clip_cfg.get("columns") or []) if c in feature_cols]
    uq = float(clip_cfg.get("upper_quantile", 0.99))
    scenarios.append(
        score_transform(
            "lag_history_spike_clip",
            lambda X: clip_spike_columns(X, clip_cols, upper_quantile=uq),
            detail={"upper_quantile": uq, "columns": clip_cols},
        )
    )

    summary: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "feature_groups": groups,
        "feature_columns": feature_cols,
        "seed": seed,
        "clean_repredict_mean_rmsle": clean_mean,
        "artifact_oof_pooled_rmsle": oof_mean,
        "artifact_metrics_mean_rmsle": artifact_mean,
        "n_folds": len(fold_payloads),
        "scenarios": scenarios,
        "method_note": (
            "Val features rebuilt with train sales + clean OOF preds as recursive "
            "history; frozen fold boosters re-predict under feature perturbations. "
            "ΔRMSLE = stressed_mean_rmsle - clean_repredict_mean_rmsle."
        ),
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", summary_path)

    # Compact CSV for notebook tables
    rows = []
    for s in scenarios:
        rows.append(
            {
                "scenario": s["scenario"],
                "clean_mean_rmsle": s["clean_mean_rmsle"],
                "stressed_mean_rmsle": s["stressed_mean_rmsle"],
                "delta_rmsle": s["delta_rmsle"],
                "relative_delta": s["relative_delta"],
                "status": s["status"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "scenarios.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run finalist robustness battery")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stress/default.yaml"),
        help="Stress config YAML",
    )
    args = parser.parse_args()
    paths = ProjectPaths()
    cfg_path = args.config if args.config.is_absolute() else paths.root / args.config
    stress_cfg = load_yaml(cfg_path)
    summary = run_stress(stress_cfg=stress_cfg, paths=paths)
    print(json.dumps({k: summary[k] for k in ("run_id", "clean_repredict_mean_rmsle", "scenarios") if k in summary}, indent=2))


if __name__ == "__main__":
    main()
