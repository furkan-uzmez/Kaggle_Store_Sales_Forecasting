"""Nested temporal Optuna HPO for LightGBM.

Protocol (hard):
  For each trial params P:
    for each outer fold f:
      train_f only (never outer val / LB / test)
      inner_train, inner_val = split_last_block(train_f, val_days, gap)
      fit model(P) on inner_train; score RMSLE on inner_val
    objective = mean(inner RMSLE)

Artifacts under outputs/hpo/<study_name>/:
  best_params.yaml, trials.csv, study_summary.json,
  environment.json, nested_protocol.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.samplers import TPESampler

# Allow importing sibling train helpers when run as a script.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import train as train_mod  # noqa: E402

from store_sales.config import ProjectPaths, load_default_config, load_yaml  # noqa: E402
from store_sales.data.split import split_last_block  # noqa: E402
from store_sales.io.artifacts import collect_environment  # noqa: E402
from store_sales.io.logging import get_logger  # noqa: E402
from store_sales.metrics.rmsle import rmsle  # noqa: E402
from store_sales.models.gbdt import (  # noqa: E402
    fit_lgbm,
    inverse_target,
    transform_target,
)

logger = get_logger(__name__)


def _suggest_params(
    trial: optuna.Trial,
    search_space: dict[str, Any],
) -> dict[str, Any]:
    """Map tuning YAML search_space specs to Optuna suggestions."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = str(spec.get("type", "")).lower()
        low = spec["low"]
        high = spec["high"]
        if kind == "log_float":
            params[name] = trial.suggest_float(name, float(low), float(high), log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(low), float(high))
        elif kind == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unsupported search_space type for {name!r}: {kind!r}")
    return params


def _score_inner_fold(
    *,
    panel: pd.DataFrame,
    outer_train_idx: np.ndarray,
    inner: dict[str, Any],
    groups: list[str],
    extras: dict[str, pd.DataFrame],
    feature_cols: list[str],
    entity_cols: list[str],
    date_col: str,
    target_col: str,
    model_params: dict[str, Any],
    early_stopping_rounds: int | None,
    target_transform: str,
    clip_negative_preds: bool,
    seed: int,
    use_gpu: bool,
    es_holdout_days: int = 15,
) -> float:
    """Fit on inner_train; return RMSLE on inner_val (outer train only)."""
    train_end = pd.Timestamp(inner["train_end"])
    val_start = pd.Timestamp(inner["val_start"])
    val_end = pd.Timestamp(inner["val_end"])
    train_idx = inner["train_idx"]
    val_idx = inner["val_idx"]

    # Guard: both partitions must be subsets of outer train.
    outer_set = set(np.asarray(outer_train_idx).tolist())
    if not set(np.asarray(train_idx).tolist()).issubset(outer_set):
        raise RuntimeError("inner_train indices escape outer train")
    if not set(np.asarray(val_idx).tolist()).issubset(outer_set):
        raise RuntimeError("inner_val indices escape outer train")

    recursive = train_mod._needs_recursive_forecast(groups)
    X_train, y_train, X_val, y_val, _meta_val = train_mod._prepare_lgbm_matrices(
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
    X_train, X_val, cat_cols, cat_maps = train_mod._encode_categoricals(X_train, X_val)

    y_train_t = transform_target(y_train, target_transform)
    X_fit, y_fit_t = X_train, y_train_t
    X_es, y_es_t = X_val, transform_target(y_val, target_transform)

    # Recursive multi-step: ES holdout from last days of inner_train (not outer val).
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
        fold_idx = np.unique(np.concatenate([train_idx, val_idx]))
        fold_panel = panel.loc[fold_idx].copy()
        y_true, y_pred, _ = train_mod._recursive_val_predict(
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
        y_true = y_val
        pred_t = model.predict(X_val)
        y_pred = inverse_target(pred_t, target_transform)
        if clip_negative_preds:
            y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    return float(rmsle(y_true, y_pred))


def evaluate_trial_params(
    *,
    params: dict[str, Any],
    train: pd.DataFrame,
    folds_meta: list[dict[str, Any]],
    splits_dir: Path,
    panel: pd.DataFrame,
    groups: list[str],
    extras: dict[str, pd.DataFrame],
    feature_cols: list[str],
    entity_cols: list[str],
    date_col: str,
    target_col: str,
    fixed_model: dict[str, Any],
    early_stopping_rounds: int | None,
    target_transform: str,
    clip_negative_preds: bool,
    seed: int,
    use_gpu: bool,
    inner_val_days: int,
    gap_days: int,
) -> tuple[float, float, list[float]]:
    """Score one param set across outer folds via inner last-block protocol."""
    model_params = {**fixed_model, **params}
    # Fixed keys that are not LGBMRegressor kwargs.
    early = early_stopping_rounds
    if "early_stopping_rounds" in model_params:
        early = int(model_params.pop("early_stopping_rounds"))
    for drop_key in ("objective", "verbosity"):
        model_params.pop(drop_key, None)

    fold_scores: list[float] = []
    for meta in folds_meta:
        fold = int(meta["fold"])
        outer_train_idx = pd.read_parquet(
            splits_dir / f"fold_{fold}_train_idx.parquet"
        )["idx"].to_numpy()

        outer_train = train.loc[outer_train_idx]
        inner = split_last_block(
            outer_train,
            date_col=date_col,
            val_days=inner_val_days,
            gap_days=gap_days,
        )

        score = _score_inner_fold(
            panel=panel,
            outer_train_idx=outer_train_idx,
            inner=inner,
            groups=groups,
            extras=extras,
            feature_cols=feature_cols,
            entity_cols=entity_cols,
            date_col=date_col,
            target_col=target_col,
            model_params=model_params,
            early_stopping_rounds=early,
            target_transform=target_transform,
            clip_negative_preds=clip_negative_preds,
            seed=seed,
            use_gpu=use_gpu,
        )
        fold_scores.append(score)
        logger.info(
            "inner fold=%s rmsle=%.6f train_end=%s val=%s..%s",
            fold,
            score,
            inner["train_end"].date(),
            inner["val_start"].date(),
            inner["val_end"].date(),
        )

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    return mean_score, std_score, fold_scores


def _persist_study_artifacts(
    *,
    study_dir: Path,
    study: optuna.Study,
    cfg: dict[str, Any],
    trial_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> None:
    study_dir.mkdir(parents=True, exist_ok=True)

    best = study.best_trial
    best_payload = {
        "study_name": cfg.get("study_name"),
        "metric": cfg.get("metric", "mean_rmsle"),
        "direction": cfg.get("direction", "minimize"),
        "best_value": float(best.value) if best.value is not None else None,
        "best_params": dict(best.params),
        "best_trial_number": int(best.number),
        "feature_groups": list(cfg.get("feature_groups") or []),
        "model_fixed": (cfg.get("model") or {}).get("fixed") or {},
        "seed": int(cfg.get("seed", 42)),
        "n_trials_completed": len(study.trials),
    }
    with (study_dir / "best_params.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best_payload, handle, sort_keys=False, default_flow_style=False)

    trials_df = pd.DataFrame(trial_rows)
    trials_df.to_csv(study_dir / "trials.csv", index=False)

    summary = {
        "study_name": cfg.get("study_name"),
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "best_value": float(best.value) if best.value is not None else None,
        "best_params": dict(best.params),
        "best_trial_number": int(best.number),
        "sampler": type(study.sampler).__name__,
        "metric": cfg.get("metric", "mean_rmsle"),
        "feature_groups": list(cfg.get("feature_groups") or []),
        "search_space": cfg.get("search_space") or {},
        "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (study_dir / "study_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (study_dir / "environment.json").write_text(
        json.dumps(collect_environment(), indent=2), encoding="utf-8"
    )
    (study_dir / "nested_protocol.json").write_text(
        json.dumps(protocol, indent=2, default=str), encoding="utf-8"
    )
    with (study_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, default_flow_style=False)

    logger.info("Wrote HPO artifacts to %s", study_dir)


def _write_experiment_config(
    *,
    paths: ProjectPaths,
    best_params: dict[str, Any],
    feature_groups: list[str],
    fixed: dict[str, Any],
    seed: int,
    early_stopping_rounds: int | None,
    target_transform: str,
    clip_negative_preds: bool,
    study_name: str,
    best_value: float | None,
) -> Path:
    """Materialize configs/experiments/020_lgbm_hpo_best.yaml for retrain."""
    exp_dir = paths.configs / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_path = exp_dir / "020_lgbm_hpo_best.yaml"

    # Prefer full fixed n_estimators for retrain (early stopping still on).
    params = {
        "n_estimators": int(fixed.get("n_estimators", 2000)),
        "learning_rate": float(best_params["learning_rate"]),
        "num_leaves": int(best_params["num_leaves"]),
        "min_child_samples": int(best_params["min_child_samples"]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
    }
    payload: dict[str, Any] = {
        "run_id": "020_lgbm_hpo_best",
        "seed": seed,
        "feature_groups": feature_groups,
        "model": {
            "name": "lightgbm",
            "params": params,
            "early_stopping_rounds": (
                int(early_stopping_rounds)
                if early_stopping_rounds is not None
                else int(fixed.get("early_stopping_rounds", 50))
            ),
        },
        "target_transform": target_transform,
        "clip_negative_preds": clip_negative_preds,
        "metrics": {"primary": "rmsle", "secondary": ["mae_log1p"]},
        "hpo_source": {
            "study_name": study_name,
            "best_inner_mean_rmsle": best_value,
            "best_params": best_params,
        },
    }
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
    logger.info("Wrote retrain experiment config %s", out_path)
    return out_path


def run_study(
    *,
    cfg: dict[str, Any],
    n_trials: int | None = None,
    retrain: bool = True,
) -> dict[str, Any]:
    """Execute nested temporal Optuna study and persist artifacts."""
    paths = ProjectPaths()
    default_cfg = load_default_config()
    seed = int(cfg.get("seed", default_cfg.get("seed", 42)))
    study_name = str(cfg.get("study_name", "lgbm_store_sales"))
    direction = str(cfg.get("direction", "minimize"))
    trials_budget = int(n_trials if n_trials is not None else cfg.get("n_trials", 40))

    path_cfg = default_cfg.get("paths") or {}
    interim_dir = paths.root / path_cfg.get("interim_dir", "data/interim")
    splits_dir = paths.root / path_cfg.get("splits_dir", "data/splits")
    outputs_dir = paths.root / path_cfg.get("outputs_dir", "outputs")
    protocol_cfg = cfg.get("protocol") or {}
    artifacts_rel = protocol_cfg.get("artifacts_dir", f"outputs/hpo/{study_name}")
    study_dir = paths.root / artifacts_rel

    entity_cols = list(default_cfg.get("entity_cols", ["store_nbr", "family"]))
    date_col = str(default_cfg.get("date_col", "date"))
    target_col = str(default_cfg.get("target_col", "sales"))
    use_gpu = bool(default_cfg.get("gpu", True))

    groups = list(cfg.get("feature_groups") or ["base"])
    model_cfg = cfg.get("model") or {}
    fixed = dict(model_cfg.get("fixed") or {})
    target_transform = str(model_cfg.get("target_transform", "log1p"))
    clip_negative_preds = bool(model_cfg.get("clip_negative_preds", True))
    early_stopping_rounds = fixed.get("early_stopping_rounds", 50)
    if early_stopping_rounds is not None:
        early_stopping_rounds = int(early_stopping_rounds)

    # HPO speed: fewer boost rounds during search; final retrain uses fixed cap.
    # Early stopping still applies so weak configs exit sooner.
    fixed_n_estimators = int(fixed.get("n_estimators", 2000))
    trial_n_estimators = int(
        cfg.get("trial_n_estimators")
        or min(fixed_n_estimators, 400)
    )
    trial_es = int(
        cfg.get("trial_early_stopping_rounds")
        or min(int(early_stopping_rounds or 50), 40)
    )

    inner_cfg = cfg.get("inner") or {}
    inner_val_days = int(inner_cfg.get("val_days", 15))
    gap_days = int(inner_cfg.get("gap_days", 0))

    search_space = dict(cfg.get("search_space") or {})

    train_path = interim_dir / "train.parquet"
    logger.info(
        "Starting nested HPO study=%s n_trials=%s seed=%s groups=%s",
        study_name,
        trials_budget,
        seed,
        groups,
    )
    train = pd.read_parquet(train_path)
    folds_meta = train_mod._load_folds_meta(splits_dir)

    feature_cols = train_mod.feature_columns_for_groups(groups)
    extras = train_mod._load_feature_extras(groups, interim_dir)
    keep = list(
        dict.fromkeys(
            [date_col, *entity_cols, target_col, "onpromotion"]
            + (["id"] if "id" in train.columns else [])
        )
    )
    panel = train[keep].copy()
    panel[date_col] = pd.to_datetime(panel[date_col])

    protocol = {
        "study_name": study_name,
        "outer": "walk_forward_manifests",
        "outer_folds": folds_meta,
        "inner": {
            "strategy": inner_cfg.get("strategy", "last_train_block"),
            "val_days": inner_val_days,
            "gap_days": gap_days,
        },
        "never_use": protocol_cfg.get(
            "never_use", ["outer_val", "competition_test", "public_lb"]
        ),
        "recursive_multi_step": bool(
            protocol_cfg.get("recursive_multi_step", True)
        )
        and train_mod._needs_recursive_forecast(groups),
        "metric": cfg.get("metric", "mean_rmsle"),
        "direction": direction,
        "feature_groups": groups,
        "search_space": search_space,
        "model_fixed": fixed,
        "trial_n_estimators_cap": trial_n_estimators,
        "trial_early_stopping_rounds": trial_es,
        "retrain_n_estimators": fixed_n_estimators,
        "retrain_early_stopping_rounds": early_stopping_rounds,
        "seed": seed,
        "notes": (
            "Inner val = last val_days unique dates of each outer train only. "
            "Outer val never enters fit or trial scoring. "
            "Trials use fewer estimators + early stopping for HPO speed; "
            "post-study retrain via train.py uses full fixed n_estimators."
        ),
    }

    trial_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        suggested = _suggest_params(trial, search_space)
        fixed_for_trial = {
            "n_estimators": trial_n_estimators,
            "n_jobs": -1,
            **{
                k: v
                for k, v in fixed.items()
                if k
                not in (
                    "n_estimators",
                    "early_stopping_rounds",
                    "objective",
                    "verbosity",
                    "n_jobs",
                )
            },
        }
        mean_score, std_score, fold_scores = evaluate_trial_params(
            params=suggested,
            train=train,
            folds_meta=folds_meta,
            splits_dir=splits_dir,
            panel=panel,
            groups=groups,
            extras=extras,
            feature_cols=feature_cols,
            entity_cols=entity_cols,
            date_col=date_col,
            target_col=target_col,
            fixed_model=fixed_for_trial,
            early_stopping_rounds=trial_es,
            target_transform=target_transform,
            clip_negative_preds=clip_negative_preds,
            seed=seed,
            use_gpu=use_gpu,
            inner_val_days=inner_val_days,
            gap_days=gap_days,
        )
        trial.set_user_attr("std_rmsle", std_score)
        trial.set_user_attr("fold_rmsle", fold_scores)
        row = {
            "trial_number": trial.number,
            "mean_rmsle": mean_score,
            "std_rmsle": std_score,
            **{f"fold_{i}_rmsle": s for i, s in enumerate(fold_scores)},
            **suggested,
            "state": "COMPLETE",
        }
        trial_rows.append(row)
        logger.info(
            "trial=%s mean_rmsle=%.6f ± %.6f params=%s",
            trial.number,
            mean_score,
            std_score,
            suggested,
        )
        return mean_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=trials_budget, show_progress_bar=False)

    elapsed = time.perf_counter() - t0
    protocol["elapsed_seconds"] = elapsed
    protocol["n_trials_ran"] = len(study.trials)
    protocol["best_value"] = (
        float(study.best_value) if study.best_value is not None else None
    )
    protocol["best_params"] = dict(study.best_params)

    _persist_study_artifacts(
        study_dir=study_dir,
        study=study,
        cfg=cfg,
        trial_rows=trial_rows,
        protocol=protocol,
    )

    exp_path = _write_experiment_config(
        paths=paths,
        best_params=dict(study.best_params),
        feature_groups=groups,
        fixed=fixed,
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
        target_transform=target_transform,
        clip_negative_preds=clip_negative_preds,
        study_name=study_name,
        best_value=float(study.best_value) if study.best_value is not None else None,
    )

    result = {
        "study_dir": str(study_dir),
        "best_params": dict(study.best_params),
        "best_value": float(study.best_value) if study.best_value is not None else None,
        "n_trials": len(study.trials),
        "elapsed_seconds": elapsed,
        "experiment_config": str(exp_path),
    }

    if retrain:
        logger.info(
            "Retrain best via train.py config=%s (full outer folds)",
            exp_path,
        )
        # Invoke train scoring path in-process for same environment.
        # train.main uses argparse; call via subprocess-equivalent logic.
        import subprocess

        cmd = [
            sys.executable,
            str(_SCRIPTS_DIR / "train.py"),
            "--config",
            str(exp_path),
            "--outputs-dir",
            str(outputs_dir),
        ]
        logger.info("Running: %s", " ".join(cmd))
        proc = subprocess.run(cmd, check=False)
        result["retrain_returncode"] = int(proc.returncode)
        if proc.returncode != 0:
            logger.error("Retrain failed with code %s", proc.returncode)
        else:
            logger.info("Retrain finished successfully")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested temporal Optuna HPO (outer manifests, inner last train block)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Tuning YAML (e.g. configs/tuning/lightgbm.yaml)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override n_trials from config (smoke: 2)",
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Skip post-study retrain via train.py",
    )
    parser.add_argument(
        "--trial-n-estimators",
        type=int,
        default=None,
        help="Cap boost rounds during HPO trials (default min(fixed, 400))",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.trial_n_estimators is not None:
        cfg["trial_n_estimators"] = int(args.trial_n_estimators)
    result = run_study(
        cfg=cfg,
        n_trials=args.n_trials,
        retrain=not args.no_retrain,
    )
    print(
        f"study={cfg.get('study_name')} best_mean_rmsle={result['best_value']:.6f} "
        f"n_trials={result['n_trials']} elapsed_s={result['elapsed_seconds']:.1f} "
        f"dir={result['study_dir']}"
    )


if __name__ == "__main__":
    main()
