"""Predict fold-model loaders for LightGBM / CatBoost / XGBoost."""

from __future__ import annotations

from pathlib import Path

import pytest

# Import from scripts/predict (same path injection as CLI).
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
import predict as predict_mod  # noqa: E402


@pytest.mark.parametrize(
    ("run_id", "model_name"),
    [
        ("020_lgbm_hpo_best", "lightgbm"),
        ("030_catboost_locked_groups", "catboost"),
        ("031_xgboost_locked_groups", "xgboost"),
    ],
)
def test_load_fold_models_from_existing_runs(run_id: str, model_name: str) -> None:
    run_dir = _ROOT / "outputs" / "runs" / run_id
    if not (run_dir / "models").is_dir():
        pytest.skip(f"run artifacts missing: {run_dir}")
    models = predict_mod._load_fold_models(run_dir, model_name)
    assert len(models) == 3
    for m in models:
        assert hasattr(m, "predict")
