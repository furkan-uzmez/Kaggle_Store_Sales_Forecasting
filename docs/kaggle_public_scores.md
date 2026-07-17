# Kaggle public leaderboard scores

**Competition:** [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)  
**Metric:** RMSLE (lower is better)  
**Recorded:** 2026-07-17  

Public scores are from the Kaggle submission UI. They are **not** the same as local walk-forward CV; use both for comparison, not as interchangeable numbers.

## Public scores by submission

| Rank (public) | Submission file | Model / source | Public score (RMSLE) | Local CV mean RMSLE (reference) |
| ---: | --- | --- | ---: | ---: |
| 1 | `outputs/submissions/submission_xgboost_031.csv` | XGBoost — `031_xgboost_locked_groups` (3-fold mean ensemble, seed 42) | **0.44139** | ~0.397 (seed 42) / multi-seed ~0.399 |
| 2 | `outputs/submissions/submission_catboost_030.csv` | CatBoost — `030_catboost_locked_groups` (3-fold mean ensemble, seed 42) | **0.46567** | ~0.412 (seed 42) / multi-seed ~0.412 |
| 3 | `outputs/submissions/submission.csv` | LightGBM locked finalist — `020_lgbm_hpo_best` / `configs/final.yaml` (3-fold mean ensemble, seed 42) | **0.47064** | multi-seed **0.3933 ± 0.0004** |

## Notes

1. **Lower RMSLE is better.** On the public LB, **XGBoost** currently ranks best among these three submissions; **LightGBM** ranks worst on public LB despite the best **local** multi-seed CV.
2. Local CV used fixed expanding walk-forward folds under `data/splits/`; public LB is the competition test window (and partial public leaderboard). Score gaps can come from distribution shift, recursive multi-step error accumulation, and HPO/selection on local folds only.
3. LightGBM remains the **repo-locked** primary path in `configs/final.yaml` until a deliberate re-lock after re-evaluation. Challenger submissions are additional artifacts for LB comparison.
4. Re-generate submissions (does not submit to Kaggle):

```bash
uv run python scripts/predict.py --config configs/final.yaml \
  --output outputs/submissions/submission.csv

uv run python scripts/predict.py \
  --config configs/experiments/030_catboost_locked_groups.yaml \
  --output outputs/submissions/submission_catboost_030.csv

uv run python scripts/predict.py \
  --config configs/experiments/031_xgboost_locked_groups.yaml \
  --output outputs/submissions/submission_xgboost_031.csv
```

5. Optional Kaggle CLI submit (manual; not automated in this project):

```bash
kaggle competitions submit -c store-sales-time-series-forecasting \
  -f outputs/submissions/submission_xgboost_031.csv \
  -m "XGBoost 031 fold-mean ensemble"
```

## Raw capture (user-reported)

| File | Public Score |
| --- | ---: |
| `submission_xgboost_031.csv` | 0.44139 |
| `submission_catboost_030.csv` | 0.46567 |
| `submission.csv` | 0.47064 |
