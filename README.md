# Store Sales — Time Series Forecasting

Leakage-safe forecasting pipeline for the Kaggle [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition.

**What it does:** predict daily unit sales for thousands of `(store_nbr, family)` series over a **15-day** horizon (Favorita grocery data), optimized for **RMSLE**, using promotions, oil, holidays, stores, and transactions—without temporal leakage.

**Who it is for:** portfolio reviewers, ML engineers, and practitioners studying walk-forward validation and GBDT forecasting on panel data.

**Headline results**

| Lens | Best figure | Notes |
| --- | --- | --- |
| Local walk-forward CV | LightGBM multi-seed **0.3933 ± 0.0004** | beats seasonal-naive-7 floor **~0.5513** |
| Public LB (snapshot 2026-07-17) | XGBoost **0.44139** | then CatBoost 0.46567, locked LGBM 0.47064 |

Local CV and public LB are **not interchangeable**. Details: [docs/kaggle_public_scores.md](docs/kaggle_public_scores.md).

Hybrid layout: notebooks decide → `src/store_sales` implements → `scripts/` execute → `outputs/` holds local evidence (gitignored).

## Table of contents

- [Features](#features)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproduce the locked pipeline](#reproduce-the-locked-pipeline)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Notebooks](#notebooks)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License and data](#license-and-data)

## Features

- **Leakage-safe design:** expanding walk-forward folds, point-in-time feature registry, no time-shuffled CV
- **Hybrid package:** reusable `src/store_sales` + thin CLI scripts + decision notebooks
- **GBDT ladder:** naive floors → LightGBM ablations → CatBoost / XGBoost challengers
- **Nested temporal HPO:** Optuna with inner validation cut from outer train only
- **Locked finalist:** multi-seed matrix, stress battery, evaluate + submission path
- **Kaggle-ready notebooks:** self-contained HPO and joint feature-group + HPO notebooks (LGBM / XGB / CatBoost)
- **Reproducible env:** Python ≥ 3.11, `uv.lock`, `pytest` leakage and schema tests

## Results

### Local walk-forward CV

Primary metric: **fold-mean RMSLE** on fixed 3-fold expanding splits from `scripts/prepare_data.py` (written under `data/splits/`, local only). Same folds and `store_sales.metrics.rmsle` for every row.

| Model | Config / source | Local CV mean RMSLE | Notes |
| --- | --- | --- | --- |
| last value | `000_last_value` | ~0.6382 | naive floor |
| seasonal naive (period=14) | `002_seasonal_naive_14` | ~0.5583 | secondary naive |
| **seasonal naive (period=7)** | `001_seasonal_naive_7` | **~0.5513** | **primary floor** |
| **LightGBM multi-seed (42/43/44)** | `020_lgbm_hpo_best` / [`configs/final.yaml`](configs/final.yaml) | **0.3933 ± 0.0004** | **locked finalist** |
| LightGBM evaluate (seed 42 OOF) | `scripts/evaluate.py` | ~0.3935 | fold-mean from locked OOF |
| XGBoost multi-seed | `031_xgboost_locked_groups` | ~0.3990 ± 0.0029 | secondary challenger |
| CatBoost multi-seed | `030_catboost_locked_groups` | ~0.4117 ± 0.0011 | tertiary challenger |

### Public leaderboard (recorded snapshot)

| Rank (among these 3) | Submission | Model | Public RMSLE | Local CV (ref.) |
| ---: | --- | --- | ---: | --- |
| 1 | `submission_xgboost_031.csv` | XGBoost `031` fold-mean ensemble | **0.44139** | multi-seed ~0.399 |
| 2 | `submission_catboost_030.csv` | CatBoost `030` fold-mean ensemble | **0.46567** | multi-seed ~0.412 |
| 3 | `submission.csv` | LightGBM locked finalist | **0.47064** | multi-seed **0.3933 ± 0.0004** |

Caveats and regenerate commands: [docs/kaggle_public_scores.md](docs/kaggle_public_scores.md).

**Locked ship path:** LightGBM via [`configs/final.yaml`](configs/final.yaml), groups `[base, calendar, promo, lag, rolling]`, `log1p` target, non-negative clip, primary seed **42**. On public LB, XGBoost currently scores better; LGBM remains the repo-locked primary until a deliberate re-lock.

### Local evidence (not in git)

After data download + reproduce steps, artifacts appear under `outputs/` (gitignored):

| Artifact | Path after reproduce |
| --- | --- |
| Multi-seed summary | `outputs/reports/multi_seed_summary.csv` |
| Locked evaluation | `outputs/final_evaluation/metrics.json` |
| Baselines vs final | `outputs/reports/final_results/baselines_vs_final.csv` |
| Stress battery | `outputs/stress/default/summary.json` |
| Submission CSV | `outputs/submissions/submission.csv` (28 512 rows, `id,sales`) |

## Requirements

- Python **≥ 3.11**
- [uv](https://docs.astral.sh/uv/) (syncs `.venv` from `uv.lock`)
- [Kaggle API](https://www.kaggle.com/docs/api) credentials for data download only (`~/.kaggle/kaggle.json` mode `600`, or `KAGGLE_USERNAME` / `KAGGLE_KEY`)
- Accepted [competition rules](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules) before download
- Optional GPU for XGBoost / CatBoost (CPU works; slower)

Do **not** commit raw CSVs or `kaggle.json`.

## Installation

From the repository root:

```bash
uv sync --all-extras
```

Optional notebook tooling (Jupytext / nbconvert):

```bash
uv sync --all-extras --group dev
```

Prefer `uv run …` so the locked environment is always used (no ad-hoc `pip install`).

## Quick start

Minimal path that proves the environment, data, and package import work.

### 1. Download competition data

```bash
mkdir -p data/raw
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw
unzip -o data/raw/store-sales-time-series-forecasting.zip -d data/raw
```

Expected files under `data/raw/`: `train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`, `sample_submission.csv`.

### 2. Smoke-load tables

```bash
uv run python -c "from pathlib import Path; from store_sales.data.load import load_raw_tables; t=load_raw_tables(Path('data/raw')); print({k: v.shape for k, v in t.items()})"
```

Expected: shapes for `train`, `test`, `stores`, `oil`, `holidays_events`, `transactions`, `sample_submission` print without error.

### 3. Prepare interim data + folds

```bash
uv run python scripts/prepare_data.py
```

Expected: `data/interim/*.parquet` and `data/splits/fold_*_{train,val}_idx.parquet` plus `folds_meta.json`.

### 4. Run tests

```bash
uv run pytest -q
```

Expected: all tests pass (leakage, point-in-time features, submission schema, metrics).

### 5. Train locked LightGBM + submission (optional full path)

```bash
uv run python scripts/train.py --config configs/experiments/020_lgbm_hpo_best.yaml
uv run python scripts/evaluate.py --config configs/final.yaml
uv run python scripts/predict.py --config configs/final.yaml
```

Expected: `outputs/runs/020_lgbm_hpo_best/`, `outputs/final_evaluation/metrics.json`, `outputs/submissions/submission.csv`.

Full training can take tens of minutes on CPU depending on hardware.

## Reproduce the locked pipeline

Ordered end-to-end path used for the portfolio lock. Skip steps you already ran.

### 1. Naive floors

```bash
uv run python scripts/train.py --config configs/experiments/000_last_value.yaml
uv run python scripts/train.py --config configs/experiments/001_seasonal_naive_7.yaml
uv run python scripts/train.py --config configs/experiments/002_seasonal_naive_14.yaml
```

Each run writes `outputs/runs/<run_id>/` with `config.yaml`, `metrics.json`, `environment.json`, `run_metadata.json`.

### 2. Locked finalist (+ optional multi-seed / stress)

```bash
uv run python scripts/train.py --config configs/experiments/020_lgbm_hpo_best.yaml

# Optional: LGBM / CatBoost / XGBoost multi-seed matrix
uv run python scripts/multi_seed.py \
  --configs configs/experiments/020_lgbm_hpo_best.yaml \
            configs/experiments/030_catboost_locked_groups.yaml \
            configs/experiments/031_xgboost_locked_groups.yaml \
  --seeds 42,43,44

# Optional: robustness battery on frozen fold models
uv run python scripts/stress_test.py --config configs/stress/default.yaml
```

Multi-seed → `outputs/reports/multi_seed_summary.csv`.  
Stress → `outputs/stress/default/summary.json`.

### 3. Evaluate (no retune) + submission

```bash
uv run python scripts/evaluate.py --config configs/final.yaml
uv run python scripts/predict.py --config configs/final.yaml
```

Evaluate writes `outputs/final_evaluation/` (metrics + horizon/segment tables).  
Predict writes **`outputs/submissions/submission.csv`** (row count matches `sample_submission`, no NaN, non-negative `sales` when clip is enabled).

### 4. Optional Kaggle upload (manual)

Not automated. Submit only after intentional review—**do not retune** against the leaderboard:

```bash
kaggle competitions submit \
  -c store-sales-time-series-forecasting \
  -f outputs/submissions/submission.csv \
  -m "locked LGBM final.yaml multi-seed 0.3933±0.0004 local CV"
```

Record public scores in [docs/kaggle_public_scores.md](docs/kaggle_public_scores.md); do not silently edit [`configs/final.yaml`](configs/final.yaml) from LB feedback.

### Delivery layers (summary)

1. **Layer 1:** contract, `prepare_data`, fixed folds, naive floors, notebook `01`, leakage tests  
2. **Layer 2:** point-in-time features, LGBM → CatBoost → XGBoost, HPO, multi-seed, stress, notebooks `02`–`03`, lock `final.yaml`  
3. **Layer 3:** `evaluate.py` once, `predict.py` → submission, notebook `04`

## Configuration

| Setting | Location | Description |
| --- | --- | --- |
| Locked finalist | [`configs/final.yaml`](configs/final.yaml) | Model, groups, seeds, transform used by evaluate/predict |
| Experiment runs | [`configs/experiments/`](configs/experiments/) | Naive floors, ablations, CB/XGB challengers |
| HPO search space | [`configs/tuning/lightgbm.yaml`](configs/tuning/lightgbm.yaml) | Nested temporal tuning (frozen before lock) |
| Stress scenarios | [`configs/stress/default.yaml`](configs/stress/default.yaml) | Noise / lag-spike / join-failure battery |
| Defaults | [`configs/default.yaml`](configs/default.yaml) | Entity/time/target contract |

### Locked config snapshot

| Field | Value |
| --- | --- |
| Config | [`configs/final.yaml`](configs/final.yaml) |
| Run id | `020_lgbm_hpo_best` |
| Model | LightGBM |
| Feature groups | `base`, `calendar`, `promo`, `lag`, `rolling` |
| Seeds | `[42, 43, 44]` (primary `42`) |
| Target | `log1p` + clip negative preds |
| Submission | `outputs/submissions/submission.csv` |

Do not change groups, params, or seeds after lock without a **new experiment id**.

### Leakage stance

- Expanding **walk-forward** folds only; never time-shuffled KFold  
- Features use history only up to each fold’s `train_end` (point-in-time registry)  
- Shared RMSLE: [`src/store_sales/metrics/rmsle.py`](src/store_sales/metrics/rmsle.py)  
- Guardrail tests: `tests/test_splits_no_leakage.py`, `tests/test_feature_point_in_time.py`, `tests/test_nested_hpo_split.py`, `tests/test_submission_schema.py`

Use **local CV for selection**; treat public LB as an external check. If CV and LB disagree, suspect leakage or shift—open a new experiment id rather than silent edits.

## Project layout

```text
Kaggle_Store_Sales_Forecasting/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── configs/                 # default, final, experiments, stress, tuning
├── data/
│   ├── raw/                 # Kaggle CSVs (gitignored)
│   ├── interim/             # cleaned parquet (gitignored)
│   └── splits/              # walk-forward fold manifests (gitignored)
├── notebooks/               # decision layer + Kaggle HPO notebooks
├── src/store_sales/         # package: data, features, models, metrics, io
├── scripts/                 # prepare / train / tune / evaluate / predict / …
├── outputs/                 # local runs & submissions (gitignored)
├── tests/
└── docs/
    └── kaggle_public_scores.md
```

## Notebooks

Notebooks decide; package code implements; scripts execute. Each notebook has a Jupytext pair (`.ipynb` + `.py:percent`). Analysis notebooks ship **without executed outputs** (no local absolute paths)—re-run after `prepare_data` to regenerate figures.

| Notebook | Role |
| --- | --- |
| [`notebooks/01_data_understanding.ipynb`](notebooks/01_data_understanding.ipynb) | Problem contract, EDA, cleaning/split decisions |
| [`notebooks/02_baseline_and_feature_design.ipynb`](notebooks/02_baseline_and_feature_design.ipynb) | Feature groups, experiment matrix, ablations |
| [`notebooks/03_model_analysis.ipynb`](notebooks/03_model_analysis.ipynb) | GBDT ladder, multi-seed, stress, lock rationale |
| [`notebooks/04_final_results.ipynb`](notebooks/04_final_results.ipynb) | Locked report, evaluate/predict, submission sanity |
| `notebooks/kaggle_*_hpo_multiseed_submission.ipynb` | Nested HPO + multi-seed + recursive submission (LGBM / XGB / CB) |
| `notebooks/kaggle_*_joint_fs_hpo_submission.ipynb` | Joint feature-group flags + HPO (LGBM / XGB / CB) |

## Development

```bash
# install (includes pytest via optional dev extras)
uv sync --all-extras

# full suite
uv run pytest

# verbose / focused
uv run pytest tests/ -v
uv run pytest tests/test_splits_no_leakage.py -v
```

`pyproject.toml` sets `testpaths = ["tests"]` and `pythonpath = ["src"]`.

Useful scripts:

| Script | Purpose |
| --- | --- |
| `scripts/prepare_data.py` | Clean + interim parquet + fold manifests |
| `scripts/train.py` | Single experiment run |
| `scripts/tune.py` | Nested temporal Optuna HPO |
| `scripts/multi_seed.py` | Multi-seed matrix over configs |
| `scripts/stress_test.py` | Robustness battery |
| `scripts/evaluate.py` | Locked final evaluation |
| `scripts/predict.py` | Submission CSV |
| `scripts/blend_oof.py` | Optional OOF blend (not locked) |

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Kaggle download 403 | Rules not accepted / bad token | Accept competition rules; check `~/.kaggle/kaggle.json` mode `600` |
| `ModuleNotFoundError: store_sales` | Env not synced / wrong cwd | Run from repo root with `uv run` after `uv sync --all-extras` |
| Missing interim / splits | `prepare_data` not run | `uv run python scripts/prepare_data.py` |
| Missing OOF for evaluate | Finalist train not run | Train `020_lgbm_hpo_best` first |
| `predict.py` fails on models | Fold boosters missing | Re-run train for locked run |
| RMSLE worse than sn7 | Leakage / wrong groups / broken folds | Re-check leakage tests; do not retune against LB |
| Public LB much worse than CV | Shift / recursive multi-step error | Compare with [docs/kaggle_public_scores.md](docs/kaggle_public_scores.md); keep experiment id discipline |

## License and data

- **Code:** [MIT License](LICENSE)
- **Competition data:** subject to [Kaggle competition rules](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules). This repository does **not** ship raw CSVs, model weights, or credentials. Download data yourself after accepting the rules.

### Further reading

| Doc | Path |
| --- | --- |
| Public LB scores (snapshot) | [docs/kaggle_public_scores.md](docs/kaggle_public_scores.md) |
| Locked config | [configs/final.yaml](configs/final.yaml) |
| Competition page | [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) |
