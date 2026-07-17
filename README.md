# Store Sales — Time Series Forecasting

Leakage-safe hybrid pipeline for the Kaggle [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition: forecast unit sales for Favorita grocery store–family series under **RMSLE**.

**Problem one-liner:** predict daily unit sales for thousands of `(store_nbr, family)` series over a **15-day** horizon, using promotions, oil, holidays, stores, and transactions—without temporal leakage.

**Headline (local CV only):** locked LightGBM multi-seed walk-forward mean RMSLE **0.3933 ± 0.0004**, beating the seasonal-naive-7 floor **~0.5513** by ~0.158. **No public leaderboard scores are claimed here.**

This is a portfolio / learning project: notebooks decide, `src/` implements, `scripts/` execute, and `outputs/` stores evidence.

## Results (local walk-forward CV)

Primary metric is **fold-mean RMSLE** on the fixed 3-fold expanding walk-forward splits in `data/splits/`. Same folds and shared `store_sales.metrics.rmsle` for every row below. Figures are **not** Kaggle public/private LB.

| Model | Config / source | Local CV mean RMSLE | Notes |
| --- | --- | --- | --- |
| last value | `000_last_value` | ~0.6382 | naive floor |
| seasonal naive (period=14) | `002_seasonal_naive_14` | ~0.5583 | secondary naive |
| **seasonal naive (period=7)** | `001_seasonal_naive_7` | **~0.5513** | **primary floor** |
| **LightGBM multi-seed (42/43/44)** | `020_lgbm_hpo_best` / `configs/final.yaml` | **0.3933 ± 0.0004** | **locked finalist** |
| LightGBM evaluate (seed 42 OOF) | `scripts/evaluate.py` | ~0.3935 | fold-mean from locked OOF |
| XGBoost multi-seed | `031_xgboost_locked_groups` | ~0.3990 ± 0.0029 | secondary challenger |
| CatBoost multi-seed | `030_catboost_locked_groups` | ~0.4117 ± 0.0011 | tertiary challenger |

Evidence files:

- Multi-seed summary: [`outputs/reports/multi_seed_summary.csv`](outputs/reports/multi_seed_summary.csv)
- Locked evaluation: [`outputs/final_evaluation/metrics.json`](outputs/final_evaluation/metrics.json)
- Baselines vs final: [`outputs/reports/final_results/baselines_vs_final.csv`](outputs/reports/final_results/baselines_vs_final.csv)
- Stress battery: [`outputs/stress/default/summary.json`](outputs/stress/default/summary.json)
- Submission artifact: [`outputs/submissions/submission.csv`](outputs/submissions/submission.csv) (28 512 rows, `id,sales`)

**Locked ship path:** single LightGBM (`configs/final.yaml`), feature groups `[base, calendar, promo, lag, rolling]`, `log1p` target, non-negative clip, primary seed **42**. Optional OOF blend of LGBM/XGB/CB was measured slightly better on pooled OOF but **not** locked (higher ops cost, worse seed stability than single LGBM).

## Requirements

- Python **≥ 3.11**
- [uv](https://docs.astral.sh/uv/) for environment + lockfile sync
- [Kaggle API credentials](https://www.kaggle.com/docs/api) for data download (`~/.kaggle/kaggle.json`, mode `600`, or `KAGGLE_USERNAME` / `KAGGLE_KEY`)
- Accept the competition rules on Kaggle before downloading
- Optional GPU for LightGBM (CPU fallback works; slower)

## Setup

```bash
# from repo root
uv sync --all-extras
```

Run tools through `uv run` so the project `.venv` and `uv.lock` stay the source of truth (no manual `pip install` path).

Optional: install PEP 735 dev tools used for notebooks (Jupytext / nbconvert):

```bash
uv sync --all-extras --group dev
```

## Data download

Competition assets use the **legacy Kaggle CLI** (full competition lifecycle). Do not commit raw CSVs or `kaggle.json`.

```bash
mkdir -p data/raw
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw
unzip -o data/raw/store-sales-time-series-forecasting.zip -d data/raw
```

Expected CSVs under `data/raw/`: `train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`, `sample_submission.csv`.

Quick load smoke check:

```bash
uv run python -c "from pathlib import Path; from store_sales.data.load import load_raw_tables; t=load_raw_tables(Path('data/raw')); print({k: v.shape for k,v in t.items()})"
```

## Reproduce (ordered)

### 1. Prepare data

Structural clean + interim parquet + expanding walk-forward fold manifests (no scaler fitting, no global winsorization):

```bash
uv run python scripts/prepare_data.py
```

Writes:

- `data/interim/*.parquet` — cleaned tables
- `data/splits/fold_*_{train,val}_idx.parquet` + `folds_meta.json` — fixed temporal folds

### 2. Score naive baselines (floors)

```bash
uv run python scripts/train.py --config configs/experiments/000_last_value.yaml
uv run python scripts/train.py --config configs/experiments/001_seasonal_naive_7.yaml
uv run python scripts/train.py --config configs/experiments/002_seasonal_naive_14.yaml
```

Each run writes `outputs/runs/<run_id>/` with `config.yaml`, `metrics.json`, `environment.json`, and `run_metadata.json`.

### 3. Train locked LightGBM finalist

```bash
uv run python scripts/train.py --config configs/experiments/020_lgbm_hpo_best.yaml
```

Optional multi-seed matrix (seeds 42/43/44; reuses existing run dirs when metrics already exist):

```bash
uv run python scripts/multi_seed.py \
  --configs configs/experiments/020_lgbm_hpo_best.yaml \
            configs/experiments/030_catboost_locked_groups.yaml \
            configs/experiments/031_xgboost_locked_groups.yaml \
  --seeds 42,43,44
```

Writes `outputs/reports/multi_seed_summary.csv`.

Optional robustness battery on frozen fold models:

```bash
uv run python scripts/stress_test.py --config configs/stress/default.yaml
```

Writes `outputs/stress/default/summary.json` (ΔRMSLE vs clean under noise / lag-spike / join-failure scenarios).

### 4. Evaluate locked config (no retune)

```bash
uv run python scripts/evaluate.py --config configs/final.yaml
```

Writes `outputs/final_evaluation/` (`metrics.json`, horizon/segment tables, `environment.json`). Primary selection metric remains fold-mean walk-forward RMSLE; horizon/segment tables are guardrails only.

### 5. Generate submission

```bash
uv run python scripts/predict.py --config configs/final.yaml
```

Default write path: **`outputs/submissions/submission.csv`**.

Sanity contract: row count matches `sample_submission`, no NaN, non-negative `sales` when clip is enabled in `final.yaml`.

### 6. Optional: upload to Kaggle (manual)

**Not run automatically in this portfolio path.** Only after you intentionally want a public score — and **without retuning** against LB:

```bash
kaggle competitions submit \
  -c store-sales-time-series-forecasting \
  -f outputs/submissions/submission.csv \
  -m "locked LGBM final.yaml multi-seed 0.3933±0.0004 local CV"
```

Requires accepted competition rules and valid API credentials. Record any public LB score separately; do not edit `configs/final.yaml` from LB feedback.

## Locked config

| Field | Value |
| --- | --- |
| Config | [`configs/final.yaml`](configs/final.yaml) |
| Run id | `020_lgbm_hpo_best` |
| Model | LightGBM |
| Feature groups | `base`, `calendar`, `promo`, `lag`, `rolling` |
| Seeds | `[42, 43, 44]` (primary `42`) |
| Target | `log1p` + clip negative preds |
| Submission | `outputs/submissions/submission.csv` |

Do not change groups, params, or seeds after lock without a **new experiment id**. If evaluation fails the sn7 floor, roll back in Layer 2 — do not silently edit `final.yaml`.

## Leakage stance

- **Walk-forward / expanding** validation only; folds are date-ordered and fixed in `data/splits/`.
- **Never** time-shuffled KFold (or any random split that mixes future into train).
- Features and baselines may use history only up to each fold’s `train_end` (point-in-time registry).
- One shared **RMSLE** implementation: `src/store_sales/metrics/rmsle.py` (competition formula). Secondary guardrail: MAE on `log1p`.
- Tests: `tests/test_splits_no_leakage.py`, `tests/test_feature_point_in_time.py`, `tests/test_nested_hpo_split.py`, `tests/test_submission_schema.py`.

## Notebooks

Decision-layer notebooks only (`01`–`04`). Notebooks decide; `src/` implements; `scripts/` execute. Jupytext pairs (`.ipynb` + `.py:percent`) for each.

| Notebook | Status | Role |
| --- | --- | --- |
| `notebooks/01_data_understanding.ipynb` | Done | Problem contract, EDA, cleaning/split decisions, exploratory naive floors |
| `notebooks/02_baseline_and_feature_design.ipynb` | Done | Feature groups, experiment matrix, ablation decisions |
| `notebooks/03_model_analysis.ipynb` | Done | GBDT ladder, multi-seed, stress, lock rationale → `final.yaml` |
| `notebooks/04_final_results.ipynb` | Done | Locked report, evaluate/predict path, `submission.csv` sanity |

## Layout (hybrid `src/store_sales`)

```text
Kaggle_Store_Sales_Forecasting/
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── default.yaml
│   ├── final.yaml              # Layer 3 lock
│   ├── experiments/            # 000–031 run configs
│   ├── stress/default.yaml
│   └── tuning/lightgbm.yaml
├── data/
│   ├── raw/                    # Kaggle CSVs (gitignored)
│   ├── interim/                # cleaned parquet
│   └── splits/                 # walk-forward fold manifests
├── notebooks/                  # 01–04 decision layer
├── src/store_sales/
│   ├── config.py
│   ├── data/                   # load, validate, clean, split, outliers
│   ├── features/               # point-in-time feature registry
│   ├── models/                 # baselines + GBDT
│   ├── metrics/                # rmsle + final_eval guards
│   ├── validation/             # walk-forward helpers
│   ├── io/                     # artifacts, logging, submission
│   ├── viz/
│   └── stress.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── tune.py
│   ├── multi_seed.py
│   ├── stress_test.py
│   ├── evaluate.py
│   ├── predict.py
│   └── blend_oof.py
├── outputs/                    # runs, reports, stress, submissions
├── tests/
└── docs/superpowers/
```

## Tests

```bash
uv run pytest
# focused:
uv run pytest tests/ -v
```

`pyproject.toml` sets `testpaths = ["tests"]` and `pythonpath = ["src"]`. Prefer `uv run pytest` so the locked env is used.

## Multi-seed & stress (pointers)

| Concern | Command / artifact |
| --- | --- |
| Multi-seed matrix | `uv run python scripts/multi_seed.py --configs … --seeds 42,43,44` → `outputs/reports/multi_seed_summary.csv` |
| Stress battery | `uv run python scripts/stress_test.py --config configs/stress/default.yaml` → `outputs/stress/default/summary.json` |
| Nested temporal HPO | `scripts/tune.py` + `outputs/hpo/lgbm_store_sales/` (frozen before lock) |
| Final report plots | `outputs/reports/final_results/` |

Trust **local CV** over public LB. If CV improves but LB drops (or vice versa), suspect leakage or shift — do not keep the change without a new experiment id.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Kaggle download 403 | Rules not accepted / bad token | Accept competition rules; check `~/.kaggle/kaggle.json` mode `600` |
| Missing interim / splits | `prepare_data` not run | `uv run python scripts/prepare_data.py` |
| Missing OOF for evaluate | Finalist train not run | Train `020_lgbm_hpo_best` first |
| `predict.py` fails on models | Fold boosters missing | Re-run train for locked run, or let predict retrain path if enabled |
| RMSLE worse than sn7 | Leakage / wrong groups / broken folds | Re-check splits tests; do not retune against LB |

## Delivery layers (summary)

1. **Layer 1:** contract, `prepare_data`, fixed folds, naive floors, notebook `01`, leakage tests.
2. **Layer 2:** point-in-time features, LGBM → CatBoost → XGBoost, HPO, multi-seed, stress, notebooks `02`–`03`, lock `configs/final.yaml`.
3. **Layer 3:** `evaluate.py` once, `predict.py` → `outputs/submissions/submission.csv`, notebook `04`. Optional manual Kaggle submit only.

## Documentation

| Doc | Path |
| --- | --- |
| Design spec | [docs/superpowers/specs/2026-07-17-store-sales-forecasting-design.md](docs/superpowers/specs/2026-07-17-store-sales-forecasting-design.md) |
| Implementation plan | [docs/superpowers/plans/2026-07-17-store-sales-forecasting.md](docs/superpowers/plans/2026-07-17-store-sales-forecasting.md) |
| Skill routing table | [.agent/ml-project-skill-routing-table.md](.agent/ml-project-skill-routing-table.md) |
| Competition clippings | [kaggle_competation_pages/](kaggle_competation_pages/) |

## License / data notice

Competition data is subject to [Kaggle competition rules](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules). Do not redistribute raw competition files or credentials.
