# Store Sales — Time Series Forecasting

Leakage-safe hybrid pipeline for the Kaggle [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition: forecast unit sales for Favorita grocery store–family series under **RMSLE**.

**Problem one-liner:** predict daily unit sales for thousands of `(store_nbr, family)` series over a **15-day** horizon, using promotions, oil, holidays, stores, and transactions—without temporal leakage.

This repo is a portfolio / learning project. **Layer 1** ships the data contract, walk-forward folds, EDA decisions, and naive CV floors. Modeling (LightGBM → CatBoost → XGBoost) and a locked submission land in Layers 2–3; **no public leaderboard scores are claimed here**.

## Requirements

- Python **≥ 3.11**
- [uv](https://docs.astral.sh/uv/) for environment + lockfile sync
- [Kaggle API credentials](https://www.kaggle.com/docs/api) for data download (`~/.kaggle/kaggle.json`, mode `600`, or `KAGGLE_USERNAME` / `KAGGLE_KEY`)
- Accept the competition rules on Kaggle before downloading

## Setup

```bash
# from repo root
uv sync --all-extras
```

Run tools through `uv run` so the project `.venv` and `uv.lock` stay the source of truth (no manual `pip install` path).

Optional: also install PEP 735 dev tools used for notebooks (Jupytext / nbconvert):

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

## Prepare data

Structural clean + interim parquet + expanding walk-forward fold manifests (no scaler fitting, no global winsorization):

```bash
uv run python scripts/prepare_data.py
```

Writes:

- `data/interim/*.parquet` — cleaned tables
- `data/splits/fold_*_{train,val}_idx.parquet` + `folds_meta.json` — fixed temporal folds

## Score naive baselines (Layer 1 floors)

All three baselines use the **same** fold manifests and shared `store_sales.metrics.rmsle`:

```bash
uv run python scripts/train.py --config configs/experiments/000_last_value.yaml
uv run python scripts/train.py --config configs/experiments/001_seasonal_naive_7.yaml
uv run python scripts/train.py --config configs/experiments/002_seasonal_naive_14.yaml
```

Each run writes `outputs/runs/<run_id>/` with `config.yaml`, `metrics.json`, `environment.json`, and `run_metadata.json`.

### Official naive floors (local CV mean RMSLE)

Scores from `scripts/train.py` on the fixed 3-fold expanding walk-forward validation windows (**local CV mean RMSLE**, not leaderboard):

| Baseline | Config | Local CV mean RMSLE |
| --- | --- | --- |
| **seasonal naive (period=7)** — primary floor | `001_seasonal_naive_7.yaml` | **~0.5513** |
| seasonal naive (period=14) | `002_seasonal_naive_14.yaml` | ~0.5583 |
| last value | `000_last_value.yaml` | ~0.6382 |

Later GBDT models must beat the **sn7 ≈ 0.5513** primary floor on the same folds. Exploratory notebook tables are not the official metric source—always re-score with `train.py`.

## Leakage stance

- **Walk-forward / expanding** validation only; folds are date-ordered and fixed in `data/splits/`.
- **Never** time-shuffled KFold (or any random split that mixes future into train).
- Features and baselines may use history only up to each fold’s `train_end`.
- One shared **RMSLE** implementation: `src/store_sales/metrics/rmsle.py` (competition formula). Secondary guardrail: MAE on `log1p`.
- Split/leakage tests live under `tests/test_splits_no_leakage.py`.

## Notebooks

Decision-layer notebooks only (`01`–`04`). Notebooks decide; `src/` implements; `scripts/` execute.

| Notebook | Status | Role |
| --- | --- | --- |
| `notebooks/01_data_understanding.ipynb` | **Done** (Layer 1) | Problem contract, EDA visuals, cleaning/split decisions, exploratory naive floors |
| `notebooks/02_baseline_and_feature_design.ipynb` | Planned (Layer 2) | Feature groups, experiment matrix |
| `notebooks/03_model_analysis.ipynb` | Planned (Layer 2) | Model comparison, lock policy |
| `notebooks/04_final_results.ipynb` | Planned (Layer 3) | Locked report **and** `submission.csv` |

Paired percent-format source for `01`: `notebooks/01_data_understanding.py` (Jupytext).

## Layout (hybrid `src/store_sales`)

```text
Kaggle_Store_Sales_Forecasting/
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── default.yaml
│   └── experiments/
│       ├── 000_last_value.yaml
│       ├── 001_seasonal_naive_7.yaml
│       └── 002_seasonal_naive_14.yaml
├── data/
│   ├── raw/                 # Kaggle CSVs (gitignored)
│   ├── interim/             # cleaned parquet (gitignored if large)
│   └── splits/              # walk-forward fold manifests
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── … (02–04 planned)
├── src/store_sales/
│   ├── config.py
│   ├── data/                # load, validate, clean, split, outliers
│   ├── models/              # baseline (GBDT in Layer 2)
│   ├── metrics/             # rmsle + guards
│   ├── validation/          # walk-forward helpers
│   ├── io/                  # run artifacts + logging
│   └── viz/                 # EDA plot helpers
├── scripts/
│   ├── prepare_data.py
│   └── train.py
├── outputs/                 # runs + reports (gitignored)
├── tests/
└── docs/superpowers/
    ├── specs/…design.md
    └── plans/…forecasting.md
```

## Tests

```bash
uv run pytest
# or focused:
uv run pytest tests/ -v
```

`pyproject.toml` sets `testpaths = ["tests"]` and `pythonpath = ["src"]`. Prefer `uv run pytest` so the locked env is used.

## Documentation

| Doc | Path |
| --- | --- |
| Design spec | [docs/superpowers/specs/2026-07-17-store-sales-forecasting-design.md](docs/superpowers/specs/2026-07-17-store-sales-forecasting-design.md) |
| Implementation plan | [docs/superpowers/plans/2026-07-17-store-sales-forecasting.md](docs/superpowers/plans/2026-07-17-store-sales-forecasting.md) |
| Skill routing table | [.agent/ml-project-skill-routing-table.md](.agent/ml-project-skill-routing-table.md) |
| Competition clippings | [kaggle_competation_pages/](kaggle_competation_pages/) |

## Delivery layers (summary)

1. **Layer 1 (this README):** contract, `prepare_data`, fixed folds, naive floors, notebook `01`, tests.
2. **Layer 2:** point-in-time features, LightGBM → CatBoost → XGBoost, notebooks `02`–`03`.
3. **Layer 3:** `configs/final.yaml`, evaluate once, notebook `04` + submission.

## License / data notice

Competition data is subject to [Kaggle competition rules](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules). Do not redistribute raw competition files or credentials.
