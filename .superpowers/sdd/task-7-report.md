# Task 7 Report — Notebook `01_data_understanding.ipynb`

## Skills read

**Required (project-local `.agent/skills/`):**
- `tabular-time-series-lifecycle`
- `tabular-time-series-eda`
- `tabular-eda-mentorship`
- `tabular-ml-visual-diagnostics`
- `tabular-time-series-imbalance-handling`
- `tabular-time-series-outlier-handling`
- `tabular-outlier-handling`
- `tabular-time-series-data-leakage-control`
- `better-jupyter-notebook` (`SKILLS.md`)
- `jupytext-notebook-workflows`
- `python-observability`

**Supporting:** `data-storytelling`, `ml-reproducibility-seed-control`

**Not used as primary path:** threshold-tuning, probability-calibration.

## Deliverables

| Path | Role |
| --- | --- |
| `notebooks/01_data_understanding.ipynb` | EDA + decision log (executed) |
| `notebooks/01_data_understanding.py` | Jupytext `py:percent` pair |
| `jupytext.toml` | Pairing policy |
| `src/store_sales/viz/__init__.py` | Viz package export |
| `src/store_sales/viz/eda_plots.py` | Reusable plot helpers |
| `outputs/reports/eda/*` | Figures + exploratory baseline CSVs (local; gitignored under `outputs/`) |

## Verification

```text
uv run pytest -q  → 17 passed
uv run jupyter nbconvert --to notebook --execute notebooks/01_data_understanding.ipynb \
  --ExecutePreprocessor.timeout=1200 --output 01_data_understanding.ipynb  → OK
ls outputs/reports/eda/ → 9 PNGs + 2 CSVs
```

**Sampling honesty:** Target hist uses a 300k-row sample for RAM; zero rates, entity stats, folds, and baselines use full panel. Dtypes downcast after load.

## Key decision log summary (for Task 6)

### Problem contract
- Unit: `(date, store_nbr, family)`; target `sales`; `T0` = last train date; `L=0`; `H=15`; metric **RMSLE** (`store_sales.metrics.rmsle`).
- Cost asymmetry: over-forecast → waste; under-forecast → stockout.

### Cleaning / oil / outliers
- **Oil:** causal **ffill** confirmed (raw ~43 missing → interim leading NA only); never bfill.
- **Holidays:** preserve `transferred`; do not treat transferred as celebrated.
- **Transactions:** history-only lag/rolling; not a free future feature on test horizon.
- **Outliers:** no invalid negatives; **preserve** rare-valid spikes; EQ 2016-04-16 = **regime** (features, not caps); **no row deletion**; no global target winsorize.

### Intermittency
- Global zero rate ~31%; high-zero families include BOOKS, BABY CARE, SCHOOL AND OFFICE SUPPLIES, HOME APPLIANCES, etc.
- Treat as intermittent demand (not SMOTE/resampling).

### Splits
- Expanding walk-forward manifests OK: folds 0–2, 15-day vals, `train_end < val_start`.
- **`gap_days`:** keep **0** for pure naives; set to **max feature lookback** when Layer 2 lags lock.

### Baselines (Task 6 must re-score officially)

Exploratory (this notebook only; origin-based seasonal lag):

| baseline | mean RMSLE | std | notes |
| --- | ---: | ---: | --- |
| **seasonal_naive_7** | **0.551** | 0.053 | **primary floor** |
| seasonal_naive_14 | 0.558 | 0.060 | secondary |
| last_value | 0.638 | 0.019 | stable but weaker |

- Implement all three; periods **7 and 14**.
- Multi-step sn must look back `k*period` into train (avoid zero-fill on days 8–15).
- Do **not** treat notebook CSVs as official Task 6 artifacts.

### Feature hypotheses (Layer 2, unvalidated)
1. Calendar: DOW, payday (15th + month-end), holidays w/ locale+transferred  
2. Lags 1/7/14/28; rolling mean/std 7/14/28 (shifted)  
3. `onpromotion` + family interactions  
4. Oil lag-1 ffilled; transactions lag/rolling (past only)  
5. Zero-streak / days-since-sale  
6. Post-EQ regime indicator  
7. Store static: type, cluster, city, state  

### Model hints
- Global multi-series GBDT after beating sn7 floor; segment errors by intermittency; no threshold/calibration primary path.

## Commit

```text
docs: EDA notebook with visual diagnostics and decision log
```

## Final status

- Task 7 complete: thin executed EDA notebook, viz helpers, decision log, exploratory naive floors for Task 6.
- pytest green; notebook executes top-to-bottom; figures under `outputs/reports/eda/`.
- Official baseline runner/artifacts deferred to Task 6.

## Review fix — markdown-before-code (visual pack)

**Finding:** After target/zero O/I/A, three code cells ran back-to-back (aggregate+panels → DOW → oil) without markdown before DOW and oil cells — broke `better-jupyter-notebook` Markdown-before-code contract.

**Fix (content-equivalent; decision-log numbers unchanged):**
- Inserted short purpose markdown before DOW seasonality code cell.
- Inserted short purpose markdown before oil missingness code cell.
- Optional minor: hist title/comment aligned with log1p-positive hist (no KDE claim).

**Verification:**

```text
uv run jupytext --sync notebooks/01_data_understanding.py  → OK
uv run python (nbformat walk): code cells without preceding markdown: [] → OK structure
```

**Commit message:** `docs: add markdown before EDA visual-pack code cells`
