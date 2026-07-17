# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-autoscroll,-collapsed,-scrolled,-trusted,-ExecuteTime
#     comment_magics: false
#     formats: ipynb,py:percent
#     notebook_metadata_filter: kernelspec,jupytext,language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Store Sales — XGBoost HPO → Multi-Seed → `submission.csv` (Kaggle-ready)
#
# **Self-contained Kaggle notebook** (no local `src/` imports). Pipeline:
#
# 1. Load competition CSVs from
#    `/kaggle/input/competitions/store-sales-time-series-forecasting`
# 2. Leakage-safe features: calendar, promo, **past-only** lag / rolling
# 3. Expanding walk-forward outer folds (3 × 15-day val windows at the train end)
# 4. **Nested temporal HPO** (Optuna): inner score = last 15 days of each **outer train only**
# 5. **Multi-seed** outer CV with seeds `{42, 43, 44}` using best params
# 6. Retrain on full train (last-15d ES holdout) per seed → recursive multi-step test predict
# 7. Mean ensemble of seed models → **`/kaggle/working/submission.csv`**
#
# **Metric:** RMSLE (lower is better).  
# **Local reference:** prior XGB seed-42 public LB ≈ **0.44139** (baseline params, no HPO).
#
# **Runtime notes (Kaggle GPU recommended):**
# - Set `CFG.n_trials` lower (e.g. 10–15) if the session is time-limited.
# - Trial trees are capped; full retrain uses more trees + early stopping.
# - Internet not required if Optuna/XGBoost are preinstalled on the image.

# %% [markdown]
# ## 0. Config & seeds

# %%
from __future__ import annotations

import gc
import json
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class CFG:
    # Paths — Kaggle first, local repo fallback
    competition: str = "store-sales-time-series-forecasting"
    # Official competitions mount (user / current Kaggle layout)
    kaggle_input_dir: str = (
        "/kaggle/input/competitions/store-sales-time-series-forecasting"
    )
    seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])

    # Walk-forward
    n_outer_folds: int = 3
    val_days: int = 15
    gap_days: int = 0
    min_train_days: int = 365
    horizon_days: int = 15

    # Features
    lags: list[int] = field(default_factory=lambda: [1, 7, 14, 28])
    roll_windows: list[int] = field(default_factory=lambda: [7, 14, 28])

    # Nested HPO
    n_trials: int = 25  # raise to 40 when GPU/time allows
    trial_n_estimators: int = 350
    trial_early_stopping: int = 40
    retrain_n_estimators: int = 1200
    retrain_early_stopping: int = 50

    # Target / post
    target_transform: str = "log1p"
    clip_negative: bool = True

    # Device
    use_gpu: bool = True
    n_jobs: int = 4

    # Smoke mode (tiny subsample — off by default; set env XGB_NOTEBOOK_SMOKE=1)
    smoke: bool = False
    smoke_max_entities: int = 40


CFG = CFG()
CFG.smoke = os.environ.get("XGB_NOTEBOOK_SMOKE", "0") == "1"
if CFG.smoke:
    CFG.n_trials = 2
    CFG.trial_n_estimators = 50
    CFG.retrain_n_estimators = 80
    CFG.min_train_days = 30
    CFG.seeds = [42, 43]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(CFG.seed)

# Resolve paths (prefer competitions/ mount, then legacy input slug, then local)
_kaggle_candidates = [
    Path(CFG.kaggle_input_dir),  # /kaggle/input/competitions/store-sales-time-series-forecasting
    Path(f"/kaggle/input/{CFG.competition}"),  # legacy Add Data mount
]
_kaggle_work = Path("/kaggle/working")
_kaggle_in = next((p for p in _kaggle_candidates if p.exists()), None)
if _kaggle_in is not None:
    INPUT_DIR = _kaggle_in
    WORK_DIR = _kaggle_work if _kaggle_work.exists() else Path("outputs/kaggle_xgb_hpo")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ON_KAGGLE = True
else:
    # Local repo: data/raw
    INPUT_DIR = Path("data/raw")
    WORK_DIR = Path("outputs/kaggle_xgb_hpo")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ON_KAGGLE = False

print(f"ON_KAGGLE={ON_KAGGLE}")
print(f"INPUT_DIR={INPUT_DIR}")
print(f"WORK_DIR={WORK_DIR}")
print(f"CFG.n_trials={CFG.n_trials} seeds={CFG.seeds}")

# %% [markdown]
# ## 1. Imports (modeling stack)

# %%
import optuna
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("xgboost", xgb.__version__)
print("optuna", optuna.__version__)

# %% [markdown]
# ## 2. Helpers — metric, transforms, memory

# %%
def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        t = df[col].dtype
        if pd.api.types.is_float_dtype(t):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(t):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    yp = np.clip(yp, 0.0, None)
    yt = np.clip(yt, 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt)) ** 2)))


def transform_y(y: np.ndarray) -> np.ndarray:
    if CFG.target_transform == "log1p":
        return np.log1p(np.clip(np.asarray(y, dtype=float), 0.0, None))
    return np.asarray(y, dtype=float)


def inverse_y(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=float)
    if CFG.target_transform == "log1p":
        out = np.expm1(out)
    if CFG.clip_negative:
        out = np.clip(out, 0.0, None)
    return out


def xgb_device() -> str:
    if not CFG.use_gpu:
        return "cpu"
    try:
        m = xgb.XGBRegressor(
            n_estimators=1, max_depth=2, tree_method="hist", device="cuda", verbosity=0
        )
        m.fit(np.zeros((8, 2), dtype=np.float32), np.zeros(8, dtype=np.float32))
        return "cuda"
    except Exception as exc:  # noqa: BLE001
        print("GPU unavailable, CPU hist:", exc)
        return "cpu"


DEVICE = xgb_device()
print("XGBoost device:", DEVICE)

# %% [markdown]
# ## 3. Load & light clean

# %%
def load_tables(input_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "train": "train.csv",
        "test": "test.csv",
        "stores": "stores.csv",
        "oil": "oil.csv",
        "holidays_events": "holidays_events.csv",
        "transactions": "transactions.csv",
        "sample_submission": "sample_submission.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for k, fn in files.items():
        path = input_dir / fn
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        out[k] = reduce_mem(df)
    # Oil: causal ffill only
    oil = out["oil"].sort_values("date").drop_duplicates("date", keep="last")
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill()
    out["oil"] = oil.reset_index(drop=True)
    return out


tables = load_tables(INPUT_DIR)
train = tables["train"].copy()
test = tables["test"].copy()
sample = tables["sample_submission"].copy()
stores = tables["stores"].copy()
oil = tables["oil"].copy()

print({k: v.shape for k, v in tables.items()})
print("train dates", train["date"].min().date(), "→", train["date"].max().date())
print("test dates", test["date"].min().date(), "→", test["date"].max().date())

if CFG.smoke:
    # Tiny entity subsample for local dry-run
    ents = (
        train[["store_nbr", "family"]]
        .drop_duplicates()
        .sample(n=min(CFG.smoke_max_entities, train[["store_nbr", "family"]].drop_duplicates().shape[0]), random_state=CFG.seed)
    )
    train = train.merge(ents, on=["store_nbr", "family"], how="inner")
    test = test.merge(ents, on=["store_nbr", "family"], how="inner")
    print("SMOKE train/test", train.shape, test.shape)

# %% [markdown]
# ## 4. Feature engineering (point-in-time)
#
# Lags/rollings use **past only** (`shift` before rolling). For multi-step test, sales after
# origin are masked and filled with recursive predictions.

# %%
ENTITY = ["store_nbr", "family"]
TARGET = "sales"
DATE = "date"


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[DATE])
    out["dow"] = d.dt.dayofweek.astype(np.int8)
    out["dom"] = d.dt.day.astype(np.int8)
    out["month"] = d.dt.month.astype(np.int8)
    out["weekofyear"] = d.dt.isocalendar().week.astype(np.int16)
    out["is_weekend"] = (out["dow"] >= 5).astype(np.int8)
    # Ecuador public-sector payday heuristic: 15th and month-end
    eom = d + pd.offsets.MonthEnd(0)
    out["is_payday"] = ((d.dt.day == 15) | (d.dt.normalize() == eom.dt.normalize())).astype(np.int8)
    # Earthquake regime (2016-04-16+)
    eq = pd.Timestamp("2016-04-16")
    out["post_eq"] = (d >= eq).astype(np.int8)
    out["days_since_eq"] = (d - eq).dt.days.clip(lower=0).astype(np.int32)
    return out


def add_promo(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["onpromotion"] = out["onpromotion"].fillna(0).astype(np.int16)
    out["onpromotion_log1p"] = np.log1p(out["onpromotion"].astype(np.float32))
    out["has_promotion"] = (out["onpromotion"] > 0).astype(np.int8)
    return out


def add_store_meta(df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:
    meta = stores_df.copy()
    # Light encoding of categoricals for trees
    for col in ["city", "state", "type"]:
        if col in meta.columns:
            meta[col] = meta[col].astype("category").cat.codes.astype(np.int16)
    if "cluster" in meta.columns:
        meta["cluster"] = meta["cluster"].astype(np.int16)
    return df.merge(meta, on="store_nbr", how="left")


def add_oil_lag(df: pd.DataFrame, oil_df: pd.DataFrame) -> pd.DataFrame:
    o = oil_df[["date", "dcoilwtico"]].sort_values("date").copy()
    o["oil_lag_1"] = o["dcoilwtico"].shift(1)
    return df.merge(o[["date", "oil_lag_1"]], on="date", how="left")


def add_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Entity-grouped lag/rolling on sales — past only (shift before rolling)."""
    out = df.sort_values(ENTITY + [DATE]).reset_index(drop=True)
    g = out.groupby(ENTITY, sort=False)[TARGET]
    for lag in CFG.lags:
        out[f"sales_lag_{lag}"] = g.shift(lag)
    shifted = g.shift(1)
    # groupby on entity keys already aligned with `out` index order
    for w in CFG.roll_windows:
        out[f"sales_roll_mean_{w}"] = (
            shifted.groupby([out["store_nbr"], out["family"]], sort=False)
            .transform(lambda s, ww=w: s.rolling(ww, min_periods=1).mean())
            .astype(np.float32)
        )
        out[f"sales_roll_std_{w}"] = (
            shifted.groupby([out["store_nbr"], out["family"]], sort=False)
            .transform(lambda s, ww=w: s.rolling(ww, min_periods=1).std())
            .astype(np.float32)
        )
    return out


def build_feature_frame(
    base: pd.DataFrame,
    *,
    stores_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    include_target: bool,
    family_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    df = base.copy()
    if not include_target and TARGET not in df.columns:
        df[TARGET] = np.nan
    df = add_calendar(df)
    df = add_promo(df)
    df = add_store_meta(df, stores_df)
    df = add_oil_lag(df, oil_df)
    df = add_lag_rolling(df)
    if family_map is not None and "family" in df.columns:
        df["family_code"] = df["family"].map(family_map).fillna(-1).astype(np.int16)
    return df


FEATURE_EXCLUDE = {
    DATE,
    TARGET,
    "id",
    "family",
}


def featurize_matrix(df: pd.DataFrame, feat_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Return X frame + numeric feature column list."""
    x = df.copy()
    if "family" in x.columns and "family_code" not in x.columns:
        x["family_code"] = x["family"].astype("category").cat.codes.astype(np.int16)
    drop = [c for c in FEATURE_EXCLUDE if c in x.columns]
    if feat_cols is None:
        feat_cols = [
            c
            for c in x.columns
            if c not in drop and c != "family" and pd.api.types.is_numeric_dtype(x[c])
        ]
    return x, feat_cols


# Global family codes from train (stable for train/test)
FAMILY_MAP = {
    f: i for i, f in enumerate(sorted(train["family"].dropna().astype(str).unique()))
}

# Build train features once (full history)
print("Building train features…")
t0 = time.time()
train_fe = build_feature_frame(
    train, stores_df=stores, oil_df=oil, include_target=True, family_map=FAMILY_MAP
)
train_fe, FEAT_COLS = featurize_matrix(train_fe)
print(f"train_fe {train_fe.shape} n_features={len(FEAT_COLS)} in {time.time()-t0:.1f}s")
print("features sample:", FEAT_COLS[:12], "…")

# %% [markdown]
# ## 5. Expanding walk-forward folds
#
# Outer folds: three successive 15-day validation blocks ending at the last train date.
# Train for fold `k` is all rows with `date <= train_end_k`.

# %%
def build_expanding_folds(
    dates: np.ndarray,
    *,
    n_folds: int,
    val_days: int,
    gap_days: int,
    min_train_days: int,
) -> list[dict[str, Any]]:
    dates = np.array(sorted(pd.to_datetime(pd.unique(dates))))
    folds: list[dict[str, Any]] = []
    end_idx = len(dates) - 1
    for fold_id in range(n_folds):
        val_end_i = end_idx - fold_id * val_days
        val_start_i = val_end_i - val_days + 1
        train_end_i = val_start_i - gap_days - 1
        if train_end_i < 0 or val_start_i < 0:
            break
        train_end = pd.Timestamp(dates[train_end_i])
        val_start = pd.Timestamp(dates[val_start_i])
        val_end = pd.Timestamp(dates[val_end_i])
        train_start = pd.Timestamp(dates[0])
        if (train_end - train_start).days + 1 < min_train_days:
            break
        folds.append(
            {
                "fold": fold_id,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
            }
        )
    folds = list(reversed(folds))
    for i, f in enumerate(folds):
        f["fold"] = i
    if not folds:
        raise RuntimeError("Could not build folds")
    return folds


OUTER_FOLDS = build_expanding_folds(
    train_fe[DATE].values,
    n_folds=CFG.n_outer_folds,
    val_days=CFG.val_days,
    gap_days=CFG.gap_days,
    min_train_days=CFG.min_train_days if not CFG.smoke else 30,
)
print(pd.DataFrame(OUTER_FOLDS))

# %% [markdown]
# ## 6. XGBoost fit / predict helpers

# %%
def make_xgb(params: dict[str, Any], seed: int, n_estimators: int) -> xgb.XGBRegressor:
    p = dict(params)
    p.setdefault("learning_rate", 0.05)
    p.setdefault("max_depth", 8)
    p.setdefault("min_child_weight", 20)
    p.setdefault("subsample", 0.8)
    p.setdefault("colsample_bytree", 0.8)
    p.setdefault("reg_lambda", 1.0)
    p.setdefault("reg_alpha", 0.0)
    return xgb.XGBRegressor(
        n_estimators=int(n_estimators),
        objective="reg:squarederror",
        tree_method="hist",
        device=DEVICE,
        random_state=int(seed),
        n_jobs=CFG.n_jobs,
        verbosity=0,
        **p,
    )


def fit_xgb(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame | None,
    y_va: np.ndarray | None,
    params: dict[str, Any],
    seed: int,
    n_estimators: int,
    early_stopping: int | None,
) -> xgb.XGBRegressor:
    model = make_xgb(params, seed=seed, n_estimators=n_estimators)
    y_tr_t = transform_y(y_tr)
    fit_kw: dict[str, Any] = {}
    if (
        X_va is not None
        and y_va is not None
        and early_stopping is not None
        and len(X_va) > 100
    ):
        model.set_params(early_stopping_rounds=int(early_stopping))
        fit_kw["eval_set"] = [(X_va, transform_y(y_va))]
        fit_kw["verbose"] = False
    model.fit(X_tr, y_tr_t, **fit_kw)
    return model


def predict_xgb(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    pred_t = model.predict(X)
    return inverse_y(pred_t)


def slice_xy(
    fe: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
    feat_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    m = fe[DATE] <= end
    if start is not None:
        m &= fe[DATE] >= start
    part = fe.loc[m]
    return part[feat_cols], part[TARGET].to_numpy(dtype=float)


def split_last_block(
    fe: pd.DataFrame,
    train_end: pd.Timestamp,
    *,
    val_days: int,
    gap_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Inner val = last val_days unique dates with date <= train_end."""
    dates = np.array(sorted(fe.loc[fe[DATE] <= train_end, DATE].unique()))
    if len(dates) < val_days + 2:
        raise RuntimeError("Not enough dates for inner split")
    val_end = pd.Timestamp(dates[-1])
    val_start = pd.Timestamp(dates[-val_days])
    # train ends gap_days before val_start on the unique-date grid
    val_start_i = int(np.searchsorted(dates, np.datetime64(val_start)))
    train_end_i = val_start_i - gap_days - 1
    if train_end_i < 0:
        raise RuntimeError("Inner train empty")
    inner_train_end = pd.Timestamp(dates[train_end_i])
    return inner_train_end, val_start, val_end

# %% [markdown]
# ## 7. Nested temporal Optuna HPO
#
# For each trial `P`: for each outer fold, score only on the **last 15 days of that fold's train**
# (never the outer validation window). Objective = mean inner RMSLE.

# %%
def evaluate_params_inner(params: dict[str, Any], seed: int = CFG.seed) -> tuple[float, list[float]]:
    scores: list[float] = []
    for fold in OUTER_FOLDS:
        outer_train_end = fold["train_end"]
        inner_tr_end, inner_va_start, inner_va_end = split_last_block(
            train_fe,
            outer_train_end,
            val_days=CFG.val_days,
            gap_days=CFG.gap_days,
        )
        X_tr, y_tr = slice_xy(train_fe, None, inner_tr_end, FEAT_COLS)
        X_va, y_va = slice_xy(train_fe, inner_va_start, inner_va_end, FEAT_COLS)
        # Drop rows with all-null lag features early in history is ok; drop y nan
        tr_ok = np.isfinite(y_tr)
        va_ok = np.isfinite(y_va)
        X_tr, y_tr = X_tr.loc[tr_ok], y_tr[tr_ok]
        X_va, y_va = X_va.loc[va_ok], y_va[va_ok]
        model = fit_xgb(
            X_tr,
            y_tr,
            X_va,
            y_va,
            params=params,
            seed=seed,
            n_estimators=CFG.trial_n_estimators,
            early_stopping=CFG.trial_early_stopping,
        )
        pred = predict_xgb(model, X_va)
        scores.append(rmsle(y_va, pred))
        del model
        gc.collect()
    return float(np.mean(scores)), scores


def run_hpo(n_trials: int) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        }
        mean_score, fold_scores = evaluate_params_inner(params, seed=CFG.seed)
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("std", float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0)
        return mean_score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=CFG.seed),
        study_name="xgb_store_sales_nested",
    )
    t0 = time.time()
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=not ON_KAGGLE)
    print(f"HPO finished {len(study.trials)} trials in {(time.time()-t0)/60:.1f} min")
    print("best value", study.best_value)
    print("best params", study.best_params)
    return study


study = run_hpo(CFG.n_trials if not CFG.smoke else 2)
BEST_PARAMS = dict(study.best_params)

# Persist HPO summary
hpo_path = WORK_DIR / "xgb_hpo_summary.json"
hpo_path.write_text(
    json.dumps(
        {
            "best_value": study.best_value,
            "best_params": BEST_PARAMS,
            "n_trials": len(study.trials),
            "device": DEVICE,
            "outer_folds": [
                {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in f.items()}
                for f in OUTER_FOLDS
            ],
        },
        indent=2,
    ),
    encoding="utf-8",
)
print("Wrote", hpo_path)

# %% [markdown]
# ## 8. Multi-seed outer walk-forward (best params)
#
# Re-score best params on **true outer val** windows with seeds `{42,43,44}`.

# %%
def outer_cv_score(params: dict[str, Any], seed: int) -> dict[str, Any]:
    fold_scores: list[float] = []
    for fold in OUTER_FOLDS:
        tr_end = fold["train_end"]
        va_start, va_end = fold["val_start"], fold["val_end"]
        X_tr, y_tr = slice_xy(train_fe, None, tr_end, FEAT_COLS)
        X_va, y_va = slice_xy(train_fe, va_start, va_end, FEAT_COLS)
        # Early stopping from last block of train (not val) to avoid optimistic ES
        inner_tr_end, es_start, es_end = split_last_block(
            train_fe, tr_end, val_days=CFG.val_days, gap_days=0
        )
        X_fit, y_fit = slice_xy(train_fe, None, inner_tr_end, FEAT_COLS)
        X_es, y_es = slice_xy(train_fe, es_start, es_end, FEAT_COLS)
        tr_ok = np.isfinite(y_fit)
        es_ok = np.isfinite(y_es)
        va_ok = np.isfinite(y_va)
        model = fit_xgb(
            X_fit.loc[tr_ok],
            y_fit[tr_ok],
            X_es.loc[es_ok],
            y_es[es_ok],
            params=params,
            seed=seed,
            n_estimators=CFG.retrain_n_estimators,
            early_stopping=CFG.retrain_early_stopping,
        )
        pred = predict_xgb(model, X_va.loc[va_ok])
        fold_scores.append(rmsle(y_va[va_ok], pred))
        del model
        gc.collect()
    return {
        "seed": seed,
        "mean_rmsle": float(np.mean(fold_scores)),
        "std_rmsle": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
        "fold_rmsle": fold_scores,
    }


multi_rows = []
for s in CFG.seeds:
    print("Outer multi-seed eval seed=", s)
    row = outer_cv_score(BEST_PARAMS, seed=s)
    print(row)
    multi_rows.append(row)

multi_df = pd.DataFrame(multi_rows)
print(multi_df)
print(
    "MULTI-SEED mean±std across seeds:",
    multi_df["mean_rmsle"].mean(),
    multi_df["mean_rmsle"].std(ddof=1),
)
multi_df.to_csv(WORK_DIR / "xgb_multiseed_outer.csv", index=False)

# %% [markdown]
# ## 9. Full-train retrain per seed + recursive test prediction
#
# Test sales are unknown. For each horizon day we:
# 1. Concatenate train history + test rows (sales NaN on test / future)
# 2. Rebuild lag/rolling features
# 3. Predict that day for all entities
# 4. Write predictions into `sales` for the next lag steps
#
# Final submission = **mean of seed models**.

# %%
def recursive_predict_test(
    model: xgb.XGBRegressor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    oil_df: pd.DataFrame,
) -> pd.DataFrame:
    hist = train_df[[DATE, "store_nbr", "family", "onpromotion", TARGET]].copy()
    te = test_df[[DATE, "store_nbr", "family", "onpromotion", "id"]].copy()
    te[TARGET] = np.nan
    panel = pd.concat([hist, te.drop(columns=["id"])], ignore_index=True)
    panel[DATE] = pd.to_datetime(panel[DATE])

    test_dates = sorted(te[DATE].unique())
    pred_chunks: list[pd.DataFrame] = []

    for day in test_dates:
        # Only need history window for lag/rolling max
        lookback = max(CFG.lags + CFG.roll_windows) + 5
        window_start = pd.Timestamp(day) - pd.Timedelta(days=lookback)
        sub = panel[(panel[DATE] >= window_start) & (panel[DATE] <= day)].copy()
        fe = build_feature_frame(
            sub,
            stores_df=stores_df,
            oil_df=oil_df,
            include_target=True,
            family_map=FAMILY_MAP,
        )
        fe, _ = featurize_matrix(fe, feat_cols=FEAT_COLS)
        for c in FEAT_COLS:
            if c not in fe.columns:
                fe[c] = 0.0
        day_rows = fe.loc[fe[DATE] == pd.Timestamp(day)].copy()
        if day_rows.empty:
            raise RuntimeError(f"No rows for test day {day}")
        X_day = day_rows[FEAT_COLS]
        pred = predict_xgb(model, X_day)
        day_keys = day_rows[["store_nbr", "family", DATE]].copy()
        day_keys["y_pred"] = pred
        # Write into panel for subsequent lags
        panel = panel.merge(
            day_keys.rename(columns={"y_pred": "_pred"}),
            on=["store_nbr", "family", DATE],
            how="left",
        )
        fill = panel["_pred"].notna()
        panel.loc[fill, TARGET] = panel.loc[fill, "_pred"]
        panel = panel.drop(columns=["_pred"])
        pred_chunks.append(day_keys)

    preds = pd.concat(pred_chunks, ignore_index=True)
    # Attach ids from test
    te2 = te.copy()
    te2[DATE] = pd.to_datetime(te2[DATE])
    out = te2.merge(preds, on=["store_nbr", "family", DATE], how="left")
    if out["y_pred"].isna().any():
        raise RuntimeError("Missing predictions for some test rows")
    return out


print("Full-train multi-seed inference…")
seed_pred_cols = []
test_ids = test[["id", "store_nbr", "family", "date"]].copy()
test_ids["date"] = pd.to_datetime(test_ids["date"])

for s in CFG.seeds:
    print("Train seed", s)
    # ES holdout = last 15 train days
    tr_end = pd.Timestamp(train_fe[DATE].max())
    inner_tr_end, es_start, es_end = split_last_block(
        train_fe, tr_end, val_days=CFG.val_days, gap_days=0
    )
    X_fit, y_fit = slice_xy(train_fe, None, inner_tr_end, FEAT_COLS)
    X_es, y_es = slice_xy(train_fe, es_start, es_end, FEAT_COLS)
    ok_f = np.isfinite(y_fit)
    ok_e = np.isfinite(y_es)
    model = fit_xgb(
        X_fit.loc[ok_f],
        y_fit[ok_f],
        X_es.loc[ok_e],
        y_es[ok_e],
        params=BEST_PARAMS,
        seed=s,
        n_estimators=CFG.retrain_n_estimators,
        early_stopping=CFG.retrain_early_stopping,
    )
    print("  best_iteration", getattr(model, "best_iteration", None))
    pred_df = recursive_predict_test(model, train, test, stores, oil)
    col = f"pred_seed_{s}"
    test_ids = test_ids.merge(
        pred_df[["id", "y_pred"]].rename(columns={"y_pred": col}),
        on="id",
        how="left",
    )
    seed_pred_cols.append(col)
    # Save model
    model_path = WORK_DIR / f"xgb_seed{s}.json"
    model.save_model(str(model_path))
    print("  saved", model_path)
    del model
    gc.collect()

test_ids["sales"] = test_ids[seed_pred_cols].mean(axis=1)
if CFG.clip_negative:
    test_ids["sales"] = test_ids["sales"].clip(lower=0.0)

# %% [markdown]
# ## 10. Submission sanity → `submission.csv`

# %%
# Full competition: align to sample_submission order. Smoke: only predicted ids.
how = "inner" if CFG.smoke else "left"
sub = sample[["id"]].merge(test_ids[["id", "sales"]], on="id", how=how)
if not CFG.smoke:
    assert len(sub) == len(sample), (len(sub), len(sample))
    assert sub["id"].tolist() == sample["id"].tolist(), "id order/set mismatch"
assert sub["sales"].notna().all(), "NaN sales"
assert np.isfinite(sub["sales"].to_numpy()).all(), "non-finite sales"
assert (sub["sales"] >= 0).all(), "negative sales"

out_path = WORK_DIR / "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote", out_path, "rows", len(sub))
print(sub.head())
print(
    "sales min/mean/max",
    float(sub["sales"].min()),
    float(sub["sales"].mean()),
    float(sub["sales"].max()),
)

# Also dump a small run card
card = {
    "model": "xgboost",
    "best_params": BEST_PARAMS,
    "hpo_best_inner_mean_rmsle": study.best_value,
    "multi_seed_outer": multi_rows,
    "multi_seed_mean": float(multi_df["mean_rmsle"].mean()),
    "multi_seed_std": float(multi_df["mean_rmsle"].std(ddof=1)) if len(multi_df) > 1 else 0.0,
    "seeds": CFG.seeds,
    "n_trials": len(study.trials),
    "device": DEVICE,
    "submission": str(out_path),
    "n_rows": int(len(sub)),
}
(WORK_DIR / "run_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
print(json.dumps(card, indent=2)[:1200])

# %% [markdown]
# ## 11. Decision log
#
# | Step | Result |
# | --- | --- |
# | Nested HPO | Best params in `xgb_hpo_summary.json`; objective = inner mean RMSLE |
# | Multi-seed outer | `xgb_multiseed_outer.csv` — prefer stable mean |
# | Submission | Seed-mean recursive preds → **`submission.csv`** |
# | Compare | Upload to Kaggle; update `docs/kaggle_public_scores.md` with new public RMSLE |
#
# **Observation:** Nested HPO never peeks outer val or public LB.  
# **Interpretation:** Local multi-seed mean is the selection signal; public LB is a check.  
# **Action:** If public improves over 0.44139 (prior XGB), consider re-locking the repo finalist.
