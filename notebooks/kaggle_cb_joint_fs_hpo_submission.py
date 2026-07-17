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
# # Store Sales — Joint Feature-Group + Nested HPO (CatBoost) → `submission.csv`
#
# **Self-contained Kaggle notebook** (no local `src/` imports).
#
# ## What this runs
# 1. Load data from `/kaggle/input/competitions/store-sales-time-series-forecasting`
# 2. Build **all** feature groups (core + optional) once
# 3. **Joint Optuna search** over:
#    - optional group flags: `oil`, `holiday`, `store_meta`, `transactions`
#    - CatBoost hyperparameters
# 4. Nested temporal scoring: **inner val = last 15d of each outer train only**
# 5. Multi-seed outer CV `{42,43,44}` with best config
# 6. Full-train retrain + recursive multi-step test → **`/kaggle/working/submission.csv`**
#
# ## Leakage rules (hard)
# - Outer val / test / LB never enter trial objective
# - Lag/rolling: past only; multi-step uses mask + recursive fill
# - Oil: lag-1 only (no same-day / bfill)
# - Transactions: past only; null post-origin values in multi-step
# - Core groups always ON: `base, calendar, promo, lag, rolling`
#
# ## Budget
# - Default `CFG.n_trials = 100` (joint FS+HPO). Smoke uses 4.
# - SQLite: `optuna_cb_joint.db` (resume with `load_if_exists`)
# - GPU: `CFG.use_gpu=True` probes CatBoost GPU, falls back to CPU

# %% [markdown]
# ## 0. Config & paths

# %%
from __future__ import annotations

import gc
import json
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class CFG:
    competition: str = "store-sales-time-series-forecasting"
    kaggle_input_dir: str = (
        "/kaggle/input/competitions/store-sales-time-series-forecasting"
    )
    seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])

    n_outer_folds: int = 3
    val_days: int = 15
    gap_days: int = 0
    min_train_days: int = 365

    lags: list[int] = field(default_factory=lambda: [1, 7, 14, 28])
    roll_windows: list[int] = field(default_factory=lambda: [7, 14, 28])

    # Joint search budget
    n_trials: int = 80  # CB slower; raise to 100 if time allows
    trial_n_estimators: int = 400
    trial_early_stopping: int = 40
    retrain_n_estimators: int = 800
    retrain_early_stopping: int = 50

    study_name: str = "cb_joint_fs_hpo"
    optuna_db_name: str = "optuna_cb_joint.db"
    optuna_load_if_exists: bool = True

    target_transform: str = "log1p"
    clip_negative: bool = True
    use_gpu: bool = True  # GPU probe + CPU fallback
    n_jobs: int = 4
    thread_count: int = 4

    smoke: bool = False
    smoke_max_entities: int = 40


CFG = CFG()
CFG.smoke = os.environ.get("CB_JOINT_SMOKE", "0") == "1"
if CFG.smoke:
    CFG.n_trials = 4
    CFG.trial_n_estimators = 40
    CFG.retrain_n_estimators = 60
    CFG.min_train_days = 30
    CFG.seeds = [42, 43]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(CFG.seed)

_kaggle_candidates = [
    Path(CFG.kaggle_input_dir),
    Path(f"/kaggle/input/{CFG.competition}"),
]
_kaggle_work = Path("/kaggle/working")
_kaggle_in = next((p for p in _kaggle_candidates if p.exists()), None)
if _kaggle_in is not None:
    INPUT_DIR = _kaggle_in
    WORK_DIR = _kaggle_work if _kaggle_work.exists() else Path("outputs/kaggle_cb_joint")
    ON_KAGGLE = True
else:
    INPUT_DIR = Path("data/raw")
    WORK_DIR = Path("outputs/kaggle_cb_joint")
    ON_KAGGLE = False
WORK_DIR.mkdir(parents=True, exist_ok=True)

OPTUNA_DB_PATH = WORK_DIR / CFG.optuna_db_name
OPTUNA_STORAGE = f"sqlite:///{OPTUNA_DB_PATH.resolve()}"

print(f"ON_KAGGLE={ON_KAGGLE}")
print(f"INPUT_DIR={INPUT_DIR}")
print(f"WORK_DIR={WORK_DIR}")
print(f"OPTUNA_DB={OPTUNA_DB_PATH}")
print(f"n_trials={CFG.n_trials} seeds={CFG.seeds}")

# %% [markdown]
# ## 1. Imports

# %%
from catboost import CatBoostRegressor
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
print("optuna", optuna.__version__)
try:
    import catboost as _cb
    print("catboost", _cb.__version__)
except Exception:
    pass

# %% [markdown]
# ## 2. Metric / transform helpers

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
    yt = np.clip(np.asarray(y_true, dtype=float), 0.0, None)
    yp = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt)) ** 2)))


def transform_y(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(y, dtype=float), 0.0, None))


def inverse_y(y: np.ndarray) -> np.ndarray:
    out = np.expm1(np.asarray(y, dtype=float))
    if CFG.clip_negative:
        out = np.clip(out, 0.0, None)
    return out


def cb_task_type() -> str:
    if not CFG.use_gpu:
        return "CPU"
    try:
        m = CatBoostRegressor(
            iterations=2, depth=2, learning_rate=0.1,
            task_type="GPU", devices="0", verbose=False, allow_writing_files=False,
        )
        m.fit(np.zeros((32, 3), dtype=np.float32), np.zeros(32, dtype=np.float32))
        return "GPU"
    except Exception as exc:  # noqa: BLE001
        print("CatBoost GPU unavailable, CPU:", exc)
        return "CPU"


TASK_TYPE = cb_task_type()
print("CatBoost task_type:", TASK_TYPE)

# %% [markdown]
# ## 3. Load data

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
holidays = tables["holidays_events"].copy()
transactions = tables["transactions"].copy()

print({k: v.shape for k, v in tables.items()})

if CFG.smoke:
    ents = (
        train[["store_nbr", "family"]]
        .drop_duplicates()
        .sample(
            n=min(
                CFG.smoke_max_entities,
                train[["store_nbr", "family"]].drop_duplicates().shape[0],
            ),
            random_state=CFG.seed,
        )
    )
    train = train.merge(ents, on=["store_nbr", "family"], how="inner")
    test = test.merge(ents, on=["store_nbr", "family"], how="inner")
    print("SMOKE", train.shape, test.shape)

# %% [markdown]
# ## 4. Feature engineering — build **all groups**, select columns per trial
#
# Core (always used): base, calendar, promo, lag, rolling  
# Optional (joint flags): oil, holiday, store_meta, transactions

# %%
ENTITY = ["store_nbr", "family"]
TARGET = "sales"
DATE = "date"

CORE_GROUPS = ["base", "calendar", "promo", "lag", "rolling"]
OPTIONAL_GROUPS = ["oil", "holiday", "store_meta", "transactions"]

# Column names per group (must match builders below)
GROUP_COLS: dict[str, list[str]] = {
    "base": ["store_nbr", "family_code", "onpromotion"],
    "calendar": [
        "dow",
        "dom",
        "month",
        "weekofyear",
        "is_weekend",
        "is_payday",
        "post_eq",
        "days_since_eq",
    ],
    "promo": ["onpromotion", "onpromotion_log1p", "has_promotion"],
    "lag": [f"sales_lag_{k}" for k in CFG.lags],
    "rolling": [
        f"sales_roll_mean_{w}" for w in CFG.roll_windows
    ]
    + [f"sales_roll_std_{w}" for w in CFG.roll_windows],
    "oil": ["oil_lag_1"],
    "holiday": [
        "is_holiday",
        "is_national_holiday",
        "is_regional_holiday",
        "is_local_holiday",
        "is_bridge",
        "is_work_day",
        "is_transfer",
    ],
    "store_meta": ["city_code", "state_code", "store_type_code", "cluster"],
    "transactions": ["transactions_lag_1", "transactions_roll_mean_7"],
}


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[DATE])
    out["dow"] = d.dt.dayofweek.astype(np.int8)
    out["dom"] = d.dt.day.astype(np.int8)
    out["month"] = d.dt.month.astype(np.int8)
    out["weekofyear"] = d.dt.isocalendar().week.astype(np.int16)
    out["is_weekend"] = (out["dow"] >= 5).astype(np.int8)
    eom = d + pd.offsets.MonthEnd(0)
    out["is_payday"] = (
        (d.dt.day == 15) | (d.dt.normalize() == eom.dt.normalize())
    ).astype(np.int8)
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
    meta["city_code"] = meta["city"].astype("category").cat.codes.astype(np.int16)
    meta["state_code"] = meta["state"].astype("category").cat.codes.astype(np.int16)
    meta["store_type_code"] = meta["type"].astype("category").cat.codes.astype(np.int16)
    meta["cluster"] = meta["cluster"].astype(np.int16)
    return df.merge(
        meta[["store_nbr", "city_code", "state_code", "store_type_code", "cluster", "city", "state"]],
        on="store_nbr",
        how="left",
    )


def add_oil_lag(df: pd.DataFrame, oil_df: pd.DataFrame) -> pd.DataFrame:
    o = oil_df[["date", "dcoilwtico"]].sort_values("date").copy()
    o["oil_lag_1"] = o["dcoilwtico"].shift(1)
    return df.merge(o[["date", "oil_lag_1"]], on="date", how="left")


def add_holiday_flags(
    df: pd.DataFrame, holidays_df: pd.DataFrame
) -> pd.DataFrame:
    """Locale-aware holidays; transferred=True is not celebrated."""
    out = df.copy()
    hol = holidays_df.copy()
    hol["date"] = pd.to_datetime(hol["date"])
    celebrated = hol.loc[~hol["transferred"].astype(bool)].copy()

    national = set(celebrated.loc[celebrated["locale"].eq("National"), "date"])
    regional = celebrated.loc[
        celebrated["locale"].eq("Regional"), ["date", "locale_name"]
    ]
    local = celebrated.loc[celebrated["locale"].eq("Local"), ["date", "locale_name"]]
    bridges = set(celebrated.loc[celebrated["type"].eq("Bridge"), "date"])
    workdays = set(celebrated.loc[celebrated["type"].eq("Work Day"), "date"])
    transfers = set(celebrated.loc[celebrated["type"].eq("Transfer"), "date"])

    out["is_national_holiday"] = out[DATE].isin(national).astype(np.int8)
    reg_keys = set(zip(regional["date"], regional["locale_name"]))
    loc_keys = set(zip(local["date"], local["locale_name"]))
    out["is_regional_holiday"] = [
        int((d, s) in reg_keys) for d, s in zip(out[DATE], out.get("state", pd.Series([""] * len(out))))
    ]
    out["is_local_holiday"] = [
        int((d, c) in loc_keys) for d, c in zip(out[DATE], out.get("city", pd.Series([""] * len(out))))
    ]
    out["is_regional_holiday"] = pd.Series(out["is_regional_holiday"], index=out.index).astype(np.int8)
    out["is_local_holiday"] = pd.Series(out["is_local_holiday"], index=out.index).astype(np.int8)
    out["is_bridge"] = out[DATE].isin(bridges).astype(np.int8)
    out["is_work_day"] = out[DATE].isin(workdays).astype(np.int8)
    out["is_transfer"] = out[DATE].isin(transfers).astype(np.int8)
    out["is_holiday"] = (
        (out["is_national_holiday"] == 1)
        | (out["is_regional_holiday"] == 1)
        | (out["is_local_holiday"] == 1)
    ).astype(np.int8)
    return out


def add_transactions(df: pd.DataFrame, tx_df: pd.DataFrame) -> pd.DataFrame:
    """Past-only store transactions (lag/rolling)."""
    out = df.copy()
    tx = (
        tx_df[["date", "store_nbr", "transactions"]]
        .drop_duplicates(["date", "store_nbr"], keep="last")
        .sort_values(["store_nbr", "date"])
        .copy()
    )
    g = tx.groupby("store_nbr", sort=False)["transactions"]
    tx["transactions_lag_1"] = g.shift(1)
    hist = g.shift(1)
    tx["transactions_roll_mean_7"] = hist.groupby(tx["store_nbr"], sort=False).transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    out = out.merge(
        tx[["date", "store_nbr", "transactions_lag_1", "transactions_roll_mean_7"]],
        on=["date", "store_nbr"],
        how="left",
    )
    return out


def add_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(ENTITY + [DATE]).reset_index(drop=True)
    g = out.groupby(ENTITY, sort=False)[TARGET]
    for lag in CFG.lags:
        out[f"sales_lag_{lag}"] = g.shift(lag)
    shifted = g.shift(1)
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


def build_all_features(
    base: pd.DataFrame,
    *,
    stores_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    family_map: dict[str, int],
    include_target: bool,
) -> pd.DataFrame:
    df = base.copy()
    if not include_target and TARGET not in df.columns:
        df[TARGET] = np.nan
    df = add_calendar(df)
    df = add_promo(df)
    df = add_store_meta(df, stores_df)
    df = add_oil_lag(df, oil_df)
    df = add_holiday_flags(df, holidays_df)
    df = add_transactions(df, tx_df)
    df = add_lag_rolling(df)
    df["family_code"] = df["family"].map(family_map).fillna(-1).astype(np.int16)
    return df


def resolve_feature_cols(
    use_oil: bool,
    use_holiday: bool,
    use_store_meta: bool,
    use_transactions: bool,
) -> list[str]:
    groups = list(CORE_GROUPS)
    if use_oil:
        groups.append("oil")
    if use_holiday:
        groups.append("holiday")
    if use_store_meta:
        groups.append("store_meta")
    if use_transactions:
        groups.append("transactions")
    cols: list[str] = []
    seen: set[str] = set()
    for g in groups:
        for c in GROUP_COLS[g]:
            if c not in seen:
                seen.add(c)
                cols.append(c)
    return cols


FAMILY_MAP = {
    f: i for i, f in enumerate(sorted(train["family"].dropna().astype(str).unique()))
}

print("Building full feature matrix (all groups)…")
t0 = time.time()
train_fe = build_all_features(
    train,
    stores_df=stores,
    oil_df=oil,
    holidays_df=holidays,
    tx_df=transactions,
    family_map=FAMILY_MAP,
    include_target=True,
)
# ensure all expected cols exist
for g, cols in GROUP_COLS.items():
    for c in cols:
        if c not in train_fe.columns:
            train_fe[c] = 0
print(f"train_fe {train_fe.shape} in {time.time()-t0:.1f}s")
print("core cols", len(resolve_feature_cols(False, False, False, False)))
print("full cols", len(resolve_feature_cols(True, True, True, True)))

# %% [markdown]
# ## 5. Expanding walk-forward folds

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
        if (train_end - pd.Timestamp(dates[0])).days + 1 < min_train_days:
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
# ## 6. CatBoost fit helpers

# %%
CB_CAT_COLS = ["family_code"]


def _prep_cb_X(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for c in CB_CAT_COLS:
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out


def fit_cb(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame | None,
    y_va: np.ndarray | None,
    params: dict[str, Any],
    seed: int,
    n_estimators: int,
    early_stopping: int | None,
) -> CatBoostRegressor:
    global TASK_TYPE
    p = dict(params)
    for k in list(p.keys()):
        if k in {"num_leaves", "min_child_samples", "colsample_bytree", "max_depth", "min_child_weight", "reg_lambda", "reg_alpha"}:
            p.pop(k, None)
    kwargs: dict[str, Any] = dict(
        iterations=int(n_estimators),
        loss_function="RMSE",
        random_seed=int(seed),
        task_type=TASK_TYPE,
        verbose=False,
        allow_writing_files=False,
        thread_count=int(getattr(CFG, "thread_count", CFG.n_jobs)),
        **p,
    )
    if TASK_TYPE == "GPU":
        kwargs["devices"] = "0"
    model = CatBoostRegressor(**kwargs)
    y_tr_t = transform_y(y_tr)
    X_tr_p = _prep_cb_X(X_tr)
    fit_kw: dict[str, Any] = {}
    cats = [c for c in CB_CAT_COLS if c in X_tr_p.columns]
    if cats:
        fit_kw["cat_features"] = cats
    if (
        X_va is not None
        and y_va is not None
        and early_stopping is not None
        and len(X_va) > 100
    ):
        fit_kw["eval_set"] = (_prep_cb_X(X_va), transform_y(y_va))
        fit_kw["early_stopping_rounds"] = int(early_stopping)
        fit_kw["use_best_model"] = True
    try:
        model.fit(X_tr_p, y_tr_t, **fit_kw)
    except Exception as exc:
        if TASK_TYPE != "GPU":
            raise
        print("CatBoost GPU fit failed, CPU retry:", exc)
        TASK_TYPE = "CPU"
        kwargs["task_type"] = "CPU"
        kwargs.pop("devices", None)
        model = CatBoostRegressor(**kwargs)
        model.fit(X_tr_p, y_tr_t, **fit_kw)
    return model


def predict_cb(model: CatBoostRegressor, X: pd.DataFrame) -> np.ndarray:
    return inverse_y(model.predict(_prep_cb_X(X)))


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
    dates = np.array(sorted(fe.loc[fe[DATE] <= train_end, DATE].unique()))
    if len(dates) < val_days + 2:
        raise RuntimeError("Not enough dates for inner split")
    val_end = pd.Timestamp(dates[-1])
    val_start = pd.Timestamp(dates[-val_days])
    val_start_i = int(np.searchsorted(dates, np.datetime64(val_start)))
    train_end_i = val_start_i - gap_days - 1
    if train_end_i < 0:
        raise RuntimeError("Inner train empty")
    return pd.Timestamp(dates[train_end_i]), val_start, val_end

# %% [markdown]
# ## 7. Joint Optuna: feature flags + hyperparameters (SQLite)
#
# Trial θ = (use_oil, use_holiday, use_store_meta, use_transactions, cb_params).  
# Score = mean RMSLE on **inner** blocks only (last 15d of each outer train).

# %%
def evaluate_theta(
    *,
    feat_cols: list[str],
    params: dict[str, Any],
    seed: int = CFG.seed,
) -> tuple[float, list[float]]:
    scores: list[float] = []
    for fold in OUTER_FOLDS:
        outer_train_end = fold["train_end"]
        inner_tr_end, inner_va_start, inner_va_end = split_last_block(
            train_fe,
            outer_train_end,
            val_days=CFG.val_days,
            gap_days=CFG.gap_days,
        )
        X_tr, y_tr = slice_xy(train_fe, None, inner_tr_end, feat_cols)
        X_va, y_va = slice_xy(train_fe, inner_va_start, inner_va_end, feat_cols)
        tr_ok = np.isfinite(y_tr)
        va_ok = np.isfinite(y_va)
        X_tr, y_tr = X_tr.loc[tr_ok], y_tr[tr_ok]
        X_va, y_va = X_va.loc[va_ok], y_va[va_ok]
        model = fit_cb(
            X_tr,
            y_tr,
            X_va,
            y_va,
            params=params,
            seed=seed,
            n_estimators=CFG.trial_n_estimators,
            early_stopping=CFG.trial_early_stopping,
        )
        pred = predict_cb(model, X_va)
        scores.append(rmsle(y_va, pred))
        del model
        gc.collect()
    return float(np.mean(scores)), scores


def _n_complete(study: optuna.Study) -> int:
    return sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)


def run_joint_hpo(n_trials: int) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        use_oil = trial.suggest_categorical("use_oil", [0, 1])
        use_holiday = trial.suggest_categorical("use_holiday", [0, 1])
        use_store_meta = trial.suggest_categorical("use_store_meta", [0, 1])
        use_transactions = trial.suggest_categorical("use_transactions", [0, 1])
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        feat_cols = resolve_feature_cols(
            bool(use_oil),
            bool(use_holiday),
            bool(use_store_meta),
            bool(use_transactions),
        )
        mean_score, fold_scores = evaluate_theta(feat_cols=feat_cols, params=params)
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("std", float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0)
        trial.set_user_attr("n_features", len(feat_cols))
        trial.set_user_attr(
            "groups",
            CORE_GROUPS
            + [
                g
                for g, flag in [
                    ("oil", use_oil),
                    ("holiday", use_holiday),
                    ("store_meta", use_store_meta),
                    ("transactions", use_transactions),
                ]
                if flag
            ],
        )
        return mean_score

    OPTUNA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Optuna storage:", OPTUNA_STORAGE)
    study = optuna.create_study(
        study_name=CFG.study_name,
        storage=OPTUNA_STORAGE,
        load_if_exists=bool(CFG.optuna_load_if_exists),
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=CFG.seed),
    )
    n_done = _n_complete(study)
    n_target = int(n_trials)
    n_remaining = max(0, n_target - n_done)
    print(f"complete={n_done} target={n_target} remaining={n_remaining}")
    t0 = time.time()
    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=not ON_KAGGLE)
    else:
        print("Budget already reached; skipping optimize.")
    print(
        f"HPO done complete={_n_complete(study)}/{n_target} "
        f"in {(time.time()-t0)/60:.1f} min"
    )
    print("best value", study.best_value)
    print("best params", study.best_params)
    print("best groups", study.best_trial.user_attrs.get("groups"))
    print("DB", OPTUNA_DB_PATH, "bytes", OPTUNA_DB_PATH.stat().st_size)
    return study


study = run_joint_hpo(CFG.n_trials if not CFG.smoke else CFG.n_trials)
bp = dict(study.best_params)
BEST_FLAGS = {
    "use_oil": bool(bp.pop("use_oil")),
    "use_holiday": bool(bp.pop("use_holiday")),
    "use_store_meta": bool(bp.pop("use_store_meta")),
    "use_transactions": bool(bp.pop("use_transactions")),
}
BEST_PARAMS = bp  # CatBoost hparams only
BEST_FEAT_COLS = resolve_feature_cols(
    BEST_FLAGS["use_oil"],
    BEST_FLAGS["use_holiday"],
    BEST_FLAGS["use_store_meta"],
    BEST_FLAGS["use_transactions"],
)
BEST_GROUPS = study.best_trial.user_attrs.get("groups") or (
    CORE_GROUPS
    + [g for g, k in [
        ("oil", "use_oil"),
        ("holiday", "use_holiday"),
        ("store_meta", "use_store_meta"),
        ("transactions", "use_transactions"),
    ] if BEST_FLAGS[k]]
)

print("BEST_FLAGS", BEST_FLAGS)
print("BEST_GROUPS", BEST_GROUPS)
print("n_features", len(BEST_FEAT_COLS))
print("BEST_PARAMS", BEST_PARAMS)

trials_df = study.trials_dataframe(
    attrs=("number", "value", "params", "user_attrs", "state")
)
trials_df.to_csv(WORK_DIR / "trials.csv", index=False)
summary = {
    "best_value": study.best_value,
    "best_flags": BEST_FLAGS,
    "best_groups": BEST_GROUPS,
    "best_params": BEST_PARAMS,
    "n_features": len(BEST_FEAT_COLS),
    "n_trials_complete": _n_complete(study),
    "n_trials_target": CFG.n_trials,
    "optuna_db": str(OPTUNA_DB_PATH),
    "study_name": CFG.study_name,
}
(WORK_DIR / "joint_hpo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("Wrote trials.csv + joint_hpo_summary.json")

# %% [markdown]
# ## 8. Multi-seed outer CV (report only — not used inside Optuna)

# %%
def outer_cv_score(seed: int) -> dict[str, Any]:
    fold_scores: list[float] = []
    for fold in OUTER_FOLDS:
        tr_end = fold["train_end"]
        va_start, va_end = fold["val_start"], fold["val_end"]
        inner_tr_end, es_start, es_end = split_last_block(
            train_fe, tr_end, val_days=CFG.val_days, gap_days=0
        )
        X_fit, y_fit = slice_xy(train_fe, None, inner_tr_end, BEST_FEAT_COLS)
        X_es, y_es = slice_xy(train_fe, es_start, es_end, BEST_FEAT_COLS)
        X_va, y_va = slice_xy(train_fe, va_start, va_end, BEST_FEAT_COLS)
        ok_f, ok_e, ok_v = np.isfinite(y_fit), np.isfinite(y_es), np.isfinite(y_va)
        model = fit_cb(
            X_fit.loc[ok_f],
            y_fit[ok_f],
            X_es.loc[ok_e],
            y_es[ok_e],
            params=BEST_PARAMS,
            seed=seed,
            n_estimators=CFG.retrain_n_estimators,
            early_stopping=CFG.retrain_early_stopping,
        )
        # Outer val: single-shot with features already built from true history
        # (for lag features this is slightly optimistic vs recursive; reported as outer CV)
        pred = predict_cb(model, X_va.loc[ok_v])
        fold_scores.append(rmsle(y_va[ok_v], pred))
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
    print("Outer multi-seed seed", s)
    row = outer_cv_score(s)
    print(row)
    multi_rows.append(row)
multi_df = pd.DataFrame(multi_rows)
multi_df.to_csv(WORK_DIR / "joint_multiseed_outer.csv", index=False)
print(
    "MULTI-SEED mean±std",
    multi_df["mean_rmsle"].mean(),
    multi_df["mean_rmsle"].std(ddof=1) if len(multi_df) > 1 else 0.0,
)

# %% [markdown]
# ## 9. Full-train retrain + recursive test → submission.csv
#
# For lag/rolling safety on the test horizon: rebuild features day-by-day and write
# predictions into sales for subsequent lags. Optional transactions are origin-masked
# (no true post-train_end transactions).

# %%
def recursive_predict_test(model: CatBoostRegressor) -> pd.DataFrame:
    hist = train[[DATE, "store_nbr", "family", "onpromotion", TARGET]].copy()
    te = test[[DATE, "store_nbr", "family", "onpromotion", "id"]].copy()
    te[TARGET] = np.nan
    panel = pd.concat([hist, te.drop(columns=["id"])], ignore_index=True)
    panel[DATE] = pd.to_datetime(panel[DATE])
    train_end = pd.Timestamp(train[DATE].max())

    # Mask transactions after train_end for multi-step (outcome-like)
    tx_safe = transactions.copy()
    tx_safe.loc[tx_safe["date"] > train_end, "transactions"] = np.nan

    test_dates = sorted(te[DATE].unique())
    chunks: list[pd.DataFrame] = []
    lookback = max(CFG.lags + CFG.roll_windows) + 7

    for day in test_dates:
        window_start = pd.Timestamp(day) - pd.Timedelta(days=lookback)
        sub = panel[(panel[DATE] >= window_start) & (panel[DATE] <= day)].copy()
        fe = build_all_features(
            sub,
            stores_df=stores,
            oil_df=oil,
            holidays_df=holidays,
            tx_df=tx_safe,
            family_map=FAMILY_MAP,
            include_target=True,
        )
        for c in BEST_FEAT_COLS:
            if c not in fe.columns:
                fe[c] = 0
        day_rows = fe.loc[fe[DATE] == pd.Timestamp(day)].copy()
        if day_rows.empty:
            raise RuntimeError(f"No rows for {day}")
        pred = predict_cb(model, day_rows[BEST_FEAT_COLS])
        keys = day_rows[["store_nbr", "family", DATE]].copy()
        keys["y_pred"] = pred
        panel = panel.merge(
            keys.rename(columns={"y_pred": "_pred"}),
            on=["store_nbr", "family", DATE],
            how="left",
        )
        fill = panel["_pred"].notna()
        panel.loc[fill, TARGET] = panel.loc[fill, "_pred"]
        panel = panel.drop(columns=["_pred"])
        chunks.append(keys)

    preds = pd.concat(chunks, ignore_index=True)
    te2 = te.copy()
    te2[DATE] = pd.to_datetime(te2[DATE])
    out = te2.merge(preds, on=["store_nbr", "family", DATE], how="left")
    if out["y_pred"].isna().any():
        raise RuntimeError("Missing test predictions")
    return out


print("Full-train multi-seed inference…")
seed_cols: list[str] = []
test_ids = test[["id", "store_nbr", "family", "date"]].copy()
test_ids["date"] = pd.to_datetime(test_ids["date"])

for s in CFG.seeds:
    print("Train seed", s)
    tr_end = pd.Timestamp(train_fe[DATE].max())
    inner_tr_end, es_start, es_end = split_last_block(
        train_fe, tr_end, val_days=CFG.val_days, gap_days=0
    )
    X_fit, y_fit = slice_xy(train_fe, None, inner_tr_end, BEST_FEAT_COLS)
    X_es, y_es = slice_xy(train_fe, es_start, es_end, BEST_FEAT_COLS)
    ok_f, ok_e = np.isfinite(y_fit), np.isfinite(y_es)
    model = fit_cb(
        X_fit.loc[ok_f],
        y_fit[ok_f],
        X_es.loc[ok_e],
        y_es[ok_e],
        params=BEST_PARAMS,
        seed=s,
        n_estimators=CFG.retrain_n_estimators,
        early_stopping=CFG.retrain_early_stopping,
    )
    bi = getattr(model, "best_iteration_", None)
    if bi is None and hasattr(model, "get_best_iteration"):
        try:
            bi = model.get_best_iteration()
        except Exception:
            bi = None
    print("  best_iteration", bi)
    pred_df = recursive_predict_test(model)
    col = f"pred_seed_{s}"
    test_ids = test_ids.merge(
        pred_df[["id", "y_pred"]].rename(columns={"y_pred": col}),
        on="id",
        how="left",
    )
    seed_cols.append(col)
    model.save_model(str(WORK_DIR / f"cb_joint_seed{s}.cbm"))
    del model
    gc.collect()

test_ids["sales"] = test_ids[seed_cols].mean(axis=1)
if CFG.clip_negative:
    test_ids["sales"] = test_ids["sales"].clip(lower=0.0)

# %% [markdown]
# ## 10. Submission sanity → `submission.csv`

# %%
how = "inner" if CFG.smoke else "left"
sub = sample[["id"]].merge(test_ids[["id", "sales"]], on="id", how=how)
if not CFG.smoke:
    assert len(sub) == len(sample)
    assert sub["id"].tolist() == sample["id"].tolist()
assert sub["sales"].notna().all()
assert np.isfinite(sub["sales"].to_numpy()).all()
assert (sub["sales"] >= 0).all()

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

card = {
    "model": "catboost_joint_fs_hpo",
    "best_flags": BEST_FLAGS,
    "best_groups": BEST_GROUPS,
    "best_params": BEST_PARAMS,
    "n_features": len(BEST_FEAT_COLS),
    "hpo_best_inner_mean_rmsle": study.best_value,
    "multi_seed_outer": multi_rows,
    "multi_seed_mean": float(multi_df["mean_rmsle"].mean()),
    "n_trials_complete": _n_complete(study),
    "n_trials_target": CFG.n_trials,
    "optuna_db": str(OPTUNA_DB_PATH),
    "submission": str(out_path),
    "n_rows": int(len(sub)),
}
(WORK_DIR / "run_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
print(json.dumps(card, indent=2)[:1500])

# %% [markdown]
# ## 11. Decision log
#
# | Step | Artifact |
# | --- | --- |
# | Joint FS+HPO | `optuna_cb_joint.db`, `trials.csv`, `joint_hpo_summary.json` |
# | Best groups | core + selected optional flags |
# | Multi-seed outer | `joint_multiseed_outer.csv` |
# | Submission | **`submission.csv`** (seed-mean recursive) |
#
# **Observation:** Feature selection and HPO ran in one nested temporal study.  
# **Interpretation:** Outer multi-seed is the honest local score (not inner-best alone).  
# **Action:** Compare public LB to prior LGBM 0.47064 / XGB 0.44139; update scores doc.
