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
# # 02 — Baseline Review & Feature Design
#
# **Layer 2 decision notebook.** Answers: do we beat seasonal-naive, which feature
# groups are justified, is PIT safe under multi-step horizons, and how will Task 12
# run nested temporal HPO?
#
# **Does not:** retrain GBDT, run long Optuna loops, create submissions, or open a
# sealed final test beyond the walk-forward manifests already scored.
#
# **Skills read (project-local `.agent/skills/`):**
# - `better-jupyter-notebook` — Markdown before code; O/I/A; no fabricated numbers
# - `jupytext-notebook-workflows` — optional `ipynb,py:percent` pairing
# - `tabular-time-series-feature-engineering` — shifted lag/rolling; known-future vs past-only
# - `tabular-time-series-feature-selection` — chronological selection; stability ≥ folds
# - `feature-selection-optimization` — selection inside validation protocol; group ablation default
# - `tabular-time-series-hpo` — nested walk-forward HPO; no random K-fold
# - `tabular-hpo-optimization` — staged search, sealed holdout discipline
# - `tabular-ml-baseline` — naive + default GBDT portfolio under frozen contract
# - `tabular-ml-visual-diagnostics` — phase-matched charts tied to decisions
# - `ml-tabular-evaluation` — paired fold metrics, artifact lineage
# - `tabular-time-series-evaluation` — RMSLE primary; FVA vs seasonal-naive; walk-forward
# - Supporting: `tabular-time-series-lifecycle` (steps 11–19), `data-storytelling`
#
# **No long manual HPO loops in cells.** Search space + protocol are defined for Task 12
# (`configs/tuning/lightgbm.yaml`); this notebook only reviews artifacts.

# %% [markdown]
# ## 0. Problem Contract (frozen from notebook 01 + Task 6–10)
#
# | Field | Decision |
# | --- | --- |
# | Task | Multi-series panel **point forecast** of unit sales |
# | Unit | `(date, store_nbr, family)` |
# | Target | `sales` (non-negative; RMSLE via `log1p`) |
# | Horizon `H` | **15 days** |
# | Lead `L` | 0 (first forecast day = `T0 + 1`) |
# | Metric | **mean_rmsle** across 3 expanding walk-forward folds |
# | Guardrail | `mae_log1p` |
# | Naive floor | seasonal-naive period=7 (`001_seasonal_naive_7`) ≈ **0.5513** |
# | Best core LGBM | `014_lgbm_plus_rolling` ≈ **0.4004** |
# | Split | manifests in `data/splits/` — do not re-split here |
# | Forbidden | Random K-fold; global pre-split selectors; unmasked mid-horizon targets in lag/rolling |

# %% [markdown]
# ## 1. Setup — seeds, paths, logging
#
# Purpose: lock seed, resolve project root, import registry helpers (no model training).

# %%
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

try:
    from IPython.display import display
except ImportError:  # plain script fallback
    def display(obj):  # type: ignore[misc]
        print(obj)

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from store_sales.config import ProjectPaths, load_default_config
from store_sales.features.lag import DEFAULT_LAGS, add_lag_features
from store_sales.features.registry import (
    ADMITTED_GROUPS,
    CORE_ABLATION_ORDER,
    FEATURE_GROUPS,
    OPTIONAL_ABLATION_GROUPS,
    build_feature_matrix,
    list_group_ablation_configs,
    mask_target_after,
)
from store_sales.features.rolling import DEFAULT_WINDOWS, add_rolling_features
from store_sales.io.logging import get_logger

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

cfg = load_default_config()
paths = ProjectPaths()
INTERIM = paths.data_interim
SPLITS = paths.data_splits
RUNS = paths.outputs / "runs"
REPORTS = paths.outputs / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

logger = get_logger("notebook.02_baseline_and_feature_design")
logger.info("ROOT=%s SEED=%s RUNS=%s", ROOT, SEED, RUNS)

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

print(f"seed={SEED}")
print(f"runs={RUNS}")
print(f"reports={REPORTS}")
print(f"horizon_days={cfg.get('horizon_days')} metric={cfg.get('metric')}")

# %% [markdown]
# ## 2. Split verification (manifests only)
#
# Purpose: re-confirm expanding walk-forward integrity before trusting ablation RMSLE.

# %%
folds_meta = json.loads((SPLITS / "folds_meta.json").read_text(encoding="utf-8"))
train = pd.read_parquet(INTERIM / "train.parquet")
train["date"] = pd.to_datetime(train["date"])

fold_rows = []
for f in folds_meta:
    fold = int(f["fold"])
    train_end = pd.Timestamp(f["train_end"])
    val_start = pd.Timestamp(f["val_start"])
    val_end = pd.Timestamp(f["val_end"])
    assert train_end < val_start, f"fold {fold}: train_end must be < val_start"
    val_days = (val_end - val_start).days + 1
    assert val_days == 15, f"fold {fold}: expected H=15, got {val_days}"
    tr_idx_path = SPLITS / f"fold_{fold}_train_idx.parquet"
    va_idx_path = SPLITS / f"fold_{fold}_val_idx.parquet"
    tr_n = len(pd.read_parquet(tr_idx_path)) if tr_idx_path.exists() else None
    va_n = len(pd.read_parquet(va_idx_path)) if va_idx_path.exists() else None
    max_tr = train.loc[train["date"] <= train_end, "date"].max()
    min_va = train.loc[(train["date"] >= val_start) & (train["date"] <= val_end), "date"].min()
    fold_rows.append(
        {
            "fold": fold,
            "train_end": str(train_end.date()),
            "val_start": str(val_start.date()),
            "val_end": str(val_end.date()),
            "val_days": val_days,
            "idx_train_rows": tr_n,
            "idx_val_rows": va_n,
            "max_train_date": str(max_tr.date()) if pd.notna(max_tr) else None,
            "min_val_date": str(min_va.date()) if pd.notna(min_va) else None,
            "leak_train_ge_val": bool(max_tr >= min_va) if pd.notna(max_tr) and pd.notna(min_va) else None,
        }
    )

fold_df = pd.DataFrame(fold_rows)
display(fold_df)
assert not fold_df["leak_train_ge_val"].any(), "chronological leakage in folds"
print("split_ok=True n_folds=", len(fold_df))

# %% [markdown]
# ### Split verification — O/I/A
#
# **Observation:** Three expanding folds with 15-day validation blocks; `train_end < val_start` on every fold; index files present.
#
# **Interpretation:** Ablation and naive metrics under `outputs/runs/` share this sealed outer design — valid for paired group comparisons.
#
# **Action:** Freeze these manifests for Task 12 outer folds; inner HPO may only carve the last 15 days from **outer train**.

# %% [markdown]
# ## 3. Baseline & LGBM ablation scoreboard
#
# Purpose: load persisted `metrics.json` (no retrain) and compute lift vs sn7 floor.

# %%
RUN_SPECS = [
    ("000_last_value", "naive", "last_value"),
    ("001_seasonal_naive_7", "naive", "sn7"),
    ("002_seasonal_naive_14", "naive", "sn14"),
    ("010_lgbm_base", "lgbm", "base"),
    ("011_lgbm_plus_calendar", "lgbm", "+calendar"),
    ("012_lgbm_plus_promo", "lgbm", "+promo"),
    ("013_lgbm_plus_lag", "lgbm", "+lag"),
    ("014_lgbm_plus_rolling", "lgbm", "+rolling"),
]

SN7_FLOOR = 0.5513  # documented floor; will overwrite from metrics if present
rows = []
for run_id, family, label in RUN_SPECS:
    mpath = RUNS / run_id / "metrics.json"
    if not mpath.exists():
        logger.warning("missing metrics: %s", mpath)
        continue
    m = json.loads(mpath.read_text(encoding="utf-8"))
    fold_rmsle = m.get("fold_rmsle") or [f["rmsle"] for f in m.get("folds", [])]
    mean_rmsle = float(m["mean_rmsle"])
    std_rmsle = float(m.get("std_rmsle", np.std(fold_rmsle, ddof=0)))
    if run_id == "001_seasonal_naive_7":
        SN7_FLOOR = mean_rmsle
    groups = m.get("feature_groups")
    rows.append(
        {
            "run_id": run_id,
            "family": family,
            "label": label,
            "feature_groups": ",".join(groups) if groups else m.get("model_name"),
            "mean_rmsle": mean_rmsle,
            "std_rmsle": std_rmsle,
            "fold_0": fold_rmsle[0] if len(fold_rmsle) > 0 else np.nan,
            "fold_1": fold_rmsle[1] if len(fold_rmsle) > 1 else np.nan,
            "fold_2": fold_rmsle[2] if len(fold_rmsle) > 2 else np.nan,
            "beats_sn7": mean_rmsle < SN7_FLOOR if family == "lgbm" else None,
            "delta_vs_sn7": mean_rmsle - SN7_FLOOR,
            "n_features": (m.get("folds") or [{}])[0].get("n_features"),
        }
    )

scoreboard = pd.DataFrame(rows)
display(scoreboard)

# Progressive ablation deltas (LGBM only)
lgbm = scoreboard.loc[scoreboard["family"] == "lgbm"].copy().reset_index(drop=True)
if len(lgbm) >= 2:
    lgbm["delta_vs_prev"] = lgbm["mean_rmsle"].diff()
    lgbm["fold_helps_vs_prev"] = None
    for i in range(1, len(lgbm)):
        helps = 0
        for c in ("fold_0", "fold_1", "fold_2"):
            if lgbm.loc[i, c] < lgbm.loc[i - 1, c]:
                helps += 1
        lgbm.loc[i, "fold_helps_vs_prev"] = helps
    display(lgbm[["run_id", "label", "mean_rmsle", "delta_vs_prev", "fold_helps_vs_prev", "beats_sn7"]])

print(f"sn7_floor={SN7_FLOOR:.6f}")
best = scoreboard.loc[scoreboard["mean_rmsle"].idxmin()]
print(f"best_run={best['run_id']} mean_rmsle={best['mean_rmsle']:.6f}")

# %% [markdown]
# ### Scoreboard — O/I/A
#
# **Observation:** Last-value ~0.638, sn7 ~0.551, sn14 ~0.558. LGBM base-only (~0.885) is **worse** than naive; adding calendar jumps to ~0.437. Progressive promo → lag → rolling reaches **~0.4004**, beating sn7 by ~0.15 RMSLE.
#
# **Interpretation:** Calendar is the dominant exogenous signal; promo/lag/rolling add stable incremental gains (each helps on 3/3 folds in the progressive chain). Base-only LGBM without temporal structure underfits autocorrelation that sn7 already captures — complexity without features is not justified.
#
# **Action:** Lock core groups `[base, calendar, promo, lag, rolling]` for Task 12 HPO. Defer optional oil/holiday/store_meta/transactions until post-HPO one-at-a-time extensions (configs 015–018 exist).

# %% [markdown]
# ## 4. Visual — CV mean RMSLE bar chart
#
# Purpose: decision chart comparing naive floor vs progressive LGBM ablations (same outer folds).

# %%
plot_df = scoreboard.copy()
plot_df["display"] = plot_df["label"]
order = plot_df["run_id"].tolist()

fig, ax = plt.subplots(figsize=(11, 4.5))
colors = []
for fam in plot_df["family"]:
    colors.append("#4C72B0" if fam == "naive" else "#55A868")
bars = ax.bar(
    np.arange(len(plot_df)),
    plot_df["mean_rmsle"],
    yerr=plot_df["std_rmsle"],
    color=colors,
    capsize=3,
    edgecolor="black",
    linewidth=0.4,
)
ax.axhline(SN7_FLOOR, color="#C44E52", linestyle="--", linewidth=1.5, label=f"sn7 floor ({SN7_FLOOR:.4f})")
ax.set_xticks(np.arange(len(plot_df)))
ax.set_xticklabels(plot_df["display"], rotation=25, ha="right")
ax.set_ylabel("mean RMSLE (3 outer folds)")
ax.set_title("Walk-forward RMSLE: naive baselines vs LGBM group ablations")
ax.legend(loc="upper right")
# Annotate best LGBM
best_lgbm_idx = plot_df.loc[plot_df["family"] == "lgbm", "mean_rmsle"].idxmin()
bi = plot_df.index.get_loc(best_lgbm_idx)
ax.annotate(
    f"{plot_df.loc[best_lgbm_idx, 'mean_rmsle']:.4f}",
    xy=(bi, plot_df.loc[best_lgbm_idx, "mean_rmsle"]),
    xytext=(0, 8),
    textcoords="offset points",
    ha="center",
    fontsize=9,
)
fig.tight_layout()
chart_path = REPORTS / "02_cv_rmsle_baselines_vs_ablations.png"
fig.savefig(chart_path, dpi=120, bbox_inches="tight")
plt.show()
plt.close(fig)
print(f"saved={chart_path}")

# %% [markdown]
# ### Chart — O/I/A
#
# **Observation:** Clear step-down from naive (~0.55–0.64) through calendar LGBM (~0.44) to full core (~0.40). Error bars (fold std) remain small relative to the sn7 gap for 011–014.
#
# **Interpretation:** Forecast Value Added of best core vs sn7 is large and consistent; further HPO is about squeezing the core, not replacing the baseline portfolio.
#
# **Action:** Use this chart in Task 12 study summary as pre-HPO reference.

# %% [markdown]
# ## 5. Feature registry audit + formula (PIT) proof
#
# Purpose: document admitted groups, formulas, and prove multi-step safety with
# `mask_target_after` on one store×family panel (no full retrain).

# %%
print("ADMITTED_GROUPS:", ADMITTED_GROUPS)
print("CORE_ABLATION_ORDER:", CORE_ABLATION_ORDER)
print("OPTIONAL_ABLATION_GROUPS:", OPTIONAL_ABLATION_GROUPS)
print("FEATURE_GROUPS sizes:", {k: len(v) for k, v in FEATURE_GROUPS.items()})
display(pd.DataFrame(list_group_ablation_configs()))

formula_rows = [
    {
        "group": "base",
        "formula": "identity: store_nbr, family, onpromotion",
        "availability": "known_future",
        "pit_rule": "onpromotion present on test horizon",
    },
    {
        "group": "calendar",
        "formula": "dayofweek, month, day, is_weekend, is_payday(15|EOM), post_eq, days_since_eq",
        "availability": "known_future",
        "pit_rule": "calendar only; no target",
    },
    {
        "group": "promo",
        "formula": "onpromotion, log1p(onpromotion), has_promotion",
        "availability": "known_future",
        "pit_rule": "transforms of known promo",
    },
    {
        "group": "lag",
        "formula": f"sales_lag_k = groupby(store,family).sales.shift(k); k in {DEFAULT_LAGS}",
        "availability": "past_target",
        "pit_rule": "mask_target_after(T0) or recursive pred fill for H>1",
    },
    {
        "group": "rolling",
        "formula": f"shift(1).rolling(w).{{mean,std}}; w in {DEFAULT_WINDOWS}",
        "availability": "past_target",
        "pit_rule": "same multi-step mask/recursive contract as lag",
    },
    {
        "group": "oil",
        "formula": "oil_lag_1 (no same-day)",
        "availability": "past_exog",
        "pit_rule": "causal ffill + lag-1",
    },
    {
        "group": "holiday",
        "formula": "locale-aware flags; transferred not celebrated",
        "availability": "known_future",
        "pit_rule": "public calendar semantics",
    },
    {
        "group": "store_meta",
        "formula": "city, state, type, cluster",
        "availability": "static",
        "pit_rule": "join on store_nbr",
    },
    {
        "group": "transactions",
        "formula": "tx lag-1 + roll mean-7",
        "availability": "past_outcome_like",
        "pit_rule": "mask post-origin extras; no future join",
    },
]
display(pd.DataFrame(formula_rows))

# --- PIT proof on one entity around fold-0 origin ---
ENTITY_STORE = 1
ENTITY_FAMILY = "GROCERY I"
ORIGIN = pd.Timestamp(folds_meta[0]["train_end"])  # 2017-07-01
WINDOW_START = ORIGIN - pd.Timedelta(days=40)
WINDOW_END = ORIGIN + pd.Timedelta(days=15)

panel = train[
    (train["store_nbr"] == ENTITY_STORE)
    & (train["family"] == ENTITY_FAMILY)
    & (train["date"] >= WINDOW_START)
    & (train["date"] <= WINDOW_END)
].copy()
panel = panel.sort_values("date").reset_index(drop=True)
assert len(panel) > 0, "entity panel empty — pick another store×family"

# Unsafe: lag/rolling on panel that still holds true post-origin sales
unsafe = add_lag_features(
    panel, entity_cols=["store_nbr", "family"], date_col="date", target_col="sales"
)
unsafe = add_rolling_features(
    unsafe, entity_cols=["store_nbr", "family"], date_col="date", target_col="sales"
)

# Safe: null sales after origin before building target-derived features
safe_panel = mask_target_after(panel, ORIGIN, date_col="date", target_col="sales")
safe = add_lag_features(
    safe_panel, entity_cols=["store_nbr", "family"], date_col="date", target_col="sales"
)
safe = add_rolling_features(
    safe, entity_cols=["store_nbr", "family"], date_col="date", target_col="sales"
)

# Compare first post-origin day (T0+1): lag_1 must equal last train sales under mask
day1 = ORIGIN + pd.Timedelta(days=1)
u1 = unsafe.loc[unsafe["date"] == day1].iloc[0]
s1 = safe.loc[safe["date"] == day1].iloc[0]
last_train_sales = panel.loc[panel["date"] == ORIGIN, "sales"].iloc[0]
true_day1_sales = panel.loc[panel["date"] == day1, "sales"].iloc[0]

# Mid-horizon day (T0+2): unsafe lag_1 sees true day1 sales; safe lag_1 is NaN
# (no recursive fill in this audit — training path fills with predictions)
day2 = ORIGIN + pd.Timedelta(days=2)
u2 = unsafe.loc[unsafe["date"] == day2].iloc[0]
s2 = safe.loc[safe["date"] == day2].iloc[0]

pit_table = pd.DataFrame(
    [
        {
            "check": "day1_lag1_equals_last_train",
            "unsafe_sales_lag_1": u1["sales_lag_1"],
            "safe_sales_lag_1": s1["sales_lag_1"],
            "expected": last_train_sales,
            "pass_safe": bool(np.isclose(s1["sales_lag_1"], last_train_sales, equal_nan=False)),
        },
        {
            "check": "day2_lag1_not_true_midhorizon_under_mask",
            "unsafe_sales_lag_1": u2["sales_lag_1"],
            "safe_sales_lag_1": s2["sales_lag_1"],
            "true_day1_sales": true_day1_sales,
            "pass_safe": bool(pd.isna(s2["sales_lag_1"])),
            "unsafe_leaks_true_day1": bool(
                np.isclose(u2["sales_lag_1"], true_day1_sales, equal_nan=False)
            ),
        },
        {
            "check": "post_origin_sales_nulled",
            "safe_day1_sales": s1["sales"],
            "pass_safe": bool(pd.isna(s1["sales"])),
        },
    ]
)
display(pit_table)
assert pit_table.loc[0, "pass_safe"]
assert pit_table.loc[1, "pass_safe"]
assert pit_table.loc[1, "unsafe_leaks_true_day1"]
assert pit_table.loc[2, "pass_safe"]

# Full matrix build with registry under mask (core groups only)
core_groups = list(CORE_ABLATION_ORDER)
masked = mask_target_after(panel, ORIGIN)
feat = build_feature_matrix(masked, groups=core_groups)
feat_cols = [c for g in core_groups for c in FEATURE_GROUPS[g]]
missing = [c for c in feat_cols if c not in feat.columns]
print(f"entity=({ENTITY_STORE}, {ENTITY_FAMILY}) origin={ORIGIN.date()} n_rows={len(feat)}")
print(f"core feature columns present={len(feat_cols) - len(missing)}/{len(feat_cols)} missing={missing}")
display(
    feat.loc[feat["date"] >= ORIGIN, ["date", "sales", "sales_lag_1", "sales_lag_7", "sales_roll_mean_7"]]
    .head(5)
)

# %% [markdown]
# ### PIT audit — O/I/A
#
# **Observation:** Without `mask_target_after`, day-2 `sales_lag_1` equals true mid-horizon sales (leak). With mask, post-origin sales are NaN and day-2 lag is NaN until recursive prediction fill (training path).
#
# **Interpretation:** Multi-step recursive forecasting is **required** for lag/rolling under H=15; building features on unmasked train∪val is invalid even if single-step lag code is correct.
#
# **Action:** Task 12 objective must call the same mask (+ recursive val) path as `scripts/train.py`. Keep core groups; optional transactions must mask post-origin extras.

# %% [markdown]
# ## 6. Formal feature selection plan
#
# Purpose: codify the four-stage selection protocol and write/verify the YAML plan artifact.
# No new training — decisions from scoreboard + stability counts.

# %%
plan_path = ROOT / "configs" / "feature_selection" / "group_ablation_plan.yaml"
with plan_path.open(encoding="utf-8") as fh:
    plan = yaml.safe_load(fh)

stages = [
    "stage_1_validity_filter",
    "stage_2_group_ablation",
    "stage_3_within_group_prune",
    "stage_4_stability",
]
for s in stages:
    assert s in plan, f"missing stage {s}"

# Stability evidence from scoreboard folds
stab_rows = []
lgbm_runs = [r for r in rows if r["family"] == "lgbm"]
for i in range(1, len(lgbm_runs)):
    prev, cur = lgbm_runs[i - 1], lgbm_runs[i]
    helps = sum(
        1
        for c in ("fold_0", "fold_1", "fold_2")
        if cur[c] < prev[c]
    )
    added = cur["label"]
    stab_rows.append(
        {
            "step": f"{prev['label']} → {cur['label']}",
            "added": added,
            "mean_prev": prev["mean_rmsle"],
            "mean_cur": cur["mean_rmsle"],
            "delta_mean": cur["mean_rmsle"] - prev["mean_rmsle"],
            "folds_helped": helps,
            "keep_ge_2of3": helps >= 2,
        }
    )
stab_df = pd.DataFrame(stab_rows)
display(stab_df)
assert stab_df["keep_ge_2of3"].all(), "a core group failed ≥2/3 stability — re-open selection"

locked = plan["stage_2_group_ablation"]["decision_as_of_task_11"]["locked_for_hpo"]
print("locked_for_hpo:", locked)
print("within_group_prune:", plan["stage_3_within_group_prune"]["default_for_task_11_12"])
print("plan_path:", plan_path)

# %% [markdown]
# ### Selection plan — O/I/A
#
# **Observation:** Validity admits all notebook-01 groups. Progressive ablation keeps every core step (each helps 3/3 folds). Within-group prune is **skipped** (22 features; lag/rolling collinearity is intentional structure).
#
# **Interpretation:** Group-level ablation is the right default for this panel (skill: feature-selection-optimization + TTS FS). Individual lag pruning would need grouped permutation and is deferred unless HPO latency demands it.
#
# **Action:** Task 12 searches hyperparameters on locked groups only. Optional groups 015–018 remain post-HPO challengers, not search dimensions.

# %% [markdown]
# ## 7. Nested HPO protocol (design only — Task 12 executes)
#
# Purpose: define outer/inner split, objective, search space, and hard leakage rules.
# **No Optuna study is run in this notebook.**

# %%
tuning_path = ROOT / "configs" / "tuning" / "lightgbm.yaml"
with tuning_path.open(encoding="utf-8") as fh:
    tuning = yaml.safe_load(fh)

assert tuning["study_name"] == "lgbm_store_sales"
assert tuning["metric"] == "mean_rmsle"
assert tuning["direction"] == "minimize"
assert tuning["n_trials"] == 40
assert tuning["inner"]["strategy"] == "last_train_block"
assert tuning["inner"]["val_days"] == 15
assert tuning["feature_groups"] == locked

# Illustrate inner split on fold 0 train only (no model fit)
f0 = folds_meta[0]
outer_train_end = pd.Timestamp(f0["train_end"])
inner_val_days = int(tuning["inner"]["val_days"])
inner_val_end = outer_train_end
inner_val_start = outer_train_end - pd.Timedelta(days=inner_val_days - 1)
inner_train_end = inner_val_start - pd.Timedelta(days=1 + int(tuning["inner"].get("gap_days", 0)))

assert inner_train_end < inner_val_start <= inner_val_end <= outer_train_end
# Outer val must not intersect inner anything
outer_val_start = pd.Timestamp(f0["val_start"])
assert inner_val_end < outer_val_start

inner_demo = pd.DataFrame(
    [
        {
            "fold": 0,
            "outer_train_end": str(outer_train_end.date()),
            "inner_train_end": str(inner_train_end.date()),
            "inner_val_start": str(inner_val_start.date()),
            "inner_val_end": str(inner_val_end.date()),
            "outer_val_start": str(outer_val_start.date()),
            "outer_val_end": f0["val_end"],
        }
    ]
)
display(inner_demo)
display(pd.DataFrame([{"param": k, **v} for k, v in tuning["search_space"].items()]))

protocol_md = f"""
### Nested protocol (Task 12)

```text
For each Optuna trial params P:
  scores = []
  For each outer fold f in folds_meta:
    train_f = rows with date <= train_end_f   # NEVER outer val
    inner_train, inner_val = last_train_block(train_f, val_days={inner_val_days})
    mask_target_after(... origin=inner_train_end) before lag/rolling
    fit LightGBM(P) on inner_train; RMSLE on inner_val (recursive if lag/rolling)
    scores.append(rmsle)
  objective = mean(scores)
After study: retrain best P with scripts/train.py on full outer folds; persist run dir.
```

| Field | Value |
| --- | --- |
| study_name | `{tuning["study_name"]}` |
| n_trials | {tuning["n_trials"]} |
| seed | {tuning["seed"]} |
| feature_groups | {tuning["feature_groups"]} |
| search_space | learning_rate, num_leaves, min_child_samples, subsample, colsample_bytree |
| never_use | outer val, competition test, public LB |
"""
print(protocol_md)
print("tuning_config:", tuning_path)

# %% [markdown]
# ### HPO design — O/I/A
#
# **Observation:** Inner validation is the last 15 days of each outer train (matches competition horizon). Outer val stays sealed for post-HPO assessment via `train.py`. Search space is five capacity/regularization knobs; `n_estimators` uses early stopping.
#
# **Interpretation:** Nested design avoids selection bias from tuning on the same blocks used to claim score (TTS HPO + tabular-hpo-optimization). Feature groups are fixed so HPO does not re-open selection.
#
# **Action:** Implement `scripts/tune.py` in Task 12 against `configs/tuning/lightgbm.yaml`. Smoke with `--n-trials 2` before full 40-trial budget.

# %% [markdown]
# ## 8. Decision log & handoff
#
# | Decision | Choice | Evidence |
# | --- | --- | --- |
# | Primary metric | mean_rmsle | competition + walk-forward mean |
# | Naive floor | sn7 ≈ 0.5513 | `001_seasonal_naive_7` |
# | Best core model (pre-HPO) | LGBM + base/calendar/promo/lag/rolling | `014` mean_rmsle ≈ 0.4004 |
# | Feature selection unit | **group** ablation | stages 1–4 YAML |
# | Locked groups for HPO | base, calendar, promo, lag, rolling | 3/3 fold stability |
# | Within-group prune | skip for now | cardinality OK |
# | Multi-step FE | `mask_target_after` + recursive preds | PIT proof cell |
# | HPO | nested Optuna 40 trials, inner last-15d | `configs/tuning/lightgbm.yaml` |
# | Optional groups | post-HPO challengers | 015–018 configs |
#
# **Next:** Task 12 — `scripts/tune.py` + nested split unit tests; no notebook-local HPO.

# %%
handoff = {
    "notebook": "02_baseline_and_feature_design",
    "sn7_floor": float(SN7_FLOOR),
    "best_core_run": "014_lgbm_plus_rolling",
    "best_core_mean_rmsle": float(
        scoreboard.loc[scoreboard["run_id"] == "014_lgbm_plus_rolling", "mean_rmsle"].iloc[0]
    ),
    "locked_feature_groups": locked,
    "tuning_config": str(tuning_path.relative_to(ROOT)),
    "selection_plan": str(plan_path.relative_to(ROOT)),
    "cv_chart": str(chart_path.relative_to(ROOT)),
    "next_task": 12,
}
print(json.dumps(handoff, indent=2))
logger.info("notebook 02 complete handoff=%s", handoff)
