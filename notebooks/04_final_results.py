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
# # 04 — Final Results & Submission
#
# **Layer 3 end.** Locked report + submission path. Answers: what we ship, why it beats
# the naive floor, how robust it is, how to reproduce `submission.csv`, and what not to
# retune after lock.
#
# **Does not:** redefine splits, open new HPO, retrain for leaderboard peek, or change
# `configs/final.yaml` feature groups / params.
#
# **Skills read (project-local `.agent/skills/`):**
# - `better-jupyter-notebook` — Markdown before code; O/I/A; no fabricated numbers
# - `jupytext-notebook-workflows` — optional `ipynb,py:percent` pairing
# - `kaggle-notebook-competition` — submission sanity; `/kaggle/input` path notes
# - `tabular-time-series-evaluation` — RMSLE primary; FVA vs seasonal-naive; walk-forward
# - `tabular-ml-visual-diagnostics` — baseline comparison + submission diagnostics
# - `tabular-time-series-lifecycle` — freeze / report / no post-lock retune (steps 22–26)
# - `readme-best-practices` — repro metadata (env, seeds, git, commands)
# - `data-storytelling` — executive summary first; numbers → decision
# - Supporting: `ml-reproducibility-seed-control`, `model-fit-diagnostics`
#
# **No retune. Artifacts only** (plus optional documented `predict.py` call).

# %% [markdown]
# ## 0. Executive summary (locked)
#
# | Field | Locked value |
# | --- | --- |
# | Competition | [Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) |
# | Unit | `(date, store_nbr, family)` unit sales |
# | Horizon `H` | **15 days** |
# | Primary metric | walk-forward **mean RMSLE** (3 expanding folds) |
# | Guardrail | `mae_log1p` / bias on log1p |
# | Naive floor (SN7) | **≈ 0.5513** (`001_seasonal_naive_7`) |
# | Finalist | LightGBM `020_lgbm_hpo_best` |
# | Feature groups | `[base, calendar, promo, lag, rolling]` |
# | Target | `log1p` train; **clip negatives** at predict |
# | Multi-seed (42/43/44) | **0.3933 ± 0.0004** |
# | Evaluate (primary seed 42 OOF) | **mean_rmsle ≈ 0.3935** |
# | vs SN7 (Δ) | **≈ −0.158 RMSLE** (beats floor → GO) |
# | Blend | **not shipped** (ops + seed stability) |
# | Submission | `outputs/submissions/submission.csv` — **28512** rows `id,sales` |
# | Config freeze | `configs/final.yaml` — **no HPO after lock** |
#
# **Ship decision:** single multi-seed-stable LightGBM finalist; document XGB secondary /
# CB tertiary; optional OOF blend remains offline-only.

# %% [markdown]
# ## 1. Problem contract & methodology freeze
#
# Purpose: restate the frozen protocol so this notebook cannot silently re-open Layer 2.

# %% [markdown]
# | Contract item | Frozen decision |
# | --- | --- |
# | Split | Expanding walk-forward manifests in `data/splits/` (3 folds, H=15) |
# | Forbidden | Random K-fold; global pre-split selectors; post-lock HPO |
# | Features | Point-in-time lag/rolling; known-future calendar/promo only |
# | Inference | Recursive multi-step over test horizon with train history |
# | Selection metric | Fold-mean RMSLE; multi-seed mean+std for finalist |
# | Assessment | `scripts/evaluate.py` once on locked OOF + horizon/segment tables |
# | Submission | `scripts/predict.py --config configs/final.yaml` |
# | Layer boundary | Notebooks decide; `src/` implements; `scripts/` execute; `outputs/` evidence |

# %% [markdown]
# ## 2. Setup — seeds, paths, logging
#
# Purpose: lock PRNG seed for plot sampling only; resolve project root; load artifacts
# (no model training).

# %%
from __future__ import annotations

import json
import os
import random
import subprocess
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

from store_sales.config import ProjectPaths, load_yaml
from store_sales.io.logging import get_logger
from store_sales.io.submission import validate_submission
from store_sales.viz.eda_plots import save_figure
from store_sales.viz.model_plots import plot_multi_seed_bars, plot_stress_deltas

SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
random.seed(SEED)
np.random.seed(SEED)

paths = ProjectPaths()
INTERIM = paths.data_interim
RUNS = paths.outputs / "runs"
REPORTS = paths.outputs / "reports"
FINAL_EVAL = paths.outputs / "final_evaluation"
STRESS = paths.outputs / "stress" / "default"
SUBMISSIONS = paths.outputs / "submissions"
FINAL_REPORTS = REPORTS / "final_results"
FINAL_REPORTS.mkdir(parents=True, exist_ok=True)

logger = get_logger("notebook.04_final_results")
logger.info("ROOT=%s SEED=%s", ROOT, SEED)

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

print(f"seed={SEED}")
print(f"root={ROOT}")
print(f"final_eval={FINAL_EVAL}")
print(f"submissions={SUBMISSIONS}")
print(f"final_reports={FINAL_REPORTS}")

# %% [markdown]
# ## 3. Load locked config + final evaluation artifacts
#
# Purpose: prove Layer 3 artifacts exist and match `configs/final.yaml`.

# %%
final_yaml = paths.configs / "final.yaml"
assert final_yaml.exists(), "missing configs/final.yaml"
locked = load_yaml(final_yaml)

metrics_path = FINAL_EVAL / "metrics.json"
assert metrics_path.exists(), f"missing {metrics_path} — run scripts/evaluate.py"
final_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

env_path = FINAL_EVAL / "environment.json"
env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}

horizon = pd.read_csv(FINAL_EVAL / "horizon_metrics.csv")
segment = pd.read_csv(FINAL_EVAL / "segment_metrics.csv")
multi_seed = pd.read_csv(REPORTS / "multi_seed_summary.csv")

print("locked run_id=", locked.get("run_id"))
print("locked model=", (locked.get("model") or {}).get("name"))
print("locked groups=", locked.get("feature_groups"))
print("primary_seed=", locked.get("primary_seed"))
print("seeds=", locked.get("seeds"))
print("blend.enabled=", (locked.get("blend") or {}).get("enabled"))
print("--- final_evaluation/metrics.json ---")
print("mean_rmsle=", final_metrics.get("mean_rmsle"))
print("std_rmsle=", final_metrics.get("std_rmsle"))
print("multi_seed.mean_across_seeds=", (final_metrics.get("multi_seed") or {}).get("mean_across_seeds"))
print("multi_seed.std_across_seeds=", (final_metrics.get("multi_seed") or {}).get("std_across_seeds"))
print("naive_floor_sn7=", final_metrics.get("naive_floor_sn7_mean_rmsle"))
print("delta_vs_naive=", final_metrics.get("delta_vs_naive_rmsle"))
print("go_no_go=", final_metrics.get("go_no_go"))

assert locked.get("run_id") == final_metrics.get("run_id") == "020_lgbm_hpo_best"
assert locked.get("primary_seed") == 42
assert (locked.get("blend") or {}).get("enabled") is False
assert final_metrics.get("go_no_go") == "GO"

display(pd.Series(final_metrics.get("multi_seed") or {}).to_frame("value"))

# %% [markdown]
# ### Locked artifacts — O/I/A
#
# **Observation:** `configs/final.yaml` and `outputs/final_evaluation/metrics.json` both
# point at LightGBM `020_lgbm_hpo_best`, primary seed 42, blend off, GO vs SN7 floor.
#
# **Interpretation:** Layer 3 assessment is frozen; this notebook only reports and ships.
#
# **Action:** Do not edit finalist params/groups; any change requires a new experiment id.

# %% [markdown]
# ## 4. Baselines vs final multi-seed LGBM
#
# Purpose: Forecast Value Added table — last-value / SN7 / SN14 vs locked multi-seed mean.

# %%
BASELINE_IDS = {
    "000_last_value": "last_value",
    "001_seasonal_naive_7": "seasonal_naive_7",
    "002_seasonal_naive_14": "seasonal_naive_14",
}


def _load_run_metrics(run_id: str) -> dict:
    p = RUNS / run_id / "metrics.json"
    assert p.exists(), f"missing baseline metrics: {p}"
    return json.loads(p.read_text(encoding="utf-8"))


baseline_rows = []
for run_id, label in BASELINE_IDS.items():
    m = _load_run_metrics(run_id)
    baseline_rows.append(
        {
            "model": label,
            "run_id": run_id,
            "mean_rmsle": m["mean_rmsle"],
            "std_rmsle": m.get("std_rmsle"),
            "mean_mae_log1p": m.get("mean_mae_log1p"),
            "source": "walk-forward fold mean",
        }
    )

lgbm_ms = multi_seed.loc[multi_seed["base_run_id"] == "020_lgbm_hpo_best"].iloc[0]
final_row = {
    "model": "lgbm_hpo_best_multiseed",
    "run_id": "020_lgbm_hpo_best",
    "mean_rmsle": float(lgbm_ms["mean_across_seeds"]),
    "std_rmsle": float(lgbm_ms["std_across_seeds"]),
    "mean_mae_log1p": final_metrics.get("mean_mae_log1p"),
    "source": "multi-seed mean (42/43/44)",
}
compare = pd.DataFrame(baseline_rows + [final_row]).sort_values("mean_rmsle")
sn7 = float(compare.loc[compare["model"] == "seasonal_naive_7", "mean_rmsle"].iloc[0])
compare["delta_vs_sn7"] = compare["mean_rmsle"] - sn7
compare["beats_sn7"] = compare["mean_rmsle"] < sn7
display(compare)

# Primary-seed evaluate number for side-by-side
print(
    "evaluate primary-seed mean_rmsle=",
    final_metrics["mean_rmsle"],
    "| multi-seed mean=",
    final_row["mean_rmsle"],
    "±",
    final_row["std_rmsle"],
)

fig, ax = plt.subplots(figsize=(9, 4.2))
plot_df = compare.sort_values("mean_rmsle", ascending=False)
colors = ["#4C78A8" if m != "lgbm_hpo_best_multiseed" else "#F58518" for m in plot_df["model"]]
ax.barh(plot_df["model"], plot_df["mean_rmsle"], color=colors, alpha=0.9)
ax.axvline(sn7, color="gray", ls="--", lw=1.2, label=f"SN7 floor={sn7:.4f}")
ax.set_xlabel("mean RMSLE (lower better)")
ax.set_title("Baselines vs locked multi-seed LightGBM")
ax.legend(loc="lower right")
fig.tight_layout()
save_figure(fig, FINAL_REPORTS / "baselines_vs_final_lgbm.png")
plt.show()
plt.close(fig)

compare.to_csv(FINAL_REPORTS / "baselines_vs_final.csv", index=False)

# %% [markdown]
# ### Baseline comparison — O/I/A
#
# **Observation:** Last-value ≈ **0.638**; SN14 ≈ **0.558**; SN7 ≈ **0.5513**; multi-seed
# LGBM ≈ **0.3933 ± 0.0004**. Evaluate primary-seed fold-mean ≈ **0.3935**. Δ vs SN7 ≈
# **−0.158** RMSLE.
#
# **Interpretation:** Clear Forecast Value Added over all naive floors. Multi-seed band is
# tight (std ≈ 0.0004) — selection is not a single-seed fluke. Primary-seed evaluate
# number sits inside that band.
#
# **Action:** Ship LGBM; keep SN7 as the published floor in README / portfolio notes.

# %% [markdown]
# ## 5. Multi-seed challengers (context only)
#
# Purpose: show LGBM vs XGB vs CB under the same locked feature groups (artifact table).

# %%
display(multi_seed)

fig, ax = plt.subplots(figsize=(8, 4))
plot_multi_seed_bars(
    multi_seed,
    ax=ax,
    save_path=FINAL_REPORTS / "multi_seed_mean_rmsle.png",
)
plt.show()
plt.close(fig)

# %% [markdown]
# ### Challengers — O/I/A
#
# **Observation:** LGBM best mean and lowest seed std; XGB secondary (~0.399); CB tertiary
# (~0.412) under identical groups.
#
# **Interpretation:** Family ranking is stable; no need to ship an ensemble for portfolio
# simplicity given LGBM seed stability.
#
# **Action:** Document secondary/tertiary only; primary submission is single LGBM.

# %% [markdown]
# ## 6. Stress battery summary
#
# Purpose: load frozen stress summary (no re-stress in this notebook).

# %%
stress_path = STRESS / "summary.json"
assert stress_path.exists(), f"missing {stress_path}"
stress = json.loads(stress_path.read_text(encoding="utf-8"))
scen = pd.DataFrame(
    [
        {
            "scenario": s["scenario"],
            "clean_mean_rmsle": s["clean_mean_rmsle"],
            "stressed_mean_rmsle": s["stressed_mean_rmsle"],
            "delta_rmsle": s["delta_rmsle"],
            "relative_delta": s["relative_delta"],
            "status": s["status"],
        }
        for s in stress["scenarios"]
    ]
)
display(scen)
print("clean_repredict_mean_rmsle=", stress.get("clean_repredict_mean_rmsle"))
print("method_note=", stress.get("method_note"))

fig, ax = plt.subplots(figsize=(8, 4))
plot_stress_deltas(scen, ax=ax, save_path=FINAL_REPORTS / "stress_deltas.png")
plt.show()
plt.close(fig)
scen.to_csv(FINAL_REPORTS / "stress_scenarios.csv", index=False)

# %% [markdown]
# ### Stress — O/I/A
#
# **Observation:** 5% rel. noise ΔRMSLE ≈ **+0.0005**; 10% ≈ **+0.0023**; lag spike clip ≈
# **+0.0009** (graceful). Oil/holiday join failure **N/A** (groups excluded). Payday
# subset RMSLE ≈ **0.409** vs clean ≈ **0.394**.
#
# **Interpretation:** Finalist tolerates moderate feature noise and lag outliers. Payday
# remains a known weak slice for monitoring, not a ship blocker.
#
# **Action:** Record stress path in final.yaml (already); no retrain for noise.

# %% [markdown]
# ## 7. Horizon & segment guardrails (evaluate artifacts)
#
# Purpose: show multi-step degradation and worst segments from Layer 3 evaluate — not for
# retuning.

# %%
display(horizon)
display(segment.head(15))

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(horizon["horizon"], horizon["rmsle"], marker="o", color="#4C78A8")
ax.set_xlabel("horizon day (1..15)")
ax.set_ylabel("OOF RMSLE")
ax.set_title("Locked finalist — RMSLE by horizon (evaluate artifact)")
ax.set_xticks(list(horizon["horizon"]))
fig.tight_layout()
save_figure(fig, FINAL_REPORTS / "horizon_rmsle.png")
plt.show()
plt.close(fig)

fam = segment.loc[segment["segment_type"] == "family"].sort_values("rmsle", ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
top = fam.head(12)
ax.barh(top["segment"].astype(str), top["rmsle"], color="#E45756", alpha=0.9)
ax.invert_yaxis()
ax.set_xlabel("RMSLE")
ax.set_title("Worst families by OOF RMSLE (evaluate artifact)")
fig.tight_layout()
save_figure(fig, FINAL_REPORTS / "worst_families_rmsle.png")
plt.show()
plt.close(fig)

print("horizon min/max rmsle=", float(horizon["rmsle"].min()), float(horizon["rmsle"].max()))
print("worst family=", fam.iloc[0]["segment"], float(fam.iloc[0]["rmsle"]))

# %% [markdown]
# ### Guardrails — O/I/A
#
# **Observation:** RMSLE rises with horizon (multi-step compounding). Error concentrates in
# intermittent / sparse families (exact ranks from evaluate `segment_metrics.csv`).
#
# **Interpretation:** Residual structure is segment/horizon, not a broken fold mean. Fits
# recursive GBDT design limits without implying leakage.
#
# **Action:** Monitor family + payday RMSLE if productionized; do not retune on these slices.

# %% [markdown]
# ## 8. Submission generation (`predict.py`)
#
# Purpose: document the production command; **load** the existing submission if present
# (already generated in Task 16). Optionally re-run predict when
# `FORCE_REGENERATE_SUBMISSION=1`.
#
# ```bash
# # From repo root (recommended ship path — load fold boosters, mean ensemble):
# uv run python scripts/predict.py --config configs/final.yaml
#
# # Optional full-train refit (not required for this report):
# uv run python scripts/predict.py --config configs/final.yaml --mode retrain
# ```
#
# Default write path: **`outputs/submissions/submission.csv`**.

# %%
SUB_PATH = SUBMISSIONS / "submission.csv"
FORCE = os.environ.get("FORCE_REGENERATE_SUBMISSION", "0") == "1"

if FORCE or not SUB_PATH.exists():
    logger.info("Running predict.py (force=%s exists=%s)", FORCE, SUB_PATH.exists())
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "predict.py"),
        "--config",
        str(ROOT / "configs" / "final.yaml"),
        "--mode",
        "load",
        "--output",
        str(SUB_PATH),
    ]
    # Prefer uv when available for project env consistency
    try:
        subprocess.run(
            ["uv", "run", "python", "scripts/predict.py", "--config", "configs/final.yaml"],
            cwd=str(ROOT),
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        subprocess.run(cmd, cwd=str(ROOT), check=True)
else:
    print(f"Using existing submission artifact: {SUB_PATH}")
    print("Set FORCE_REGENERATE_SUBMISSION=1 to re-call scripts/predict.py")

assert SUB_PATH.exists(), f"missing submission: {SUB_PATH}"
sub = pd.read_csv(SUB_PATH)
sample = pd.read_parquet(INTERIM / "sample_submission.parquet")
validate_submission(sub, sample)

print("submission_path=", SUB_PATH.resolve())
print("rows=", len(sub), "cols=", list(sub.columns))
print("nan=", int(sub.isna().sum().sum()), "negatives=", int((sub["sales"] < 0).sum()))
print(sub["sales"].describe())
display(sub.head(8))
display(sub.tail(4))

# %% [markdown]
# ### Submission command — O/I/A
#
# **Observation:** Artifact at `outputs/submissions/submission.csv` has **28512** rows,
# columns `id,sales`, ids aligned to sample, **0 NaN / 0 negatives** after clip policy.
#
# **Interpretation:** Schema and non-negativity contract match Kaggle sample_submission.
#
# **Action:** Upload this file as-is; do not post-process with LB feedback.

# %% [markdown]
# ## 9. Submission sanity plots
#
# Purpose: distribution / level checks only — no labels on test; no metric fishing.

# %%
# Join store/family/date for panel diagnostics
test = pd.read_parquet(INTERIM / "test.parquet")
test["date"] = pd.to_datetime(test["date"])
sub_panel = test[["id", "date", "store_nbr", "family"]].merge(sub, on="id", how="inner")
assert len(sub_panel) == len(sub)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 1) prediction histogram (log1p)
axes[0].hist(np.log1p(sub["sales"].to_numpy()), bins=60, color="steelblue", alpha=0.9)
axes[0].set_xlabel("log1p(sales_pred)")
axes[0].set_ylabel("count")
axes[0].set_title("Submission prediction density")

# 2) daily aggregate forecast
daily = sub_panel.groupby("date", as_index=False)["sales"].sum()
axes[1].plot(daily["date"], daily["sales"], marker="o", color="#F58518")
axes[1].set_title("Aggregate daily predicted sales")
axes[1].set_xlabel("date")
axes[1].tick_params(axis="x", rotation=30)

# 3) zero-rate by family (top intermittent)
zero_by_fam = (
    sub_panel.assign(is_zero=lambda d: d["sales"] <= 1e-12)
    .groupby("family")["is_zero"]
    .mean()
    .sort_values(ascending=False)
    .head(12)
)
axes[2].barh(zero_by_fam.index.astype(str), zero_by_fam.values, color="#54A24B", alpha=0.9)
axes[2].invert_yaxis()
axes[2].set_xlabel("fraction pred ≈ 0")
axes[2].set_title("Highest near-zero prediction rate by family")

fig.tight_layout()
save_figure(fig, FINAL_REPORTS / "submission_sanity.png")
plt.show()
plt.close(fig)

print("test date range:", sub_panel["date"].min().date(), "→", sub_panel["date"].max().date())
print("n_days=", sub_panel["date"].nunique(), "n_entities=", sub_panel.groupby(["store_nbr", "family"]).ngroups)
print("pred mean/median/max=", float(sub["sales"].mean()), float(sub["sales"].median()), float(sub["sales"].max()))
print("frac_zero_pred=", float((sub["sales"] <= 1e-12).mean()))

# %% [markdown]
# ### Sanity plots — O/I/A
#
# **Observation:** Predictions are non-negative with a long right tail (typical retail
# panel); daily totals show weekly-ish structure; some families have high near-zero rates
# (intermittency).
#
# **Interpretation:** No schema collapse, no negative mass, no single constant fill. Shape
# is consistent with train intermittency + clip policy.
#
# **Action:** Proceed to Kaggle upload when desired; public LB is optional and must not
# trigger retune (Task 18).

# %% [markdown]
# ## 10. Limitations
#
# | Limitation | Detail | Mitigation / note |
# | --- | --- | --- |
# | Recursive multi-step | Error compounds over H=15 | Horizon table; lags ≥ weekly structure |
# | Intermittent families | High zero mass / sparse demand | Segment metrics; RMSLE not MAPE |
# | Payday / month-end | Higher RMSLE on payday subset | Stress scenario; monitor only |
# | Oil / holiday excluded | Join fragility removed with groups | N/A in stress; not in locked set |
# | Single model ship | Blend slightly better OOF (~0.0008) | Rejected for seed std + ops |
# | No conformal intervals | Point forecast only | Out of scope for this portfolio |
# | Earthquake regime | All 2017 val folds post-2016-04 shock | Cannot contrast pre/post in OOF |
# | Public LB | Not used for selection | Trust walk-forward CV |

# %% [markdown]
# ## 11. Reproducibility metadata
#
# Purpose: environment packages, seeds, git revision, exact commands for portfolio audit.

# %%
def _git(cmd: list[str]) -> str:
    try:
        out = subprocess.run(
            ["git", *cmd],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"<unavailable: {exc}>"


git_head = _git(["rev-parse", "HEAD"])
git_short = _git(["rev-parse", "--short", "HEAD"])
git_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
git_status = _git(["status", "--porcelain"])
dirty = bool(git_status.strip())

repro = {
    "project_root": str(ROOT),
    "git_head": git_head,
    "git_short": git_short,
    "git_branch": git_branch,
    "git_dirty": dirty,
    "seed_primary": int(locked.get("primary_seed", 42)),
    "seeds_multi": list(locked.get("seeds") or [42, 43, 44]),
    "run_id": locked.get("run_id"),
    "config": "configs/final.yaml",
    "evaluate_mean_rmsle": final_metrics.get("mean_rmsle"),
    "multi_seed_mean_rmsle": (final_metrics.get("multi_seed") or {}).get("mean_across_seeds"),
    "multi_seed_std_rmsle": (final_metrics.get("multi_seed") or {}).get("std_across_seeds"),
    "naive_floor_sn7": final_metrics.get("naive_floor_sn7_mean_rmsle"),
    "submission_path": str(SUB_PATH.relative_to(ROOT)),
    "submission_rows": int(len(sub)),
    "environment": env,
    "commands": {
        "prepare": "uv run python scripts/prepare_data.py",
        "train_finalist": "uv run python scripts/train.py --config configs/experiments/020_lgbm_hpo_best.yaml",
        "evaluate": "uv run python scripts/evaluate.py --config configs/final.yaml",
        "predict": "uv run python scripts/predict.py --config configs/final.yaml",
        "stress": "uv run python scripts/stress_test.py --config configs/stress/default.yaml",
    },
}
(FINAL_REPORTS / "repro_metadata.json").write_text(
    json.dumps(repro, indent=2, default=str) + "\n",
    encoding="utf-8",
)
print(json.dumps({k: v for k, v in repro.items() if k != "environment"}, indent=2, default=str))
print("--- environment packages ---")
print(json.dumps(env.get("packages") or {}, indent=2))
print("python_version=", env.get("python_version"))
print("platform=", env.get("platform"))

# %% [markdown]
# ### Repro — O/I/A
#
# **Observation:** Seeds `{42,43,44}` with primary 42; evaluate + multi-seed numbers and
# package versions captured from `outputs/final_evaluation/environment.json`; git HEAD
# recorded (workspace may include untracked docs/agent files).
#
# **Interpretation:** Another machine can rebuild interim data, rescore OOF, and regenerate
# submission from locked YAML without guessing hyperparameters.
#
# **Action:** Prefer lockfile (`uv.lock`) + commands above for portfolio reproduction.

# %% [markdown]
# ## 12. Kaggle platform path notes
#
# This repository is a **local hybrid portfolio** (`src/` + scripts + notebooks). It is
# **not** a self-contained Kaggle upload kernel.
#
# | Context | Paths |
# | --- | --- |
# | Local (this project) | `data/raw/`, `data/interim/`, `outputs/submissions/submission.csv` |
# | Kaggle competition data | `/kaggle/input/store-sales-time-series-forecasting/` |
# | Kaggle notebook working | `/kaggle/working/submission.csv` |
#
# If porting inference to a Kaggle notebook:
#
# 1. Package fold boosters + code as a **Kaggle Dataset** (or inline helpers).
# 2. Point inputs at `/kaggle/input/...` only — **no** `sys.path` to a local laptop repo.
# 3. Write `submission.csv` under `/kaggle/working`.
# 4. Disable internet for offline inference; use "Save & Run All".
# 5. Assert `len(submission) == len(sample_submission)` and no NaNs before commit.
#
# Local ship path remains:
# `uv run python scripts/predict.py --config configs/final.yaml` →
# **`outputs/submissions/submission.csv`**.

# %% [markdown]
# ## 13. Session summary (Layer 3 lock)
#
# | Item | Value |
# | --- | --- |
# | Finalist | LightGBM `020_lgbm_hpo_best` |
# | Multi-seed RMSLE | **0.3933 ± 0.0004** |
# | Evaluate mean RMSLE | **≈ 0.3935** |
# | SN7 floor | **≈ 0.5513** |
# | Δ vs SN7 | **≈ −0.158** (GO) |
# | Stress | Graceful under noise/clip; payday weak slice noted |
# | Submission | **`outputs/submissions/submission.csv`** (28512 × `id,sales`) |
# | Predict command | `uv run python scripts/predict.py --config configs/final.yaml` |
# | Retune after lock? | **No** |
#
# Next (Task 18): README polish with these locked numbers; optional one-shot public LB
# record **without** parameter changes.
