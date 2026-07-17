"""Model-analysis plot helpers for notebook 03.

Thin wrappers around matplotlib/seaborn for residual packs, horizon curves,
segment bars, multi-seed tables, and light group-importance charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from store_sales.viz.eda_plots import save_figure


def plot_residual_hist(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    ax: Axes | None = None,
    title: str = "Residuals (log1p)",
    bins: int = 60,
    save_path: Path | str | None = None,
) -> Axes:
    """Histogram of log1p residual = log1p(pred) - log1p(true)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    resid = np.log1p(np.clip(yp, 0, None)) - np.log1p(np.clip(yt, 0, None))
    ax.hist(resid[np.isfinite(resid)], bins=bins, color="steelblue", alpha=0.85)
    ax.axvline(0.0, color="black", ls="--", lw=0.9)
    ax.set_xlabel("log1p(y_pred) - log1p(y_true)")
    ax.set_ylabel("count")
    ax.set_title(title)
    if save_path is not None:
        save_figure(ax.figure, save_path)
    return ax


def plot_pred_vs_actual(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    ax: Axes | None = None,
    title: str = "Predicted vs actual (log1p)",
    max_points: int = 8000,
    seed: int = 42,
    save_path: Path | str | None = None,
) -> Axes:
    """Scatter of log1p actual vs log1p pred (subsampled for readability)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    yt = np.log1p(np.clip(np.asarray(y_true, dtype=float), 0, None))
    yp = np.log1p(np.clip(np.asarray(y_pred, dtype=float), 0, None))
    n = len(yt)
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        yt, yp = yt[idx], yp[idx]
    ax.scatter(yt, yp, s=4, alpha=0.25, color="teal", edgecolors="none")
    lo = float(np.nanmin([yt.min(), yp.min()]))
    hi = float(np.nanmax([yt.max(), yp.max()]))
    ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1.0)
    ax.set_xlabel("log1p(y_true)")
    ax.set_ylabel("log1p(y_pred)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if save_path is not None:
        save_figure(ax.figure, save_path)
    return ax


def plot_horizon_rmsle(
    horizon_df: pd.DataFrame,
    *,
    horizon_col: str = "horizon",
    metric_col: str = "rmsle",
    ax: Axes | None = None,
    title: str = "RMSLE by forecast horizon",
    save_path: Path | str | None = None,
) -> Axes:
    """Line chart of RMSLE vs horizon step (1..H)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    d = horizon_df.sort_values(horizon_col)
    ax.plot(d[horizon_col], d[metric_col], marker="o", color="darkorange")
    ax.set_xlabel("horizon day (1 = first forecast day)")
    ax.set_ylabel(metric_col)
    ax.set_title(title)
    ax.set_xticks(list(d[horizon_col]))
    if save_path is not None:
        save_figure(ax.figure, save_path)
    return ax


def plot_segment_metric_bars(
    segment_df: pd.DataFrame,
    *,
    segment_col: str,
    metric_col: str = "rmsle",
    ax: Axes | None = None,
    title: str | None = None,
    top_n: int | None = 20,
    ascending: bool = False,
    save_path: Path | str | None = None,
) -> Axes:
    """Horizontal bars of a metric by segment (worst first by default)."""
    d = segment_df[[segment_col, metric_col]].dropna().copy()
    d = d.sort_values(metric_col, ascending=ascending)
    if top_n is not None:
        d = d.tail(top_n) if ascending else d.head(top_n)
        d = d.sort_values(metric_col, ascending=True)
    if ax is None:
        height = max(3.5, 0.28 * len(d))
        _, ax = plt.subplots(figsize=(8, height))
    ax.barh(d[segment_col].astype(str), d[metric_col], color="coral")
    ax.set_xlabel(metric_col)
    ax.set_ylabel(segment_col)
    ax.set_title(title or f"{metric_col} by {segment_col}")
    if save_path is not None:
        save_figure(ax.figure, save_path)
    return ax


def plot_multi_seed_bars(
    summary_df: pd.DataFrame,
    *,
    model_col: str = "base_run_id",
    mean_col: str = "mean_across_seeds",
    std_col: str = "std_across_seeds",
    ax: Axes | None = None,
    title: str = "Multi-seed mean RMSLE ± std",
    save_path: Path | str | None = None,
) -> Axes:
    """Bar chart of mean RMSLE across seeds with error bars."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    d = summary_df.sort_values(mean_col)
    labels = d[model_col].astype(str).tolist()
    means = d[mean_col].to_numpy(dtype=float)
    stds = (
        d[std_col].to_numpy(dtype=float)
        if std_col in d.columns
        else np.zeros(len(d))
    )
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("mean RMSLE")
    ax.set_title(title)
    if save_path is not None:
        save_figure(ax.figure, save_path)
    return ax


def plot_group_importance(
    importance: Mapping[str, float] | pd.Series,
    *,
    ax: Axes | None = None,
    title: str = "Feature-group importance (gain sum)",
    save_path: Path | str | None = None,
) -> Axes:
    """Horizontal bars for grouped gain / permutation importance."""
    if isinstance(importance, pd.Series):
        s = importance.sort_values(ascending=True)
    else:
        s = pd.Series(dict(importance)).sort_values(ascending=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3.0, 0.4 * len(s))))
    s.plot(kind="barh", ax=ax, color="seagreen")
    ax.set_xlabel("importance")
    ax.set_ylabel("feature group")
    ax.set_title(title)
    if save_path is not None:
        save_figure(ax.figure, save_path)
    return ax


def plot_fold_rmsle_stability(
    fold_metrics: Sequence[float] | pd.Series,
    *,
    ax: Axes | None = None,
    title: str = "Fold RMSLE stability",
    save_path: Path | str | None = None,
) -> Axes:
    """Bar chart of per-fold RMSLE with mean line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    vals = np.asarray(fold_metrics, dtype=float)
    folds = np.arange(len(vals))
    ax.bar(folds, vals, color="slateblue", alpha=0.85)
    ax.axhline(float(np.mean(vals)), color="black", ls="--", lw=1.0, label="mean")
    ax.set_xlabel("fold")
    ax.set_ylabel("RMSLE")
    ax.set_title(title)
    ax.set_xticks(folds)
    ax.legend(loc="best")
    if save_path is not None:
        save_figure(ax.figure, save_path)
    return ax


def plot_stress_deltas(
    scenarios_df: pd.DataFrame,
    *,
    scenario_col: str = "scenario",
    delta_col: str = "delta_rmsle",
    ax: Axes | None = None,
    title: str = "Stress ΔRMSLE vs clean",
    save_path: Path | str | None = None,
) -> Axes:
    """Bar chart of RMSLE degradation under stress scenarios."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    d = scenarios_df.dropna(subset=[delta_col]).copy()
    colors = ["salmon" if v > 0 else "seagreen" for v in d[delta_col]]
    ax.barh(d[scenario_col].astype(str), d[delta_col], color=colors)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("ΔRMSLE (stressed − clean)")
    ax.set_title(title)
    if save_path is not None:
        save_figure(ax.figure, save_path)
    return ax


__all__ = [
    "save_figure",
    "plot_residual_hist",
    "plot_pred_vs_actual",
    "plot_horizon_rmsle",
    "plot_segment_metric_bars",
    "plot_multi_seed_bars",
    "plot_group_importance",
    "plot_fold_rmsle_stability",
    "plot_stress_deltas",
]
