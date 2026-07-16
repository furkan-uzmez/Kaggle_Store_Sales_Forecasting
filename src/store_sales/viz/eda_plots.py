"""Reusable EDA plot helpers for Store Sales notebooks.

Functions return a matplotlib ``Axes`` (or figure) and optionally save under
``outputs/reports/eda/``. Notebooks stay thin: call these helpers, then write
Observation / Interpretation / Action notes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def save_figure(fig: Figure, path: Path | str, *, dpi: int = 120) -> Path:
    """Save figure to disk and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return out


def plot_target_distribution(
    sales: pd.Series | np.ndarray,
    *,
    ax: Axes | None = None,
    title: str = "Sales distribution (log1p)",
    save_path: Path | str | None = None,
) -> Axes:
    """Histogram + KDE of log1p(sales) with zero-mass annotation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    s = pd.Series(np.asarray(sales, dtype=float)).dropna()
    zero_rate = float((s <= 0).mean()) if len(s) else float("nan")
    positive = s[s > 0]
    if len(positive):
        # Hist only by default — KDE on multi-100k samples is RAM-heavy in notebooks
        ax.hist(np.log1p(positive), bins=60, color="steelblue", alpha=0.85, edgecolor="none")
    ax.set_xlabel("log1p(sales) for sales > 0")
    ax.set_ylabel("count")
    ax.set_title(f"{title} | zero-rate={zero_rate:.3%}")
    if save_path is not None:
        save_figure(ax.figure, save_path)
    return ax


def plot_zero_mass_by_family(
    zero_rate: pd.Series,
    *,
    ax: Axes | None = None,
    title: str = "Zero-sales rate by family",
    save_path: Path | str | None = None,
    top_n: int | None = None,
) -> Axes:
    """Horizontal bar chart of zero-sales rate by family."""
    if ax is None:
        height = max(4, 0.28 * len(zero_rate))
        _, ax = plt.subplots(figsize=(8, height))
    z = zero_rate.sort_values(ascending=True)
    if top_n is not None:
        z = z.tail(top_n)
    z.plot(kind="barh", ax=ax, color="coral")
    ax.set_xlabel("P(sales == 0)")
    ax.set_ylabel("family")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    if save_path is not None:
        save_figure(ax.figure, save_path)
    return ax


def plot_aggregate_sales_over_time(
    daily: pd.DataFrame,
    *,
    date_col: str = "date",
    value_col: str = "sales",
    event_dates: Sequence[pd.Timestamp | str] | None = None,
    event_labels: Sequence[str] | None = None,
    ax: Axes | None = None,
    title: str = "Aggregate daily sales",
    save_path: Path | str | None = None,
) -> Axes:
    """Line plot of national daily sales with optional event markers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    d = daily.sort_values(date_col)
    ax.plot(d[date_col], d[value_col], lw=0.9, color="steelblue")
    if event_dates:
        labels = list(event_labels) if event_labels is not None else [""] * len(event_dates)
        for dt, lab in zip(event_dates, labels, strict=False):
            ts = pd.Timestamp(dt)
            ax.axvline(ts, color="crimson", ls="--", lw=1.0, alpha=0.8)
            if lab:
                ax.text(
                    ts,
                    ax.get_ylim()[1] * 0.95,
                    lab,
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=8,
                    color="crimson",
                )
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        save_figure(ax.figure, save_path)
    return ax


def plot_store_family_panels(
    panel: pd.DataFrame,
    *,
    date_col: str = "date",
    value_col: str = "sales",
    entity_cols: Sequence[str] = ("store_nbr", "family"),
    n_cols: int = 2,
    title: str = "Sample store × family series",
    save_path: Path | str | None = None,
) -> Figure:
    """Small-multiples time series for selected entities."""
    keys = list(entity_cols)
    groups = list(panel.groupby(keys, sort=False))
    n = len(groups)
    n_cols = max(1, n_cols)
    n_rows = int(np.ceil(n / n_cols)) if n else 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 2.6 * n_rows),
        squeeze=False,
        sharex=False,
    )
    for i, (key, g) in enumerate(groups):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        g = g.sort_values(date_col)
        ax.plot(g[date_col], g[value_col], lw=0.8)
        ax.set_title(str(key), fontsize=9)
        ax.grid(True, alpha=0.25)
    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].axis("off")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path)
    return fig


def plot_dow_seasonality(
    frame: pd.DataFrame,
    *,
    sales_col: str = "sales",
    dow_col: str = "dow",
    ax: Axes | None = None,
    title: str = "Sales by day-of-week",
    save_path: Path | str | None = None,
) -> Axes:
    """Box/violin of sales by day-of-week (expects precomputed dow 0=Mon..6=Sun)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    order = list(range(7))
    plot_df = frame[[dow_col, sales_col]].dropna()
    # Cap extreme tails for readability; raw data unchanged
    q99 = plot_df[sales_col].quantile(0.99) if len(plot_df) else 0
    plot_df = plot_df[plot_df[sales_col] <= q99]
    sns.boxplot(
        data=plot_df,
        x=dow_col,
        y=sales_col,
        order=order,
        ax=ax,
        showfliers=False,
        color="lightsteelblue",
    )
    ax.set_xlabel("day of week (0=Mon … 6=Sun)")
    ax.set_ylabel(sales_col)
    ax.set_title(title + f" (values ≤ p99={q99:.1f} for display)")
    if save_path is not None:
        save_figure(ax.figure, save_path)
    return ax


def plot_oil_missingness(
    oil: pd.DataFrame,
    *,
    date_col: str = "date",
    value_col: str = "dcoilwtico",
    ax: Axes | None = None,
    title: str = "Oil price + missing gaps",
    save_path: Path | str | None = None,
) -> Axes:
    """Oil series with missing points highlighted."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3.5))
    o = oil.sort_values(date_col).copy()
    ax.plot(o[date_col], o[value_col], lw=0.9, color="darkgreen", label="dcoilwtico")
    missing = o[o[value_col].isna()]
    if len(missing):
        ymin = float(o[value_col].min(skipna=True)) if o[value_col].notna().any() else 0.0
        ax.scatter(
            missing[date_col],
            np.full(len(missing), ymin),
            marker="|",
            s=80,
            color="red",
            label=f"missing (n={len(missing)})",
        )
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        save_figure(ax.figure, save_path)
    return ax


def plot_fold_date_ranges(
    folds: Iterable[dict[str, Any]],
    *,
    ax: Axes | None = None,
    title: str = "Walk-forward fold date ranges",
    save_path: Path | str | None = None,
) -> Axes:
    """Horizontal bars for train_end / val_start / val_end per fold."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5))
    rows = list(folds)
    for i, f in enumerate(rows):
        fold = int(f.get("fold", i))
        train_end = pd.Timestamp(f["train_end"])
        val_start = pd.Timestamp(f["val_start"])
        val_end = pd.Timestamp(f["val_end"])
        # Train bar: approximate from train_end - 365d for visual only if train_start absent
        train_start = pd.Timestamp(f["train_start"]) if "train_start" in f else train_end - pd.Timedelta(days=365)
        ax.barh(
            fold,
            (train_end - train_start).days,
            left=train_start.toordinal(),
            height=0.35,
            color="steelblue",
            label="train" if i == 0 else None,
        )
        ax.barh(
            fold,
            (val_end - val_start).days + 1,
            left=val_start.toordinal(),
            height=0.35,
            color="orange",
            label="val" if i == 0 else None,
        )
    ax.set_yticks([int(f.get("fold", i)) for i, f in enumerate(rows)])
    ax.set_yticklabels([f"fold {int(f.get('fold', i))}" for i, f in enumerate(rows)])
    ax.set_title(title)
    ax.legend(loc="upper left")
    # Readable x ticks from ordinals
    xticks = ax.get_xticks()
    labels = []
    for x in xticks:
        try:
            labels.append(pd.Timestamp.fromordinal(int(x)).strftime("%Y-%m"))
        except (ValueError, OverflowError):
            labels.append("")
    ax.set_xticklabels(labels)
    ax.set_xlabel("calendar time")
    if save_path is not None:
        save_figure(ax.figure, save_path)
    return ax


def plot_intermittency_by_family(
    rates: pd.Series,
    *,
    ax: Axes | None = None,
    title: str = "Intermittency (zero rate) by family",
    save_path: Path | str | None = None,
) -> Axes:
    """Alias-style bar for intermittency rates (sorted)."""
    return plot_zero_mass_by_family(
        rates,
        ax=ax,
        title=title,
        save_path=save_path,
    )
