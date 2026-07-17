"""Expanding walk-forward folds and fold-manifest IO.

Splits on unique dates (panel-safe). Never shuffles. Invariant per fold:
``max(train_dates) < min(val_dates)`` with ``gap_days`` calendar buffer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_expanding_folds(
    train: pd.DataFrame,
    *,
    date_col: str,
    n_folds: int,
    val_days: int,
    gap_days: int,
    min_train_days: int,
) -> list[dict[str, Any]]:
    """Build expanding-window walk-forward folds ending at the latest dates.

    Validation windows walk backward from the last unique date; folds are then
    renumbered chronologically (fold 0 = earliest val window).

    Indices reference the *input* ``train`` frame (original index preserved).
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if val_days < 1:
        raise ValueError(f"val_days must be >= 1, got {val_days}")
    if gap_days < 0:
        raise ValueError(f"gap_days must be >= 0, got {gap_days}")
    if min_train_days < 1:
        raise ValueError(f"min_train_days must be >= 1, got {min_train_days}")
    if date_col not in train.columns:
        raise KeyError(f"date_col {date_col!r} not in train columns")

    # Keep original index so callers can .loc into the input frame.
    df = train.sort_values(date_col)
    dates = np.array(sorted(pd.to_datetime(df[date_col]).unique()))
    folds: list[dict[str, Any]] = []
    end_idx = len(dates) - 1

    for fold_id in range(n_folds):
        val_end_i = end_idx - fold_id * val_days
        val_start_i = val_end_i - val_days + 1
        train_end_i = val_start_i - gap_days - 1
        if train_end_i < 0 or val_start_i < 0 or val_end_i < 0:
            break
        if val_end_i >= len(dates):
            break

        train_start_date = pd.Timestamp(dates[0])
        train_end_date = pd.Timestamp(dates[train_end_i])
        val_start_date = pd.Timestamp(dates[val_start_i])
        val_end_date = pd.Timestamp(dates[val_end_i])

        # Span of training calendar days (inclusive).
        train_span_days = (train_end_date - train_start_date).days + 1
        if train_span_days < min_train_days:
            break

        date_series = pd.to_datetime(df[date_col])
        train_mask = date_series <= train_end_date
        val_mask = (date_series >= val_start_date) & (date_series <= val_end_date)

        train_idx = df.index[train_mask].to_numpy()
        val_idx = df.index[val_mask].to_numpy()
        if len(train_idx) == 0 or len(val_idx) == 0:
            break

        # Leakage invariant (always enforced at build time).
        train_dates = date_series.loc[train_idx]
        val_dates = date_series.loc[val_idx]
        if train_dates.max() >= val_dates.min():
            raise RuntimeError(
                f"Temporal leakage in fold construction: "
                f"train_max={train_dates.max()} val_min={val_dates.min()}"
            )
        gap_observed = (val_dates.min() - train_dates.max()).days
        if gap_observed < gap_days:
            raise RuntimeError(
                f"gap_days={gap_days} not respected: observed gap={gap_observed}"
            )

        folds.append(
            {
                "fold": fold_id,
                "train_end": train_end_date,
                "val_start": val_start_date,
                "val_end": val_end_date,
                "train_idx": train_idx,
                "val_idx": val_idx,
            }
        )

    # Chronological fold order: earliest validation window first.
    folds = list(reversed(folds))
    for i, fold in enumerate(folds):
        fold["fold"] = i

    if not folds:
        raise ValueError(
            "Could not build any folds; check panel length and cv config"
        )
    return folds


def split_last_block(
    train_df: pd.DataFrame,
    *,
    date_col: str = "date",
    val_days: int = 15,
    gap_days: int = 0,
) -> dict[str, Any]:
    """Split an outer-train frame into inner_train + last-block inner_val.

    Used by nested temporal HPO: evaluation windows are carved only from the
    outer training slice so outer val / LB / test never enter the trial loop.

    Parameters
    ----------
    train_df:
        Outer-fold training panel (already excludes outer validation).
    date_col:
        Timestamp column.
    val_days:
        Number of unique dates in the inner validation block (trailing).
    gap_days:
        Number of unique dates dropped between inner_train and inner_val
        (embargo). ``0`` places val immediately after train.

    Returns
    -------
    dict with keys:
        ``train_idx``, ``val_idx`` (index arrays into ``train_df``),
        ``train_end``, ``val_start``, ``val_end`` (Timestamps).

    Invariants
    ----------
    - ``set(val_idx) ⊆ train_df.index`` and ``set(train_idx) ⊆ train_df.index``
    - ``max(train dates) < min(val dates)``
    - gap of at least ``gap_days`` unique dates between partitions when gap>0
    """
    if val_days < 1:
        raise ValueError(f"val_days must be >= 1, got {val_days}")
    if gap_days < 0:
        raise ValueError(f"gap_days must be >= 0, got {gap_days}")
    if date_col not in train_df.columns:
        raise KeyError(f"date_col {date_col!r} not in train_df columns")
    if len(train_df) == 0:
        raise ValueError("train_df is empty; cannot build inner split")

    dates = np.array(sorted(pd.to_datetime(train_df[date_col]).unique()))
    need = val_days + gap_days + 1  # at least one train day
    if len(dates) < need:
        raise ValueError(
            f"insufficient unique dates for inner split: have {len(dates)}, "
            f"need at least {need} (val_days={val_days}, gap_days={gap_days})"
        )

    val_end_i = len(dates) - 1
    val_start_i = val_end_i - val_days + 1
    train_end_i = val_start_i - gap_days - 1
    if train_end_i < 0 or val_start_i < 0:
        raise ValueError(
            f"insufficient history for inner split: "
            f"val_days={val_days}, gap_days={gap_days}, n_dates={len(dates)}"
        )

    train_end = pd.Timestamp(dates[train_end_i])
    val_start = pd.Timestamp(dates[val_start_i])
    val_end = pd.Timestamp(dates[val_end_i])

    date_series = pd.to_datetime(train_df[date_col])
    train_mask = date_series <= train_end
    val_mask = (date_series >= val_start) & (date_series <= val_end)

    train_idx = train_df.index[train_mask].to_numpy()
    val_idx = train_df.index[val_mask].to_numpy()
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("inner split produced empty train or val partition")

    train_dates = date_series.loc[train_idx]
    val_dates = date_series.loc[val_idx]
    if train_dates.max() >= val_dates.min():
        raise RuntimeError(
            f"Temporal leakage in inner split: "
            f"train_max={train_dates.max()} val_min={val_dates.min()}"
        )
    if gap_days > 0:
        # Count unique dates strictly between partitions.
        between = dates[(dates > train_dates.max()) & (dates < val_dates.min())]
        if len(between) < gap_days:
            raise RuntimeError(
                f"gap_days={gap_days} not respected: observed gap dates={len(between)}"
            )

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
    }


def save_fold_manifests(folds: list[dict[str, Any]], splits_dir: Path) -> None:
    """Write per-fold index parquets and folds_meta.json under splits_dir."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    meta: list[dict[str, Any]] = []
    for fold in folds:
        i = int(fold["fold"])
        pd.DataFrame({"idx": fold["train_idx"]}).to_parquet(
            splits_dir / f"fold_{i}_train_idx.parquet", index=False
        )
        pd.DataFrame({"idx": fold["val_idx"]}).to_parquet(
            splits_dir / f"fold_{i}_val_idx.parquet", index=False
        )
        meta.append(
            {
                "fold": i,
                "train_end": str(pd.Timestamp(fold["train_end"]).date()),
                "val_start": str(pd.Timestamp(fold["val_start"]).date()),
                "val_end": str(pd.Timestamp(fold["val_end"]).date()),
            }
        )
    (splits_dir / "folds_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
