"""Submission schema validation and CSV writers for Kaggle Store Sales."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("id", "sales")


def validate_submission(
    sub: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    """Assert ``sub`` matches ``sample_submission`` schema and id coverage.

    Checks:
    - columns include ``id`` and ``sales``
    - row count equals sample
    - id set equals sample (order may differ)
    - all ``sales`` values are finite

    Raises
    ------
    ValueError
        On any schema / coverage / finite-value violation.
    """
    if not isinstance(sub, pd.DataFrame) or not isinstance(sample, pd.DataFrame):
        raise ValueError("submission and sample must be DataFrames")

    missing = [c for c in REQUIRED_COLUMNS if c not in sub.columns]
    if missing:
        raise ValueError(
            f"submission missing required columns {missing}; "
            f"got {list(sub.columns)}"
        )

    if "id" not in sample.columns:
        raise ValueError("sample_submission must contain column 'id'")

    n_sub, n_sample = len(sub), len(sample)
    if n_sub != n_sample:
        raise ValueError(
            f"submission row count {n_sub} != sample row count {n_sample}"
        )

    sub_ids = set(pd.Series(sub["id"]).tolist())
    sample_ids = set(pd.Series(sample["id"]).tolist())
    if sub_ids != sample_ids:
        only_sub = sorted(sub_ids - sample_ids)[:5]
        only_sample = sorted(sample_ids - sub_ids)[:5]
        raise ValueError(
            "submission id set does not match sample_submission "
            f"(extra examples={only_sub}, missing examples={only_sample})"
        )

    sales = pd.to_numeric(sub["sales"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(sales).all():
        n_bad = int((~np.isfinite(sales)).sum())
        raise ValueError(f"submission sales must be finite; non-finite count={n_bad}")


def align_submission_to_sample(
    preds: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    id_col: str = "id",
    sales_col: str = "sales",
) -> pd.DataFrame:
    """Return ``id,sales`` frame in sample row order.

    ``preds`` must contain ``id_col`` and a sales prediction column (default
    ``sales``). Extra columns are dropped.
    """
    if id_col not in preds.columns:
        raise ValueError(f"predictions missing id column {id_col!r}")
    if sales_col not in preds.columns:
        raise ValueError(f"predictions missing sales column {sales_col!r}")

    ordered = sample[[id_col]].merge(
        preds[[id_col, sales_col]],
        on=id_col,
        how="left",
        validate="one_to_one",
    )
    out = ordered.rename(columns={sales_col: "sales"})[["id", "sales"]].copy()
    out["id"] = out["id"].astype(sample[id_col].dtype, copy=False)
    out["sales"] = pd.to_numeric(out["sales"], errors="coerce").astype(float)
    return out


def write_submission(
    sub: pd.DataFrame,
    path: Path | str,
    *,
    sample: pd.DataFrame | None = None,
) -> Path:
    """Validate (optional sample) and write CSV with columns ``id,sales``.

    Returns the resolved output path.
    """
    path = Path(path)
    frame = sub[["id", "sales"]].copy() if set(REQUIRED_COLUMNS).issubset(sub.columns) else sub
    if sample is not None:
        if not set(REQUIRED_COLUMNS).issubset(frame.columns):
            raise ValueError(
                f"submission must have columns {list(REQUIRED_COLUMNS)} before write"
            )
        validate_submission(frame, sample)
    else:
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"submission missing columns {missing}")
        sales = pd.to_numeric(frame["sales"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(sales).all():
            raise ValueError("submission sales must be finite")

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame[list(REQUIRED_COLUMNS)]
    frame.to_csv(path, index=False)
    return path


def clip_negative_sales(sales: Any, *, enabled: bool) -> np.ndarray:
    """Clip negative predictions when policy is enabled."""
    arr = np.asarray(sales, dtype=float)
    if enabled:
        return np.clip(arr, a_min=0.0, a_max=None)
    return arr
