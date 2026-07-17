"""Locked final evaluation helpers: OOF re-score, horizon/segment guards, naive gate.

No training or HPO — score existing OOF predictions only.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from store_sales.metrics.guards import mae_log1p
from store_sales.metrics.rmsle import rmsle

NAIVE_FLOOR_SN7_RMSLE = 0.5513


def bias_log1p(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean signed error on log1p scale: mean(log1p(pred) - log1p(true)).

    Positive ⇒ systematic over-prediction on the log1p scale.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")
    if yt.size == 0:
        raise ValueError("Empty arrays")
    return float(np.mean(np.log1p(yp) - np.log1p(yt)))


def annotate_horizon(
    oof: pd.DataFrame,
    fold_val_start: Mapping[int, pd.Timestamp],
    *,
    date_col: str = "date",
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Add 1-based horizon day within each fold validation window."""
    out = oof.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    starts = {int(k): pd.Timestamp(v) for k, v in fold_val_start.items()}

    def _h(row: pd.Series) -> int:
        start = starts[int(row[fold_col])]
        return int((row[date_col] - start).days) + 1

    out["horizon"] = out.apply(_h, axis=1)
    return out


def annotate_payday(oof: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    """Flag public-sector-style payday (15th or last day of month)."""
    out = oof.copy()
    dates = pd.to_datetime(out[date_col])
    is_month_end = (dates + pd.Timedelta(days=1)).dt.day.eq(1)
    out["is_payday"] = (dates.dt.day.eq(15) | is_month_end).astype(int)
    return out


def annotate_zero_target(
    oof: pd.DataFrame, *, target_col: str = "y_true"
) -> pd.DataFrame:
    out = oof.copy()
    out["is_zero"] = (out[target_col].to_numpy(dtype=float) <= 0).astype(int)
    return out


def fold_metrics_from_oof(
    oof: pd.DataFrame,
    *,
    fold_col: str = "fold",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> dict[str, Any]:
    """Per-fold and mean RMSLE / MAE_log1p / bias from OOF rows."""
    fold_ids = sorted(int(f) for f in oof[fold_col].unique())
    fold_rmsle: list[float] = []
    fold_mae: list[float] = []
    fold_bias: list[float] = []
    fold_rows: list[dict[str, Any]] = []

    for fold in fold_ids:
        g = oof.loc[oof[fold_col] == fold]
        yt = g[y_true_col].to_numpy(dtype=float)
        yp = g[y_pred_col].to_numpy(dtype=float)
        r = rmsle(yt, yp)
        m = mae_log1p(yt, yp)
        b = bias_log1p(yt, yp)
        fold_rmsle.append(r)
        fold_mae.append(m)
        fold_bias.append(b)
        fold_rows.append(
            {
                "fold": fold,
                "n": int(len(g)),
                "rmsle": r,
                "mae_log1p": m,
                "bias_log1p": b,
            }
        )

    yt_all = oof[y_true_col].to_numpy(dtype=float)
    yp_all = oof[y_pred_col].to_numpy(dtype=float)
    mean_r = float(np.mean(fold_rmsle)) if fold_rmsle else float("nan")
    std_r = float(np.std(fold_rmsle, ddof=1)) if len(fold_rmsle) > 1 else 0.0

    return {
        "mean_rmsle": mean_r,
        "std_rmsle": std_r,
        "fold_rmsle": fold_rmsle,
        "mean_mae_log1p": float(np.mean(fold_mae)) if fold_mae else float("nan"),
        "fold_mae_log1p": fold_mae,
        "mean_bias_log1p": float(np.mean(fold_bias)) if fold_bias else float("nan"),
        "fold_bias_log1p": fold_bias,
        "pooled_oof_rmsle": rmsle(yt_all, yp_all),
        "pooled_oof_mae_log1p": mae_log1p(yt_all, yp_all),
        "pooled_oof_bias_log1p": bias_log1p(yt_all, yp_all),
        "folds": fold_rows,
        "n_oof": int(len(oof)),
    }


def horizon_metrics(
    oof: pd.DataFrame,
    *,
    horizon_col: str = "horizon",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    """RMSLE / MAE_log1p / bias by forecast horizon step."""
    if horizon_col not in oof.columns:
        raise KeyError(f"missing {horizon_col}; call annotate_horizon first")
    rows: list[dict[str, Any]] = []
    for h, g in oof.groupby(horizon_col, sort=True):
        yt = g[y_true_col].to_numpy(dtype=float)
        yp = g[y_pred_col].to_numpy(dtype=float)
        rows.append(
            {
                "horizon": int(h),
                "n": int(len(g)),
                "rmsle": rmsle(yt, yp),
                "mae_log1p": mae_log1p(yt, yp),
                "bias_log1p": bias_log1p(yt, yp),
            }
        )
    return pd.DataFrame(rows)


def segment_metrics(
    oof: pd.DataFrame,
    *,
    by: str,
    min_n: int = 50,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    """RMSLE / MAE_log1p / bias for one categorical segment column."""
    if by not in oof.columns:
        raise KeyError(f"missing segment column {by!r}")
    rows: list[dict[str, Any]] = []
    for key, g in oof.groupby(by, observed=True, sort=False):
        if len(g) < min_n:
            continue
        yt = g[y_true_col].to_numpy(dtype=float)
        yp = g[y_pred_col].to_numpy(dtype=float)
        rows.append(
            {
                "segment_type": by,
                "segment": key,
                "n": int(len(g)),
                "rmsle": rmsle(yt, yp),
                "mae_log1p": mae_log1p(yt, yp),
                "bias_log1p": bias_log1p(yt, yp),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame(
            columns=[
                "segment_type",
                "segment",
                "n",
                "rmsle",
                "mae_log1p",
                "bias_log1p",
            ]
        )
    return out.sort_values("rmsle", ascending=False).reset_index(drop=True)


def gate_vs_naive(
    *,
    mean_rmsle: float,
    naive_floor: float = NAIVE_FLOOR_SN7_RMSLE,
) -> dict[str, Any]:
    """Primary go/no-go: locked model must beat seasonal-naive-7 floor."""
    beats = bool(mean_rmsle < naive_floor)
    return {
        "naive_floor_sn7_mean_rmsle": float(naive_floor),
        "beats_naive_floor": beats,
        "delta_vs_naive_rmsle": float(mean_rmsle - naive_floor),
        "go_no_go": "GO" if beats else "NO_GO",
    }


def build_segment_table(
    oof: pd.DataFrame,
    segment_cols: list[str],
    *,
    min_n: int = 50,
) -> pd.DataFrame:
    """Stack multi-column segment metrics into one long table."""
    frames = [segment_metrics(oof, by=col, min_n=min_n) for col in segment_cols]
    if not frames:
        return pd.DataFrame(
            columns=[
                "segment_type",
                "segment",
                "n",
                "rmsle",
                "mae_log1p",
                "bias_log1p",
            ]
        )
    return pd.concat(frames, ignore_index=True)
