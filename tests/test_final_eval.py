"""Locked final evaluation: OOF re-score, horizon/segment guards, naive gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from store_sales.metrics.final_eval import (
    annotate_horizon,
    bias_log1p,
    fold_metrics_from_oof,
    gate_vs_naive,
    horizon_metrics,
    segment_metrics,
)
from store_sales.metrics.rmsle import rmsle


def _tiny_oof() -> pd.DataFrame:
    # Two folds, 3-day horizons, two families
    rows = []
    for fold, start in [(0, "2017-07-02"), (1, "2017-07-17")]:
        base = pd.Timestamp(start)
        for h in range(3):
            for fam, y, p in [
                ("A", 1.0, 1.0),
                ("B", 4.0, 2.0),
            ]:
                rows.append(
                    {
                        "store_nbr": 1,
                        "family": fam,
                        "date": base + pd.Timedelta(days=h),
                        "fold": fold,
                        "y_true": y,
                        "y_pred": p,
                    }
                )
    return pd.DataFrame(rows)


def test_bias_log1p_zero_when_perfect():
    y = np.array([0.0, 1.0, 3.0])
    assert bias_log1p(y, y) == pytest.approx(0.0)


def test_bias_log1p_positive_when_overpredict():
    y_true = np.array([1.0, 1.0])
    # log1p(pred) - log1p(true) = 1  ⇒  pred = exp(log1p(true) + 1) - 1
    y_pred = np.expm1(np.log1p(y_true) + 1.0)
    assert bias_log1p(y_true, y_pred) == pytest.approx(1.0)


def test_annotate_horizon_day_index_from_val_start():
    oof = _tiny_oof()
    fold_start = {
        0: pd.Timestamp("2017-07-02"),
        1: pd.Timestamp("2017-07-17"),
    }
    out = annotate_horizon(oof, fold_start)
    assert set(out["horizon"].unique()) == {1, 2, 3}
    day0 = out[(out["fold"] == 0) & (out["date"] == pd.Timestamp("2017-07-02"))]
    assert (day0["horizon"] == 1).all()


def test_horizon_metrics_has_rmsle_and_bias_per_step():
    oof = annotate_horizon(
        _tiny_oof(),
        {0: pd.Timestamp("2017-07-02"), 1: pd.Timestamp("2017-07-17")},
    )
    hm = horizon_metrics(oof)
    assert list(hm.columns) == ["horizon", "n", "rmsle", "mae_log1p", "bias_log1p"]
    assert len(hm) == 3
    h1 = hm.loc[hm["horizon"] == 1].iloc[0]
    g = oof[oof["horizon"] == 1]
    assert h1["rmsle"] == pytest.approx(
        rmsle(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
    )


def test_segment_metrics_by_family():
    oof = _tiny_oof()
    sm = segment_metrics(oof, by="family", min_n=1)
    assert set(sm["segment"]) == {"A", "B"}
    assert (sm["segment_type"] == "family").all()
    assert "rmsle" in sm.columns


def test_fold_metrics_from_oof_mean_matches_hand():
    oof = _tiny_oof()
    folds = fold_metrics_from_oof(oof)
    assert len(folds["fold_rmsle"]) == 2
    f0 = oof[oof["fold"] == 0]
    assert folds["fold_rmsle"][0] == pytest.approx(
        rmsle(f0["y_true"].to_numpy(), f0["y_pred"].to_numpy())
    )


def test_gate_vs_naive_beats_floor():
    decision = gate_vs_naive(mean_rmsle=0.39, naive_floor=0.5513)
    assert decision["beats_naive_floor"] is True
    assert decision["delta_vs_naive_rmsle"] == pytest.approx(0.39 - 0.5513)
    assert decision["go_no_go"] == "GO"


def test_gate_vs_naive_fails_when_worse():
    decision = gate_vs_naive(mean_rmsle=0.60, naive_floor=0.5513)
    assert decision["beats_naive_floor"] is False
    assert decision["go_no_go"] == "NO_GO"
