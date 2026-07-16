import numpy as np
import pytest

from store_sales.metrics.rmsle import rmsle


def test_rmsle_perfect_prediction_is_zero():
    y = np.array([0.0, 1.0, 10.5])
    assert rmsle(y, y) == pytest.approx(0.0)


def test_rmsle_known_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    # hand: mean((log1p diffs)**2) then sqrt
    expected = float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))
    assert rmsle(y_true, y_pred) == pytest.approx(expected)


def test_rmsle_rejects_length_mismatch():
    with pytest.raises(ValueError):
        rmsle(np.array([1.0, 2.0]), np.array([1.0]))
