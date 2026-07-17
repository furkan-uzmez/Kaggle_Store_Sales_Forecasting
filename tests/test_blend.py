"""Mean and OOF-weighted blend helpers (leakage-safe: weights from OOF only)."""

from __future__ import annotations

import numpy as np
import pytest

from store_sales.models.blend import fit_nonneg_blend_weights, mean_blend, weighted_blend


def test_mean_blend_equal_average():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 4.0, 5.0])
    out = mean_blend([a, b])
    assert np.allclose(out, np.array([2.0, 3.0, 4.0]))


def test_mean_blend_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        mean_blend([np.ones(3), np.ones(2)])


def test_weighted_blend_and_nonneg_weights():
    # y ≈ 0.7 * p0 + 0.3 * p1; non-negative least squares should recover ~that.
    rng = np.random.default_rng(0)
    n = 200
    p0 = rng.normal(size=n)
    p1 = rng.normal(size=n)
    y = 0.7 * p0 + 0.3 * p1 + rng.normal(0, 0.01, size=n)
    oof = np.column_stack([p0, p1])
    w = fit_nonneg_blend_weights(oof, y)
    assert w.shape == (2,)
    assert np.all(w >= -1e-9)
    assert abs(w.sum() - 1.0) < 1e-6
    blended = weighted_blend([p0, p1], w)
    assert blended.shape == (n,)
    # Closer to truth than either alone on average squared error.
    mse_blend = float(np.mean((blended - y) ** 2))
    mse0 = float(np.mean((p0 - y) ** 2))
    mse1 = float(np.mean((p1 - y) ** 2))
    assert mse_blend < mse0
    assert mse_blend < mse1
