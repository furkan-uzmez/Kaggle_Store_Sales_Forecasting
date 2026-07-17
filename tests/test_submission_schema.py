"""Submission schema contract: id/sales columns, row count, id set, finite values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from store_sales.io.submission import validate_submission


def test_validate_submission_ok():
    sub = pd.DataFrame({"id": [1, 2], "sales": [0.0, 1.5]})
    sample = pd.DataFrame({"id": [1, 2], "sales": [0.0, 0.0]})
    validate_submission(sub, sample)


def test_validate_submission_bad_columns():
    sub = pd.DataFrame({"id": [1], "pred": [0.0]})
    sample = pd.DataFrame({"id": [1], "sales": [0.0]})
    with pytest.raises(ValueError):
        validate_submission(sub, sample)


def test_validate_submission_row_count_mismatch():
    sub = pd.DataFrame({"id": [1], "sales": [0.0]})
    sample = pd.DataFrame({"id": [1, 2], "sales": [0.0, 0.0]})
    with pytest.raises(ValueError, match="row count"):
        validate_submission(sub, sample)


def test_validate_submission_id_set_mismatch():
    sub = pd.DataFrame({"id": [1, 99], "sales": [0.0, 1.0]})
    sample = pd.DataFrame({"id": [1, 2], "sales": [0.0, 0.0]})
    with pytest.raises(ValueError, match="id"):
        validate_submission(sub, sample)


def test_validate_submission_non_finite_sales():
    sub = pd.DataFrame({"id": [1, 2], "sales": [0.0, np.nan]})
    sample = pd.DataFrame({"id": [1, 2], "sales": [0.0, 0.0]})
    with pytest.raises(ValueError, match="finite"):
        validate_submission(sub, sample)


def test_validate_submission_inf_sales():
    sub = pd.DataFrame({"id": [1, 2], "sales": [0.0, np.inf]})
    sample = pd.DataFrame({"id": [1, 2], "sales": [0.0, 0.0]})
    with pytest.raises(ValueError, match="finite"):
        validate_submission(sub, sample)
