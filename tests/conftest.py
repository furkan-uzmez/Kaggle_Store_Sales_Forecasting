from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tiny_panel() -> pd.DataFrame:
    """Two entities, 40 consecutive days, for unit tests."""
    rows = []
    for store, family in [(1, "A"), (1, "B")]:
        for day in range(40):
            rows.append(
                {
                    "date": pd.Timestamp("2017-01-01") + pd.Timedelta(days=day),
                    "store_nbr": store,
                    "family": family,
                    "onpromotion": day % 5,
                    "sales": float((day % 7) + store),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_project_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "interim").mkdir(parents=True)
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / "outputs").mkdir(parents=True)
    (tmp_path / "configs").mkdir(parents=True)
    return tmp_path
