"""Leakage-safe outlier *policy helpers* (flag / describe only).

Do **not** delete rare-but-valid sales spikes in prepare. Winsorization, if
chosen later in notebook ``01``, must be fit fold-locally on train only —
never global train+test caps.
"""

from __future__ import annotations

import pandas as pd

# Quantiles for EDA tail inspection only (not fitted preprocessing state).
_TAIL_QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)


def flag_invalid_sales(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask of domain-invalid sales (e.g. negative).

    Index-aligned with ``df``. Does not mutate or drop rows.
    """
    if "sales" not in df.columns:
        raise ValueError("df must contain a 'sales' column")
    return (df["sales"] < 0).astype(bool)


def describe_sales_tails(df: pd.DataFrame) -> pd.DataFrame:
    """Return sales quantile summary for EDA (not a transform pipeline).

    Does not mutate ``df`` and does not delete spikes. Use only for diagnostics;
    do not derive global caps from full-history quantiles for modeling.
    """
    if "sales" not in df.columns:
        raise ValueError("df must contain a 'sales' column")
    quantiles = df["sales"].quantile(list(_TAIL_QUANTILES))
    return quantiles.to_frame(name="sales")
