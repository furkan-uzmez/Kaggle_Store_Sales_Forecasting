"""Known-future calendar features (safe at prediction time)."""

from __future__ import annotations

import pandas as pd

# Ecuador earthquake (competition regime marker) — calendar-known event date.
EARTHQUAKE_DATE = pd.Timestamp("2016-04-16")

CALENDAR_FEATURE_NAMES: list[str] = [
    "dayofweek",
    "month",
    "day",
    "is_weekend",
    "is_payday",
    "post_eq",
    "days_since_eq",
]


def add_calendar_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Add DOW/month/weekend/payday and post-earthquake regime markers.

    Payday rule (notebook 01): day == 15 or calendar month-end.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    out["dayofweek"] = dates.dt.dayofweek.astype("int16")
    out["month"] = dates.dt.month.astype("int16")
    out["day"] = dates.dt.day.astype("int16")
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("int8")
    # Month-end: next day is day 1 of a month
    is_month_end = (dates + pd.Timedelta(days=1)).dt.day.eq(1)
    out["is_payday"] = (dates.dt.day.eq(15) | is_month_end).astype("int8")
    out["post_eq"] = (dates >= EARTHQUAKE_DATE).astype("int8")
    out["days_since_eq"] = (dates - EARTHQUAKE_DATE).dt.days.astype("int32")
    return out
