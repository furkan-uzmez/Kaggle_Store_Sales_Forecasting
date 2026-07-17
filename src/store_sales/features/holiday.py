"""Holiday features with locale + transferred semantics."""

from __future__ import annotations

import pandas as pd

HOLIDAY_FEATURE_NAMES: list[str] = [
    "is_holiday",
    "is_national_holiday",
    "is_regional_holiday",
    "is_local_holiday",
    "is_bridge",
    "is_work_day",
    "is_transfer",
]


def add_holiday_features(
    df: pd.DataFrame,
    holidays: pd.DataFrame,
    stores: pd.DataFrame | None = None,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Attach locale-aware holiday flags; ``transferred=True`` is not celebrated.

    Rules (notebook 01 / competition):
    - National: apply to all stores on date when not transferred.
    - Regional: match ``locale_name`` to store ``state``.
    - Local: match ``locale_name`` to store ``city``.
    - Transfer / Bridge / Work Day: separate markers; Transfer alone is not a
      celebration day for the original description.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    hol = holidays.copy()
    hol[date_col] = pd.to_datetime(hol[date_col])
    if "transferred" not in hol.columns:
        raise ValueError("holidays missing required column: transferred")
    celebrated = hol.loc[~hol["transferred"].astype(bool)].copy()

    national_dates = set(
        celebrated.loc[celebrated["locale"].eq("National"), date_col]
    )
    regional = celebrated.loc[celebrated["locale"].eq("Regional"), [date_col, "locale_name"]]
    local = celebrated.loc[celebrated["locale"].eq("Local"), [date_col, "locale_name"]]

    type_dates = {
        "Bridge": set(celebrated.loc[celebrated["type"].eq("Bridge"), date_col]),
        "Work Day": set(celebrated.loc[celebrated["type"].eq("Work Day"), date_col]),
        "Transfer": set(celebrated.loc[celebrated["type"].eq("Transfer"), date_col]),
    }

    # Attach store city/state for locale matching when available
    if stores is not None and "store_nbr" in out.columns:
        meta = stores[["store_nbr", "city", "state"]].drop_duplicates("store_nbr")
        out = out.merge(meta, on="store_nbr", how="left", suffixes=("", "_store"))
        city_col = "city"
        state_col = "state"
    else:
        city_col = state_col = None

    out["is_national_holiday"] = out[date_col].isin(national_dates).astype("int8")

    if state_col and not regional.empty:
        reg_keys = set(zip(regional[date_col], regional["locale_name"]))
        out["is_regional_holiday"] = [
            int((d, s) in reg_keys)
            for d, s in zip(out[date_col], out[state_col])
        ]
        out["is_regional_holiday"] = out["is_regional_holiday"].astype("int8")
    else:
        out["is_regional_holiday"] = 0

    if city_col and not local.empty:
        loc_keys = set(zip(local[date_col], local["locale_name"]))
        out["is_local_holiday"] = [
            int((d, c) in loc_keys)
            for d, c in zip(out[date_col], out[city_col])
        ]
        out["is_local_holiday"] = out["is_local_holiday"].astype("int8")
    else:
        out["is_local_holiday"] = 0

    out["is_bridge"] = out[date_col].isin(type_dates["Bridge"]).astype("int8")
    out["is_work_day"] = out[date_col].isin(type_dates["Work Day"]).astype("int8")
    out["is_transfer"] = out[date_col].isin(type_dates["Transfer"]).astype("int8")
    out["is_holiday"] = (
        (out["is_national_holiday"] == 1)
        | (out["is_regional_holiday"] == 1)
        | (out["is_local_holiday"] == 1)
    ).astype("int8")

    # Drop temporarily joined store geo if we added duplicates of existing cols only once
    return out
