"""Feature group registry, matrix builder, and ablation helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from store_sales.features.base import BASE_FEATURE_NAMES, add_base_features
from store_sales.features.calendar import CALENDAR_FEATURE_NAMES, add_calendar_features
from store_sales.features.holiday import HOLIDAY_FEATURE_NAMES, add_holiday_features
from store_sales.features.lag import DEFAULT_LAGS, add_lag_features, lag_feature_names
from store_sales.features.oil import OIL_FEATURE_NAMES, add_oil_features
from store_sales.features.promo import PROMO_FEATURE_NAMES, add_promo_features
from store_sales.features.rolling import (
    DEFAULT_STATS,
    DEFAULT_WINDOWS,
    add_rolling_features,
    rolling_feature_names,
)
from store_sales.features.store_meta import (
    STORE_META_FEATURE_NAMES,
    add_store_meta_features,
)
from store_sales.features.transactions import (
    TRANSACTIONS_FEATURE_NAMES,
    add_transaction_features,
)

# Notebook 01 admitted all of these for Layer 2 experiments.
ADMITTED_GROUPS: tuple[str, ...] = (
    "base",
    "calendar",
    "promo",
    "lag",
    "rolling",
    "oil",
    "holiday",
    "store_meta",
    "transactions",
)

# Progressive ablation spine (finite YAML-ready path).
CORE_ABLATION_ORDER: tuple[str, ...] = (
    "base",
    "calendar",
    "promo",
    "lag",
    "rolling",
)

OPTIONAL_ABLATION_GROUPS: tuple[str, ...] = (
    "oil",
    "holiday",
    "store_meta",
    "transactions",
)

ENTITY_COLS: tuple[str, ...] = ("store_nbr", "family")
DATE_COL = "date"
TARGET_COL = "sales"


def mask_target_after(
    df: pd.DataFrame,
    origin: pd.Timestamp | str,
    *,
    date_col: str = DATE_COL,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Null ``target_col`` for rows with ``date_col`` strictly after ``origin``.

    Returns a copy. Use for multi-step (H>1) feature builds from forecast origin
    T0: if the panel is train∪val (or train∪horizon) with true post-origin sales,
    lag/rolling would otherwise see those values. Call this before
    :func:`build_feature_matrix` (or lag/rolling builders), or use recursive
    fill of predicted sales after each step instead.
    """
    out = df.copy()
    origin_ts = pd.Timestamp(origin)
    dates = pd.to_datetime(out[date_col])
    out.loc[dates > origin_ts, target_col] = float("nan")
    return out


FEATURE_GROUPS: dict[str, list[str]] = {
    "base": list(BASE_FEATURE_NAMES),
    "calendar": list(CALENDAR_FEATURE_NAMES),
    "promo": list(PROMO_FEATURE_NAMES),
    "lag": lag_feature_names(TARGET_COL, DEFAULT_LAGS),
    "rolling": rolling_feature_names(TARGET_COL, DEFAULT_WINDOWS, DEFAULT_STATS),
    "oil": list(OIL_FEATURE_NAMES),
    "holiday": list(HOLIDAY_FEATURE_NAMES),
    "store_meta": list(STORE_META_FEATURE_NAMES),
    "transactions": list(TRANSACTIONS_FEATURE_NAMES),
}


def list_group_ablation_configs(
    base_groups: Sequence[str] | None = None,
) -> list[dict]:
    """Emit finite YAML-ready feature-group sets for progressive ablation.

    Path: ``base`` → ``+calendar`` → ``+promo`` → ``+lag`` → ``+rolling``,
    then each optional group admitted by notebook 01 (and present in
    ``base_groups``) as a one-at-a-time extension of the full core.
    """
    admitted = set(base_groups) if base_groups is not None else set(ADMITTED_GROUPS)
    configs: list[dict] = []

    cumulative: list[str] = []
    for group in CORE_ABLATION_ORDER:
        if group not in admitted:
            continue
        cumulative.append(group)
        configs.append({"feature_groups": list(cumulative)})

    core_full = list(cumulative)
    for opt in OPTIONAL_ABLATION_GROUPS:
        if opt in admitted:
            configs.append({"feature_groups": core_full + [opt]})

    return configs


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    groups: Sequence[str],
    extras: Mapping[str, pd.DataFrame] | None = None,
    entity_cols: Sequence[str] = ENTITY_COLS,
    date_col: str = DATE_COL,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Build a PIT-safe feature matrix for the requested groups.

    Target-derived groups (lag, rolling) use only sales values present in
    ``df`` (NaNs propagate through shifts). They do **not** invent a multi-step
    origin contract: for H>1 forecasts from origin T0 on a panel that still
    holds true post-origin sales, call :func:`mask_target_after` with that
    origin first (or recursively fill with model predictions). Building on
    unmasked train∪val leaks mid-horizon targets into lag_1/rollings.

    Parameters
    ----------
    df:
        Panel with at least entity keys and ``date``; ``sales`` required for
        lag/rolling groups. Available target values only — mask after origin
        for multi-step PIT when needed.
    groups:
        Subset of ``FEATURE_GROUPS`` keys to materialize.
    extras:
        Optional tables: ``oil``, ``holidays`` / ``holidays_events``,
        ``stores``, ``transactions``.
    """
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")

    extras = dict(extras or {})
    out = df.copy()
    entity_cols = list(entity_cols)

    if "base" in groups:
        out = add_base_features(out)
    if "calendar" in groups:
        out = add_calendar_features(out, date_col=date_col)
    if "promo" in groups:
        out = add_promo_features(out)
    if "lag" in groups:
        if target_col not in out.columns:
            raise ValueError(f"lag group requires column {target_col!r}")
        out = add_lag_features(
            out,
            entity_cols=entity_cols,
            date_col=date_col,
            target_col=target_col,
            lags=DEFAULT_LAGS,
        )
    if "rolling" in groups:
        if target_col not in out.columns:
            raise ValueError(f"rolling group requires column {target_col!r}")
        out = add_rolling_features(
            out,
            entity_cols=entity_cols,
            date_col=date_col,
            target_col=target_col,
            windows=DEFAULT_WINDOWS,
            stats=DEFAULT_STATS,
            shift=1,
        )
    if "oil" in groups:
        oil = extras.get("oil")
        if oil is None:
            raise ValueError("oil group requires extras['oil']")
        out = add_oil_features(out, oil, date_col=date_col, lag=1)
    if "holiday" in groups:
        holidays = extras.get("holidays_events", extras.get("holidays"))
        if holidays is None:
            raise ValueError(
                "holiday group requires extras['holidays'] or extras['holidays_events']"
            )
        stores = extras.get("stores")
        out = add_holiday_features(
            out, holidays, stores=stores, date_col=date_col
        )
    if "store_meta" in groups:
        stores = extras.get("stores")
        if stores is None:
            raise ValueError("store_meta group requires extras['stores']")
        out = add_store_meta_features(out, stores)
    if "transactions" in groups:
        tx = extras.get("transactions")
        if tx is None:
            raise ValueError("transactions group requires extras['transactions']")
        out = add_transaction_features(out, tx, date_col=date_col)

    return out
