import pandas as pd

from store_sales.models.baseline import last_value_predict, seasonal_naive_predict


def _panel():
    return pd.DataFrame(
        {
            "date": pd.date_range("2017-01-01", periods=14, freq="D"),
            "store_nbr": 1,
            "family": "A",
            "sales": list(range(14)),
        }
    )


def test_last_value_uses_most_recent_history():
    df = _panel()
    hist, fut = df.iloc[:10], df.iloc[10:]
    pred = last_value_predict(
        history=hist,
        future=fut,
        entity_cols=["store_nbr", "family"],
        date_col="date",
        target_col="sales",
    )
    assert pred.iloc[0] == 9.0  # last hist sales


def test_seasonal_naive_period_7():
    df = _panel()
    hist, fut = df.iloc[:10], df.iloc[10:]
    pred = seasonal_naive_predict(
        history=hist,
        future=fut,
        entity_cols=["store_nbr", "family"],
        date_col="date",
        target_col="sales",
        period=7,
    )
    assert pred.iloc[0] == 3.0


def test_seasonal_naive_multi_step_looks_back_k_periods():
    """Horizon day beyond one period must use k*period lag into history, not val zeros."""
    # history: 14 days (0..13), future: days 14..20 (relative offsets)
    dates = pd.date_range("2017-01-01", periods=21, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 1,
            "family": "A",
            "sales": list(range(21)),
        }
    )
    hist, fut = df.iloc[:14], df.iloc[14:]
    # train_end = 2017-01-14 (sales 13); first future day offset 1 → k=1 → lag 7 → sales 7
    # day index 20 (2017-01-21): delta from train_end = 7 → k=1 → lag 7 → 2017-01-14 sales 13
    # day index 21 would need k=2; extend: for fut day at sales index 20 is day 21 of range
    # Use a future day with delta=8: 2017-01-22 is not in panel. Rebuild with longer hist/fut.
    dates = pd.date_range("2017-01-01", periods=25, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 1,
            "family": "A",
            "sales": list(range(25)),
        }
    )
    hist, fut = df.iloc[:10], df.iloc[10:]
    # train_end = day 9 (2017-01-10, sales=9)
    # future day 17 (iloc 17): date 2017-01-18, delta=8, k=ceil(8/7)=2, lag=14
    # lookup 2017-01-18 - 14d = 2017-01-04 → sales=3
    pred = seasonal_naive_predict(
        history=hist,
        future=fut,
        entity_cols=["store_nbr", "family"],
        date_col="date",
        target_col="sales",
        period=7,
    )
    target_row = fut["date"] == pd.Timestamp("2017-01-18")
    assert pred.loc[fut.index[target_row]].iloc[0] == 3.0
