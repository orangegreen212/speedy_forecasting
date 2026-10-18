import numpy as np
import pandas as pd

from speedy.data import to_wide
from speedy.features import date_features
from speedy.forecasters import (
    HOLT_GRID,
    holt_fc,
    naive_drift_fc,
    naive_fc,
    ridge_fc,
)


def declining_history(weeks):
    """Store 1 falls about 20% a year, store 2 is flat."""
    index = pd.date_range("2010-02-05", periods=weeks, freq="7D")
    history = pd.DataFrame(
        {1: 1e5 * np.exp(-0.004 * np.arange(weeks)), 2: np.full(weeks, 1e5)},
        index=index,
    )
    types = pd.Series({1: "A", 2: "A"})
    return history, types


def future_dates(history, weeks):
    start = history.index[-1] + pd.Timedelta(days=7)
    return pd.date_range(start, periods=weeks, freq="7D")


def test_naive_uses_same_week_last_year(panel):
    wide = to_wide(panel)
    forecast = naive_fc(
        wide.iloc[:100], [wide.index[60] + pd.Timedelta(days=364)]
    )
    assert np.allclose(forecast.iloc[0].to_numpy(), wide.iloc[60].to_numpy())


def test_naive_beyond_one_year_rolls_back_whole_years(panel):
    wide = to_wide(panel).iloc[:100]
    date = wide.index[-1] + pd.Timedelta(days=364 * 2)
    forecast = naive_fc(wide, [date])
    source = wide.loc[date - pd.Timedelta(days=364 * 2)]
    assert np.allclose(forecast.iloc[0].to_numpy(), source.to_numpy())


def test_drift_pulls_a_declining_store_down_and_d0_is_plain_naive():
    history, types = declining_history(110)
    dates = future_dates(history, 30)
    plain = naive_fc(history, dates)
    no_drift = naive_drift_fc(history, types, dates, 0.0)
    drift = naive_drift_fc(history, types, dates, 0.8)
    assert np.array_equal(plain.to_numpy(), no_drift.to_numpy())
    assert (drift[1] < plain[1]).all()
    assert np.allclose(drift[2], plain[2], rtol=0.15)


def test_holt_extends_a_declining_store_and_damps_it():
    history, types = declining_history(120)
    dates = future_dates(history, 60)
    forecast = holt_fc(
        history,
        types,
        date_features(history.index),
        date_features(dates),
        "h2",
    )
    last_actual = history[1].iloc[-1]
    assert forecast[1].iloc[0] < last_actual * 1.05
    assert forecast[1].iloc[-1] < forecast[1].iloc[0]  # keeps falling...
    undamped = last_actual * np.exp(-0.004 * 60)
    assert forecast[1].iloc[-1] > undamped  # ...but slower than undamped
    assert abs(forecast[2].iloc[-1] / 1e5 - 1) < 0.05  # flat stays flat


def reference_holt(series, shape, alpha, beta, phi):
    """Plain loop version of the damped Holt recursion for one store."""
    level, trend = series[:13].mean(), 0.0
    for value in series[13:]:
        previous = level
        level = alpha * value + (1 - alpha) * (previous + phi * trend)
        trend = beta * (level - previous) + (1 - beta) * phi * trend
    return level, trend


def test_vectorised_holt_matches_the_one_store_loop(panel):
    wide = to_wide(panel)
    train = wide.iloc[:100]
    types = pd.Series(dict.fromkeys(wide.columns, "A"))
    dates = future_dates(train, 5)
    x_train, x_future = date_features(train.index), date_features(dates)
    forecast = holt_fc(train, types, x_train, x_future, "h1")

    alpha, beta, phi = HOLT_GRID["h1"]
    damping = np.cumsum(phi ** np.arange(1, 6))
    all_stores = np.log(train)
    centred = all_stores - all_stores.mean()
    from speedy.forecasters import _seasonal_shape

    shape_train, shape_future = _seasonal_shape(
        centred, x_train, x_future, 10.0
    )
    for store in train.columns:
        series = all_stores[store].to_numpy() - shape_train
        level, trend = reference_holt(series, shape_train, alpha, beta, phi)
        expected = np.exp(level + trend * damping + shape_future)
        assert np.allclose(forecast[store].to_numpy(), expected, rtol=1e-10)


def test_ridge_forecast_is_positive_and_seasonal(panel):
    wide = to_wide(panel)
    train = wide.iloc[:100]
    types = pd.Series(dict.fromkeys(wide.columns, "A"))
    dates = future_dates(train, 52)
    forecast = ridge_fc(
        train, types, date_features(train.index), date_features(dates), 1.0
    )
    assert (forecast > 0).all().all()
    assert forecast[1].max() / forecast[1].min() > 1.1
