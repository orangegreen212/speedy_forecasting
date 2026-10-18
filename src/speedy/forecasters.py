"""Base forecasters.

Every function takes the sales history as a weeks x stores table and
returns a forecast table with the forecast weeks as rows and the same
stores as columns.
"""

import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

from .features import HOLIDAYS, SEASONAL_COLUMNS, thanksgiving

# Weight on the store's own recent growth in "naive with drift". The rest
# goes to the growth of the store type. "d0" means no drift at all.
DRIFT_GRID = {"d0": 0.0, "d5": 0.5, "d8": 0.8}

# Damped Holt smoothing: (level alpha, trend beta, damping phi), per week.
HOLT_GRID = {
    "h0": (0.1, 0.05, 0.90),
    "h1": (0.3, 0.10, 0.95),
    "h2": (0.2, 0.05, 0.98),
}

DAYS_PER_YEAR_OF_WEEKS = 364  # 52 whole weeks keep the weekday
HOLT_WARMUP_WEEKS = 13
RIDGE_SLOPE_CAP = 0.15
RIDGE_SLOPE_SHRINK = 0.5


def _same_week_last_year(date: pd.Timestamp) -> pd.Timestamp:
    """The week one year earlier (364 days back).

    Thanksgiving moves by up to a week between years, so around it the
    week is aligned on Thanksgiving instead of on the calendar.
    """
    base = date - pd.Timedelta(days=DAYS_PER_YEAR_OF_WEEKS)
    offset = (thanksgiving(date.year) - date).days
    if -14 <= offset <= 21:
        candidate = thanksgiving(date.year - 1) - pd.Timedelta(days=offset)
        weekday_gap = (date.weekday() - candidate.weekday()) % 7
        candidate += pd.Timedelta(days=weekday_gap)
        if abs((candidate - base).days) <= 7:
            return candidate
    return base


def _naive_sources(history: pd.DataFrame, dates):
    """For each forecast date: the history week to copy and its age.

    A date more than one year ahead is mapped back by whole 52-week years
    until it lands inside the history. The age is the number of years back.
    """
    last_week = history.index[-1]
    sources, years_back = [], []
    for date in pd.DatetimeIndex(dates):
        source, years = _same_week_last_year(date), 1
        while source > last_week:
            source -= pd.Timedelta(days=DAYS_PER_YEAR_OF_WEEKS)
            years += 1
        sources.append(source)
        years_back.append(years)
    return sources, np.array(years_back)


def naive_fc(history: pd.DataFrame, dates) -> pd.DataFrame:
    """Seasonal naive: sales of the same week last year."""
    sources, _ = _naive_sources(history, dates)
    forecast = history.reindex(sources)
    forecast.index = pd.DatetimeIndex(dates)
    return forecast


def naive_drift_fc(
    history, types, dates, shrink=0.5, window=26, growth_limits=(0.8, 1.2)
) -> pd.DataFrame:
    """Seasonal naive with drift.

    Last year's sales of the same week, times the recent year-on-year
    growth of the store (last `window` weeks against the same weeks one
    year earlier). The store growth is mixed with the growth of its store
    type (`shrink` is the weight of the store's own growth), limited to
    `growth_limits`, and damped for dates more than one year ahead.
    """
    sources, years_back = _naive_sources(history, dates)
    base = history.reindex(sources)
    base.index = pd.DatetimeIndex(dates)
    if shrink == 0 or len(history) < window + 52:
        return base

    recent = history.iloc[-window:]
    shifted = [
        d - pd.Timedelta(days=DAYS_PER_YEAR_OF_WEEKS) for d in recent.index
    ]
    prior = history.reindex(shifted)
    store_growth = recent.sum() / prior.sum().to_numpy()

    type_of_store = types.reindex(history.columns)
    type_growth = recent.sum().groupby(type_of_store).sum() / (
        prior.sum().groupby(type_of_store).sum()
    )
    type_growth = type_growth.reindex(type_of_store.to_numpy())
    growth = (
        shrink * store_growth + (1 - shrink) * type_growth.to_numpy()
    ).clip(*growth_limits)

    # One year back: full growth. Two years back: 1.5 times (damped).
    exponent = 1 + 0.5 * (years_back - 1)
    return base * (growth.to_numpy()[None, :] ** exponent[:, None])


def _columns_by_type(history: pd.DataFrame, types: pd.Series):
    """Yield (store type, store columns of that type)."""
    type_of_store = types.reindex(history.columns).to_numpy()
    for kind in types.unique():
        yield kind, history.columns[type_of_store == kind]


def _seasonal_shape(demeaned, x_train, x_future, alpha):
    """Seasonal and holiday shape (log scale) shared by one store type.

    A ridge regression of the cross-store mean of the demeaned log sales
    on the seasonal features.
    """
    model = Ridge(alpha=alpha).fit(
        x_train[SEASONAL_COLUMNS].to_numpy(),
        demeaned.mean(axis=1).to_numpy(),
    )
    return (
        model.predict(x_train[SEASONAL_COLUMNS].to_numpy()),
        model.predict(x_future[SEASONAL_COLUMNS].to_numpy()),
    )


def holt_fc(
    history, types, x_train, x_future, key="h1", ridge_alpha=10.0
) -> pd.DataFrame:
    """Damped local trend per store.

    The shared seasonal shape is removed from each log series. Damped
    Holt smoothing then follows the current level and slope of the
    deseasonalised series, and the seasonal shape is added back. This lets
    a store that grows or declines keep doing so, with the trend fading.
    """
    alpha, beta, phi = HOLT_GRID[key]
    forecast = pd.DataFrame(
        index=x_future.index, columns=history.columns, dtype=float
    )
    horizon = np.round(
        (x_future.index - history.index[-1]).days.to_numpy() / 7
    ).astype(int)
    damping = np.cumsum(phi ** np.arange(1, horizon.max() + 1))[horizon - 1]

    for _, columns in _columns_by_type(history, types):
        log_sales = np.log(history[columns])
        demeaned = log_sales - log_sales.mean()
        shape_train, shape_future = _seasonal_shape(
            demeaned, x_train, x_future, ridge_alpha
        )
        series = log_sales.to_numpy() - shape_train[:, None]
        level = series[:HOLT_WARMUP_WEEKS].mean(axis=0)
        trend = np.zeros(series.shape[1])
        for week in range(HOLT_WARMUP_WEEKS, len(series)):
            previous = level
            level = alpha * series[week] + (1 - alpha) * (
                previous + phi * trend
            )
            trend = beta * (level - previous) + (1 - beta) * phi * trend
        forecast[columns] = np.exp(
            level[None, :]
            + trend[None, :] * damping[:, None]
            + shape_future[:, None]
        )
    return forecast


def ridge_fc(history, types, x_train, x_future, alpha=10.0) -> pd.DataFrame:
    """Shared seasonal shape plus store level plus a shrunk, frozen trend.

    Each store type has one seasonal shape. A store adds its own level and
    a linear trend (capped, halved, and not extrapolated past the end of
    the training data).
    """
    forecast = pd.DataFrame(
        index=x_future.index, columns=history.columns, dtype=float
    )
    trend_train = x_train["trend"].to_numpy()
    trend_future = np.minimum(x_future["trend"].to_numpy(), trend_train.max())

    for _, columns in _columns_by_type(history, types):
        log_sales = np.log(history[columns])
        level = log_sales.mean()
        demeaned = log_sales - level
        shape_train, shape_future = _seasonal_shape(
            demeaned, x_train, x_future, alpha
        )
        residual = demeaned.to_numpy() - shape_train[:, None]
        slope = np.polyfit(trend_train, residual, 1)[0]
        slope = np.clip(slope, -RIDGE_SLOPE_CAP, RIDGE_SLOPE_CAP)
        slope = slope * RIDGE_SLOPE_SHRINK
        intercept = residual.mean(axis=0) - slope * trend_train.mean()
        forecast[columns] = np.exp(
            level.to_numpy()[None, :]
            + shape_future[:, None]
            + intercept[None, :]
            + trend_future[:, None] * slope[None, :]
        )
    return forecast


def _fit_prophet(store, week_index, sales, future_dates, fourier, cps):
    """Fit Prophet on the log sales of one store and forecast."""
    from prophet import Prophet

    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.ERROR)
    model = Prophet(
        growth="linear",
        yearly_seasonality=fourier,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=HOLIDAYS,
        changepoint_prior_scale=cps,
    )
    model.fit(pd.DataFrame({"ds": week_index, "y": np.log(sales)}))
    predicted = model.predict(pd.DataFrame({"ds": future_dates}))
    return store, np.exp(predicted["yhat"].to_numpy())


def prophet_fc(history, dates, fourier=5, cps=0.05, n_jobs=1) -> pd.DataFrame:
    """One Prophet model per store, fitted on log sales."""
    dates = pd.DatetimeIndex(dates)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_prophet)(
            store,
            history.index,
            history[store].to_numpy(),
            dates,
            fourier,
            cps,
        )
        for store in history.columns
    )
    forecast = pd.DataFrame(dict(results), index=dates)
    return forecast[history.columns]
