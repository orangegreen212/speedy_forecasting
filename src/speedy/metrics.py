"""Accuracy metrics, bootstrap confidence intervals and prediction bands.

Sign convention: Bias = (forecast - actual) / actual, so a negative bias
means the forecast was too LOW. Weekly totals are summed over stores
before the error is taken.
"""

import numpy as np
import pandas as pd


def metrics(actual: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """Error metrics for weeks x stores tables of actuals and forecasts."""
    error = forecast - actual
    total = actual.sum().sum()
    return {
        "MAE": float(np.abs(error).to_numpy().mean()),
        "RMSE": float(np.sqrt((error**2).to_numpy().mean())),
        "WMAPE": float(np.abs(error).sum().sum() / total),
        "MAPE": float((np.abs(error) / actual).mean().mean()),
        "Bias": float(error.sum().sum() / total),
        "Store_annual_APE": float((np.abs(error.sum()) / actual.sum()).mean()),
        "Weekly_total_MAPE": float(
            (np.abs(error.sum(axis=1)) / actual.sum(axis=1)).mean()
        ),
    }


def wmape(actual, forecast) -> float:
    """Weighted MAPE: total absolute error divided by total sales."""
    return np.abs(forecast - actual).sum() / actual.sum()


def mae(actual, forecast) -> float:
    """Mean absolute error per store-week, in sales units (USD)."""
    return np.abs(forecast - actual).mean()


def rmse(actual, forecast) -> float:
    """Root mean squared error per store-week, in sales units (USD).

    Squaring punishes big misses (holiday weeks) more than MAE does.
    """
    return np.sqrt(((forecast - actual) ** 2).mean())


def bias(actual, forecast) -> float:
    return (forecast - actual).sum() / actual.sum()


def weekly_total_mape(actual, forecast) -> float:
    total_error = np.abs(forecast.sum(axis=1) - actual.sum(axis=1))
    return (total_error / actual.sum(axis=1)).mean()


STATISTICS = {
    "MAE": mae,
    "RMSE": rmse,
    "WMAPE": wmape,
    "Bias": bias,
    "Weekly_total_MAPE": weekly_total_mape,
}


def _resample_weeks(n_weeks: int, block: int, n_draws: int, seed: int):
    """Yield week positions of moving-block bootstrap samples."""
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_weeks / block))
    for _ in range(n_draws):
        starts = rng.integers(0, n_weeks - block + 1, size=n_blocks)
        weeks = np.concatenate([np.arange(s, s + block) for s in starts])
        yield weeks[:n_weeks]


def block_bootstrap(
    actual,
    forecasts: dict,
    stat="WMAPE",
    block=6,
    n=1000,
    seed=0,
    diff_against=None,
) -> dict:
    """95% confidence intervals with a moving-block bootstrap over weeks.

    Weeks are resampled in blocks because weekly errors are autocorrelated;
    the stores of one week always stay together.
    Returns {model: (value, lower95, upper95)}. With `diff_against` set to
    a model name the statistic is the paired difference
    stat(reference) - stat(model), so a positive value favours the model.
    """
    statistic = STATISTICS[stat]
    actual_values = actual.to_numpy()
    forecast_values = {k: v.to_numpy() for k, v in forecasts.items()}

    def evaluate(name, weeks):
        a = actual_values[weeks]
        f = forecast_values[name][weeks]
        if diff_against is None:
            return statistic(a, f)
        reference = forecast_values[diff_against][weeks]
        return statistic(a, reference) - statistic(a, f)

    n_weeks = len(actual_values)
    all_weeks = np.arange(n_weeks)
    draws = {name: [] for name in forecast_values}
    for weeks in _resample_weeks(n_weeks, block, n, seed):
        for name in draws:
            draws[name].append(evaluate(name, weeks))

    return {
        name: (
            float(evaluate(name, all_weeks)),
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for name, values in draws.items()
    }


def interval_multipliers(actual, forecast, low=0.1, high=0.9):
    """Factors (low, high) so that forecast * factor is a prediction band.

    Taken from the quantiles of log(actual / forecast) of past forecasts.
    """
    log_ratio = np.log(actual.to_numpy() / forecast.to_numpy()).ravel()
    return (
        float(np.exp(np.quantile(log_ratio, low))),
        float(np.exp(np.quantile(log_ratio, high))),
    )


def coverage(actual, forecast, multipliers) -> float:
    """Share of actuals that fall inside forecast * multipliers."""
    inside = (actual >= forecast * multipliers[0]) & (
        actual <= forecast * multipliers[1]
    )
    return float(inside.mean().mean())
