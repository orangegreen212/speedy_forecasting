"""Plot helpers used by the notebooks.

Style rules: one colour per meaning (actual = dark grey, forecast = blue,
benchmark = orange), bars start at zero, every axis has a label with its
unit, and the legend appears only when there is more than one series.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACTUAL = "#333333"
FORECAST = "#1f77b4"
BENCHMARK = "#ff7f0e"
BAND = "#aec7e8"
NEUTRAL = "#7f7f7f"


def _style(ax, title, xlabel=None, ylabel=None, legend=False):
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(frameon=False)
    return ax


def plot_total_sales(wide: pd.DataFrame, types: pd.Series):
    """Total weekly sales, overall and by store type ($ millions)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=False)
    (wide.sum(axis=1) / 1e6).plot(ax=axes[0], color=ACTUAL)
    _style(axes[0], "All stores", "Week", "Sales per week ($M)")
    by_type = wide.T.groupby(types).sum().T / 1e6
    by_type.plot(ax=axes[1])
    _style(
        axes[1], "By store type", "Week", "Sales per week ($M)", legend=True
    )
    fig.tight_layout()
    return fig


def plot_holiday_effect(relative: dict, title: str, ax=None):
    """Bars of sales relative to a normal week (1.0), starting at zero.

    `relative` maps weeks from the holiday week (0 = the holiday week,
    negative = before it) to the mean sales relative to the store average.
    """
    ax = ax or plt.subplots(figsize=(6, 3.5))[1]
    weeks = sorted(k for k, v in relative.items() if pd.notna(v))
    ax.bar(
        [str(k) for k in weeks], [relative[k] for k in weeks], color=FORECAST
    )
    ax.axhline(1, color=ACTUAL, lw=1, ls="--", label="normal week")
    ax.legend(frameon=False, loc="upper left")
    return _style(
        ax, title, "Weeks from the holiday week", "Sales / store average"
    )


def plot_store_trends(wide: pd.DataFrame, stores):
    """13-week average sales of some stores ($ thousands)."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for store in stores:
        (wide[store].rolling(13).mean() / 1e3).plot(
            ax=ax, label=f"Store {store}"
        )
    _style(
        ax,
        "Some stores grow or shrink steadily",
        "Week",
        "13-week average sales ($K)",
        legend=True,
    )
    return fig


def plot_oof_total(actual, blend, naive):
    """Backtest: actual company total against blend and naive."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(
        actual.index, actual.sum(axis=1) / 1e6, color=ACTUAL, label="Actual"
    )
    ax.plot(
        blend.index,
        blend.sum(axis=1) / 1e6,
        color=FORECAST,
        label="Blend (walk-forward)",
    )
    ax.plot(
        naive.index,
        naive.sum(axis=1) / 1e6,
        color=BENCHMARK,
        alpha=0.7,
        label="Seasonal naive",
    )
    _style(
        ax,
        "Backtest: company weekly total",
        "Week",
        "Sales per week ($M)",
        legend=True,
    )
    return fig


def plot_quarter_errors(errors):
    """Bars of actual / forecast - 1 for each 13-week block, in %."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(range(1, len(errors) + 1), np.array(errors) * 100, color=FORECAST)
    ax.axhline(0, color=ACTUAL, lw=1)
    _style(
        ax,
        "Positive = forecast was too low",
        "13-week block (oldest first)",
        "Actual / forecast - 1 (%)",
    )
    return fig


def plot_store_error_hist(actual, blend):
    """Histogram of the error of each store's total over the backtest."""
    errors = (actual.sum() / blend.sum() - 1) * 100
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(errors, bins=15, color=FORECAST, edgecolor="white")
    ax.axvline(0, color=ACTUAL, lw=1)
    _style(
        ax,
        "Error of each store's 65-week total",
        "Actual / forecast - 1 (%)",
        "Number of stores",
    )
    return fig


def plot_company_forecast(actual_total, forecast_total):
    """Actuals, forecast and 80% band of the company total ($M per week).

    `actual_total` is a Series indexed by week; `forecast_total` has the
    columns Date, Forecast, Lower80, Upper80.
    """
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(
        actual_total.index, actual_total / 1e6, color=ACTUAL, label="Actual"
    )
    dates = forecast_total["Date"]
    ax.fill_between(
        dates,
        forecast_total["Lower80"] / 1e6,
        forecast_total["Upper80"] / 1e6,
        color=BAND,
        alpha=0.6,
        label="80% range",
    )
    ax.plot(
        dates,
        forecast_total["Forecast"] / 1e6,
        color=FORECAST,
        label="Forecast",
    )
    ax.axvline(actual_total.index[-1], color=NEUTRAL, ls=":", lw=1)
    ax.text(
        actual_total.index[-1],
        ax.get_ylim()[1],
        " last actual week",
        va="top",
        fontsize=8,
        color=NEUTRAL,
    )
    _style(
        ax,
        "Company weekly sales: actual and forecast",
        "Week",
        "Sales per week ($M)",
        legend=True,
    )
    return fig


def plot_store_examples(actual, forecast, stores):
    """Small multiples: actual and forecast for a few stores ($K)."""
    n_cols = 3
    n_rows = int(np.ceil(len(stores) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(13, 3 * n_rows), sharex=True
    )
    for ax, store in zip(np.ravel(axes), stores, strict=False):
        ax.plot(actual.index, actual[store] / 1e3, color=ACTUAL, lw=1)
        ax.plot(forecast.index, forecast[store] / 1e3, color=FORECAST, lw=1)
        _style(ax, f"Store {store}", None, "Sales per week ($K)")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(
        "Dark = actual, blue = forecast", x=0.01, ha="left", fontsize=10
    )
    fig.tight_layout()
    return fig


def plot_growth_check(observed, forecast, label_stores):
    """Forecast growth against growth observed one year earlier (%)."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(observed * 100, forecast * 100, s=16, color=FORECAST)
    low = min(observed.min(), forecast.min()) * 100 - 1
    high = max(observed.max(), forecast.max()) * 100 + 1
    ax.plot([low, high], [low, high], color=NEUTRAL, ls="--", lw=1)
    for store in label_stores:
        ax.annotate(str(store), (observed[store] * 100, forecast[store] * 100))
    _style(
        ax,
        "Does the forecast continue each store's trend?",
        "Growth seen in the data, 2012 vs 2011 (%)",
        "Forecast growth, 2013 vs 2012 (%)",
    )
    return fig
