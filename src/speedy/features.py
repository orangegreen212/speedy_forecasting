"""The single feature contract of the project.

Only calendar information is used. Temperature, CPI, fuel price and
markdowns are not known for 2013, and using their real values in a backtest
would leak the future into the model.
"""

import numpy as np
import pandas as pd
from dateutil.easter import easter

from .config import T0

K_FOURIER = 8
THANKSGIVING_LAGS = range(-3, 3)
CHRISTMAS_LAGS = range(-2, 4)
FIXED_HOLIDAYS = ("superbowl", "laborday", "easter")
HOLIDAY_YEARS = range(2010, 2016)

SEASONAL_COLUMNS = (
    [f"{kind}{k}" for k in range(1, K_FOURIER + 1) for kind in ("s", "c")]
    + [f"th{lag}" for lag in THANKSGIVING_LAGS]
    + [f"xm{lag}" for lag in CHRISTMAS_LAGS]
    + list(FIXED_HOLIDAYS)
)
FEATURE_COLUMNS = SEASONAL_COLUMNS + ["trend"]


def thanksgiving(year: int) -> pd.Timestamp:
    """Fourth Thursday of November."""
    first = pd.Timestamp(year, 11, 1)
    days_to_first_thursday = (3 - first.weekday()) % 7
    return first + pd.Timedelta(days=days_to_first_thursday + 21)


def week_end(date: pd.Timestamp) -> pd.Timestamp:
    """Friday that ends the week containing `date` (data weeks end Friday)."""
    return date + pd.Timedelta(days=(4 - date.weekday()) % 7)


def _first_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(weekday - first.weekday()) % 7)


def holiday_frame(years=HOLIDAY_YEARS) -> pd.DataFrame:
    """Holiday weeks (column `ds`) in the format Prophet expects."""
    day = pd.Timedelta
    rows = []
    for year in years:
        thanks = week_end(thanksgiving(year))
        xmas = week_end(pd.Timestamp(year, 12, 25))
        holidays = {
            "thanks_m2": thanks - day(days=14),
            "thanks_m1": thanks - day(days=7),
            "thanks": thanks,
            "thanks_p1": thanks + day(days=7),
            "xmas_m2": xmas - day(days=14),
            "xmas_m1": xmas - day(days=7),
            "xmas": xmas,
            "xmas_p1": xmas + day(days=7),
            "superbowl": week_end(_first_weekday(year, 2, 6)),
            "laborday": week_end(_first_weekday(year, 9, 0)),
            "easter": pd.Timestamp(easter(year)) - day(days=2),
        }
        rows.extend(holidays.items())
    return pd.DataFrame(rows, columns=["holiday", "ds"])


HOLIDAYS = holiday_frame()


def date_features(dates, t0: pd.Timestamp = T0) -> pd.DataFrame:
    """Calendar features for week-ending dates (the index is the dates)."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    features = {}

    year_fraction = dates.dayofyear.to_numpy() / 365.25
    for k in range(1, K_FOURIER + 1):
        features[f"s{k}"] = np.sin(2 * np.pi * k * year_fraction)
        features[f"c{k}"] = np.cos(2 * np.pi * k * year_fraction)

    days_to_thanks = np.array([(thanksgiving(d.year) - d).days for d in dates])
    weeks_after_thanks = np.floor(-days_to_thanks / 7).astype(int)
    for lag in THANKSGIVING_LAGS:
        features[f"th{lag}"] = (weeks_after_thanks == lag).astype(float)

    days_to_xmas = np.array(
        [(pd.Timestamp(d.year, 12, 25) - d).days for d in dates]
    )
    for lag in CHRISTMAS_LAGS:
        in_week = (days_to_xmas >= 7 * lag) & (days_to_xmas < 7 * lag + 7)
        features[f"xm{lag}"] = in_week.astype(float)

    for name in FIXED_HOLIDAYS:
        holiday_weeks = HOLIDAYS.loc[HOLIDAYS.holiday == name, "ds"]
        features[name] = dates.isin(holiday_weeks).astype(float)

    features["trend"] = (dates - t0).days.to_numpy() / 365.25
    return pd.DataFrame(features, index=dates)[FEATURE_COLUMNS]


def build_features(df: pd.DataFrame, t0: pd.Timestamp = T0) -> pd.DataFrame:
    """Add the feature columns to a table with `Store` and `Date` columns.

    Row order and index are kept and other columns (e.g. `Type`) are passed
    through. The function is pure: no state and no target is used.
    """
    for column in ("Store", "Date"):
        if column not in df.columns:
            raise ValueError(f"build_features needs column '{column}'")
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    unique_dates = out["Date"].drop_duplicates().sort_values()
    features = date_features(unique_dates, t0)
    mapped = features.reindex(out["Date"]).set_axis(out.index)
    stale = [c for c in FEATURE_COLUMNS if c in out.columns]
    out = pd.concat([out.drop(columns=stale), mapped], axis=1)
    if out[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("NaN in features")
    return out
