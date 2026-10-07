"""The single feature contract. Both EDA and modelling consume build_features(); nothing else creates features.

Only deterministic calendar information is used (no Temperature/CPI/MarkDown): these are unknown for 2013 and
using their realised values in a backtest would leak the future.
"""
import numpy as np
import pandas as pd
from dateutil.easter import easter
from .config import T0

K_FOURIER = 8
SEASONAL_COLUMNS = ([f"{f}{k}" for k in range(1, K_FOURIER + 1) for f in ("s", "c")]
                    + [f"th{l}" for l in range(-3, 3)] + [f"xm{l}" for l in range(-2, 4)]
                    + ["superbowl", "laborday", "easter"])
FEATURE_COLUMNS = SEASONAL_COLUMNS + ["trend"]


def thanksgiving(year: int) -> pd.Timestamp:
    d = pd.Timestamp(year, 11, 1)
    return d + pd.Timedelta(days=(3 - d.weekday()) % 7 + 21)


def week_end(dt: pd.Timestamp) -> pd.Timestamp:
    """Friday ending the week that contains dt (weeks in the data end on Friday)."""
    return dt + pd.Timedelta(days=(4 - dt.weekday()) % 7)


def holiday_frame(years=range(2010, 2016)) -> pd.DataFrame:
    rows = []
    for yr in years:
        th = week_end(thanksgiving(yr)); xm = week_end(pd.Timestamp(yr, 12, 25))
        feb1 = pd.Timestamp(yr, 2, 1)
        sb = week_end(feb1 + pd.Timedelta(days=(6 - feb1.weekday()) % 7))
        sep1 = pd.Timestamp(yr, 9, 1)
        lb = week_end(sep1 + pd.Timedelta(days=(0 - sep1.weekday()) % 7))
        ea = pd.Timestamp(easter(yr)) - pd.Timedelta(days=2)
        d = pd.Timedelta
        for n, dt in {"thanks_m2": th - d(days=14), "thanks_m1": th - d(days=7), "thanks": th, "thanks_p1": th + d(days=7),
                      "xmas_m2": xm - d(days=14), "xmas_m1": xm - d(days=7), "xmas": xm, "xmas_p1": xm + d(days=7),
                      "superbowl": sb, "laborday": lb, "easter": ea}.items():
            rows.append((n, dt))
    return pd.DataFrame(rows, columns=["holiday", "ds"])


def date_features(dates, t0: pd.Timestamp = T0) -> pd.DataFrame:
    """Calendar features for a set of week-ending dates (index = dates)."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    F = {}
    doy = dates.dayofyear.to_numpy() / 365.25
    for k in range(1, K_FOURIER + 1):
        F[f"s{k}"] = np.sin(2 * np.pi * k * doy); F[f"c{k}"] = np.cos(2 * np.pi * k * doy)
    th = np.array([(thanksgiving(d.year) - d).days for d in dates])
    for lag in range(-3, 3):
        F[f"th{lag}"] = (np.floor(-th / 7).astype(int) == lag).astype(float)
    xm = np.array([(pd.Timestamp(d.year, 12, 25) - d).days for d in dates])
    for lag in range(-2, 4):
        F[f"xm{lag}"] = ((xm >= 7 * lag) & (xm < 7 * lag + 7)).astype(float)
    hol = holiday_frame()
    for n in ("superbowl", "laborday", "easter"):
        F[n] = dates.isin(hol.loc[hol.holiday == n, "ds"]).astype(float)
    F["trend"] = (dates - t0).days.to_numpy() / 365.25
    return pd.DataFrame(F, index=dates)[FEATURE_COLUMNS]


def build_features(df: pd.DataFrame, t0: pd.Timestamp = T0) -> pd.DataFrame:
    """Row-wise feature builder. Input needs Store, Date (Type is passed through if present).
    Output keeps the input row order and index, adds FEATURE_COLUMNS. Pure function: no state, no target."""
    for c in ("Store", "Date"):
        if c not in df.columns:
            raise ValueError(f"build_features needs column '{c}'")
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    feats = date_features(out["Date"].drop_duplicates().sort_values(), t0)
    mapped = feats.reindex(out["Date"]).set_axis(out.index)
    out = pd.concat([out.drop(columns=[c for c in FEATURE_COLUMNS if c in out.columns]), mapped], axis=1)
    if out[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("NaN in features")
    return out
