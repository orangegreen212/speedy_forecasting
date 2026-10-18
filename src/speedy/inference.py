"""Inference API: load the saved model and forecast future weeks.

No raw data is needed, only the saved model file.
"""

import joblib
import numpy as np
import pandas as pd

REQUIRED_KEYS = ("pipeline", "metadata", "calibration")


def load_artifact(path) -> dict:
    """Load a saved model.

    Warning: joblib files are pickles and can run code when loaded. Only
    load files that you trained yourself.
    """
    artifact = joblib.load(path)
    for key in REQUIRED_KEYS:
        if key not in artifact:
            raise ValueError(f"artefact is missing '{key}'")
    return artifact


def future_frame(artifact, end_date) -> pd.DataFrame:
    """Every (week, store) from the week after training up to `end_date`."""
    model = artifact["pipeline"].named_steps["model"]
    start = model.last_train_date_ + pd.Timedelta(days=7)
    dates = pd.date_range(start, pd.Timestamp(end_date), freq="7D")
    if len(dates) == 0:
        raise ValueError(f"end_date must be after {start.date()}")
    stores = model.types_
    return pd.DataFrame(
        {
            "Date": np.repeat(dates, len(stores)),
            "Store": np.tile(stores.index, len(dates)),
            "Type": np.tile(stores.to_numpy(), len(dates)),
        }
    )


def forecast_weekly(artifact, end_date) -> pd.DataFrame:
    """Store x week forecasts with an 80% band.

    Columns: Date, Store, Forecast, Lower80, Upper80.
    """
    frame = future_frame(artifact, end_date)
    forecast = artifact["pipeline"].predict(frame)
    low, high = artifact["calibration"]["store_week_mult"]
    out = frame[["Date", "Store"]].assign(Forecast=forecast.to_numpy())
    out["Lower80"] = out["Forecast"] * low
    out["Upper80"] = out["Forecast"] * high
    return out


def forecast_totals(artifact, weekly: pd.DataFrame) -> pd.DataFrame:
    """Company totals per week with an 80% band.

    The band comes from the backtest errors of the weekly total
    (actual / forecast - 1, 10% and 90% quantiles).
    """
    low, high = artifact["calibration"]["weekly_total_err_q"]
    totals = weekly.groupby("Date", as_index=False)["Forecast"].sum()
    totals["Lower80"] = totals["Forecast"] * (1 + low)
    totals["Upper80"] = totals["Forecast"] * (1 + high)
    return totals
