"""Stable inference API: load the persisted artefact and score future weeks. No access to raw data needed."""
import joblib, numpy as np, pandas as pd


def load_artifact(path):
    """NOTE: joblib/pickle files can execute code on load. Only load artefacts you trained yourself."""
    art = joblib.load(path)
    for k in ("pipeline", "metadata", "calibration"):
        if k not in art:
            raise ValueError(f"artefact is missing '{k}'")
    return art


def future_frame(artifact, end_date) -> pd.DataFrame:
    model = artifact["pipeline"].named_steps["model"]
    start = model.last_train_date_ + pd.Timedelta(days=7)
    dates = pd.date_range(start, pd.Timestamp(end_date), freq="7D")
    if len(dates) == 0:
        raise ValueError(f"end_date must be after {start.date()}")
    stores = model.types_
    return pd.DataFrame({"Date": np.repeat(dates, len(stores)), "Store": np.tile(stores.index, len(dates)),
                         "Type": np.tile(stores.to_numpy(), len(dates))})


def forecast_weekly(artifact, end_date) -> pd.DataFrame:
    """Store x week forecasts with 80% bands: columns Date, Store, Forecast, Lower80, Upper80."""
    X = future_frame(artifact, end_date)
    f = artifact["pipeline"].predict(X)
    lo, hi = artifact["calibration"]["store_week_mult"]
    out = X[["Date", "Store"]].assign(Forecast=f.to_numpy())
    out["Lower80"] = out["Forecast"] * lo; out["Upper80"] = out["Forecast"] * hi
    return out


def forecast_totals(artifact, weekly: pd.DataFrame) -> pd.DataFrame:
    """Company weekly totals with 80% band from the backtested weekly-total error quantiles (actual/forecast-1)."""
    q10, q90 = artifact["calibration"]["weekly_total_err_q"]
    t = weekly.groupby("Date", as_index=False)["Forecast"].sum()
    t["Lower80"] = t["Forecast"] * (1 + q10); t["Upper80"] = t["Forecast"] * (1 + q90)
    return t
