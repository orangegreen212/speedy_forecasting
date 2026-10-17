"""Data-cleaning policies. Everything here is FIT ON THE TRAINING SLICE ONLY (called inside each fold / inside fit),
so cleaning can never carry information from the forecast period into the model.

Policies
* Returns (negative dept-week rows): kept NET. The reported revenue figure is net of returns; the effect is ~0.001% of sales.
* Anomalies: a store-week is flagged only if it is extreme AFTER removing the store's own level and the move that all stores
  made in that week (robust z by MAD > z) AND it is not a holiday week (stores legitimately differ in holiday response).
  Flagged points are winsorised to +-clip robust sigmas (not deleted, not imputed from the future).
"""
import numpy as np
import pandas as pd
from .features import date_features

HOLIDAY_COLUMNS = ["th-1", "th0", "th1", "xm-2", "xm-1", "xm0", "xm1", "xm2", "xm3", "superbowl", "laborday", "easter"]


def protected_weeks(dates) -> np.ndarray:
    return (date_features(dates)[HOLIDAY_COLUMNS].sum(axis=1) > 0).to_numpy()


def robust_z(hist: pd.DataFrame):
    """Returns (z, R, med, scale): z of log-sales after removing store level and the common weekly movement."""
    L = np.log(hist); R = L - L.mean(); R = R.sub(R.median(axis=1), axis=0)
    med = R.median(); scale = 1.4826 * (R - med).abs().median()
    return (R - med) / scale, L, R, med, scale


def winsorize_anomalies(hist: pd.DataFrame, z: float = 5.0, clip: float = 3.5):
    """hist: weeks x stores (positive sales). Returns (cleaned, report DataFrame of what was changed)."""
    Z, L, R, med, scale = robust_z(hist)
    prot = protected_weeks(hist.index)
    flag = pd.DataFrame((Z.abs() > z).to_numpy() & (~prot)[:, None], index=hist.index, columns=hist.columns)
    if not flag.to_numpy().any():
        return hist.copy(), pd.DataFrame(columns=["Date", "Store", "z", "original", "cleaned"])
    Rc = med + np.sign(Z) * clip * scale                      # DataFrame: clipped residual
    Lc = L.where(~flag, L - R + Rc)                           # only flagged cells change
    clean = np.exp(Lc)
    rep = flag.stack(); rep = rep[rep].index.to_frame(index=False); rep.columns = ["Date", "Store"]
    rep["z"] = [float(Z.loc[d, s]) for d, s in zip(rep.Date, rep.Store)]
    rep["original"] = [float(hist.loc[d, s]) for d, s in zip(rep.Date, rep.Store)]
    rep["cleaned"] = [float(clean.loc[d, s]) for d, s in zip(rep.Date, rep.Store)]
    return clean, rep
