"""Base forecasters (seasonal naive, pooled ridge, Prophet) and the blended sklearn estimator."""
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import Ridge
from .features import FEATURE_COLUMNS, SEASONAL_COLUMNS, thanksgiving, holiday_frame, build_features


# --------------------------------------------------------------------------- base models (weeks x stores in/out)
def _source_date(d: pd.Timestamp, k: int = 1) -> pd.Timestamp:
    """Same week k years earlier (364*k days), re-aligned around Thanksgiving, which drifts by a week."""
    base = d - pd.Timedelta(days=364 * k)
    off = (thanksgiving(d.year) - d).days
    if -14 <= off <= 21:
        cand = thanksgiving(d.year - k) - pd.Timedelta(days=off)
        cand = cand + pd.Timedelta(days=(d.weekday() - cand.weekday()) % 7)
        if abs((cand - base).days) <= 7:
            return cand
    return base


def naive_fc(hist: pd.DataFrame, dates) -> pd.DataFrame:
    out = pd.DataFrame(index=pd.DatetimeIndex(dates), columns=hist.columns, dtype=float)
    for d in out.index:
        s = _source_date(d)
        while s > hist.index[-1]:          # beyond one year ahead: roll back by whole 52-week years
            s -= pd.Timedelta(days=364)
        if s in hist.index:
            out.loc[d] = hist.loc[s].to_numpy()
    return out


def ridge_fc(hist, types, Xtr, Xf, alpha=10.0, use_trend=True) -> pd.DataFrame:
    """Per store type: pooled seasonal/holiday shape (ridge on log sales) + store level + shrunk, frozen store trend."""
    out = pd.DataFrame(index=Xf.index, columns=hist.columns, dtype=float)
    for g in types.unique():
        cols = [c for c in hist.columns if types[c] == g]
        L = np.log(hist[cols]); lvl = L.mean(); R = L - lvl
        m = Ridge(alpha=alpha).fit(Xtr[SEASONAL_COLUMNS].to_numpy(), R.mean(axis=1).to_numpy())
        shape_tr = m.predict(Xtr[SEASONAL_COLUMNS].to_numpy()); shape_f = m.predict(Xf[SEASONAL_COLUMNS].to_numpy())
        tt = Xtr["trend"].to_numpy(); ttf = np.minimum(Xf["trend"].to_numpy(), tt.max())
        for c in cols:
            res = R[c].to_numpy() - shape_tr
            if use_trend:
                slope = np.clip(np.polyfit(tt, res, 1)[0], -0.15, 0.15) * 0.5
                a = res.mean() - slope * tt.mean()
                out[c] = np.exp(lvl[c] + shape_f + a + slope * ttf)
            else:
                out[c] = np.exp(lvl[c] + shape_f + res.mean())
    return out


def _fit_one_prophet(args):
    c, ds, yv, fd, hol, fourier, cps = args
    from prophet import Prophet
    for n in ("cmdstanpy", "prophet"):
        logging.getLogger(n).setLevel(logging.ERROR)
    m = Prophet(growth="linear", yearly_seasonality=fourier, weekly_seasonality=False, daily_seasonality=False,
                holidays=hol, changepoint_prior_scale=cps)
    m.fit(pd.DataFrame({"ds": ds, "y": np.log(yv)}))
    return c, np.exp(m.predict(pd.DataFrame({"ds": fd}))["yhat"].to_numpy())


def prophet_fc(hist, dates, fourier=5, cps=0.05, n_jobs=1) -> pd.DataFrame:
    from joblib import Parallel, delayed
    dates = pd.DatetimeIndex(dates); hol = holiday_frame()
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_one_prophet)((c, hist.index, hist[c].to_numpy(), dates, hol, fourier, cps))
                                  for c in hist.columns)
    return pd.DataFrame({c: v for c, v in res}, index=dates)[hist.columns]


# --------------------------------------------------------------------------- sklearn components
class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Pipeline step 1: stateless wrapper around build_features()."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return build_features(X)


class BlendForecaster(BaseEstimator, RegressorMixin):
    """Pipeline step 2. fit(X, y): X = feature frame (Store, Date, Type, FEATURE_COLUMNS), y = weekly sales.
    predict(X) returns forecasts aligned to the rows of X (any future dates, known stores)."""

    def __init__(self, w_naive=0.5, w_ridge=0.2, w_prophet=0.3, alpha=10.0, use_trend=True,
                 fourier=5, cps=0.05, n_jobs=1):
        self.w_naive = w_naive; self.w_ridge = w_ridge; self.w_prophet = w_prophet; self.alpha = alpha
        self.use_trend = use_trend; self.fourier = fourier; self.cps = cps; self.n_jobs = n_jobs

    def _check_weights(self):
        w = [self.w_naive, self.w_ridge, self.w_prophet]
        if min(w) < 0 or abs(sum(w) - 1) > 1e-9:
            raise ValueError(f"blend weights must be >=0 and sum to 1, got {w}")

    def fit(self, X, y):
        self._check_weights()
        d = X[["Store", "Date"]].copy(); d["y"] = np.asarray(y, dtype=float)
        hist = d.pivot(index="Date", columns="Store", values="y").sort_index()
        if hist.isna().any().any():
            raise ValueError("training data must be rectangular (every store, every week)")
        self.history_ = hist
        self.types_ = X.drop_duplicates("Store").set_index("Store")["Type"].reindex(hist.columns)
        self.train_features_ = X.drop_duplicates("Date").set_index("Date")[FEATURE_COLUMNS].sort_index()
        self.last_train_date_ = hist.index[-1]
        return self

    def components(self, X) -> dict:
        Xf = X.drop_duplicates("Date").set_index("Date")[FEATURE_COLUMNS].sort_index()
        comp = {"naive": naive_fc(self.history_, Xf.index),
                "ridge": ridge_fc(self.history_, self.types_, self.train_features_, Xf, self.alpha, self.use_trend)}
        if self.w_prophet > 0:
            comp["prophet"] = prophet_fc(self.history_, Xf.index, self.fourier, self.cps, self.n_jobs)
        return comp

    def predict(self, X):
        if not hasattr(self, "history_"):
            raise RuntimeError("fit first")
        unknown = set(X["Store"]) - set(self.history_.columns)
        if unknown:
            raise ValueError(f"unknown stores: {sorted(unknown)}")
        comp = self.components(X)
        wide = self.w_naive * comp["naive"] + self.w_ridge * comp["ridge"]
        if "prophet" in comp:
            wide = wide + self.w_prophet * comp["prophet"]
        wide = wide.rename_axis(index="Date", columns="Store")
        s = wide.reset_index().melt(id_vars="Date", var_name="Store", value_name="yhat").set_index(["Date", "Store"])["yhat"]
        vals = s.reindex(pd.MultiIndex.from_arrays([pd.to_datetime(X["Date"]), X["Store"]])).to_numpy()
        if np.isnan(vals).any():
            raise ValueError("NaN forecast (date too far before history?)")
        return pd.Series(vals, index=X.index, name="Forecast")
