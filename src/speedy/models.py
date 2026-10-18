"""scikit-learn components: the feature step and the blended forecaster."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from .cleaning import winsorize_anomalies
from .features import FEATURE_COLUMNS, build_features
from .forecasters import (
    DRIFT_GRID,
    holt_fc,
    naive_drift_fc,
    prophet_fc,
    ridge_fc,
)


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Pipeline step 1: a stateless wrapper around `build_features`."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return build_features(X)


class BlendForecaster(BaseEstimator, RegressorMixin):
    """Pipeline step 2: a weighted blend of the base forecasters.

    fit(X, y): X is the feature table (Store, Date, Type, features) and y
    the weekly sales. The table must contain every store in every week.
    predict(X): forecasts for any future dates of the known stores, in the
    row order of X.
    """

    def __init__(
        self,
        w_naive=0.5,
        w_ridge=0.2,
        w_prophet=0.3,
        w_holt=0.0,
        alpha=10.0,
        holt_key="h1",
        drift_key="d0",
        fourier=5,
        cps=0.05,
        n_jobs=1,
        clean_z=5.0,
        store_scale=None,
    ):
        self.w_naive = w_naive
        self.w_ridge = w_ridge
        self.w_prophet = w_prophet
        self.w_holt = w_holt
        self.alpha = alpha
        self.holt_key = holt_key
        self.drift_key = drift_key
        self.fourier = fourier
        self.cps = cps
        self.n_jobs = n_jobs
        self.clean_z = clean_z
        self.store_scale = store_scale

    def _check_weights(self):
        weights = [self.w_naive, self.w_ridge, self.w_prophet, self.w_holt]
        if min(weights) < 0 or abs(sum(weights) - 1) > 1e-9:
            raise ValueError(
                f"blend weights must be >=0 and sum to 1, got {weights}"
            )

    def fit(self, X, y):
        self._check_weights()
        data = X[["Store", "Date"]].copy()
        data["y"] = np.asarray(y, dtype=float)
        history = data.pivot(
            index="Date", columns="Store", values="y"
        ).sort_index()
        if history.isna().any().any():
            raise ValueError(
                "training data must be rectangular (every store, every week)"
            )
        self.raw_history_ = history
        if self.clean_z:
            history, self.anomalies_ = winsorize_anomalies(
                history, z=self.clean_z
            )
        self.history_ = history
        self.types_ = (
            X.drop_duplicates("Store")
            .set_index("Store")["Type"]
            .reindex(history.columns)
        )
        self.train_features_ = (
            X.drop_duplicates("Date")
            .set_index("Date")[FEATURE_COLUMNS]
            .sort_index()
        )
        self.last_train_date_ = history.index[-1]
        return self

    def components(self, X) -> dict:
        """Forecast table of every base model that has a positive weight."""
        future = (
            X.drop_duplicates("Date")
            .set_index("Date")[FEATURE_COLUMNS]
            .sort_index()
        )
        train = self.train_features_
        components = {}
        if self.w_naive > 0:
            components["naive"] = naive_drift_fc(
                self.history_,
                self.types_,
                future.index,
                DRIFT_GRID[self.drift_key],
            )
        if self.w_ridge > 0:
            components["ridge"] = ridge_fc(
                self.history_, self.types_, train, future, self.alpha
            )
        if self.w_prophet > 0:
            components["prophet"] = prophet_fc(
                self.history_,
                future.index,
                self.fourier,
                self.cps,
                self.n_jobs,
            )
        if self.w_holt > 0:
            components["holt"] = holt_fc(
                self.history_,
                self.types_,
                train,
                future,
                self.holt_key,
                self.alpha,
            )
        return components

    def predict(self, X):
        if not hasattr(self, "history_"):
            raise RuntimeError("fit first")
        unknown = set(X["Store"]) - set(self.history_.columns)
        if unknown:
            raise ValueError(f"unknown stores: {sorted(unknown)}")

        weights = {
            "naive": self.w_naive,
            "ridge": self.w_ridge,
            "prophet": self.w_prophet,
            "holt": self.w_holt,
        }
        blend = sum(
            weights[name] * forecast
            for name, forecast in self.components(X).items()
        )
        if self.store_scale:
            scale = pd.Series(self.store_scale).reindex(blend.columns)
            blend = blend * scale.fillna(1.0).to_numpy()

        rows = blend.index.get_indexer(pd.to_datetime(X["Date"]))
        cols = blend.columns.get_indexer(X["Store"])
        values = blend.to_numpy()[rows, cols]
        if np.isnan(values).any():
            raise ValueError("NaN forecast (date too far before history?)")
        return pd.Series(values, index=X.index, name="Forecast")
