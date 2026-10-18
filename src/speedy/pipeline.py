"""The model pipeline: calendar features followed by the blended model."""

from sklearn.pipeline import Pipeline

from .models import BlendForecaster, FeatureBuilder

TENTHS = 10


def build_pipeline(**model_params) -> Pipeline:
    """Features (stateless) -> blend of the base forecasters."""
    return Pipeline(
        [
            ("features", FeatureBuilder()),
            ("model", BlendForecaster(**model_params)),
        ]
    )


def pipeline_from_config(config, store_scale=None) -> Pipeline:
    """Pipeline for a `BlendConfig` (weights in tenths, see evaluate.py)."""
    return build_pipeline(
        alpha=config.ridge_alpha,
        holt_key=config.holt_key,
        drift_key=config.drift_key,
        w_naive=config.w_naive / TENTHS,
        w_ridge=config.w_ridge / TENTHS,
        w_prophet=config.w_prophet / TENTHS,
        w_holt=config.w_holt / TENTHS,
        store_scale=store_scale,
    )
