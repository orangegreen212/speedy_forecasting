from sklearn.pipeline import Pipeline
from .models import FeatureBuilder, BlendForecaster


def build_pipeline(**model_params) -> Pipeline:
    """features (stateless, build_features) -> model (blend of naive / ridge / Prophet)."""
    return Pipeline([("features", FeatureBuilder()), ("model", BlendForecaster(**model_params))])
