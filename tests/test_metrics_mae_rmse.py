import numpy as np
import pandas as pd

from speedy.metrics import metrics


def test_mae_and_rmse_known_values():
    actual = pd.DataFrame({"a": [10.0, 10.0], "b": [10.0, 10.0]})
    forecast = pd.DataFrame({"a": [12.0, 10.0], "b": [10.0, 6.0]})
    result = metrics(actual, forecast)
    assert result["MAE"] == np.mean([2, 0, 0, 4])
    assert np.isclose(result["RMSE"], np.sqrt(np.mean([4, 0, 0, 16])))
    assert result["RMSE"] >= result["MAE"]
