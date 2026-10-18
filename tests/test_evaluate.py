import numpy as np
import pandas as pd
import pytest

from speedy.data import store_types, to_wide
from speedy.evaluate import (
    BlendConfig,
    apply_factor,
    bias_factor,
    candidate_cfgs,
    holiday_week_mask,
    make_fold,
    run_evaluation,
    select_cfg,
    walk_forward,
)
from speedy.forecasters import _same_week_last_year
from speedy.metrics import (
    block_bootstrap,
    coverage,
    interval_multipliers,
    metrics,
)


def simple_frames():
    actual = pd.DataFrame(
        {"s1": [100.0, 200.0, 100.0, 200.0], "s2": [100.0] * 4}
    )
    return actual, actual * 1.1


def test_metrics_known_values():
    actual, forecast = simple_frames()
    result = metrics(actual, forecast)
    assert np.isclose(result["WMAPE"], 0.1)
    assert np.isclose(result["Bias"], 0.1)
    assert np.isclose(result["MAPE"], 0.1)


def test_negative_bias_means_forecast_too_low():
    actual, forecast = simple_frames()
    assert metrics(actual, forecast * 0.5)["Bias"] < 0


def test_bootstrap_is_deterministic_and_contains_the_point_estimate():
    rng = np.random.default_rng(1)
    actual = pd.DataFrame(rng.uniform(90, 110, (40, 3)))
    forecast = actual * rng.uniform(0.9, 1.1, actual.shape)
    first = block_bootstrap(actual, {"m": forecast}, n=200, seed=3)
    second = block_bootstrap(actual, {"m": forecast}, n=200, seed=3)
    assert first == second
    point, low, high = first["m"]
    assert low <= point <= high


def test_paired_difference_is_zero_for_identical_models():
    actual, _ = simple_frames()
    actual = pd.concat([actual] * 5, ignore_index=True)
    forecast = actual * 1.1
    result = block_bootstrap(actual, {"x": forecast}, n=50, diff_against="x")
    assert result["x"] == (0.0, 0.0, 0.0)


def test_candidate_weights_sum_to_ten_and_switches_work():
    assert all(sum(c[3:]) == 10 for c in candidate_cfgs())
    assert all(c.w_prophet == 0 for c in candidate_cfgs(prophet=False))
    no_trend = candidate_cfgs(allow_trend=False)
    assert all(c.w_holt == 0 and c.drift_key == "d0" for c in no_trend)


def test_config_to_list_is_json_friendly():
    config = BlendConfig(1.0, "h1", "d5", 3, 2, 3, 2)
    assert config.to_list() == [1.0, "h1", "d5", 3, 2, 3, 2]


def test_interval_multipliers_order_and_coverage():
    rng = np.random.default_rng(0)
    forecast = pd.DataFrame(rng.uniform(90, 110, (200, 5)))
    actual = forecast * rng.lognormal(0, 0.05, forecast.shape)
    low, high = interval_multipliers(actual, forecast)
    assert low < 1 < high
    assert abs(coverage(actual, forecast, (low, high)) - 0.8) < 0.03


def test_bias_factor_known_values_and_shrinkage():
    actual = pd.DataFrame({1: [110.0] * 4, 2: [110.0] * 4, 3: [90.0] * 4})
    forecast = pd.DataFrame({1: [100.0] * 4, 2: [100.0] * 4, 3: [100.0] * 4})
    types = pd.Series({1: "A", 2: "A", 3: "B"})
    full = bias_factor(actual, forecast, types, 1.0)
    half = bias_factor(actual, forecast, types, 0.5)
    assert np.allclose(full.to_numpy(), [1.1, 1.1, 0.9])
    assert np.allclose(half.to_numpy(), [1.05, 1.05, 0.95])
    assert np.allclose(
        apply_factor(forecast, full).iloc[0].to_numpy(), [110, 110, 90]
    )


def test_holiday_mask():
    mask = holiday_week_mask(
        pd.to_datetime(["2012-11-23", "2012-05-11", "2012-12-28"])
    )
    assert mask.tolist() == [True, False, True]


@pytest.mark.parametrize("origin", [78, 104, 125])
def test_scrambling_the_future_does_not_change_fold_forecasts(panel, origin):
    wide, types = to_wide(panel), store_types(panel)
    scrambled = wide.copy()
    noise = np.random.default_rng(1).uniform(
        0.2, 5, scrambled.iloc[origin:].shape
    )
    scrambled.iloc[origin:] = scrambled.iloc[origin:] * noise
    a = make_fold(wide, types, origin, 13, prophet=False)
    b = make_fold(scrambled, types, origin, 13, prophet=False)
    for name in a.components:
        # float noise (~1e-15) is fine; a leak would move forecasts by percents
        assert np.allclose(
            a.components[name].to_numpy(),
            b.components[name].to_numpy(),
            rtol=1e-9,
            atol=0,
        ), name


def test_naive_never_reads_beyond_training_end(panel):
    wide = to_wide(panel)
    for origin in (60, 85, 100):
        last = wide.index[origin - 1]
        for date in wide.index[origin : origin + 58]:
            source = _same_week_last_year(date)
            while source > last:
                source -= pd.Timedelta(days=364)
            assert source <= last


def test_walk_forward_uses_only_earlier_folds(panel):
    wide, types = to_wide(panel), store_types(panel)
    origins = range(78, 143, 13)
    folds = [make_fold(wide, types, o, 13, prophet=False) for o in origins]
    _, _, configs = walk_forward(folds, prophet=False)
    assert len(configs) == len(folds)
    for j in range(1, len(folds)):
        assert configs[j] == select_cfg(folds[:j], prophet=False)


def test_full_evaluation_smoke_and_adoption_rule(panel):
    result = run_evaluation(panel, prophet=False, n_boot=30)
    result.pop("_arrays")
    study = result["bias_correction"]
    assert study["adopted_lam"] in (0.0, 0.5, 1.0)
    assert sum(result["final_cfg"][3:]) == 10
    low, high = result["calibration"]["store_week_mult"]
    assert low < 1 < high
    assert len(result["store_scale"]) == panel["Store"].nunique()
    if study["adopted_lam"] > 0:  # adopted only when the CI excludes zero
        key = str(study["adopted_lam"])
        assert study["candidates"][key]["WMAPE_gain_vs_uncorrected"][1] > 0
