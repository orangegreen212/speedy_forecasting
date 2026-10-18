import joblib
import numpy as np
import pandas as pd
import pytest

from speedy.evaluate import BlendConfig
from speedy.inference import forecast_totals, forecast_weekly, load_artifact
from speedy.pipeline import build_pipeline, pipeline_from_config

CALIBRATION = {
    "store_week_mult": (0.95, 1.1),
    "weekly_total_err_q": (-0.01, 0.07),
    "annual_planning_range": (-0.01, 0.05),
}


def fit_pipe(panel, **kwargs):
    params = {"w_naive": 0.5, "w_ridge": 0.5, "w_prophet": 0.0, **kwargs}
    pipe = build_pipeline(**params)
    return pipe.fit(panel, panel["Weekly_Sales"])


def one_row(date, store, store_type):
    return pd.DataFrame(
        {"Date": [pd.Timestamp(date)], "Store": [store], "Type": [store_type]}
    )


def test_weights_must_sum_to_one(panel):
    with pytest.raises(ValueError, match="sum to 1"):
        build_pipeline(w_naive=0.5, w_ridge=0.5, w_prophet=0.5).fit(
            panel, panel["Weekly_Sales"]
        )


def test_blend_with_holt_runs_and_weights_are_validated(panel):
    pipe = fit_pipe(panel, w_naive=0.4, w_ridge=0.3, w_holt=0.3)
    assert pipe.predict(one_row("2012-11-02", 1, "A")).iloc[0] > 0
    with pytest.raises(ValueError):
        fit_pipe(panel, w_naive=0.4, w_ridge=0.3, w_holt=0.4)


def test_pipeline_predict_shape_positive_and_aligned(panel):
    pipe = fit_pipe(panel)
    future = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2012-11-02", "2012-11-09"] * 2),
            "Store": [1, 1, 4, 4],
            "Type": ["A", "A", "B", "B"],
        }
    )
    out = pipe.predict(future)
    assert len(out) == 4
    assert (out > 0).all()
    assert (out.index == future.index).all()


def test_unknown_store_raises(panel):
    pipe = fit_pipe(panel)
    with pytest.raises(ValueError, match="unknown stores"):
        pipe.predict(one_row("2012-11-02", 99, "A"))


def test_predict_before_fit_raises():
    pipe = build_pipeline()
    with pytest.raises(RuntimeError):
        pipe.named_steps["model"].predict(one_row("2012-11-02", 1, "A"))


def test_scrambling_the_future_changes_nothing(panel):
    origin = pd.Timestamp("2011-12-30")
    train = panel[panel["Date"] <= origin]
    future = panel[panel["Date"] > origin][["Store", "Date", "Type"]]
    before = fit_pipe(train).predict(future.reset_index(drop=True))

    scrambled = panel.copy()
    scrambled.loc[scrambled["Date"] > origin, "Weekly_Sales"] *= 7.0
    after = fit_pipe(scrambled[scrambled["Date"] <= origin]).predict(
        future.reset_index(drop=True)
    )
    assert np.array_equal(before.to_numpy(), after.to_numpy())


def test_model_never_sees_target_after_origin(panel):
    train = panel[panel["Date"] <= "2011-12-30"]
    model = fit_pipe(train).named_steps["model"]
    assert model.last_train_date_ == pd.Timestamp("2011-12-30")


def test_store_scale_multiplies_predictions(panel):
    plain = fit_pipe(panel)
    scaled = fit_pipe(panel, store_scale={1: 2.0})
    rows = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2012-11-02"] * 2),
            "Store": [1, 2],
            "Type": ["A", "A"],
        }
    )
    a, b = plain.predict(rows), scaled.predict(rows)
    assert np.isclose(b.iloc[0], 2 * a.iloc[0])
    assert np.isclose(b.iloc[1], a.iloc[1])


def test_pipeline_from_config_uses_the_weights():
    config = BlendConfig(0.1, "h0", "d8", 6, 0, 2, 2)
    model = pipeline_from_config(config).named_steps["model"]
    assert (model.w_naive, model.w_ridge) == (0.6, 0.0)
    assert (model.w_prophet, model.w_holt) == (0.2, 0.2)
    assert (model.holt_key, model.drift_key, model.alpha) == ("h0", "d8", 0.1)


def test_save_load_roundtrip_gives_identical_forecast(panel, tmp_path):
    artifact = {
        "pipeline": fit_pipe(panel),
        "metadata": {"trained_through": "x"},
        "calibration": CALIBRATION,
    }
    joblib.dump(artifact, tmp_path / "m.joblib")
    loaded = load_artifact(tmp_path / "m.joblib")
    pd.testing.assert_frame_equal(
        forecast_weekly(artifact, "2012-12-28"),
        forecast_weekly(loaded, "2012-12-28"),
    )


def test_load_artifact_rejects_incomplete_files(tmp_path):
    joblib.dump({"pipeline": None}, tmp_path / "bad.joblib")
    with pytest.raises(ValueError, match="missing"):
        load_artifact(tmp_path / "bad.joblib")


def test_forecast_schema_bands_and_dates(panel):
    artifact = {
        "pipeline": fit_pipe(panel),
        "metadata": {},
        "calibration": CALIBRATION,
    }
    weekly = forecast_weekly(artifact, "2012-12-28")
    assert list(weekly.columns) == [
        "Date", "Store", "Forecast", "Lower80", "Upper80",
    ]  # fmt: skip
    assert weekly["Date"].min() == pd.Timestamp("2012-11-02")
    assert weekly["Date"].max() == pd.Timestamp("2012-12-28")
    assert weekly["Store"].nunique() == 6
    inside = (weekly["Lower80"] < weekly["Forecast"]) & (
        weekly["Forecast"] < weekly["Upper80"]
    )
    assert inside.all()

    totals = forecast_totals(artifact, weekly)
    assert len(totals) == weekly["Date"].nunique()
    assert np.isclose(totals["Forecast"].sum(), weekly["Forecast"].sum())
    with pytest.raises(ValueError, match="end_date"):
        forecast_weekly(artifact, "2012-10-01")
