import os, numpy as np, pandas as pd, pytest, joblib
from speedy.models import naive_fc, BlendForecaster
from speedy.pipeline import build_pipeline
from speedy.data import to_wide, validate_panel
from speedy.inference import forecast_weekly, forecast_totals, load_artifact


def fit_pipe(panel, **kw):
    p = build_pipeline(w_naive=0.5, w_ridge=0.5, w_prophet=0.0, **kw); p.fit(panel, panel["Weekly_Sales"]); return p


def test_naive_uses_same_week_last_year(panel):
    wide = to_wide(panel)
    f = naive_fc(wide.iloc[:100], [wide.index[60] + pd.Timedelta(days=364)])
    assert np.allclose(f.iloc[0].to_numpy(), wide.iloc[60].to_numpy())


def test_naive_beyond_one_year_rolls_back_whole_years(panel):
    wide = to_wide(panel).iloc[:100]; d = wide.index[-1] + pd.Timedelta(days=364 * 2)
    f = naive_fc(wide, [d]); assert np.allclose(f.iloc[0].to_numpy(), wide.loc[d - pd.Timedelta(days=364 * 2)].to_numpy())


def test_weights_must_sum_to_one(panel):
    with pytest.raises(ValueError):
        build_pipeline(w_naive=0.5, w_ridge=0.5, w_prophet=0.5).fit(panel, panel["Weekly_Sales"])


def test_pipeline_predict_shape_positive_and_aligned(panel):
    p = fit_pipe(panel)
    fut = pd.DataFrame({"Date": pd.to_datetime(["2012-11-02", "2012-11-09"] * 2), "Store": [1, 1, 4, 4], "Type": ["A", "A", "B", "B"]})
    out = p.predict(fut)
    assert len(out) == 4 and (out > 0).all() and (out.index == fut.index).all()


def test_unknown_store_raises(panel):
    p = fit_pipe(panel)
    with pytest.raises(ValueError):
        p.predict(pd.DataFrame({"Date": [pd.Timestamp("2012-11-02")], "Store": [99], "Type": ["A"]}))


def test_no_leakage_scrambling_future_changes_nothing(panel):
    origin = pd.Timestamp("2011-12-30"); tr = panel[panel.Date <= origin]
    fut = panel[(panel.Date > origin)][["Store", "Date", "Type"]].reset_index(drop=True)
    a = fit_pipe(tr).predict(fut)
    scrambled = panel.copy(); scrambled.loc[scrambled.Date > origin, "Weekly_Sales"] *= 7.0
    b = fit_pipe(scrambled[scrambled.Date <= origin]).predict(fut)
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_model_never_sees_target_after_origin(panel):
    tr = panel[panel.Date <= "2011-12-30"]; p = fit_pipe(tr)
    assert p.named_steps["model"].last_train_date_ == pd.Timestamp("2011-12-30")


def test_save_load_roundtrip_identical_forecast(panel, tmp_path):
    p = fit_pipe(panel); art = {"pipeline": p, "metadata": {"trained_through": "x"},
                                "calibration": {"store_week_mult": (0.95, 1.1), "weekly_total_err_q": (-0.01, 0.07), "annual_planning_range": (-0.01, 0.05)}}
    joblib.dump(art, tmp_path / "m.joblib"); art2 = load_artifact(tmp_path / "m.joblib")
    a = forecast_weekly(art, "2012-12-28"); b = forecast_weekly(art2, "2012-12-28")
    pd.testing.assert_frame_equal(a, b)


def test_forecast_schema_bands_and_dates(panel):
    art = {"pipeline": fit_pipe(panel), "metadata": {}, "calibration": {"store_week_mult": (0.95, 1.1), "weekly_total_err_q": (-0.01, 0.07), "annual_planning_range": (-0.01, 0.05)}}
    w = forecast_weekly(art, "2012-12-28")
    assert list(w.columns) == ["Date", "Store", "Forecast", "Lower80", "Upper80"]
    assert w.Date.min() == pd.Timestamp("2012-11-02") and w.Date.max() == pd.Timestamp("2012-12-28")
    assert ((w.Lower80 < w.Forecast) & (w.Forecast < w.Upper80)).all() and w.Store.nunique() == 6
    t = forecast_totals(art, w); assert len(t) == w.Date.nunique() and np.isclose(t.Forecast.sum(), w.Forecast.sum())
    with pytest.raises(ValueError):
        forecast_weekly(art, "2012-10-01")


def test_validate_panel_catches_problems(panel):
    with pytest.raises(ValueError): validate_panel(pd.concat([panel, panel.iloc[:1]]))
    bad = panel.copy(); bad.loc[0, "Weekly_Sales"] = -1
    with pytest.raises(ValueError): validate_panel(bad)
    with pytest.raises(ValueError): validate_panel(panel.iloc[1:])


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="Prophet fit is slow; set RUN_SLOW=1")
@pytest.mark.slow
def test_prophet_blend_runs(panel):
    p = build_pipeline(w_naive=0.4, w_ridge=0.3, w_prophet=0.3); p.fit(panel[panel.Store <= 2], panel[panel.Store <= 2].Weekly_Sales)
    out = p.predict(pd.DataFrame({"Date": [pd.Timestamp("2012-11-02")], "Store": [1], "Type": ["A"]}))
    assert out.iloc[0] > 0
