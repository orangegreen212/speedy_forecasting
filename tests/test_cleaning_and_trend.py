import numpy as np, pandas as pd, pytest
from speedy.cleaning import winsorize_anomalies
from speedy.data import load_panel, data_quality_report, to_wide
from speedy.features import date_features
from speedy.models import holt_fc, BlendForecaster, HOLT_GRID
from speedy.pipeline import build_pipeline


def wide_of(panel):
    return to_wide(panel)


def test_non_holiday_anomaly_is_winsorised_and_only_flagged_cells_change(
    panel,
):
    w = wide_of(panel)
    w2 = w.copy()
    d = pd.Timestamp("2011-09-02")
    w2.loc[d, 1] *= 0.4
    c, rep = winsorize_anomalies(w2)
    flagged = set(zip(rep.Date, rep.Store))
    assert (d, 1) in flagged and c.loc[d, 1] > w2.loc[d, 1]
    keep = np.ones(w2.shape, dtype=bool)
    for dd, ss in flagged:
        keep[w2.index.get_loc(dd), list(w2.columns).index(ss)] = False
    assert np.allclose(c.to_numpy()[keep], w2.to_numpy()[keep], rtol=1e-12)


def test_holiday_weeks_are_protected(panel):
    w = wide_of(panel).copy()
    d = pd.Timestamp("2011-11-25")
    w.loc[d, 2] *= 0.3  # Black Friday week
    c, rep = winsorize_anomalies(w)
    assert (d, 2) not in set(zip(rep.Date, rep.Store)) and np.allclose(
        c.loc[d, 2], w.loc[d, 2]
    )


def test_cleaning_uses_only_given_slice(panel):
    w = wide_of(panel)
    a, _ = winsorize_anomalies(w.iloc[:100])
    w2 = w.copy()
    w2.iloc[100:] *= 3
    b, _ = winsorize_anomalies(w2.iloc[:100])
    pd.testing.assert_frame_equal(a, b)


def test_holt_extends_a_declining_store_and_damps_it():
    idx = pd.date_range("2010-02-05", periods=120, freq="7D")
    types = pd.Series({1: "A", 2: "A"})
    hist = pd.DataFrame(
        {1: 1e5 * np.exp(-0.004 * np.arange(120)), 2: np.full(120, 1e5)},
        index=idx,
    )  # store 1 declines ~20%/yr
    fd = pd.date_range(idx[-1] + pd.Timedelta(days=7), periods=60, freq="7D")
    f = holt_fc(hist, types, date_features(idx), date_features(fd), "h2")
    assert (
        f[1].iloc[0] < hist[1].iloc[-1] * 1.05 and f[1].iloc[-1] < f[1].iloc[0]
    )  # keeps falling...
    assert f[1].iloc[-1] > hist[1].iloc[-1] * np.exp(
        -0.004 * 60
    )  # ...but less than the undamped path
    assert abs(f[2].iloc[-1] / 1e5 - 1) < 0.05  # a flat store stays flat


def test_blend_with_holt_runs_and_weights_validated(panel):
    p = build_pipeline(
        w_naive=0.4, w_ridge=0.3, w_prophet=0.0, w_holt=0.3
    ).fit(panel, panel.Weekly_Sales)
    out = p.predict(
        pd.DataFrame(
            {"Date": [pd.Timestamp("2012-11-02")], "Store": [1], "Type": ["A"]}
        )
    )
    assert out.iloc[0] > 0
    with pytest.raises(ValueError):
        build_pipeline(
            w_naive=0.4, w_ridge=0.3, w_prophet=0.0, w_holt=0.4
        ).fit(panel, panel.Weekly_Sales)


def test_returns_policy_and_quality_report(tmp_path):
    dates = pd.date_range("2010-02-05", periods=6, freq="7D").strftime(
        "%d/%m/%Y"
    )
    rows = (
        [(1, 1, d, 100.0, False) for d in dates]
        + [(1, 2, dates[0], -10.0, False)]
        + [(1, 2, d, 50.0, False) for d in dates[1:]]
    )
    pd.DataFrame(
        rows, columns=["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"]
    ).to_csv(tmp_path / "sales.csv", index=False)
    pd.DataFrame({"Store": [1], "Type": ["A"], "Size": [1]}).to_csv(
        tmp_path / "stores.csv", index=False
    )
    pd.DataFrame(
        {
            "Store": [1],
            "Date": ["05/02/2010"],
            "Temperature": [1.0],
            "IsHoliday": [False],
        }
    ).to_csv(tmp_path / "features.csv", index=False)
    net, clip, drop = (
        load_panel(tmp_path, r).Weekly_Sales.sum()
        for r in ("net", "clip", "drop")
    )
    assert net == 100 * 6 + 50 * 5 - 10 and clip == net + 10 and drop == clip
    with pytest.raises(ValueError):
        load_panel(tmp_path, "bogus")


def test_drift_pulls_a_declining_store_down_and_d0_equals_plain_naive():
    from speedy.models import naive_fc, naive_drift_fc

    idx = pd.date_range("2010-02-05", periods=110, freq="7D")
    types = pd.Series({1: "A", 2: "A"})
    hist = pd.DataFrame(
        {1: 1e5 * np.exp(-0.004 * np.arange(110)), 2: np.full(110, 1e5)},
        index=idx,
    )
    fd = pd.date_range(idx[-1] + pd.Timedelta(days=7), periods=30, freq="7D")
    plain = naive_fc(hist, fd)
    d0 = naive_drift_fc(hist, types, fd, 0.0)
    d8 = naive_drift_fc(hist, types, fd, 0.8)
    assert (
        np.array_equal(plain.to_numpy(), d0.to_numpy())
        and (d8[1] < plain[1]).all()
    )
    assert np.allclose(d8[2], plain[2], rtol=0.15)
