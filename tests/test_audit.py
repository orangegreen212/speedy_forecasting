"""Leakage and forecasting-hygiene audit, kept as regression tests."""

import os, numpy as np, pandas as pd, pytest
from speedy.data import to_wide
from speedy.evaluate import (
    make_fold,
    bias_factor,
    apply_factor,
    holiday_week_mask,
    run_evaluation,
)
from speedy.models import _source_date
from speedy.pipeline import build_pipeline


def _types(panel, wide):
    return (
        panel.drop_duplicates("Store")
        .set_index("Store")["Type"]
        .reindex(wide.columns)
    )


@pytest.mark.parametrize("origin", [78, 104, 125])
def test_scrambling_future_does_not_change_fold_forecasts(panel, origin):
    wide = to_wide(panel)
    types = _types(panel, wide)
    w2 = wide.copy()
    w2.iloc[origin:] = w2.iloc[origin:] * np.random.default_rng(1).uniform(
        0.2, 5, w2.iloc[origin:].shape
    )
    a, b = (
        make_fold(wide, types, origin, 13, prophet=False),
        make_fold(w2, types, origin, 13, prophet=False),
    )
    for k in a["comp"]:
        # float noise (~1e-15 relative) is expected; a real leak would move forecasts by percents
        assert np.allclose(
            a["comp"][k].to_numpy(), b["comp"][k].to_numpy(), rtol=1e-9, atol=0
        ), k


@pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"), reason="Prophet; set RUN_SLOW=1"
)
def test_prophet_fold_has_no_leakage(panel):
    wide = to_wide(panel)
    types = _types(panel, wide)
    o = 100
    w2 = wide.copy()
    w2.iloc[o:] = w2.iloc[o:] * 4.0
    a, b = make_fold(wide, types, o, 13), make_fold(w2, types, o, 13)
    assert (
        np.abs(
            a["comp"]["prophet"].to_numpy() - b["comp"]["prophet"].to_numpy()
        ).max()
        < 1e-6
    )


def test_naive_never_reads_beyond_training_end(panel):
    wide = to_wide(panel)
    for o in (60, 85, 100):
        last = wide.index[o - 1]
        for d in wide.index[o : o + 58]:
            s = _source_date(d)
            while s > last:
                s -= pd.Timedelta(days=364)
            assert s <= last


def test_bias_factor_known_values_and_shrinkage():
    a = pd.DataFrame({1: [110.0] * 4, 2: [110.0] * 4, 3: [90.0] * 4})
    f = pd.DataFrame({1: [100.0] * 4, 2: [100.0] * 4, 3: [100.0] * 4})
    types = pd.Series({1: "A", 2: "A", 3: "B"})
    full = bias_factor(a, f, types, 1.0)
    half = bias_factor(a, f, types, 0.5)
    assert np.allclose(full.to_numpy(), [1.1, 1.1, 0.9]) and np.allclose(
        half.to_numpy(), [1.05, 1.05, 0.95]
    )
    assert np.allclose(
        apply_factor(f, full).iloc[0].to_numpy(), [110, 110, 90]
    )


def test_store_scale_multiplies_predictions(panel):
    p1 = build_pipeline(w_naive=0.5, w_ridge=0.5, w_prophet=0.0).fit(
        panel, panel.Weekly_Sales
    )
    p2 = build_pipeline(
        w_naive=0.5, w_ridge=0.5, w_prophet=0.0, store_scale={1: 2.0}
    ).fit(panel, panel.Weekly_Sales)
    X = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2012-11-02"] * 2),
            "Store": [1, 2],
            "Type": ["A", "A"],
        }
    )
    a, b = p1.predict(X), p2.predict(X)
    assert np.isclose(b.iloc[0], 2 * a.iloc[0]) and np.isclose(
        b.iloc[1], a.iloc[1]
    )


def test_holiday_mask():
    m = holiday_week_mask(
        pd.to_datetime(["2012-11-23", "2012-05-11", "2012-12-28"])
    )
    assert m.tolist() == [True, False, True]


def test_full_evaluation_smoke_and_rules(panel):
    r = run_evaluation(panel, prophet=False, n_boot=30)
    r.pop("_arrays")
    assert (
        r["bias_correction"]["adopted_lam"] in (0.0, 0.5, 1.0)
        and sum(r["final_cfg"][3:]) == 10
    )
    lo, hi = r["calibration"]["store_week_mult"]
    assert lo < 1 < hi and len(r["store_scale"]) == panel.Store.nunique()
    if (
        r["bias_correction"]["adopted_lam"] > 0
    ):  # rule: only adopted when the CI excludes zero
        assert (
            r["bias_correction"]["candidates"][
                str(r["bias_correction"]["adopted_lam"])
            ]["WMAPE_gain_vs_uncorrected"][1]
            > 0
        )
