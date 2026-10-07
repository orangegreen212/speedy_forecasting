import pandas as pd, pytest
from speedy.features import build_features, date_features, thanksgiving, FEATURE_COLUMNS, week_end


def test_thanksgiving_dates():
    assert thanksgiving(2012) == pd.Timestamp("2012-11-22") and thanksgiving(2013) == pd.Timestamp("2013-11-28")


def test_week_end_is_friday():
    assert week_end(pd.Timestamp("2012-11-22")) == pd.Timestamp("2012-11-23")


def test_contract_columns_and_no_nan(panel):
    f = build_features(panel[["Store", "Date", "Type"]])
    assert list(f.columns) == ["Store", "Date", "Type"] + FEATURE_COLUMNS
    assert not f[FEATURE_COLUMNS].isna().any().any() and len(f) == len(panel)


def test_thanksgiving_flag_on_correct_week():
    f = date_features(["2012-11-16", "2012-11-23", "2012-11-30"])
    assert f.loc["2012-11-23", "th0"] == 1 and f.loc["2012-11-16", "th0"] == 0
    assert f.loc["2012-11-16", "th-1"] == 1 and f.loc["2012-11-30", "th1"] == 1


def test_features_are_pure_row_wise(panel):
    """A row's features must not depend on which other rows are passed (no hidden state, no target)."""
    full = build_features(panel[["Store", "Date"]]); part = build_features(panel[["Store", "Date"]].iloc[50:60])
    pd.testing.assert_frame_equal(full.iloc[50:60][FEATURE_COLUMNS], part[FEATURE_COLUMNS])


def test_input_not_mutated_and_order_kept(panel):
    x = panel[["Store", "Date"]].sample(frac=1, random_state=1); before = x.copy()
    out = build_features(x)
    pd.testing.assert_frame_equal(x, before); assert (out.index == x.index).all()


def test_missing_column_raises(panel):
    with pytest.raises(ValueError):
        build_features(panel[["Store"]])
