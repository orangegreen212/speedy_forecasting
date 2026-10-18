import pandas as pd
import pytest

from speedy.features import (
    FEATURE_COLUMNS,
    build_features,
    date_features,
    thanksgiving,
    week_end,
)


def test_thanksgiving_dates():
    assert thanksgiving(2012) == pd.Timestamp("2012-11-22")
    assert thanksgiving(2013) == pd.Timestamp("2013-11-28")


def test_week_end_is_friday():
    assert week_end(pd.Timestamp("2012-11-22")) == pd.Timestamp("2012-11-23")


def test_contract_columns_and_no_nan(panel):
    features = build_features(panel[["Store", "Date", "Type"]])
    assert list(features.columns) == ["Store", "Date", "Type"] + (
        FEATURE_COLUMNS
    )
    assert not features[FEATURE_COLUMNS].isna().any().any()
    assert len(features) == len(panel)


def test_thanksgiving_flag_on_correct_week():
    f = date_features(["2012-11-16", "2012-11-23", "2012-11-30"])
    assert f.loc["2012-11-23", "th0"] == 1
    assert f.loc["2012-11-16", "th0"] == 0
    assert f.loc["2012-11-16", "th-1"] == 1
    assert f.loc["2012-11-30", "th1"] == 1


def test_fixed_holidays_fall_on_the_expected_weeks():
    f = date_features(["2012-02-10", "2012-09-07", "2012-04-06"])
    assert f.loc["2012-02-10", "superbowl"] == 1
    assert f.loc["2012-09-07", "laborday"] == 1
    assert f.loc["2012-04-06", "easter"] == 1


def test_features_are_pure_row_wise(panel):
    """A row's features must not depend on the other rows that are passed."""
    columns = ["Store", "Date"]
    full = build_features(panel[columns])
    part = build_features(panel[columns].iloc[50:60])
    pd.testing.assert_frame_equal(
        full.iloc[50:60][FEATURE_COLUMNS], part[FEATURE_COLUMNS]
    )


def test_input_not_mutated_and_order_kept(panel):
    shuffled = panel[["Store", "Date"]].sample(frac=1, random_state=1)
    before = shuffled.copy()
    out = build_features(shuffled)
    pd.testing.assert_frame_equal(shuffled, before)
    assert (out.index == shuffled.index).all()


def test_missing_column_raises(panel):
    with pytest.raises(ValueError):
        build_features(panel[["Store"]])
