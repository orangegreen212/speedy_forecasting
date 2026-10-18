import numpy as np
import pandas as pd

from speedy.cleaning import winsorize_anomalies
from speedy.data import to_wide


def test_non_holiday_anomaly_is_clipped_and_only_it_changes(panel):
    wide = to_wide(panel)
    damaged = wide.copy()
    date = pd.Timestamp("2011-09-02")
    damaged.loc[date, 1] *= 0.4

    cleaned, report = winsorize_anomalies(damaged)

    flagged = set(zip(report.Date, report.Store, strict=True))
    assert (date, 1) in flagged
    assert cleaned.loc[date, 1] > damaged.loc[date, 1]
    unchanged = np.ones(damaged.shape, dtype=bool)
    for flagged_date, store in flagged:
        row = damaged.index.get_loc(flagged_date)
        col = list(damaged.columns).index(store)
        unchanged[row, col] = False
    assert np.allclose(
        cleaned.to_numpy()[unchanged],
        damaged.to_numpy()[unchanged],
        rtol=1e-12,
    )


def test_holiday_weeks_are_protected(panel):
    wide = to_wide(panel).copy()
    black_friday_week = pd.Timestamp("2011-11-25")
    wide.loc[black_friday_week, 2] *= 0.3

    cleaned, report = winsorize_anomalies(wide)

    assert (black_friday_week, 2) not in set(
        zip(report.Date, report.Store, strict=True)
    )
    assert np.allclose(
        cleaned.loc[black_friday_week, 2], wide.loc[black_friday_week, 2]
    )


def test_cleaning_uses_only_the_given_slice(panel):
    wide = to_wide(panel)
    first, _ = winsorize_anomalies(wide.iloc[:100])
    changed_future = wide.copy()
    changed_future.iloc[100:] *= 3
    second, _ = winsorize_anomalies(changed_future.iloc[:100])
    pd.testing.assert_frame_equal(first, second)


def test_smooth_data_gives_an_empty_report(panel):
    wide = to_wide(panel)
    weekly_shape = (wide / wide.mean()).mean(axis=1)
    smooth = pd.DataFrame(
        np.outer(weekly_shape, wide.mean()),
        index=wide.index,
        columns=wide.columns,
    )
    cleaned, report = winsorize_anomalies(smooth)
    assert report.empty
    pd.testing.assert_frame_equal(cleaned, smooth)
