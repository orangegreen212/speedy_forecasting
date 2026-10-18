import pandas as pd
import pytest

from speedy.data import (
    data_quality_report,
    load_panel,
    parse_dates,
    validate_panel,
)


def test_parse_dates_reads_iso_and_day_first_without_swapping():
    iso = parse_dates(pd.Series(["2010-02-05", "2012-10-26"]))
    day_first = parse_dates(pd.Series(["05/02/2010", "26/10/2012"]))
    expected = pd.to_datetime(["2010-02-05", "2012-10-26"])
    assert (iso == expected).all()
    assert (day_first == expected).all()


def test_parse_dates_rejects_unknown_format():
    with pytest.raises(ValueError):
        parse_dates(pd.Series(["Feb 5 2010"]))


def test_returns_policy(tmp_path):
    dates = pd.date_range("2010-02-05", periods=6, freq="7D")
    dates = dates.strftime("%d/%m/%Y")
    rows = [(1, 1, d, 100.0) for d in dates]
    rows += [(1, 2, dates[0], -10.0)]
    rows += [(1, 2, d, 50.0) for d in dates[1:]]
    columns = ["Store", "Dept", "Date", "Weekly_Sales"]
    pd.DataFrame(rows, columns=columns).to_csv(
        tmp_path / "sales.csv", index=False
    )
    pd.DataFrame({"Store": [1], "Type": ["A"]}).to_csv(
        tmp_path / "stores.csv", index=False
    )

    totals = {
        policy: load_panel(tmp_path, policy)["Weekly_Sales"].sum()
        for policy in ("net", "clip", "drop")
    }
    assert totals["net"] == 100 * 6 + 50 * 5 - 10
    assert totals["clip"] == totals["net"] + 10
    assert totals["drop"] == totals["clip"]
    with pytest.raises(ValueError):
        load_panel(tmp_path, "bogus")


def test_missing_file_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="SPEEDY_DATA_DIR"):
        load_panel(tmp_path)


def test_data_quality_report(raw_data_dir):
    report = data_quality_report(raw_data_dir)
    assert report["negative_rows"] == 1
    assert report["dept_count_range"] == [1, 2]
    assert report["features_rows_after_sales_end"] == 4
    assert report["features_last_date"] > "2012-10-26"
    assert -1 <= report["dept_count_vs_relative_sales_corr"] <= 1


def test_validate_panel_catches_problems(panel):
    with pytest.raises(ValueError, match="duplicate"):
        validate_panel(pd.concat([panel, panel.iloc[:1]]))
    negative = panel.copy()
    negative.loc[0, "Weekly_Sales"] = -1
    with pytest.raises(ValueError, match="non-positive"):
        validate_panel(negative)
    with pytest.raises(ValueError, match="rectangular"):
        validate_panel(panel.iloc[1:])
