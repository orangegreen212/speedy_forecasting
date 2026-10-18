"""Loading and validating the raw files."""

from pathlib import Path

import pandas as pd

PANEL_COLUMNS = ["Store", "Date", "Type", "Weekly_Sales"]
RETURN_POLICIES = ("net", "clip", "drop")
ONE_WEEK = pd.Timedelta(days=7)


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse dates written as 2010-02-05 or as 05/02/2010 (day first).

    The format is given explicitly. A guessed day-first format would
    silently swap day and month for ISO dates (2010-02-05 -> 2010-05-02).
    """
    for date_format in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return pd.to_datetime(values, format=date_format)
        except ValueError:
            continue
    raise ValueError("dates must look like 2010-02-05 or 05/02/2010")


def read_csv(data_dir, name: str) -> pd.DataFrame:
    """Read one raw file and parse its `Date` column if it has one."""
    path = Path(data_dir) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found in {data_dir}. "
            "Set SPEEDY_DATA_DIR or put the files in data/raw/."
        )
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = parse_dates(df["Date"])
    return df


def load_panel(data_dir, returns: str = "net") -> pd.DataFrame:
    """Return one row per (Store, Date): department sales summed per store.

    `returns` says what to do with negative department-week rows (returns):
    "net" keeps them in the sum (reported revenue is net of returns),
    "clip" sets them to 0 and "drop" removes the rows.
    """
    if returns not in RETURN_POLICIES:
        raise ValueError("returns must be net, clip or drop")
    sales = read_csv(data_dir, "sales.csv")
    stores = read_csv(data_dir, "stores.csv")[["Store", "Type"]]
    if returns == "clip":
        sales["Weekly_Sales"] = sales["Weekly_Sales"].clip(lower=0)
    elif returns == "drop":
        sales = sales[sales["Weekly_Sales"] >= 0]
    panel = (
        sales.groupby(["Store", "Date"], as_index=False)["Weekly_Sales"]
        .sum()
        .merge(stores, on="Store", how="left")
    )
    panel = panel[PANEL_COLUMNS].sort_values(["Date", "Store"])
    return validate_panel(panel.reset_index(drop=True))


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError unless the panel is clean; return it unchanged."""
    missing = set(PANEL_COLUMNS) - set(panel.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if panel.duplicated(["Store", "Date"]).any():
        raise ValueError("duplicate (Store, Date) rows")
    if panel["Type"].isna().any():
        raise ValueError("stores without Type")
    if (panel["Weekly_Sales"] <= 0).any():
        raise ValueError(
            "non-positive store-week sales (log model needs positive values)"
        )
    wide = to_wide(panel)
    if wide.isna().any().any():
        raise ValueError("panel is not rectangular: a store misses a week")
    if (wide.index.to_series().diff().dropna() != ONE_WEEK).any():
        raise ValueError("weeks are not consecutive")
    return panel


def to_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Weeks x stores table of sales."""
    return panel.pivot(
        index="Date", columns="Store", values="Weekly_Sales"
    ).sort_index()


def store_types(panel: pd.DataFrame) -> pd.Series:
    """Store type indexed by store number, in the order of `to_wide`."""
    types = panel.drop_duplicates("Store").set_index("Store")["Type"]
    return types.reindex(to_wide(panel).columns)


def data_quality_report(data_dir) -> dict:
    """Descriptive numbers for the EDA notebook (not used for fitting)."""
    from .cleaning import winsorize_anomalies

    raw = read_csv(data_dir, "sales.csv")
    negative = raw[raw["Weekly_Sales"] < 0]
    report = {
        "dept_rows": int(len(raw)),
        "duplicates": int(raw.duplicated(["Store", "Dept", "Date"]).sum()),
        "negative_rows": int(len(negative)),
        "negative_share_of_rows": float(len(negative) / len(raw)),
        "returns_amount_share_of_sales": float(
            -negative["Weekly_Sales"].sum() / raw["Weekly_Sales"].sum()
        ),
        "zero_rows": int((raw["Weekly_Sales"] == 0).sum()),
    }

    dept_counts = raw.groupby(["Store", "Date"])["Dept"].nunique()
    report["dept_count_range"] = [
        int(dept_counts.min()),
        int(dept_counts.max()),
    ]
    store_week = raw.groupby(["Store", "Date"])["Weekly_Sales"].sum()
    relative_sales = store_week / store_week.groupby("Store").transform("mean")
    report["dept_count_vs_relative_sales_corr"] = float(
        dept_counts.corr(relative_sales)
    )

    panel = load_panel(data_dir)
    clipped = load_panel(data_dir, "clip")
    report["net_vs_clip_total_diff"] = float(
        clipped["Weekly_Sales"].sum() / panel["Weekly_Sales"].sum() - 1
    )
    _, anomalies = winsorize_anomalies(to_wide(panel))
    report["anomalies_non_holiday"] = anomalies_to_records(anomalies)

    features = read_csv(data_dir, "features.csv")
    after_sales_end = features["Date"] > raw["Date"].max()
    report["features_last_date"] = str(features["Date"].max().date())
    report["features_rows_after_sales_end"] = int(after_sales_end.sum())
    missing_share = features.isna().mean()
    report["features_missing_share"] = {
        column: float(share)
        for column, share in missing_share.items()
        if share > 0
    }
    return report


def anomalies_to_records(anomalies: pd.DataFrame) -> list:
    """Anomaly table -> list of dicts that can be saved as JSON."""
    if anomalies.empty:
        return []
    records = anomalies.assign(Date=anomalies["Date"].astype(str))
    return records.round(3).to_dict("records")
