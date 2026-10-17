"""Loading and validating the raw files."""
from pathlib import Path
import pandas as pd

PANEL_COLUMNS = ["Store", "Date", "Type", "Weekly_Sales"]


def load_panel(data_dir, returns: str = "net") -> pd.DataFrame:
    """Return one row per (Store, Date): department sales summed to store-week, plus store Type.
    returns: "net" (default; negative dept-week rows = returns stay in the sum, as in reported net revenue),
             "clip" (negatives set to 0 before summing) or "drop" (negative rows removed)."""
    if returns not in ("net", "clip", "drop"):
        raise ValueError("returns must be net, clip or drop")
    data_dir = Path(data_dir)
    for f in ("sales.csv", "stores.csv"):
        if not (data_dir / f).exists():
            raise FileNotFoundError(f"{f} not found in {data_dir}. Set SPEEDY_DATA_DIR or put the files in data/raw/.")
    sales = pd.read_csv(data_dir / "sales.csv")
    sales["Date"] = pd.to_datetime(sales["Date"], dayfirst=True)
    if returns == "clip":
        sales["Weekly_Sales"] = sales["Weekly_Sales"].clip(lower=0)
    elif returns == "drop":
        sales = sales[sales["Weekly_Sales"] >= 0]
    stores = pd.read_csv(data_dir / "stores.csv")[["Store", "Type"]]
    panel = sales.groupby(["Store", "Date"], as_index=False)["Weekly_Sales"].sum().merge(stores, on="Store", how="left")
    return validate_panel(panel[PANEL_COLUMNS].sort_values(["Date", "Store"]).reset_index(drop=True))


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    missing = set(PANEL_COLUMNS) - set(panel.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if panel.duplicated(["Store", "Date"]).any():
        raise ValueError("duplicate (Store, Date) rows")
    if panel["Type"].isna().any():
        raise ValueError("stores without Type")
    if (panel["Weekly_Sales"] <= 0).any():
        raise ValueError("non-positive store-week sales (log model needs positive values)")
    wide = to_wide(panel)
    if wide.isna().any().any():
        raise ValueError("panel is not rectangular: some stores miss some weeks")
    if (wide.index.to_series().diff().dropna() != pd.Timedelta(days=7)).any():
        raise ValueError("weeks are not consecutive")
    return panel


def to_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """weeks x stores matrix of sales."""
    return panel.pivot(index="Date", columns="Store", values="Weekly_Sales").sort_index()


def data_quality_report(data_dir) -> dict:
    """Descriptive EDA numbers used in the notebook and stored with the model run (not used for fitting)."""
    from .cleaning import winsorize_anomalies
    data_dir = Path(data_dir)
    raw = pd.read_csv(data_dir / "sales.csv"); raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True)
    neg = raw[raw.Weekly_Sales < 0]
    rep = {"dept_rows": int(len(raw)), "duplicates": int(raw.duplicated(["Store", "Dept", "Date"]).sum()),
           "negative_rows": int(len(neg)), "negative_share_of_rows": float(len(neg) / len(raw)),
           "returns_amount_share_of_sales": float(-neg.Weekly_Sales.sum() / raw.Weekly_Sales.sum()),
           "zero_rows": int((raw.Weekly_Sales == 0).sum())}
    dc = raw.groupby(["Store", "Date"]).Dept.nunique().unstack(0)
    rep["dept_count_range"] = [int(dc.min().min()), int(dc.max().max())]
    panel = load_panel(data_dir); wide = to_wide(panel)
    rep["net_vs_clip_total_diff"] = float(load_panel(data_dir, "clip").Weekly_Sales.sum() / panel.Weekly_Sales.sum() - 1)
    _, a = winsorize_anomalies(wide)
    rep["anomalies_non_holiday"] = a.assign(Date=a.Date.astype(str)).round(3).to_dict("records") if len(a) else []
    f = pd.read_csv(data_dir / "features.csv"); f["Date"] = pd.to_datetime(f["Date"], dayfirst=True)
    rep["features_rows_after_sales_end"] = int((f.Date > raw.Date.max()).sum())     # leakage hazard: file contains 'future' rows
    rep["features_missing_share"] = {k: float(v) for k, v in f.isna().mean().items() if v > 0}
    return rep
