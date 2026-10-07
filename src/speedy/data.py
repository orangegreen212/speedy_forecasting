"""Loading and validating the raw files."""
from pathlib import Path
import pandas as pd

PANEL_COLUMNS = ["Store", "Date", "Type", "Weekly_Sales"]


def load_panel(data_dir) -> pd.DataFrame:
    """Return one row per (Store, Date): department sales summed to store-week, plus store Type."""
    data_dir = Path(data_dir)
    for f in ("sales.csv", "stores.csv"):
        if not (data_dir / f).exists():
            raise FileNotFoundError(f"{f} not found in {data_dir}. Set SPEEDY_DATA_DIR or put the files in data/raw/.")
    sales = pd.read_csv(data_dir / "sales.csv")
    sales["Date"] = pd.to_datetime(sales["Date"], dayfirst=True)
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
