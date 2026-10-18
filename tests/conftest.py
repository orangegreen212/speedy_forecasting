import numpy as np
import pandas as pd
import pytest

from speedy.features import thanksgiving


@pytest.fixture(scope="session")
def panel():
    """Synthetic panel: 6 stores (types A and B), 143 weeks from 2010-02-05.

    Each store has a yearly cycle and a Thanksgiving bump plus small noise.
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-02-05", periods=143, freq="7D")
    rows = []
    for store in range(1, 7):
        level = 1e5 * (1 + store / 3)
        store_type = "A" if store <= 3 else "B"
        for date in dates:
            season = 1 + 0.25 * np.sin(2 * np.pi * date.dayofyear / 365.25)
            is_thanksgiving = 0 <= (thanksgiving(date.year) - date).days < 7
            bump = 1.6 if is_thanksgiving else 1.0
            noise = rng.lognormal(0, 0.03)
            rows.append(
                (store, date, store_type, level * season * bump * noise)
            )
    columns = ["Store", "Date", "Type", "Weekly_Sales"]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture()
def raw_data_dir(panel, tmp_path):
    """The synthetic panel written as raw sales/stores/features files."""
    dates = panel["Date"].dt.strftime("%d/%m/%Y")
    sales = pd.DataFrame(
        {
            "Store": panel["Store"],
            "Dept": 1,
            "Date": dates,
            "Weekly_Sales": panel["Weekly_Sales"],
            "IsHoliday": False,
        }
    )
    negative_return = sales.iloc[[0]].assign(Dept=2, Weekly_Sales=-10.0)
    pd.concat([sales, negative_return]).to_csv(
        tmp_path / "sales.csv", index=False
    )

    stores = panel.drop_duplicates("Store")[["Store", "Type"]]
    stores.assign(Size=100000).to_csv(tmp_path / "stores.csv", index=False)

    later = pd.date_range(
        panel["Date"].max() + pd.Timedelta(days=7), periods=4, freq="7D"
    )
    features = pd.DataFrame(
        {"Store": 1, "Date": later.strftime("%d/%m/%Y"), "Temperature": 1.0}
    )
    features.to_csv(tmp_path / "features.csv", index=False)
    return tmp_path
