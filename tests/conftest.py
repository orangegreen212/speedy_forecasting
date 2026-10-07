import numpy as np, pandas as pd, pytest
from speedy.features import thanksgiving


@pytest.fixture(scope="session")
def panel():
    """Synthetic rectangular panel: 6 stores (types A/B), 143 Fridays from 2010-02-05, yearly cycle + Thanksgiving bump."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-02-05", periods=143, freq="7D")
    rows = []
    for s in range(1, 7):
        level = 1e5 * (1 + s / 3); typ = "A" if s <= 3 else "B"
        for d in dates:
            seas = 1 + 0.25 * np.sin(2 * np.pi * d.dayofyear / 365.25)
            bump = 1.6 if 0 <= (thanksgiving(d.year) - d).days < 7 else 1.0
            rows.append((s, d, typ, level * seas * bump * rng.lognormal(0, 0.03)))
    return pd.DataFrame(rows, columns=["Store", "Date", "Type", "Weekly_Sales"])
