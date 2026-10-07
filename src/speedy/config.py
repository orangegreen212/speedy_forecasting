"""Paths and constants. Everything is repository-relative and can be overridden by environment variables."""
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SPEEDY_DATA_DIR", ROOT / "data" / "raw"))
MODEL_DIR = Path(os.environ.get("SPEEDY_MODEL_DIR", ROOT / "models"))
OUTPUT_DIR = Path(os.environ.get("SPEEDY_OUTPUT_DIR", ROOT / "outputs"))
MODEL_PATH = MODEL_DIR / "forecaster.joblib"

# Fixed time origin for the 'trend' feature. Fixed (not "first row of the training slice") so that a feature
# value never depends on which rows happen to be passed in.
T0 = pd.Timestamp("2010-02-05")
