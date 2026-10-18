"""Paths and constants.

All paths are relative to the repository. They can be overridden with the
environment variables SPEEDY_DATA_DIR, SPEEDY_MODEL_DIR and SPEEDY_OUTPUT_DIR.
"""

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SPEEDY_DATA_DIR", ROOT / "data" / "raw"))
MODEL_DIR = Path(os.environ.get("SPEEDY_MODEL_DIR", ROOT / "models"))
OUTPUT_DIR = Path(os.environ.get("SPEEDY_OUTPUT_DIR", ROOT / "outputs"))
APP_DIR = ROOT / "app_artifacts"
MODEL_PATH = MODEL_DIR / "forecaster.joblib"

# Fixed start date of the "trend" feature. It is fixed (and not "the first
# row we are given") so that a feature value never depends on the other rows.
T0 = pd.Timestamp("2010-02-05")
