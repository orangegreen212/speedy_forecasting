"""Root entry point, so `streamlit run app.py` works.

The dashboard itself lives in app/streamlit_app.py.
"""

import runpy
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
runpy.run_path(str(DASHBOARD), run_name="__main__")
