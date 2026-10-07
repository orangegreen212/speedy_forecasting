"""Root entry point so `streamlit run app.py` and a Streamlit Cloud main file of `app.py` both work.
The dashboard itself lives in app/streamlit_app.py."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "app" / "streamlit_app.py"), run_name="__main__")
