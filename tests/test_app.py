import pytest
from pathlib import Path
pytest.importorskip("streamlit")
ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(not (ROOT / "app_artifacts" / "evaluation.json").exists(), reason="run python -m speedy.train first")
def test_app_runs_from_bundle_only():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception and len(at.metric) == 3 and len(at.tabs) == 4
