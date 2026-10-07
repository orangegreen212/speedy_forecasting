import os, pytest
pytest.importorskip("streamlit")
from pathlib import Path
from speedy.config import MODEL_PATH


@pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="train the model first (python -m speedy.train)")
def test_app_runs_without_exception():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.metric) >= 2 and len(at.tabs) == 4
