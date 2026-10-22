from pathlib import Path

import pytest

pytest.importorskip("streamlit")
ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(
    not (ROOT / "app_artifacts" / "evaluation.json").exists(),
    reason="run python -m speedy.train first",
)
@pytest.mark.parametrize("entry", ["app.py", "app/streamlit_app.py"])
def test_app_runs_from_bundle_only(entry):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / entry), default_timeout=60).run()
    assert not app.exception
    assert len(app.metric) == 7
    assert len(app.tabs) == 4
    assert all("$" not in m.value for m in app.metric)
