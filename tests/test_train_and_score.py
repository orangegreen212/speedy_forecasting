import json

import speedy
from speedy.score import main as score_main
from speedy.train import train_and_save


def test_package_has_a_version():
    assert speedy.__version__


def test_train_then_score_end_to_end(raw_data_dir, tmp_path):
    model_dir = tmp_path / "models"
    out_dir = tmp_path / "outputs"
    app_dir = tmp_path / "app"
    artifact, result, arrays = train_and_save(
        raw_data_dir,
        model_dir,
        prophet=False,
        n_boot=20,
        out_dir=out_dir,
        app_dir=app_dir,
        end_date="2012-12-28",
    )
    assert artifact["metadata"]["n_stores"] == 6
    assert "oof_actual" in arrays
    for name in (
        "forecast_by_store.csv",
        "forecast_total.csv",
        "evaluation.json",
        "actuals_store_week.csv",
    ):
        assert (app_dir / name).exists(), name
    saved = json.loads((out_dir / "evaluation.json").read_text())
    assert saved["metadata"]["version"] == speedy.__version__
    assert (out_dir / "data_quality.json").exists()

    score_dir = tmp_path / "scored"
    score_main([
        "--model", str(model_dir / "forecaster.joblib"),
        "--end-date", "2012-12-28",
        "--out-dir", str(score_dir),
    ])  # fmt: skip
    assert (score_dir / "forecast_total.csv").exists()
