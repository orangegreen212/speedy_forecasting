"""Train, evaluate and save the model.

python -m speedy.train [--data-dir D] [--model-dir M] [--no-prophet]
"""

import argparse
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from . import __version__
from .config import APP_DIR, DATA_DIR, MODEL_DIR, OUTPUT_DIR
from .data import data_quality_report, load_panel
from .evaluate import BlendConfig, run_evaluation
from .inference import forecast_totals, forecast_weekly
from .pipeline import pipeline_from_config

END_DATE = "2013-12-06"


def write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, default=float))


def fit_artifact(panel, result) -> dict:
    """Fit the chosen blend on all data; return the model with metadata."""
    pipeline = pipeline_from_config(
        BlendConfig(*result["final_cfg"]), result["store_scale"]
    )
    pipeline.fit(panel, panel["Weekly_Sales"])
    metadata = {
        "version": __version__,
        "trained_through": str(panel["Date"].max().date()),
        "n_weeks": int(panel["Date"].nunique()),
        "n_stores": int(panel["Store"].nunique()),
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
        "params": pipeline.named_steps["model"].get_params(),
    }
    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "calibration": result["calibration"],
    }


def train_and_save(
    data_dir=DATA_DIR,
    model_dir=MODEL_DIR,
    prophet=True,
    n_boot=1000,
    out_dir=OUTPUT_DIR,
    app_dir=APP_DIR,
    end_date=END_DATE,
    n_jobs=1,
):
    """Evaluate, fit on all data and save the model and its reports.

    Returns (artifact, evaluation results, forecast and actual tables).
    """
    panel = load_panel(data_dir)
    result = run_evaluation(
        panel, prophet=prophet, n_boot=n_boot, n_jobs=n_jobs
    )
    arrays = result.pop("_arrays")

    artifact = fit_artifact(panel, result)
    result["metadata"] = artifact["metadata"]

    model_dir, out_dir = Path(model_dir), Path(out_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_dir / "forecaster.joblib")
    write_json(out_dir / "data_quality.json", data_quality_report(data_dir))
    write_json(out_dir / "evaluation.json", result)
    export_bundle(
        artifact, result, panel, Path(app_dir), end_date, arrays=arrays
    )
    return artifact, result, arrays


def export_bundle(
    artifact, result, panel, bundle_dir, end_date, arrays=None
) -> None:
    """Write everything the Streamlit app needs.

    The app then runs without the model, Prophet or the raw files: ready
    forecasts, the evaluation and weekly actuals per store.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    weekly = forecast_weekly(artifact, end_date)
    totals = forecast_totals(artifact, weekly)
    weekly.to_csv(bundle_dir / "forecast_by_store.csv", index=False)
    totals.to_csv(bundle_dir / "forecast_total.csv", index=False)
    write_json(bundle_dir / "evaluation.json", result)
    actuals = panel[["Store", "Date", "Weekly_Sales"]]
    actuals.to_csv(bundle_dir / "actuals_store_week.csv", index=False)
    if arrays is not None:
        backtest_table(arrays).to_csv(
            bundle_dir / "backtest_store_week.csv", index=False
        )


def backtest_table(arrays) -> pd.DataFrame:
    """Backtest weeks: actual, baseline and our forecast per store-week.

    Baseline = "same as last year" (seasonal naive). Our forecast = blend.
    """

    def long(table, name):
        table = table.rename_axis(index="Date", columns="Store")
        return table.stack().rename(name)

    return pd.concat(
        [
            long(arrays["oof_actual"], "Actual"),
            long(arrays["oof_benchmarks"]["seasonal_naive"], "Baseline"),
            long(arrays["oof_blend"], "Ours"),
        ],
        axis=1,
    ).reset_index()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument(
        "--no-prophet",
        action="store_true",
        help="faster: blend without Prophet",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="parallel Prophet fits (default 1)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    artifact, result, _ = train_and_save(
        Path(args.data_dir),
        Path(args.model_dir),
        prophet=not args.no_prophet,
        n_jobs=args.n_jobs,
    )
    print("saved", Path(args.model_dir) / "forecaster.joblib")
    print(json.dumps(artifact["metadata"], indent=2, default=str))
    print(
        "selected config (alpha, holt, drift, weights in tenths: "
        "naive, ridge, prophet, holt):",
        result["final_cfg"],
    )


if __name__ == "__main__":
    main()
