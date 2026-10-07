"""Train + evaluate + persist.   python -m speedy.train [--data-dir D] [--model-dir M] [--no-prophet]"""
import argparse, json, platform
import joblib, numpy as np, pandas as pd, sklearn
from . import __version__
from .config import DATA_DIR, MODEL_DIR, OUTPUT_DIR
from .data import load_panel
from .evaluate import run_evaluation
from .pipeline import build_pipeline


def train_and_save(data_dir=DATA_DIR, model_dir=MODEL_DIR, prophet=True, n_boot=1000, out_dir=OUTPUT_DIR):
    panel = load_panel(data_dir)
    res = run_evaluation(panel, prophet=prophet, n_boot=n_boot)
    alpha, wn, wr, wp = res["final_cfg"]
    pipe = build_pipeline(alpha=alpha, w_naive=wn / 10, w_ridge=wr / 10, w_prophet=wp / 10)
    pipe.fit(panel, panel["Weekly_Sales"])
    meta = {"version": __version__, "trained_through": str(panel["Date"].max().date()), "n_weeks": int(panel["Date"].nunique()),
            "n_stores": int(panel["Store"].nunique()), "sklearn": sklearn.__version__, "python": platform.python_version(),
            "params": pipe.named_steps["model"].get_params()}
    artifact = {"pipeline": pipe, "metadata": meta, "calibration": res["calibration"]}
    model_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_dir / "forecaster.joblib")
    arrays = res.pop("_arrays")
    res["metadata"] = meta
    (out_dir / "evaluation.json").write_text(json.dumps(res, indent=2, default=float))
    return artifact, res, arrays


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(DATA_DIR)); ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--no-prophet", action="store_true", help="faster, blend of naive+ridge only")
    a = ap.parse_args(argv)
    from pathlib import Path
    art, res, _ = train_and_save(Path(a.data_dir), Path(a.model_dir), prophet=not a.no_prophet)
    print("saved", Path(a.model_dir) / "forecaster.joblib"); print(json.dumps(art["metadata"], indent=2, default=str))
    print("selected cfg (alpha, w_naive, w_ridge, w_prophet x10):", res["final_cfg"])


if __name__ == "__main__":
    main()
