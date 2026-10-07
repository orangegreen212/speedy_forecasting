# Speedy Forecasting: weekly store sales forecast

Forecasts weekly sales for each of 45 stores (blend of seasonal-naive, pooled ridge and Prophet), with prediction bands and honest backtest uncertainty.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
# put features.csv, sales.csv, stores.csv in data/raw/  (git-ignored)  OR  export SPEEDY_DATA_DIR=/path/to/folder
python -m speedy.train                          # evaluate + fit + save models/forecaster.joblib, outputs/evaluation.json
python -m speedy.score --end-date 2013-12-06    # outputs/forecast_by_store.csv, outputs/forecast_total.csv
streamlit run app.py        # dashboard; needs app_artifacts/ (created by train, committed)
pytest -q                                       # unit tests (RUN_SLOW=1 adds the Prophet test)
```
Environment variables: `SPEEDY_DATA_DIR`, `SPEEDY_MODEL_DIR`, `SPEEDY_OUTPUT_DIR`. No absolute paths in the code.

## Layout
| Path | Purpose |
|---|---|
| `src/speedy/features.py` | **single feature contract**: `build_features()` (pure, calendar-only) |
| `src/speedy/models.py` | base forecasters + `FeatureBuilder` and `BlendForecaster` (sklearn estimators) |
| `src/speedy/pipeline.py` | `build_pipeline()` = `Pipeline([features, model])` |
| `src/speedy/evaluate.py` | walk-forward folds, nested blend selection, block-bootstrap CIs, band calibration |
| `src/speedy/train.py` / `score.py` / `inference.py` | training CLI, scoring CLI, stable inference API |
| `notebooks/01_eda.ipynb`, `02_modelling.ipynb` | EDA and evidence; both import the package, define no features |
| `docs/business_memo.md` | recommendation first, methodology as evidence |
| `app/streamlit_app.py`, `app_artifacts/` | Streamlit dashboard: totals, single store, accuracy with CIs, CSV download |
| `tests/` | unit tests incl. feature purity, no-leakage, save/load round trip |

## Design decisions
* **No data leakage by construction:** features are calendar-only; exogenous columns (CPI, temperature, MarkDowns) are unknown for the forecast period and are not used. A test scrambles all post-origin data and asserts forecasts are bit-identical.
* **Selection is evaluated, not just done:** walk-forward nested selection of blend weights; headline metrics with block-bootstrap 95% CIs; paired test against the naive benchmark.
* **Persisted artefact** (`joblib`): pipeline + metadata (trained-through date, versions, params) + calibration (band multipliers). Only load artefacts you trained yourself (pickle).
* Raw data and archives are never committed (`.gitignore`).

## Deploy the dashboard (Streamlit Community Cloud)
1. Run `python -m speedy.train` locally, then commit `app_artifacts/` (forecasts, evaluation.json, weekly actuals). The model file and raw data stay out of git.
2. New app -> your repo -> Main file path `app.py` (or `app/streamlit_app.py`). The root `requirements.txt` is light (no Prophet); dev deps are in `requirements-dev.txt`.
3. `app_artifacts/actuals_store_week.csv` is company sales data aggregated to store-week: delete it before pushing to a public repo (the app then shows forecasts without history).
