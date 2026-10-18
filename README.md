# Speedy Forecasting: weekly sales forecast for 45 stores

Forecasts weekly sales for each store from 2012-11-02 to 2013-12-06 and says
how accurate the forecast is. The final model is a blend of three models:
seasonal naive with drift (60%), Prophet (20%) and a damped Holt trend
(20%). A ridge model was tested and got zero weight.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
# put features.csv, sales.csv, stores.csv in data/raw/
# or: export SPEEDY_DATA_DIR=/path/to/folder
python -m speedy.train                        # backtest, fit, save model
python -m speedy.score --end-date 2013-12-06  # forecast CSV files
streamlit run app.py                          # dashboard
pytest -q                                     # tests (RUN_SLOW=1: Prophet)
ruff check . && ruff format --check .         # style (PEP8, 79 columns)
```
Environment variables: `SPEEDY_DATA_DIR`, `SPEEDY_MODEL_DIR`,
`SPEEDY_OUTPUT_DIR`. The notebooks need `python -m speedy.train` to be run
once first (section 5 of notebook 02 loads the saved model).

## Layout
| Path | Purpose |
|---|---|
| `src/speedy/features.py` | the single feature contract (calendar only) |
| `src/speedy/forecasters.py` | base models: naive, naive with drift, ridge, Holt, Prophet |
| `src/speedy/models.py` | scikit-learn parts: `FeatureBuilder`, `BlendForecaster` |
| `src/speedy/pipeline.py` | `build_pipeline`, `pipeline_from_config` |
| `src/speedy/metrics.py` | WMAPE, bias, bootstrap intervals, prediction bands |
| `src/speedy/evaluate.py` | walk-forward folds, blend selection, full evaluation |
| `src/speedy/data.py`, `cleaning.py` | loading, checks, outlier handling |
| `src/speedy/train.py`, `score.py`, `inference.py` | commands and inference API |
| `src/speedy/plots.py` | plot helpers used by the notebooks |
| `notebooks/` | `01_eda` (look at the data), `02_modelling` (accuracy and forecast) |
| `docs/business_memo.md` | recommendation for the CFO |
| `app/streamlit_app.py`, `app_artifacts/` | dashboard and the files it reads |
| `tests/` | unit tests, leakage tests, train/score smoke test |

## Data decisions
* **Returns** (negative department rows, about 0.3% of rows) stay in the
  sales: reported revenue is net, and the effect on totals is about 0.001%.
* **Outliers:** only non-holiday extremes are capped, using the training
  data only. This changes the accuracy by less than 0.01 points.
* **`features.csv` is not used.** It contains rows up to July 2013, which
  could not exist on 2012-12-10. Temperature, CPI and markdowns are also
  unknown for 2013.
* **Dates** are read with an explicit format (`2010-02-05` or
  `05/02/2010`). A guessed day-first format would swap day and month for
  ISO dates.

## Design decisions
* **No leakage by design.** Features are calendar-only. Tests change all
  data after the forecast origin and check that the forecasts do not move.
* **The selection is part of the test.** Blend weights for a backtest block
  come only from earlier blocks. Headline numbers have 95% block-bootstrap
  intervals, and the blend is compared with seasonal naive on the same weeks.
* **Saved model** (`joblib`) holds the pipeline, metadata and band
  calibration. It is a pickle: only load files you trained yourself.
* Raw data, models and outputs are not committed (`.gitignore`).

## Deploy the dashboard (Streamlit Community Cloud)
1. Run `python -m speedy.train` locally and commit `app_artifacts/`.
2. New app -> your repo -> main file `app.py`. The root `requirements.txt`
   is light (no Prophet).
3. `app_artifacts/actuals_store_week.csv` is company sales data. Delete it
   before pushing to a public repo; the app then shows forecasts only.
