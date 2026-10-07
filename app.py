"""Streamlit front-end:  streamlit run app.py
Reads the persisted model (models/forecaster.joblib) and outputs/evaluation.json. Run `python -m speedy.train` first.
Raw data is optional: if found, actual sales are drawn on the charts."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import pandas as pd, streamlit as st
from speedy.config import MODEL_PATH, OUTPUT_DIR, DATA_DIR
from speedy.inference import load_artifact, forecast_weekly, forecast_totals

st.set_page_config(page_title="Sales forecast 2013", layout="wide")


@st.cache_resource
def get_artifact(path):
    return load_artifact(path)


@st.cache_data
def get_actuals(data_dir):
    try:
        from speedy.data import load_panel
        return load_panel(data_dir)
    except Exception:
        return None


if not Path(MODEL_PATH).exists():
    st.error(f"Model not found at {MODEL_PATH}. Run `python -m speedy.train` first."); st.stop()
art = get_artifact(str(MODEL_PATH)); meta = art["metadata"]
ev_path = OUTPUT_DIR / "evaluation.json"; ev = json.loads(ev_path.read_text()) if ev_path.exists() else None
actuals = get_actuals(str(DATA_DIR))

st.title("Weekly sales forecast")
st.caption(f"Model v{meta['version']} trained through {meta['trained_through']} on {meta['n_stores']} stores, {meta['n_weeks']} weeks.")

end = st.sidebar.date_input("Forecast until", pd.Timestamp("2013-12-06"), min_value=pd.Timestamp(meta["trained_through"]) + pd.Timedelta(days=7),
                            max_value=pd.Timestamp("2014-12-31"))
w = forecast_weekly(art, end); t = forecast_totals(art, w)
stores = sorted(w.Store.unique()); store = st.sidebar.selectbox("Store", stores)
last_year = int(t.Date.dt.year.max()); ty = t[t.Date.dt.year == last_year]["Forecast"].sum()
lo, hi = art["calibration"]["annual_planning_range"]

c1, c2, c3 = st.columns(3)
c1.metric(f"{last_year} forecast ({(t.Date.dt.year == last_year).sum()} weeks in horizon)", f"${ty/1e6:,.0f}M")
c2.metric("Indicative outcome range (80%)", f"${ty*(1+lo)/1e6:,.0f}M - ${ty*(1+hi)/1e6:,.0f}M")
if ev:
    m = ev["oof_metrics"]["blend_walk_forward"]; ci = ev["oof_ci"]["WMAPE"]["blend_walk_forward"]
    c3.metric("Typical store-week error", f"{m['WMAPE']:.1%}", f"95% CI {ci[1]:.1%} - {ci[2]:.1%}", delta_color="off")
st.info("Backtests under-forecast in every quarter, so the base forecast is conservative; the range above is indicative (only 5 quarters). "
        "Data ends 2012-10-26: the 2012 holiday season is not yet seen.")

tab1, tab2, tab3, tab4 = st.tabs(["Company total", "Single store", "Accuracy", "Download"])
with tab1:
    d = t.set_index("Date")[["Forecast", "Lower80", "Upper80"]] / 1e6
    if actuals is not None:
        a = actuals.groupby("Date").Weekly_Sales.sum() / 1e6; d = pd.concat([a.rename("Actual"), d], axis=1)
    st.line_chart(d, y=[c for c in ["Actual", "Forecast", "Lower80", "Upper80"] if c in d.columns]); st.caption("$M per week")
with tab2:
    d = w[w.Store == store].set_index("Date")[["Forecast", "Lower80", "Upper80"]] / 1e3
    if actuals is not None:
        a = actuals[actuals.Store == store].set_index("Date").Weekly_Sales / 1e3; d = pd.concat([a.rename("Actual"), d], axis=1)
    st.line_chart(d); st.caption(f"Store {store}, $K per week, 80% band for a single store-week (about -4% / +12%)")
with tab3:
    if not ev:
        st.warning("outputs/evaluation.json not found: run `python -m speedy.train`.")
    else:
        rows = [{"model": k, **{s: f"{ev['oof_ci'][s][k][0]:.2%}  [{ev['oof_ci'][s][k][1]:.2%}, {ev['oof_ci'][s][k][2]:.2%}]" for s in ev["oof_ci"]}}
                for k in ev["oof_metrics"]]
        st.subheader("Out-of-fold, last 65 weeks (walk-forward selection, 95% block-bootstrap CI)"); st.dataframe(pd.DataFrame(rows).set_index("model"))
        d = ev["oof_skill_vs_naive_WMAPE_diff"]["naive_minus_blend"]
        st.write(f"Blend beats seasonal naive by {d[0]:.2%} WMAPE (95% CI {d[1]:.2%} to {d[2]:.2%}).")
        st.write("Quarter total error (actual/forecast - 1):", [f"{x:+.1%}" for x in ev["quarter_total_error"]])
        cc = ev["coverage_check"]; st.write(f"80% band coverage on later, unseen weeks: {cc['coverage_80']:.0%} (target 80%).")
        st.subheader("Single 58-week year-ahead check (one sample)"); st.dataframe(pd.DataFrame({k: v for k, v in ev["year_ahead"].items() if k != "note"}).T)
with tab4:
    st.download_button("forecast_by_store.csv", w.to_csv(index=False), "forecast_by_store.csv", "text/csv")
    st.download_button("forecast_total.csv", t.to_csv(index=False), "forecast_total.csv", "text/csv")
