"""Streamlit dashboard.   streamlit run app/streamlit_app.py
Self-contained: reads only ../app_artifacts/ (precomputed forecasts, evaluation.json, optional actuals).
No model file, Prophet or raw data needed, so it runs on Streamlit Community Cloud."""
import json
from pathlib import Path
import pandas as pd, streamlit as st

B = Path(__file__).resolve().parents[1] / "app_artifacts"
st.set_page_config(page_title="Sales forecast 2013", layout="wide")
for f in ("forecast_by_store.csv", "forecast_total.csv", "evaluation.json"):
    if not (B / f).exists():
        st.error(f"Missing {B / f}. Run `python -m speedy.train` locally and commit the app_artifacts/ folder."); st.stop()


@st.cache_data
def load():
    w = pd.read_csv(B / "forecast_by_store.csv", parse_dates=["Date"]); t = pd.read_csv(B / "forecast_total.csv", parse_dates=["Date"])
    ev = json.loads((B / "evaluation.json").read_text())
    a = pd.read_csv(B / "actuals_store_week.csv", parse_dates=["Date"]) if (B / "actuals_store_week.csv").exists() else None
    return w, t, ev, a


w, t, ev, actuals = load(); meta = ev["metadata"]
st.title("Weekly sales forecast")
st.caption(f"Model v{meta['version']} trained through {meta['trained_through']} on {meta['n_stores']} stores, {meta['n_weeks']} weeks. Forecasts are precomputed.")

end = st.sidebar.date_input("Forecast until", t.Date.max().date(), min_value=t.Date.min().date(), max_value=t.Date.max().date())
w = w[w.Date <= pd.Timestamp(end)]; t = t[t.Date <= pd.Timestamp(end)]
store = st.sidebar.selectbox("Store", sorted(w.Store.unique()))
yr = int(t.Date.dt.year.max()); ty = t.loc[t.Date.dt.year == yr, "Forecast"].sum(); lo, hi = ev["calibration"]["annual_planning_range"]

c1, c2, c3 = st.columns(3)
c1.metric(f"{yr} forecast ({(t.Date.dt.year == yr).sum()} weeks in horizon)", f"${ty/1e6:,.0f}M")
c2.metric("Indicative outcome range (80%)", f"${ty*(1+lo)/1e6:,.0f}M - ${ty*(1+hi)/1e6:,.0f}M")
ci = ev["oof_ci"]["WMAPE"]["blend_walk_forward"]
c3.metric("Typical store-week error", f"{ci[0]:.1%}", f"95% CI {ci[1]:.1%} - {ci[2]:.1%}", delta_color="off")
st.info("Backtests under-forecast in every quarter, so the base forecast is conservative; the range is indicative (only 5 quarters). "
        "Data ends 2012-10-26: the 2012 holiday season is not yet seen.")

tab1, tab2, tab3, tab4 = st.tabs(["Company total", "Single store", "Accuracy", "Download"])
with tab1:
    d = t.set_index("Date")[["Forecast", "Lower80", "Upper80"]] / 1e6
    if actuals is not None:
        d = pd.concat([(actuals.groupby("Date").Weekly_Sales.sum() / 1e6).rename("Actual"), d], axis=1)
    st.line_chart(d); st.caption("$M per week")
with tab2:
    d = w[w.Store == store].set_index("Date")[["Forecast", "Lower80", "Upper80"]] / 1e3
    if actuals is not None:
        d = pd.concat([(actuals[actuals.Store == store].set_index("Date").Weekly_Sales / 1e3).rename("Actual"), d], axis=1)
    st.line_chart(d); st.caption(f"Store {store}, $K per week; 80% band for a single store-week (about -4% / +12%)")
with tab3:
    rows = [{"model": k, **{s: f"{ev['oof_ci'][s][k][0]:.2%}  [{ev['oof_ci'][s][k][1]:.2%}, {ev['oof_ci'][s][k][2]:.2%}]" for s in ev["oof_ci"]}}
            for k in ev["oof_metrics"]]
    st.subheader("Out-of-fold, last 65 weeks (walk-forward selection, 95% block-bootstrap CI)"); st.dataframe(pd.DataFrame(rows).set_index("model"))
    d = ev["oof_skill_vs_naive_WMAPE_diff"]["naive_minus_blend"]
    st.write(f"Blend beats seasonal naive by {d[0]:.2%} WMAPE (95% CI {d[1]:.2%} to {d[2]:.2%}).")
    st.write("Quarter total error (actual/forecast - 1):", [f"{x:+.1%}" for x in ev["quarter_total_error"]])
    st.write(f"80% band coverage on later, unseen weeks: {ev['coverage_check']['coverage_80']:.0%} (target 80%).")
    st.subheader("Single 58-week year-ahead check (one sample)"); st.dataframe(pd.DataFrame({k: v for k, v in ev["year_ahead"].items() if k != "note"}).T)
with tab4:
    st.download_button("forecast_by_store.csv", w.to_csv(index=False), "forecast_by_store.csv", "text/csv")
    st.download_button("forecast_total.csv", t.to_csv(index=False), "forecast_total.csv", "text/csv")
