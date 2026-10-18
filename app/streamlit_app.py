"""Streamlit dashboard:  streamlit run app/streamlit_app.py

Self-contained: it reads only ../app_artifacts/ (ready forecasts, the
evaluation and optional actuals). No model file, Prophet or raw data is
needed, so it runs on Streamlit Community Cloud.
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

BUNDLE = Path(__file__).resolve().parents[1] / "app_artifacts"
REQUIRED = ("forecast_by_store.csv", "forecast_total.csv", "evaluation.json")
ACTUAL_COLOR, FORECAST_COLOR, BAND_COLOR = "#333333", "#1f77b4", "#aec7e8"

st.set_page_config(page_title="Sales forecast 2013", layout="wide")
for name in REQUIRED:
    if not (BUNDLE / name).exists():
        st.error(
            f"Missing {BUNDLE / name}. Run `python -m speedy.train` "
            "and commit the app_artifacts/ folder."
        )
        st.stop()


@st.cache_data
def load_bundle():
    by_store = pd.read_csv(
        BUNDLE / "forecast_by_store.csv", parse_dates=["Date"]
    )
    total = pd.read_csv(BUNDLE / "forecast_total.csv", parse_dates=["Date"])
    evaluation = json.loads((BUNDLE / "evaluation.json").read_text())
    actuals_path = BUNDLE / "actuals_store_week.csv"
    actuals = (
        pd.read_csv(actuals_path, parse_dates=["Date"])
        if actuals_path.exists()
        else None
    )
    return by_store, total, evaluation, actuals


def forecast_chart(actual: pd.Series | None, forecast: pd.DataFrame, unit):
    """Actual line, forecast line and a shaded 80% band."""
    forecast = forecast.reset_index()
    base = alt.Chart(forecast).encode(x=alt.X("Date:T", title="Week"))
    band = base.mark_area(color=BAND_COLOR, opacity=0.6).encode(
        y=alt.Y("Lower80:Q", title=unit), y2="Upper80:Q"
    )
    line = base.mark_line(color=FORECAST_COLOR).encode(y="Forecast:Q")
    layers = [band, line]
    if actual is not None:
        history = actual.rename("Actual").reset_index()
        layers.append(
            alt.Chart(history)
            .mark_line(color=ACTUAL_COLOR)
            .encode(x="Date:T", y="Actual:Q")
        )
    return alt.layer(*layers).properties(height=380)


def accuracy_table(evaluation: dict) -> pd.DataFrame:
    """One row per model: each metric with its 95% interval."""
    rows = []
    for model in evaluation["oof_metrics"]:
        row = {"model": model}
        for stat, table in evaluation["oof_ci"].items():
            value, low, high = table[model]
            row[stat] = f"{value:.2%}  [{low:.2%}, {high:.2%}]"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


by_store, total, evaluation, actuals = load_bundle()
meta = evaluation["metadata"]
st.title("Weekly sales forecast")
st.caption(
    f"Model v{meta['version']} trained through {meta['trained_through']} on "
    f"{meta['n_stores']} stores and {meta['n_weeks']} weeks. "
    "Forecasts are precomputed."
)

end = st.sidebar.date_input(
    "Forecast until",
    total["Date"].max().date(),
    min_value=total["Date"].min().date(),
    max_value=total["Date"].max().date(),
)
by_store = by_store[by_store["Date"] <= pd.Timestamp(end)]
total = total[total["Date"] <= pd.Timestamp(end)]
store = st.sidebar.selectbox("Store", sorted(by_store["Store"].unique()))

year = int(total["Date"].dt.year.max())
in_year = total[total["Date"].dt.year == year]
year_total = in_year["Forecast"].sum()
low, high = evaluation["calibration"]["annual_planning_range"]
wmape, wmape_low, wmape_high = evaluation["oof_ci"]["WMAPE"][
    "blend_walk_forward"
]

col1, col2, col3 = st.columns(3)
col1.metric(
    f"{year} forecast ({len(in_year)} weeks in horizon)",
    f"${year_total / 1e6:,.0f}M",
)
col2.metric(
    "Indicative outcome range (80%)",
    f"${year_total * (1 + low) / 1e6:,.0f}M - "
    f"${year_total * (1 + high) / 1e6:,.0f}M",
)
col3.metric(
    "Typical store-week error",
    f"{wmape:.1%}",
    f"95% CI {wmape_low:.1%} - {wmape_high:.1%}",
    delta_color="off",
)
st.info(
    "In the backtest the forecast was too low in most quarters, so the "
    "base number is on the cautious side. The range uses only 5 quarters, "
    "so treat it as a guide. The data ends on 2012-10-26: the 2012 "
    "holiday season has not been seen yet."
)

tab_total, tab_store, tab_accuracy, tab_download = st.tabs(
    ["Company total", "Single store", "Accuracy", "Download"]
)
with tab_total:
    actual_total = (
        actuals.groupby("Date")["Weekly_Sales"].sum() / 1e6
        if actuals is not None
        else None
    )
    st.altair_chart(
        forecast_chart(
            actual_total,
            total.set_index("Date")[["Forecast", "Lower80", "Upper80"]] / 1e6,
            "Sales per week ($M)",
        ),
        use_container_width=True,
    )
with tab_store:
    one = by_store[by_store["Store"] == store].set_index("Date")
    actual_store = (
        actuals[actuals["Store"] == store].set_index("Date")["Weekly_Sales"]
        / 1e3
        if actuals is not None
        else None
    )
    st.altair_chart(
        forecast_chart(
            actual_store,
            one[["Forecast", "Lower80", "Upper80"]] / 1e3,
            "Sales per week ($K)",
        ),
        use_container_width=True,
    )
    mult_low, mult_high = evaluation["calibration"]["store_week_mult"]
    st.caption(
        f"Store {store}. The band is the 80% range for one store-week: "
        f"{mult_low - 1:+.0%} to {mult_high - 1:+.0%} around the forecast."
    )
with tab_accuracy:
    st.subheader("Backtest of the last 65 weeks (95% confidence intervals)")
    st.dataframe(accuracy_table(evaluation))
    st.caption(
        "Bias = (forecast - actual) / actual. Negative means the "
        "forecast was too low."
    )
    diff = evaluation["oof_skill_vs_naive_WMAPE_diff"]["naive_minus_blend"]
    st.write(
        f"The blend beats the seasonal naive forecast by {diff[0]:.2%} "
        f"WMAPE (95% CI {diff[1]:.2%} to {diff[2]:.2%})."
    )
    errors = [f"{x:+.1%}" for x in evaluation["quarter_total_error"]]
    st.write(
        "Quarter total error, actual / forecast - 1 (positive = "
        "forecast too low):",
        errors,
    )
    study = evaluation["bias_correction"]
    if study["adopted_lam"] == 0:
        st.write(
            "An extra bias correction was tested and not used: its "
            "gain could not be told apart from zero."
        )
    holiday = evaluation.get("holiday_weeks")
    if holiday:
        gap = holiday["naive_minus_blend"]
        st.write(
            f"Holiday weeks ({holiday['n_weeks']} weeks, one season): blend "
            f"{holiday['blend_WMAPE']:.1%} vs naive "
            f"{holiday['naive_WMAPE']:.1%}. The difference interval "
            f"({gap[1]:+.1%} to {gap[2]:+.1%}) includes zero, so we cannot "
            "say which is better."
        )
    check = evaluation["coverage_check"]
    st.write(
        f"The 80% band held {check['coverage_80']:.0%} of the time on "
        "later, unseen weeks (target 80%)."
    )
    st.subheader("One 58-week check (a single sample)")
    year_ahead = {
        k: v for k, v in evaluation["year_ahead"].items() if k != "note"
    }
    st.dataframe(pd.DataFrame(year_ahead).T)
with tab_download:
    for label, table in (
        ("forecast_by_store.csv", by_store),
        ("forecast_total.csv", total),
    ):
        st.download_button(label, table.to_csv(index=False), label, "text/csv")
