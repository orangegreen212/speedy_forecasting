"""Streamlit dashboard:  streamlit run app.py

Self-contained: it reads only ../app_artifacts/ (ready forecasts, the
evaluation and the weekly actuals). No model file, Prophet or raw data is
needed, so it also runs on Streamlit Community Cloud.

Money is written as "USD 2.35B" and not with a dollar sign, because two
dollar signs in one text are shown as a formula by Streamlit.
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

BUNDLE = Path(__file__).resolve().parents[1] / "app_artifacts"
REQUIRED = (
    "forecast_by_store.csv",
    "forecast_total.csv",
    "evaluation.json",
    "actuals_store_week.csv",
)
ACTUAL_COLOR, FORECAST_COLOR, BAND_COLOR = "#333333", "#1f77b4", "#aec7e8"
BENCHMARK_COLOR, GOOD_COLOR, WARN_COLOR = "#9aa5b1", "#1f77b4", "#d9822b"
MODEL_NAMES = {
    "seasonal_naive": "Same as last year",
    "naive_with_drift_d5": "Last year + store growth",
    "holt_alone": "Trend model (Holt)",
    "prophet": "Prophet",
    "blend_walk_forward": "Our forecast (blend)",
}

st.set_page_config(
    page_title="Sales forecast 2013", page_icon="📈", layout="wide"
)
for name in REQUIRED:
    if not (BUNDLE / name).exists():
        st.error(
            f"Missing {BUNDLE / name}. Run `python -m speedy.train` "
            "and keep the app_artifacts/ folder."
        )
        st.stop()


def usd(value: float) -> str:
    """Money as text: USD 2.35B or USD 480M."""
    if abs(value) >= 1e9:
        return f"USD {value / 1e9:,.2f}B"
    return f"USD {value / 1e6:,.0f}M"


@st.cache_data
def load_bundle():
    by_store = pd.read_csv(
        BUNDLE / "forecast_by_store.csv", parse_dates=["Date"]
    )
    total = pd.read_csv(BUNDLE / "forecast_total.csv", parse_dates=["Date"])
    actuals = pd.read_csv(
        BUNDLE / "actuals_store_week.csv", parse_dates=["Date"]
    )
    evaluation = json.loads((BUNDLE / "evaluation.json").read_text())
    return by_store, total, actuals, evaluation


def growth_vs_last_year(by_store: pd.DataFrame, actuals: pd.DataFrame, year):
    """Forecast of `year` against the same weeks one year earlier.

    Only forecast weeks whose last-year week exists in the actuals are
    compared (weeks 364 days back), so both sides cover the same weeks.
    """
    wide = actuals.pivot(index="Date", columns="Store", values="Weekly_Sales")
    forecast = by_store[by_store["Date"].dt.year == year]
    last_year_dates = forecast["Date"] - pd.Timedelta(days=364)
    comparable = forecast[last_year_dates.isin(wide.index).to_numpy()]
    this_year = comparable.groupby("Store")["Forecast"].sum()
    last_year = wide.loc[
        comparable["Date"].drop_duplicates() - pd.Timedelta(days=364)
    ].sum()
    return (this_year / last_year - 1), comparable["Date"].nunique()


def line_layers(history, forecast, zoom_start, unit):
    """Actual line, forecast line, shaded likely range and a divider."""
    history = history[history["Date"] >= zoom_start].assign(Series="Actual")
    line_data = pd.concat(
        [
            history.rename(columns={"Value": "Sales"})[
                ["Date", "Sales", "Series"]
            ],
            forecast.rename(columns={"Forecast": "Sales"})[
                ["Date", "Sales"]
            ].assign(Series="Forecast"),
        ]
    )
    colors = alt.Color(
        "Series:N",
        scale=alt.Scale(
            domain=["Actual", "Forecast"], range=[ACTUAL_COLOR, FORECAST_COLOR]
        ),
        legend=alt.Legend(title=None, orient="top", symbolType="stroke"),
    )
    x_axis = alt.X("Date:T", title=None, axis=alt.Axis(format="%b %Y"))
    band = (
        alt.Chart(forecast)
        .mark_area(color=BAND_COLOR, opacity=0.55)
        .encode(
            x=x_axis,
            y=alt.Y("Lower80:Q", title=unit, scale=alt.Scale(zero=False)),
            y2="Upper80:Q",
        )
    )
    lines = (
        alt.Chart(line_data)
        .mark_line(strokeWidth=2)
        .encode(
            x=x_axis,
            y=alt.Y("Sales:Q", scale=alt.Scale(zero=False)),
            color=colors,
            tooltip=[
                alt.Tooltip("Date:T", title="Week ending"),
                alt.Tooltip("Series:N", title=""),
                alt.Tooltip("Sales:Q", title=unit, format=",.1f"),
            ],
        )
    )
    divider = (
        alt.Chart(pd.DataFrame({"Date": [history["Date"].max()]}))
        .mark_rule(strokeDash=[4, 4], color="#7f7f7f")
        .encode(x="Date:T")
    )
    label = (
        alt.Chart(
            pd.DataFrame(
                {
                    "Date": [history["Date"].max()],
                    "Text": ["  forecast starts"],
                }
            )
        )
        .mark_text(align="left", baseline="top", color="#7f7f7f", dy=4)
        .encode(x="Date:T", y=alt.value(0), text="Text:N")
    )
    return alt.layer(band, lines, divider, label).properties(height=380)


METRICS = {
    "WMAPE": {
        "title": "Typical error of one store-week (WMAPE)",
        "format": ".1%",
        "axis": "%",
    },
    "MAE": {
        "title": "MAE: average miss of one store-week (USD)",
        "format": ",.0f",
        "axis": ",.0f",
    },
    "RMSE": {
        "title": "RMSE: like MAE, but big misses count more (USD)",
        "format": ",.0f",
        "axis": ",.0f",
    },
}


def accuracy_bars(evaluation: dict, stat: str = "WMAPE"):
    """Error of each model with its 95% interval (from zero)."""
    spec = METRICS[stat]
    rows = [
        {
            "Model": MODEL_NAMES[name],
            "Error": evaluation["oof_ci"][stat][name][0],
            "Low": evaluation["oof_ci"][stat][name][1],
            "High": evaluation["oof_ci"][stat][name][2],
            "Ours": name == "blend_walk_forward",
        }
        for name in MODEL_NAMES
    ]
    data = pd.DataFrame(rows)
    order = list(data["Model"])
    base = alt.Chart(data).encode(
        y=alt.Y(
            "Model:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelLimit=260),
        ),
    )
    bars = base.mark_bar().encode(
        x=alt.X(
            "Error:Q",
            title=spec["title"],
            axis=alt.Axis(format=spec["axis"], tickCount=8),
        ),
        color=alt.condition(
            alt.datum.Ours,
            alt.value(GOOD_COLOR),
            alt.value(BENCHMARK_COLOR),
        ),
        tooltip=[
            alt.Tooltip("Model:N"),
            alt.Tooltip("Error:Q", format=spec["format"]),
            alt.Tooltip("Low:Q", title="95% low", format=spec["format"]),
            alt.Tooltip("High:Q", title="95% high", format=spec["format"]),
        ],
    )
    whiskers = base.mark_rule(color="#1f2933").encode(x="Low:Q", x2="High:Q")
    labels = base.mark_text(align="left", dx=6, color="#1f2933").encode(
        x="High:Q", text=alt.Text("Error:Q", format=spec["format"])
    )
    return alt.layer(bars, whiskers, labels).properties(height=220)


def base_vs_ours_table(evaluation: dict) -> pd.DataFrame:
    """MAE, RMSE and WMAPE of the base model and ours, side by side."""
    metrics = evaluation["oof_metrics"]
    base, ours = metrics["seasonal_naive"], metrics["blend_walk_forward"]
    rows = []
    for stat in ("MAE", "RMSE", "WMAPE"):
        rows.append(
            {
                "Metric": stat,
                "Base (same as last year)": base[stat],
                "Our forecast (blend)": ours[stat],
                "Change": ours[stat] / base[stat] - 1,
            }
        )
    return pd.DataFrame(rows).set_index("Metric")


def quarter_bars(evaluation: dict):
    """Actual / forecast - 1 for each of the five 13-week test blocks."""
    errors = evaluation["quarter_total_error"]
    data = pd.DataFrame(
        {
            "Block": [f"Block {i}" for i in range(1, len(errors) + 1)],
            "Error": errors,
            "Direction": [
                "Forecast too low" if e > 0 else "Forecast too high"
                for e in errors
            ],
        }
    )
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "Block:N",
                sort=None,
                title="13-week test block (oldest first)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "Error:Q",
                title="Actual / forecast - 1",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(
                    domain=["Forecast too low", "Forecast too high"],
                    range=[GOOD_COLOR, WARN_COLOR],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["Block", alt.Tooltip("Error:Q", format="+.1%")],
        )
        .properties(height=240)
    )


by_store, total, actuals, evaluation = load_bundle()
meta = evaluation["metadata"]
year = int(total["Date"].dt.year.max())
in_year = total[total["Date"].dt.year == year]
year_total = in_year["Forecast"].sum()
range_low, range_high = evaluation["calibration"]["annual_planning_range"]
wmape, wmape_low, wmape_high = evaluation["oof_ci"]["WMAPE"][
    "blend_walk_forward"
]
naive_wmape = evaluation["oof_ci"]["WMAPE"]["seasonal_naive"][0]
last_actual = actuals["Date"].max()

st.title("Weekly sales forecast")
st.caption(
    f"45 stores, weekly. Based on sales up to {last_actual:%d %b %Y}. "
    "The forecast is precomputed by the model."
)

col1, col2, col3 = st.columns(3)
col1.metric(
    f"Forecast for {year}",
    usd(year_total),
    help=f"Sum of {len(in_year)} weekly forecasts, all stores "
    f"({in_year['Date'].min():%d %b} to {in_year['Date'].max():%d %b}). "
    "The last 3 weeks of December are not included.",
)
col2.metric(
    "Likely range (80%)",
    f"{usd(year_total * (1 + range_low))} to "
    f"{usd(year_total * (1 + range_high))}",
    help="Based on the errors of 5 test quarters only. Use it as a guide.",
)
col3.metric(
    "Typical error of one store-week",
    f"{wmape:.1%}",
    help="Total size of the errors divided by total sales, measured on the "
    f"last 65 weeks. 95% interval: {wmape_low:.1%} to {wmape_high:.1%}.",
)
col1.caption(f"{len(in_year)} weeks: only part of the year.")
col2.caption("Where the real number will probably be.")
col3.caption(f"'Same as last year' would be off by {naive_wmape:.1%}.")

tab_company, tab_stores, tab_accuracy, tab_download = st.tabs(
    ["Company total", "Stores", "How accurate is it?", "Download"]
)

with tab_company:
    st.markdown("**All 45 stores together, sales per week**")
    view = st.radio(
        "Show",
        ["Whole history", "Last year and forecast"],
        horizontal=True,
        label_visibility="collapsed",
    )
    start = (
        actuals["Date"].min()
        if view == "Whole history"
        else last_actual - pd.Timedelta(days=364)
    )
    history = (
        actuals.groupby("Date", as_index=False)["Weekly_Sales"]
        .sum()
        .rename(columns={"Weekly_Sales": "Value"})
    )
    history["Value"] /= 1e6
    forecast = total.copy()
    for column in ("Forecast", "Lower80", "Upper80"):
        forecast[column] = forecast[column] / 1e6
    st.altair_chart(
        line_layers(history, forecast, start, "Sales per week (USD M)"),
        use_container_width=True,
    )
    st.caption(
        "Dark line: real sales. Blue line: forecast. Light blue area: where "
        "the weekly total will probably be (80% of the time). The two big "
        "peaks every year are Thanksgiving and Christmas."
    )

with tab_stores:
    growth, n_weeks = growth_vs_last_year(by_store, actuals, year)
    table = (
        by_store[by_store["Date"].dt.year == year]
        .groupby("Store")["Forecast"]
        .sum()
        .div(1e6)
        .rename("Forecast (USD M)")
        .to_frame()
        .join(growth.rename("Change vs last year"))
        .reset_index()
    )
    st.markdown("**Pick a store**")
    picker, _ = st.columns([1, 3])
    store = picker.selectbox(
        "Store",
        sorted(by_store["Store"].unique()),
        label_visibility="collapsed",
    )
    row = table[table["Store"] == store].iloc[0]
    left, right = st.columns(2)
    left.metric(
        f"Store {store}: forecast for {year}",
        usd(row["Forecast (USD M)"] * 1e6),
    )
    right.metric(
        "Change vs the same weeks last year",
        f"{row['Change vs last year']:+.1%}",
        help=f"Compares {n_weeks} forecast weeks with the same weeks of "
        "the year before.",
    )
    store_history = (
        actuals[actuals["Store"] == store][["Date", "Weekly_Sales"]]
        .rename(columns={"Weekly_Sales": "Value"})
        .assign(Value=lambda d: d["Value"] / 1e3)
    )
    store_forecast = by_store[by_store["Store"] == store].copy()
    for column in ("Forecast", "Lower80", "Upper80"):
        store_forecast[column] = store_forecast[column] / 1e3
    st.altair_chart(
        line_layers(
            store_history,
            store_forecast[["Date", "Forecast", "Lower80", "Upper80"]],
            actuals["Date"].min(),
            "Sales per week (USD thousand)",
        ),
        use_container_width=True,
    )
    mult_low, mult_high = evaluation["calibration"]["store_week_mult"]
    st.caption(
        "The light area is wider than for the company total, because one "
        f"store in one week is harder to predict: {mult_low - 1:+.0%} to "
        f"{mult_high - 1:+.0%} around the forecast."
    )
    st.markdown("**All stores**")
    st.dataframe(
        table.sort_values("Change vs last year"),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Store": st.column_config.NumberColumn(format="%d"),
            "Forecast (USD M)": st.column_config.NumberColumn(format="%.1f"),
            "Change vs last year": st.column_config.ProgressColumn(
                format="%+.1f%%",
                min_value=-0.2,
                max_value=0.2,
                help="Sorted from the biggest fall to the biggest rise",
            ),
        },
    )

with tab_accuracy:
    st.markdown(
        "To test the model we pretended not to know the last 65 weeks and "
        "forecast them, then compared with what really happened."
    )
    st.markdown("**1. Base model vs our forecast: MAE and RMSE**")
    if "MAE" not in evaluation["oof_metrics"]["seasonal_naive"]:
        st.warning(
            "evaluation.json has no MAE/RMSE. Commit the new "
            "app_artifacts/evaluation.json (or run python -m speedy.train)."
        )
    else:
        scores = evaluation["oof_metrics"]
        base_scores = scores["seasonal_naive"]
        our_scores = scores["blend_walk_forward"]
        ci = evaluation["oof_ci"]
        cols = st.columns(2)
        for col, stat in zip(cols, ("MAE", "RMSE"), strict=True):
            change = our_scores[stat] / base_scores[stat] - 1
            col.metric(
                f"{stat}, our forecast",
                f"USD {our_scores[stat]:,.0f}",
                delta=f"{change:+.1%} vs base (USD {base_scores[stat]:,.0f})",
                delta_color="inverse",
                help=f"Per store-week. 95% interval: "
                f"USD {ci[stat]['blend_walk_forward'][1]:,.0f} to "
                f"USD {ci[stat]['blend_walk_forward'][2]:,.0f}. "
                "Base = 'same as last year'.",
            )
        compare = base_vs_ours_table(evaluation).reset_index()
        shown = compare.copy()
        for column in ("Base (same as last year)", "Our forecast (blend)"):
            shown[column] = [
                f"{v:.2%}" if m == "WMAPE" else f"USD {v:,.0f}"
                for m, v in zip(
                    compare["Metric"], compare[column], strict=True
                )
            ]
        shown["Change"] = compare["Change"].map("{:+.1%}".format)
        st.dataframe(shown, hide_index=True, use_container_width=True)
        st.caption(
            "MAE and RMSE are in dollars per store-week (lower is better). "
            "RMSE is always larger than MAE; the more it differs from MAE, "
            "the more the model misses by a lot in a few weeks (holidays). "
            "WMAPE is the same error as a share of sales."
        )
    st.markdown("**2. Is our forecast better than a simple one?**")
    available = [m for m in METRICS if m in evaluation["oof_ci"]]
    stat = st.radio(
        "Metric",
        available,
        horizontal=True,
        format_func=lambda k: {
            "WMAPE": "WMAPE (%)",
            "MAE": "MAE (USD)",
            "RMSE": "RMSE (USD)",
        }[k],
        label_visibility="collapsed",
    )
    st.altair_chart(accuracy_bars(evaluation, stat), use_container_width=True)
    gain = evaluation["oof_skill_vs_naive_WMAPE_diff"]["naive_minus_blend"]
    st.caption(
        "Lower is better. The black line is the 95% range of the number. "
        f"Our forecast is {gain[0] * 100:.1f} points better than 'same as "
        f"last year' ({gain[1] * 100:.1f} to {gain[2] * 100:.1f}), "
        "so the gain is real."
    )
    st.markdown("**3. Is the forecast too high or too low?**")
    st.altair_chart(quarter_bars(evaluation), use_container_width=True)
    st.caption(
        "Blue bars: the real sales were higher than forecast. In the "
        "first four blocks the forecast was too low, so the number is on the "
        "cautious side. In the last block it was 0.2% too high."
    )
    st.markdown("**4. Things to keep in mind**")
    holiday = evaluation["holiday_weeks"]
    check = evaluation["coverage_check"]
    st.markdown(
        f"- On the **company weekly total** our forecast is only slightly "
        f"better than 'same as last year'; we cannot say it is clearly "
        f"better.\n"
        f"- In **holiday weeks** ({holiday['n_weeks']} weeks, one season) "
        f"our forecast is not better ({holiday['blend_WMAPE']:.1%} vs "
        f"{holiday['naive_WMAPE']:.1%}).\n"
        f"- The light area held **{check['coverage_80']:.0%}** of the time on "
        "later weeks (target 80%).\n"
        f"- The model was trained only on data up to "
        f"{meta['trained_through']}: it has not seen the 2012 holiday "
        "season."
    )
    with st.expander("Technical table (all metrics, 95% intervals)"):
        rows = []
        for model in evaluation["oof_metrics"]:
            row_data = {"Model": model}
            for stat, table_ci in evaluation["oof_ci"].items():
                value, low, high = table_ci[model]
                if stat in ("MAE", "RMSE"):
                    row_data[stat] = f"{value:,.0f} [{low:,.0f}, {high:,.0f}]"
                else:
                    row_data[stat] = f"{value:.2%} [{low:.2%}, {high:.2%}]"
            rows.append(row_data)
        st.dataframe(pd.DataFrame(rows).set_index("Model"))
        st.caption(
            "MAE and RMSE are in USD per store-week. "
            "Bias = (forecast - actual) / actual. Negative means the "
            "forecast was too low."
        )

with tab_download:
    st.markdown("Forecast tables as CSV files.")
    st.download_button(
        "Download forecast by store",
        by_store.to_csv(index=False),
        "forecast_by_store.csv",
        "text/csv",
    )
    st.download_button(
        "Download company total",
        total.to_csv(index=False),
        "forecast_total.csv",
        "text/csv",
    )
