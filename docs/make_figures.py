"""Make the pictures used in README.md.

Reads only the files in app_artifacts/, so it needs no model and no raw data.
Run from the repo root:  python docs/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "app_artifacts"
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

NAVY, ORANGE, GREY, MUTED = "#1F2A44", "#E07A2F", "#9AA5B1", "#5B6475"
BAND = "#F3C9A8"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": "#D5D9E3",
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E6E9F0",
        "axes.axisbelow": True,
    }
)

evaluation = json.loads((BUNDLE / "evaluation.json").read_text())
actuals = pd.read_csv(BUNDLE / "actuals_store_week.csv", parse_dates=["Date"])
total = pd.read_csv(BUNDLE / "forecast_total.csv", parse_dates=["Date"])
backtest = pd.read_csv(
    BUNDLE / "backtest_store_week.csv", parse_dates=["Date"]
)


def finish(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / name)


# 1. Forecast ---------------------------------------------------------------
history = actuals.groupby("Date")["Weekly_Sales"].sum() / 1e6
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(history.index, history, color=NAVY, lw=1.8, label="Actual sales")
ax.fill_between(
    total["Date"],
    total["Lower80"] / 1e6,
    total["Upper80"] / 1e6,
    color=BAND,
    alpha=0.7,
    lw=0,
    label="Likely range (80%)",
)
ax.plot(
    total["Date"],
    total["Forecast"] / 1e6,
    color=ORANGE,
    lw=2,
    label="Forecast",
)
ax.axvline(history.index.max(), color=MUTED, ls=":", lw=1)
ax.text(
    history.index.max(),
    ax.get_ylim()[1],
    " data ends 26 Oct 2012",
    color=MUTED,
    va="top",
    fontsize=9,
)
ax.set_ylabel("Weekly sales, all 45 stores (USD M)")
ax.set_title(
    "Weekly sales and the forecast to Dec 2013", loc="left", color=NAVY
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend(frameon=False, loc="upper left", ncol=3)
finish(fig, "01_forecast.png")

# 2. Backtest: actual vs simple vs ours -------------------------------------
weekly = backtest.groupby("Date")[["Actual", "Baseline", "Ours"]].sum() / 1e6
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(weekly.index, weekly["Actual"], color=NAVY, lw=2, label="Actual")
ax.plot(
    weekly.index,
    weekly["Baseline"],
    color=GREY,
    lw=1.8,
    ls="--",
    label="Simple: same as last year",
)
ax.plot(
    weekly.index, weekly["Ours"], color=ORANGE, lw=2.2, label="Our forecast"
)
ax.set_ylabel("Weekly sales, all stores (USD M)")
ax.set_title(
    "Test on the last 65 weeks (each block forecast from earlier data only)",
    loc="left",
    color=NAVY,
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend(frameon=False, loc="upper left", ncol=3)
finish(fig, "02_backtest.png")

# 3. Errors of every model --------------------------------------------------
names = {
    "seasonal_naive": "Same as last year",
    "naive_with_drift_d5": "Last year + growth",
    "holt_alone": "Holt trend",
    "prophet": "Prophet",
    "blend_walk_forward": "Our blend",
}
panels = [
    ("MAE", "MAE (USD per store-week)", "{:,.0f}"),
    ("RMSE", "RMSE (USD per store-week)", "{:,.0f}"),
    ("WMAPE", "WMAPE (% of sales)", "{:.1%}"),
]
order = sorted(names, key=lambda k: -evaluation["oof_ci"]["MAE"][k][0])
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, (stat, title, fmt) in zip(axes, panels, strict=True):
    values = [evaluation["oof_ci"][stat][k] for k in order]
    colors = [ORANGE if k == "blend_walk_forward" else GREY for k in order]
    y = range(len(order))
    ax.barh(list(y), [v[0] for v in values], color=colors, height=0.6)
    ax.errorbar(
        [v[0] for v in values],
        list(y),
        xerr=[
            [v[0] - v[1] for v in values],
            [v[2] - v[0] for v in values],
        ],
        fmt="none",
        ecolor=NAVY,
        lw=1.2,
        capsize=3,
    )
    for i, v in enumerate(values):
        ax.text(
            v[2],
            i,
            "  " + fmt.format(v[0]),
            va="center",
            fontsize=9,
            color=NAVY,
        )
    ax.set_title(title, loc="left", color=NAVY, fontsize=11)
    ax.set_xlim(0, max(v[2] for v in values) * 1.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    if stat == "WMAPE":
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0%}")
        )
    else:
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}K")
        )
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(list(range(len(order))))
axes[0].set_yticklabels([names[k] for k in order])
fig.suptitle(
    "Lower is better. Black lines: 95% interval.",
    x=0.01,
    ha="left",
    fontsize=10,
    color=MUTED,
    y=1.02,
)
finish(fig, "03_errors.png")

# 4. The blend --------------------------------------------------------------
weights = [
    ("Same week last year\n+ store growth", 60, NAVY),
    ("Prophet\n(one per store)", 20, ORANGE),
    ("Damped Holt\ntrend", 20, GREY),
]
fig, ax = plt.subplots(figsize=(10, 1.9))
left = 0
for label, w, color in weights:
    ax.barh(0, w, left=left, color=color, height=0.55, edgecolor="white")
    ax.text(
        left + w / 2,
        0,
        f"{w}%",
        ha="center",
        va="center",
        color="white",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        left + w / 2,
        -0.5,
        label,
        ha="center",
        va="top",
        fontsize=10,
        color=NAVY,
    )
    left += w
ax.set_xlim(0, 100)
ax.set_ylim(-1.2, 0.5)
ax.axis("off")
ax.set_title(
    "The forecast is a weighted average of three models",
    loc="left",
    color=NAVY,
)
finish(fig, "04_blend.png")

# 5. Bias by quarter --------------------------------------------------------
errors = evaluation["quarter_total_error"]
labels = [f"Block {i + 1}" for i in range(len(errors))]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.bar(
    labels,
    [e * 100 for e in errors],
    color=[ORANGE if e > 0 else NAVY for e in errors],
    width=0.6,
)
for i, e in enumerate(errors):
    ax.text(
        i,
        e * 100 + (0.15 if e >= 0 else -0.15),
        f"{e:+.1%}",
        ha="center",
        va="bottom" if e >= 0 else "top",
        color=NAVY,
        fontsize=10,
    )
ax.axhline(0, color=MUTED, lw=1)
ax.set_ylabel("Actual / forecast - 1 (%)\n(above 0 = forecast too low)")
ax.set_title(
    "Forecast was too low in older blocks, close to zero now",
    loc="left",
    color=NAVY,
)
ax.set_ylim(-1.2, 4.6)
finish(fig, "05_bias_by_block.png")
