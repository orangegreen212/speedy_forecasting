"""Scoring command.

python -m speedy.score --end-date 2013-12-27 [--model PATH]
                       [--out-dir DIR]
"""

import argparse
from pathlib import Path

from .config import MODEL_PATH, OUTPUT_DIR
from .inference import forecast_totals, forecast_weekly, load_artifact


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    artifact = load_artifact(args.model)
    weekly = forecast_weekly(artifact, args.end_date)
    totals = forecast_totals(artifact, weekly)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out_dir / "forecast_by_store.csv", index=False)
    totals.to_csv(out_dir / "forecast_total.csv", index=False)

    low, high = artifact["calibration"]["annual_planning_range"]
    year = totals["Date"].dt.year.max()
    in_year = totals[totals["Date"].dt.year == year]
    year_total = in_year["Forecast"].sum()
    trained_through = artifact["metadata"]["trained_through"]
    print(
        f"model trained through {trained_through}; forecast "
        f"{totals['Date'].min().date()}..{totals['Date'].max().date()} "
        f"({len(totals)} weeks)"
    )
    print(f"whole horizon total: ${totals['Forecast'].sum() / 1e6:,.0f}M")
    print(
        f"calendar {year}, {len(in_year)} weeks in the horizon: "
        f"${year_total / 1e6:,.0f}M. Backtest range for "
        f"actual/forecast-1: [{low:+.1%}, {high:+.1%}] -> "
        f"${year_total * (1 + low) / 1e6:,.0f}M .. "
        f"${year_total * (1 + high) / 1e6:,.0f}M "
        "(80%, based on 5 quarters: indicative)"
    )
    print(
        "written:",
        out_dir / "forecast_by_store.csv",
        out_dir / "forecast_total.csv",
    )


if __name__ == "__main__":
    main()
