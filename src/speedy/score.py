"""Scoring entry point.   python -m speedy.score --end-date 2013-12-06 [--model PATH] [--out-dir DIR]"""
import argparse
from pathlib import Path
from .config import MODEL_PATH, OUTPUT_DIR
from .inference import load_artifact, forecast_weekly, forecast_totals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(MODEL_PATH)); ap.add_argument("--end-date", required=True)
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR)); a = ap.parse_args(argv)
    art = load_artifact(a.model); w = forecast_weekly(art, a.end_date); t = forecast_totals(art, w)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    w.to_csv(out / "forecast_by_store.csv", index=False); t.to_csv(out / "forecast_total.csv", index=False)
    lo, hi = art["calibration"]["annual_planning_range"]
    yr = t.Date.dt.year.max(); ty = t[t.Date.dt.year == yr]["Forecast"].sum()
    print(f"model trained through {art['metadata']['trained_through']}; forecast {t.Date.min().date()}..{t.Date.max().date()} ({len(t)} weeks)")
    print(f"whole horizon total: ${t['Forecast'].sum()/1e6:,.0f}M")
    print(f"calendar {yr} weeks in horizon ({(t.Date.dt.year == yr).sum()} wks): ${ty/1e6:,.0f}M | "
          f"backtest says actual/forecast-1 in [{lo:+.1%}, {hi:+.1%}] -> ${ty*(1+lo)/1e6:,.0f}M .. ${ty*(1+hi)/1e6:,.0f}M (80%, n=5 quarters: indicative)")
    print("written:", out / "forecast_by_store.csv", out / "forecast_total.csv")


if __name__ == "__main__":
    main()
