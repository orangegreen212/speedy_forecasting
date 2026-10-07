# 2013 sales forecast: recommendation for the CFO

## Recommendation
* **Base guidance on $2.32B of sales for 4 Jan - 6 Dec 2013 (49 weeks, 45 stores)**, essentially flat on the same weeks of 2012.
* **Treat $2.32B as a conservative base, not a midpoint.** In every one of the last five quarters the model came in 1-4% *below* actual sales. Our indicative 80% range for the outcome is **$2.33B to $2.42B** (0.5% to 4.3% above the base). This range rests on only five quarters, so use it as a guide, not a guarantee.
* **Run the forecast in shadow mode until the November-December 2012 actuals arrive**, then decide whether to publish. Our data stops on 26 Oct 2012, so the 2012 holiday season, the biggest weeks of the year, is not yet seen.

## Why we are fairly (not fully) confident
* A typical store-week is off by about **4.9%** (95% CI 4.4% to 5.5%); a naive "same as last year" forecast is off by 5.6% (5.2% to 6.2%). The improvement is small but real: 0.7 points (CI 0.2 to 1.2).
* The whole company's weekly total is off by about **3.0%** (CI 2.4% to 3.8%); each store's total over the 65 backtest weeks by about 3.2%.
* Single 58-week "year-ahead" test (the real horizon): 5.3% store-week error and a 3.1% under-forecast. **This is one year, so it is a data point, not a distribution.**
* The 80% store-week band is about -4% / +12% around the forecast. On later, unseen weeks it contained 89% of actuals (target 80%), so it is slightly wide.

## What an error costs
* **Over-forecast** (actual below plan) risks missing published guidance: the costly direction. **Under-forecast** means conservative guidance: embarrassing only if repeated. Our known bias is in the safe direction.
* Store-level numbers are less reliable than the total (errors partly cancel across stores). Use store forecasts for planning, the total for guidance.

## Monitoring and criteria for going from shadow to production (proposed thresholds, to agree)
Track every week as actuals arrive: rolling 13-week WMAPE, bias, share of actuals inside the 80% band, and the same numbers for the naive benchmark.
* **Go live** after 8+ weeks of shadow data with: WMAPE <= 5.5%, |bias| <= 4%, band coverage 70-90%, and better than naive.
* **Investigate / retrain** if for 4 consecutive weeks the company total misses by more than 8%, or coverage drops under 60%, or bias flips sign persistently.
* **Always retrain** when Nov-Dec 2012 actuals land and before any quarterly update.

## What we do not model
Store openings/closures, promotions (MarkDowns only exist from Nov 2010 and are unknown for 2013), macro shocks, department mix. Only 2.7 years of history, so a change in growth would not be visible yet.

## Technical evidence (for questions)
* Blend of seasonal-naive (holiday-aligned), pooled ridge by store type, and per-store Prophet; weights 40/20/40, ridge alpha 10.
* Five contiguous 13-week out-of-fold blocks; weights for each block chosen only from earlier blocks; CIs by moving-block bootstrap over weeks.
* Leakage controls: calendar-only features, strict time splits, test that scrambling post-origin data leaves forecasts unchanged.
* Weights tuned on all folds gave no gain over equal weights on the 58-week check (5.31% vs 5.28%), so the result does not depend on tuning.

## Likely questions
* *"Why not use CPI / unemployment?"* Their 2013 values are unknown; using realised values in a backtest would flatter the model.
* *"Why is the range above the base?"* Because the backtests under-forecast every quarter; we report that instead of hiding it.
* *"How sure are you the blend beats naive?"* Paired CI excludes zero on the out-of-fold weeks (0.2 to 1.2 points), but the gain is modest.
