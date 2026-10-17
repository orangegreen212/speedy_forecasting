# 2013 sales forecast: recommendation for the CFO

## Recommendation
* **Base guidance on about $2.35B of sales for 4 Jan - 6 Dec 2013 (49 weeks, 45 stores)**: $2,348M, about 1.5% above the same weeks of 2012. Indicative 80% range for the outcome: **$2.32B to $2.46B** (-1.2% to +4.9% around the base). The range rests on five quarters only: a guide, not a guarantee.
* **Run in shadow mode until the November-December 2012 actuals arrive**, then decide whether to publish. Our data stops on 26 Oct 2012, so the biggest weeks of the year have not been seen. Also, modelling choices were made while looking at 2011-2012, so the accuracy below is probably a little optimistic. Only new data gives a clean test.

## Why we are fairly (not fully) confident
* A typical store-week is off by about **4.4%** (95% CI 3.8% to 5.3%); a naive "same as last year" forecast is off by 5.6% (5.2% to 6.2%). The gain is 1.2 points (CI 0.5 to 1.8).
* Each store's total over the test period is off by about **2.2%**, against 4.1% for the naive forecast. The company weekly total is off by about 2.9% (2.1% to 3.8%).
* **Store trends are now handled.** The first version lagged stores that steadily grow or decline (store 36 falls about 17% a year but was forecast -4%). Seasonal naive *with drift* plus a damped trend model fixed most of it: forecast growth now tracks each store's observed growth (correlation 0.86, was 0.49) and the store-week error fell from 4.9% to 4.4%.
* **Bias:** the forecast was too low in all earlier quarters; after the trend fix the quarter errors fade to about zero (+3.7%, +3.5%, +2.0%, +0.2%, -0.2%). Overall bias -1.8% (CI -3.0% to -0.8%). A further bias correction was tested and **not adopted** (no proven gain).
* Single 58-week "year-ahead" test (the real horizon): 5.3% vs 5.9% naive, forecast 3.5% low. **One year, with weights that saw part of those weeks: a data point, not a proof.**
* The 80% store-week band is about -4% / +12%. On later unseen weeks it held 86% of the time (target 80%). For the company weekly total, 80% of backtest weeks fell between 0.3% below and 7% above the forecast.

## Weak spots to say out loud
* **Holiday weeks:** blend 6.5% vs naive 5.7% on the 7 holiday weeks we can test. Within noise (CI about -4 to +3 points), one season only, so no holiday-specific tuning.
* **Long horizon:** errors grow with distance, so late-2013 numbers are less reliable than the headline.
* **Stores with strong trends** (14 and 36 falling about 12-13%, 35 rising about 9%): check for closures, remodels or new competitors; the model only continues recent history, damped.

## What an error costs
* **Over-forecast** (actual below plan) risks missing published guidance: the costly direction. **Under-forecast** means conservative guidance. Remaining bias is in the safe direction.
* Store numbers are less reliable than the total (errors partly cancel). Use stores for planning, the total for guidance.

## Monitoring and criteria for going from shadow to production (proposed thresholds, to agree)
Track weekly: rolling 13-week WMAPE, bias, share of actuals inside the 80% band, and the same for the naive benchmark.
* **Go live** after 8+ weeks of shadow data with WMAPE <= 5.5%, |bias| <= 4%, band coverage 70-90%, and better than naive.
* **Investigate / retrain** if for 4 consecutive weeks the company total misses by more than 8%, coverage drops under 60%, or bias flips sign persistently.
* **Always retrain** when Nov-Dec 2012 actuals land; re-check holiday weeks against naive.

## Data decisions
* **Returns** (negative lines, 0.3% of rows) stay netted in sales: reported revenue is net, and the effect on totals is 0.001%.
* **Anomalies:** one genuine non-holiday anomaly (store 18, 2 Sep 2011: sales shifted between weeks). Such cells are capped using training data only; holiday-week differences are real behaviour and left alone. No material effect on accuracy (4.47% vs 4.48%).
* **`features.csv` is not used:** it contains rows up to July 2013, i.e. information that cannot exist on 10 Dec 2012.

## What we do not model
Store openings/closures, promotions, macro shocks, department mix. Only 2.7 years of history.

## Technical evidence (for questions)
* Blend: seasonal naive with store drift 60%, Prophet 20%, damped Holt trend 20% (ridge got zero weight).
* Five contiguous 13-week out-of-fold blocks; weights for each block chosen only from earlier blocks; CIs by moving-block bootstrap over weeks; paired bootstrap versus naive.
* Leakage tests in `tests/test_audit.py`: scrambling all post-origin data leaves forecasts unchanged (float noise only), including Prophet; cleaning and drift use the training slice only; calendar-only features.

## Likely questions
* *"Why not CPI / unemployment?"* Their 2013 values are unknown today; realised values in a backtest would flatter the model.
* *"Did you tune on the test set?"* Weights were selected walk-forward; model design was not, which is why we recommend shadow mode.
* *"Why not fix holiday weeks?"* One season of holiday data cannot separate a real weakness from noise.
