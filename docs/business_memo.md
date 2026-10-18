# 2013 sales forecast: recommendation for the CFO

## Recommendation
* **Use about $2.35B as the sales guidance for 4 Jan - 6 Dec 2013** (49
  weeks, 45 stores): $2,348M. This is about 1.5% above the same Jan-Oct
  weeks of 2012. Indicative 80% range: **$2.32B to $2.46B** (-1.2% to +4.9%
  around the base). The range comes from only five quarters of errors: it is
  a guide, not a guarantee.
* **This is not the full calendar year.** The task stops on 2013-12-10, so
  the last three weeks of December, the biggest of the year, are not in
  this number. Do not publish it as "2013 sales".
* **Run in shadow mode until the Nov-Dec 2012 actuals arrive**, then decide
  whether to publish. The data stops on 26 Oct 2012, so the biggest weeks of
  the year have not been seen. Also, model choices were made while looking at
  2011-2012, so the accuracy below is probably a little optimistic.

## How accurate is it?
Tested by forecasting the last 65 weeks in five blocks of 13 weeks.
* **One store, one week:** off by about **4.4%** of sales (95% CI 3.8% to
  5.3%). "Same as last year" is off by 5.6% (5.2% to 6.2%). The gain is about
  1.2 points (CI 0.5 to 1.8), so it is real.
* **One store, whole test period:** off by about **2.2%**, against 4.1% for
  "same as last year".
* **Company weekly total:** off by about **2.9%** (2.1% to 3.8%), against
  3.1% for "same as last year". **Here the blend is not clearly better**, and
  in the one 58-week test it was slightly worse (3.8% vs 3.6%). Guidance on
  the total is only as good as the simple benchmark.
* **Trends:** adding a gentle store trend cut the store-week error from 4.9%
  to 4.4% and the store-total error from 3.2% to 2.2%. The forecast growth of
  a store now follows its recent growth (correlation 0.86). This shows the
  model continues trends, not that the trends will continue.
* **Direction of the error:** the forecast was too low in the first four
  blocks (actual / forecast - 1 = +3.7%, +3.5%, +2.0%, +0.2%) and 0.2% too
  high in the last. Overall bias is -1.8% (CI -3.0% to -0.8%). The latest two
  blocks look unbiased, but two blocks are not enough to say the bias is
  gone, and the 58-week test was still 3.5% too low.
* **58-week test (the real horizon):** 5.3% vs 5.9% for naive, forecast 3.5%
  too low. One year only, with settings that had seen part of it: a data
  point, not a proof.
* **Bands:** the 80% store-week band is about -4% / +12% around the forecast
  (it is lopsided because the forecast tends to be low). On later unseen
  weeks it held 86% of the time (target 80%). For the company weekly total,
  the 80% band is -0.3% / +7.3%.

## Weak spots
* **Holiday weeks:** blend 6.5% vs naive 5.7% in the 7 holiday weeks we can
  test. This is within noise (difference interval about -4 to +3 points) and
  it is one season, so we did not tune for holidays. Holiday weeks carry most
  of the money, so this matters.
* **Long horizon:** errors grow with distance, so late-2013 numbers are less
  reliable. Nov-Dec 2013 is copied from Nov-Dec 2011 (2012 is not seen).
* **Stores with strong trends.** In the data (2012 vs 2011, same 43 weeks),
  store 36 fell 17%, store 14 fell 9%, stores 38 and 44 rose 14% and 12%.
  The forecast for 2013 is -12% (36), -13% (14) and +9.6% (44). Store 35 is
  forecast +9% although it grew only 2%: check it before use.

## What an error costs
* **Forecast too high** (actual below plan) risks missing published
  guidance: the costly direction. **Too low** means cautious guidance. Past
  errors were mostly in the cautious direction.
* Store numbers are less reliable than the total in percent terms of a
  single week, but over a long period store errors are small (2.2%). Use
  stores for planning and the total for guidance.

## Monitoring and rules for going from shadow to production (to agree)
Track every week: 13-week WMAPE, bias, share of actuals inside the 80% band,
and the same for the naive forecast.
* **Go live** after 8+ weeks of shadow data with WMAPE <= 5.5%, |bias| <= 4%,
  band coverage 70-90%, and better than naive.
* **Investigate or retrain** if for 4 weeks in a row the company total
  misses by more than 8%, coverage falls under 60%, or the bias changes sign
  and stays.
* **Always retrain** when the Nov-Dec 2012 actuals arrive.

## Data decisions
* **Returns** (negative lines, 0.3% of rows) stay in the sales: the effect on
  totals is about 0.001%.
* **Outliers:** one real non-holiday outlier (store 18, 2 Sep 2011). Such
  cells are capped using training data only. No material effect (4.47% vs
  4.48%).
* **`features.csv` is not used:** it has rows up to July 2013, which cannot
  exist on 10 Dec 2012.

## What we do not model
Store openings and closures, promotions, macro shocks, department mix.
Only 2.7 years of history.

## Technical notes
* Blend: seasonal naive with store drift 60%, Prophet 20%, damped Holt 20%.
* Five 13-week blocks; weights for a block come only from earlier blocks;
  95% intervals from a moving-block bootstrap over weeks; paired bootstrap
  against naive.
* Leakage tests: changing all data after the forecast origin does not move
  the forecasts.
* The "range" uses the spread of five quarterly errors for a 49-week total.
  It is only a rough guide.

## Likely questions
* *Why not CPI or unemployment?* Their 2013 values are unknown today, and
  using real values in a backtest would flatter the model.
* *Did you tune on the test set?* Blend weights were chosen walk-forward;
  the model design was not. That is why we suggest shadow mode.
* *Why not fix the holiday weeks?* One holiday season cannot separate a real
  weakness from noise.
