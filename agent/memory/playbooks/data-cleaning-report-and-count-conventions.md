---
title: Data-cleaning reports and count conventions: what "rows removed", "valid observations", "non-trading days", "outliers", "duplicates" and "sample window" mean, how counts chain, timestamps and de-duplication rules, warm-up periods
category: data-engineering
tags: cleaning report, rows removed, rows after cleaning, duplicates, dedupe, outliers, winsorize, non-trading days, weekends, holidays, missing values, forward fill, valid observations, observation count, sample window, warm-up, lookback, rolling window, first valid date, last valid date, inclusive, exclusive, timestamp, utc, timezone, end of day, resample, rebalance dates, counts, reconciliation, audit
summary: Counts are the most frequently pinned numbers in graders and the easiest to get wrong by one step: each cleaning step removes rows from the OUTPUT of the previous step and the report lists them in the task's order; "non-trading days" are rows whose date is not in the exchange calendar (or weekends when no calendar is given), "outliers" are rows outside the stated rule applied to the cleaned-but-not-yet-filtered data, duplicates are judged on the task's key columns keeping the first (or last) occurrence as stated; "valid observations" of a rolling statistic start after the warm-up (window - 1 rows dropped), and a sample window given as dates is inclusive on both ends unless the task says otherwise.
triggers: text:rows removed, text:cleaning report, text:num_valid_observations, text:valid observations
---
## When this applies
"Report the number of rows removed at each step", "n_valid_observations", "clean the price file (duplicates, missing,
non-trading days, outliers)", "warm-up of 60 days", "rebalance monthly", "the first and last date used", timestamp
alignment of intraday / crypto data, de-duplicating snapshots.

## Chain the counts in the order the task lists the steps
n_raw -> after dropping exact duplicates -> after dropping missing prices -> after removing non-trading days ->
after outlier filter -> n_after_cleaning.  Each "n_<step>_removed" counts rows removed BY THAT STEP ONLY, and the sum of
removed plus n_after equals n_raw (print this as an invariant).  If the task lists steps in a different order, follow
its order - the outlier count depends on what was removed before it.

## Definitions to pin down from the task text
- Duplicates: on which columns (date? date + ticker? full row?) and which occurrence is kept (keep="first" unless told).
  Snapshot de-duplication in market data usually means "same (venue, market, timestamp)" or "same timestamp rounded to
  the task's resolution" - rounding changes the count.
- Missing: NaN in which columns; "missing" rows versus rows to forward-fill (ffill is not a removal; count it separately
  if asked).
- Non-trading days: dates not in the exchange calendar (pandas_market_calendars if available; else business days
  minus the task's holiday list; else weekends only) - state which rule you used.  Rows with zero volume are NOT
  non-trading days unless the task defines them so.
- Outliers: the rule as written (|return| > 50 %, > k standard deviations, outside percentile bounds); computed on which
  series (simple or log return, price level), and whether the threshold is applied once or iteratively.  Winsorizing
  replaces values and removes nothing.
- Returns: pct_change versus log; the first return row is NaN and is dropped - this reduces counts by one; say whether
  "number of returns" or "number of prices" is reported.
- Rolling statistics: a window of w gives its first valid value at row w (min_periods = w); "valid observations" =
  n - (w - 1); with an additional lag/shift it is n - w.  A 36-month lookback with monthly data drops 35 (or 36) rows -
  check the task's example if it gives one.
- Sample window given as dates: filter date >= start and date <= end (inclusive) unless "before"/"up to but excluding";
  with timestamps, "2023-12-31" as an end means 2023-12-31 23:59:59 for intraday data unless the task says end of day
  00:00.  Timezone: convert to the task's zone (UTC unless stated) before any date-based filtering; a 16:00 vs 00:00
  end point changes the last-day count.
- Rebalance dates: "monthly" = last trading day of each month (or first, as stated); the number of rebalances equals the
  number of month-ends inside the window AFTER the warm-up, which is often one less than the number of months.

## Reporting
- Report counts as integers, dates as ISO strings in the task's format, and keep the first/last date actually used.
- If the task's own example states a count (e.g. "the clean file has 1234 rows"), reproduce exactly that pipeline.

## Invariants worth printing
- n_raw == n_after + sum(removed by each step); first_date <= last_date and both inside the requested window;
  no duplicates remain on the key columns; no NaN in the cleaned columns; rolling-statistic length == n - window + 1.
