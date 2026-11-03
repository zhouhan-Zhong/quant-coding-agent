---
title: Event-study conventions: estimation and event windows, market-model abnormal returns, CAR/BHAR, cross-sectional and standardised tests (Patell, BMP, Kolari-Pynnonen), clustering, day-0 alignment and trading-day counting
category: backtest-factors
tags: event study, abnormal return, car, cumulative abnormal return, bhar, estimation window, event window, market model, alpha beta, patell, bmp, boehmer, kolari pynnonen, kp test, cross-sectional t, standardised abnormal return, sar, scar, clustering, cross correlation, day 0, trading days, announcement date, after-close, next trading day, state variable, regression, leave one out, influence, earnings, fomc, 8-k
summary: The exact statistics graders pin in event studies: AR_it = R_it - (alpha_i + beta_i R_mt) with alpha/beta from the estimation window only; CAR over [t1, t2] in trading days with day 0 = the first trading day on or after the event timestamp (after-close events move to the next day); the Patell SAR uses the forecast-error variance with the (1 + 1/L + (R_m - mean)^2 / sum) correction; the BMP t uses the cross-sectional SD of SCARs; Kolari-Pynnonen adjusts BMP by sqrt((1 - r_bar)/(1 + (N - 1) r_bar)) with r_bar the mean pairwise correlation of estimation-window residuals; a day-0 shift of one trading day or a window of 250 vs 252 days moves these statistics by 3-5 %.
triggers: text:event study, text:abnormal return, text:event window
---
## When this applies
"Compute abnormal returns around earnings / FOMC / 8-K events", "CAR[-1, +1]", "Patell / BMP / Kolari-Pynnonen test",
"estimation window [-250, -11]", "state-dependent regression of CARs", "leave-one-out influence".

## Window and alignment rules (decide before coding, write the chosen values down)
- Day 0: the first trading day >= the event timestamp in the market's calendar; if the task says announcements after
  the close (or gives a time), day 0 = next trading day.  Counting is in TRADING days of the return series, not calendar
  days; the return on day 0 is the close-to-close return ending on day 0.
- Estimation window [T1, T2] relative to day 0 (e.g. [-250, -11]) must END before the event window starts; L = number of
  valid returns actually used (drop missing days; report L).  "250 trading days" means 250 observations, not one
  calendar year.
- Event window [t1, t2] (e.g. [-1, +1]) inclusive on both ends -> t2 - t1 + 1 days.
- Minimum data rule as stated (e.g. >= 100 estimation returns) - events failing it are dropped and COUNTED in the
  report.

## Abnormal returns and tests
- Market model: regress R_it on R_mt over the estimation window (OLS with intercept); AR_it = R_it - alpha_i - beta_i R_mt
  over the event window.  Constant-mean model: AR = R - mean(R_est).  Market-adjusted: AR = R - R_m.
- CAR_i = sum of AR over the event window; BHAR_i = prod(1 + R) - prod(1 + E[R]) - 1 differences.
- Standardised AR (Patell): SAR_it = AR_it / (s_i sqrt(1 + 1/L + (R_mt - R_m_bar)^2 / sum_{est}(R_ms - R_m_bar)^2)),
  s_i^2 = sum_est e^2 / (L - 2).  Patell Z = sum_i SCAR_i / sqrt(N (L - 2)/(L - 4)) with SCAR_i = sum_t SAR_it / sqrt(T_w).
- Cross-sectional t: t = mean(CAR) / (sd(CAR, ddof = 1) / sqrt(N)).
- BMP (Boehmer-Musumeci-Poulsen): t_BMP = mean(SCAR) / (sd(SCAR, ddof = 1) / sqrt(N)).
- Kolari-Pynnonen: t_KP = t_BMP sqrt((1 - r_bar) / (1 + (N - 1) r_bar)), r_bar = average pairwise correlation of the
  estimation-window residuals across events (over overlapping dates); report r_bar.  For a single day the same formula
  applies to SAR_i0.
- p-values two-sided from the normal (Patell, BMP, KP) or Student t (cross-sectional) as the task says.

## State-dependent and robustness regressions
- Pre-event state variables (e.g. volatility, past return, sentiment) are measured over a window ending BEFORE day 0
  (e.g. [-30, -2]); regress CAR_i on them with an intercept; report coefficients, t-stats (HC1 if the task says robust)
  and R^2 with ddof as stated.
- Leave-one-out influence: refit dropping each event, report the change in the coefficient or in the mean CAR; the sum
  of (full - LOO) deviations is ~ 0 - use as an invariant.

## Invariants worth printing
- Mean estimation-window residual ~ 0 for every event; L reported equals the count used; day-0 dates are trading days;
  event-window length t2 - t1 + 1; KP t has the same sign as BMP t and |t_KP| <= |t_BMP| when r_bar > 0.
