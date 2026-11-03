---
title: Pairs trading and statistical arbitrage conventions: Engle-Granger and Johansen cointegration, hedge ratios, spread z-scores, OU half-life, Kalman-filter dynamic hedge ratio, signal timing and spread P&L
category: backtest-factors
tags: pairs trading, statistical arbitrage, stat arb, cointegration, engle-granger, johansen, adf, augmented dickey fuller, unit root, coint, hedge ratio, spread, z-score, rolling mean, entry threshold, exit threshold, stop loss, mean reversion, ornstein uhlenbeck, ou, half-life, kalman filter, dynamic hedge ratio, state space, dollar neutral, market neutral, spread pnl
summary: Hedge ratio from an OLS of y on x (with intercept, in the price units the task names; the regression direction matters); Engle-Granger tests the residual with MacKinnon critical values (statsmodels coint), not plain ADF tables; Johansen trace/max-eigen statistics against cvt/cvm columns; z = (spread - rolling mean) / rolling std over the stated window; positions decided at t earn the return of t+1; spread P&L = q_{t-1} (dy - beta dx); OU half-life = -ln 2 / ln(1 + b) from ds_t = a + b s_{t-1}; the Kalman hedge ratio uses the one-step prediction error e_t and its variance S_t (z = e_t / sqrt(S_t)).
triggers: text:cointegrat, text:pairs trading, text:pair trading, text:kalman, text:engle-granger, text:johansen
---
## When this applies
"Test the pair for cointegration", "estimate the hedge ratio", "trade the spread when its z-score exceeds 2 and exit at 0.5",
"half-life of mean reversion", "Kalman filter hedge ratio", "report the strategy Sharpe / number of trades".

## Cointegration tests
- Engle-Granger: OLS y_t = alpha + beta x_t + e_t (levels, or log prices when the task says log), then a unit-root test of e_t.
  `statsmodels.tsa.stattools.coint(y, x, trend="c")` returns (t-stat, p-value, crit) with MacKinnon critical values for
  residuals; `adfuller(resid)` p-values are too optimistic for an estimated spread.  Regressing x on y gives another beta
  and statistic: use the direction the task states (dependent = first asset named).
- Johansen: `coint_johansen(np.column_stack([y, x]), det_order=0, k_ar_diff=1)`; trace statistics `lr1` vs `cvt[:, 0..2]`
  (90/95/99 %), max-eigen `lr2` vs `cvm`; rank r = number of trace statistics above the chosen critical value, tested in
  order r = 0, 1, ...; the cointegrating vector is `evec[:, 0]`, normalised so the first asset's weight is 1.
- ADF on a single series: `adfuller(x, regression="c", autolag="AIC")` unless the task fixes `maxlag`; p < 0.05 rejects
  the unit root.

## Spread, signals and P&L
- Spread s_t = y_t - beta x_t (- alpha if the task includes it).  In-sample beta for the whole sample is look-ahead; a
  rolling or expanding beta uses data up to t only.
- z_t = (s_t - mean_W(s)) / std_W(s), pandas `rolling(W).mean()/std()` (std ddof=1; window includes t).  If the task says
  "using information up to t-1", shift the rolling statistics by one.
- Rules (typical): z > entry -> short the spread (sell 1 y, buy beta x); z < -entry -> long; exit when |z| < exit (or z
  crosses 0); optional stop at |z| > stop.  Hold the position until an exit, do not re-enter on the same bar unless stated.
- Timing: the position decided on bar t applies to the change from t to t+1: pnl_t = q_{t-1} * (dy_t - beta dx_t).
  Return-based variants use weights (+1/(1+|beta|), -beta/(1+|beta|)) or dollar-neutral notionals; transaction costs are
  charged on |q_t - q_{t-1}| (per unit or in bps of traded notional, as stated).
- Count a "trade" as an entry (or as a round trip) exactly as the task defines; report the number of entries separately
  from exits when asked.

## Mean reversion speed
- AR(1) on the spread: ds_t = a + b s_{t-1} + eps (OLS with intercept).  Discrete half-life = -ln 2 / ln(1 + b); the
  continuous approximation -ln 2 / b is close when |b| is small.  b >= 0 means no mean reversion (report inf / NaN).
- OU mapping with step dt: kappa = -ln(1 + b) / dt, mu = -a / b, sigma = std(eps) * sqrt(2 kappa / (1 - (1 + b)^2)).

## Kalman-filter hedge ratio
- State theta_t = [beta_t, alpha_t] follows a random walk with covariance Q = delta / (1 - delta) * I (delta e.g. 1e-4, or
  the task's value); observation y_t = [x_t, 1] . theta_t + v_t, Var(v) = R.
- Each step: P = P + Q; yhat = H theta; e = y - yhat; S = H P H' + R; K = P H' / S; theta = theta + K e; P = P - K H P.
  Initial theta_0 and P_0 as stated (commonly zeros and zeros or a large diagonal).
- Signals use the standardised innovation e_t / sqrt(S_t) (entry when it exceeds 1 or the stated threshold); trade the
  hedge ratio the task names (commonly the prior estimate, before observing y_t) and keep that choice everywhere.

## Invariants worth printing
- Residual mean ~ 0 for the in-sample OLS spread; reported p-value consistent with the statistic and critical values.
- Half-life > 0 exactly when b < 0; positions take values only in {-1, 0, +1} (or the stated sizes) and change only on
  signal bars; cumulative P&L equals the sum of per-bar P&L.
