---
title: Time-series model conventions: ARIMA orders, trend terms and forecasts in statsmodels, information-criterion selection, Ljung-Box and stationarity tests, exponential-kernel Hawkes intensity, log-likelihood recursion, branching ratio, thinning simulation and kernel parameterisations
category: numerics
tags: arima, arma, ar, ma, sarimax, statsmodels, order, trend, drift, constant, forecast, prediction interval, aic, bic, model selection, ljung box, autocorrelation, acf, pacf, adf, kpss, stationarity, differencing, hawkes process, self exciting, point process, intensity, exponential kernel, branching ratio, baseline intensity, log likelihood, compensator, ogata thinning, event arrivals, order flow
summary: statsmodels ARIMA(y, order=(p, d, q), trend=...) includes a constant only when trend='c' (and with d >= 1 a drift needs trend='t'); forecasts via res.get_forecast(h).predicted_mean / conf_int(alpha); select orders by AIC/BIC over the stated grid; Ljung-Box with the stated lags (model_df for residuals); ADF null = unit root, KPSS null = stationary; Hawkes with lambda(t) = mu + sum alpha e^{-beta (t - t_i)} has branching ratio alpha/beta (< 1), mean intensity mu / (1 - alpha/beta), log-likelihood sum log lambda(t_i) - mu T - (alpha/beta) sum (1 - e^{-beta (T - t_i)}) computed with the recursion A_i = e^{-beta (t_i - t_{i-1})} (1 + A_{i-1}).
triggers: text:arima, text:hawkes, text:self-exciting, text:ljung
---
## When this applies
"Fit an ARIMA(1,1,1) and forecast 10 steps", "choose the order by AIC", "test residual autocorrelation",
"fit a Hawkes process to trade arrival times", "branching ratio / endogeneity of order flow", "simulate a Hawkes process".

## ARIMA in statsmodels
- `from statsmodels.tsa.arima.model import ARIMA`; `res = ARIMA(y, order=(p, d, q), trend="c").fit()`.  trend: "n" none,
  "c" constant (only allowed when d = 0), "t" linear trend (with d = 1 this is the drift).  Default trend is "c" for d = 0
  and "n" for d >= 1.
- Forecast: `fc = res.get_forecast(steps=h)`; `fc.predicted_mean`, `fc.conf_int(alpha=0.05)`.  Forecasts of a differenced
  model are returned in levels.
- Parameters: `res.params` (names ar.L1, ma.L1, const, sigma2); log-likelihood `res.llf`, `res.aic`, `res.bic`.  OLS AR fits
  (statsmodels AutoReg) and MLE ARIMA give slightly different coefficients: use the estimator the task names.
- Order selection: loop over the stated (p, q) grid (and d), keep the minimum AIC (ties -> fewer parameters); skip fits
  that fail to converge and report them.
- Residual tests: `acorr_ljungbox(res.resid, lags=[L], model_df=p + q)` (a DataFrame with lb_stat, lb_pvalue); ADF `adfuller(y, regression="c",
  autolag="AIC")` (null: unit root); KPSS `kpss(y, regression="c", nlags="auto")` (null: stationary).
- Returns vs prices: model log returns (d = 0) or log prices (d = 1) as stated; annualise forecasts only if asked.

## Hawkes process (exponential kernel)
- Intensity lambda(t) = mu + sum_{t_i < t} alpha exp(-beta (t - t_i)), mu > 0, alpha >= 0, beta > 0.
  Branching ratio n = alpha / beta (stationary iff n < 1); stationary mean intensity = mu / (1 - n); kernel half-life =
  ln 2 / beta.  Another common parameterisation uses alpha beta exp(-beta s) (then n = alpha) - read which one the task uses.
- Log-likelihood on [0, T] with sorted event times: A_1 = 0, A_i = exp(-beta (t_i - t_{i-1})) (1 + A_{i-1});
  lambda(t_i) = mu + alpha A_i; LL = sum_i ln lambda(t_i) - mu T - (alpha / beta) sum_i (1 - exp(-beta (T - t_i))).
  Maximise with scipy.optimize.minimize on -LL (L-BFGS-B, bounds mu, alpha, beta > 0, optionally alpha < beta).
- Simulation (Ogata thinning): with current intensity bound lambda* (intensity right after the last event), draw
  w ~ Exp(lambda*), t = t + w, accept with probability lambda(t) / lambda*; update after each accepted event.  Use the
  stated seed and horizon.
- Time units: seconds vs minutes vs days change mu and beta by the conversion factor; report in the task's units.
- Goodness of fit: compensator increments Lambda(t_i) - Lambda(t_{i-1}) should be iid Exp(1) (KS test).

## Invariants worth printing
- Forecast length h; confidence intervals widen with the horizon; Ljung-Box p-values in [0, 1]; Hawkes fit has n < 1,
  fitted mean intensity close to the event count / T.
