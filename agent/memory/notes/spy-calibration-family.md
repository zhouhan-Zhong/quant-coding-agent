---
title: The SPY-calibration task family (calibrate from daily closes, price, summarise)
category: derivatives-pricing
tags: spy_daily.csv, calibration.json, summary.json, sigma, S0, log returns, annualized volatility, r = 0.05, dividend yield, moneyness grid, maturities, closed form, monte carlo, parity, atm
summary: Shared conventions of the many option-pricing tasks that calibrate sigma and S0 from a daily price CSV, price a moneyness x maturity grid, validate with Monte Carlo and write calibration.json / <prices>.csv / summary.json.
triggers: file:spy_daily.csv, text:calibration.json
---
## When this applies
A task gives a daily OHLCV CSV (SPY, AAPL/MSFT, SP500 ...) and asks to "calibrate" (sigma from log returns, S0 =
last close, r and dividend yield given), price options on a grid of moneyness m and maturities T, compare a closed
form with Monte Carlo, and write `calibration.json`, one or more price CSVs and `summary.json`.

## Conventions that reproduce the oracles
- Returns: `x = np.diff(np.log(close))` on the chronologically sorted closes; `n_returns = len(x)` (prices - 1);
  `return_mean = x.mean()`, `return_std = x.std(ddof=1)`, `sigma = return_std * sqrt(252)`; `S0 = close[-1]`.
  Skewness/kurtosis via scipy.stats.skew / kurtosis(fisher=True) (bias-corrected = False default) when asked.
- Rates: r is stated (commonly 0.05); the dividend yield is stated per task (0.0, 0.01, 0.013, 0.015 ...) — read the
  task, never assume. Report them in calibration.json when the template lists them.
- Grid: strikes K = S0 * m for the listed moneyness values (sometimes K = S0 / m — read carefully); maturities as
  listed; rows "sorted by T then K ascending"; columns in the template order; `moneyness` column keeps the exact
  grid values (0.95, 1.0, 1.05 ...), K unrounded.
- Closed forms: Black–Scholes with continuous yield q everywhere (d1 uses r - q, S e^{-qT}); Greeks with q; parity
  C - P = S e^{-qT} - K e^{-rT}. Digital/barrier/lookback/Asian/compound closed forms per their playbooks.
- Monte Carlo validation: `np.random.seed(42)` (or default_rng(42) if the task says) BEFORE drawing; GBM with the
  risk-neutral drift r - q - sigma^2/2; daily steps (252/yr) for path-dependent payoffs; report `mc_price` and
  `mc_std_err = std(ddof=1)/sqrt(n)`; a `*_validates` flag means |cf - mc| <= k * std_err (k and any absolute floor as
  the task states). Discrete monitoring makes barrier/lookback MC biased relative to continuous formulas — the
  tasks usually accept it or ask for the Broadie–Glasserman–Kou correction; follow the text.
- summary.json: counts (`n_*` ints), max errors (float), flags (bool), ATM values (`atm_*_T025/T050/T100` taken from
  the moneyness == 1.0 rows of the CSV you wrote), monotonicity flags computed with a tolerance (1e-10).
- Typical tolerances: ATM oracle prices rtol 1-5%; parity/decomposition errors < 1e-6..1e-2; MC within 3-4 std errors;
  monotonic in strike with 0.01 slack.

## Pitfalls
- Using simple returns, ddof=0, or 250/260 days changes sigma by ~1% and fails tight oracles.
- Forgetting the dividend yield in d1/d2 (or in delta: delta_call - delta_put = e^{-qT}).
- Mixing K = S0*m with K = S0/m; rounding K; writing moneyness as K/S0 recomputed (floating noise) instead of the grid value.
- Writing summary values from a separate recomputation instead of the CSV rows -> cross-file mismatch.
- Monte Carlo without a seed, or seeding after the first draw; antithetic variates change the std-error semantics.
