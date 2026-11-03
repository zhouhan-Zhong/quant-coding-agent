---
title: Portfolio construction conventions: weight normalisation and constraints, risk parity / inverse volatility, CVaR-targeted sizing, Kelly fractions, regime scaling, rebalancing calendar, turnover and transaction costs, performance statistics
category: portfolio
tags: portfolio, weights, long only, long short, gross exposure, net exposure, normalisation, risk parity, equal risk contribution, inverse volatility, minimum variance, mean variance, cvar optimisation, kelly, fractional kelly, half kelly, position sizing, leverage cap, volatility target, regime, scaling factor, absorption ratio, eigenvalue, rebalance, monthly, calendar, turnover, transaction costs, bps, drawdown, sharpe, annualisation, 252, median terminal wealth, backtest
summary: Where portfolio tasks go wrong: weights that must sum to one (long-only) or to zero net / one gross (long-short) and whether the task caps leverage; inverse-volatility and risk-parity weights defined on the trailing window with ddof as stated; a volatility target applied as w x (sigma_target / sigma_portfolio) with the cap stated; Kelly fraction f = mu / sigma^2 per asset (or Sigma^{-1} mu for the vector) with the "fractional Kelly" multiplier; regime scaling multiplies weights by the task's factor per regime label; the backtest uses weights decided at t with returns of t+1 (no look-ahead), rebalances on the stated calendar, charges costs = turnover x bps/1e4, and reports Sharpe with the stated annualisation and the stated risk-free; terminal wealth is reported in the units the task fixes (relative, starting at 1, or an initial capital).
---
## When this applies
"Risk-parity / inverse-vol / minimum-variance weights with a rolling window", "size positions by Kelly / fractional Kelly
/ a VaR or CVaR budget", "scale exposure by regime", "backtest with monthly rebalancing and transaction costs", "report
Sharpe, max drawdown, turnover, terminal wealth".

## Weights
- Long-only: w >= 0, sum w = 1 after every rule (clip then renormalise, in that order unless the task says the
  opposite); long-short: sum |w| = 1 (gross) or sum w = 0 (dollar neutral) - read which; leverage cap: scale the whole
  vector so that sum |w| <= cap.
- Inverse volatility: w_i proportional to 1 / sigma_i with sigma_i from the trailing window (ddof as stated, annualised or
  not does not matter for the ratio); equal risk contribution (risk parity): solve w_i (Sigma w)_i = (1/n) w' Sigma w
  (iterative or scipy minimise on sum (RC_i - 1/n)^2) - inverse vol is only the special case of zero correlation.
- Minimum variance: w = Sigma^{-1} 1 / (1' Sigma^{-1} 1) (unconstrained) or the constrained QP when bounds are given;
  the task's covariance estimator (sample with ddof, EWMA with lambda, shrinkage) decides the numbers.
- Volatility targeting: w_scaled = w x min(cap, sigma_target / sigma_hat) with sigma_hat the trailing portfolio vol
  (annualised with the stated factor) measured BEFORE the period it is applied to.
- Kelly: single asset f* = (mu - r) / sigma^2 with per-period moments (not annualised, or both annualised - be
  consistent); vector f* = Sigma^{-1} (mu - r); fractional Kelly = k f* (k = 0.5 for half-Kelly); cap as stated.
  "Growth-optimal" statistics use log wealth.
- CVaR budget sizing: scale positions so that the trailing historical CVaR_alpha of the portfolio equals the budget;
  CVaR from the trailing window of portfolio returns using the empirical tail mean (losses positive).
- Regime scaling: regime label from the trailing indicator (e.g. absorption ratio z-score or volatility quantile)
  computed with data up to t; weights multiplied by the task's factor for that regime; report the fraction of periods in
  each regime (invariant: fractions sum to one).

## Backtest mechanics
- Decide weights at the close of t from data through t; earn r_{t+1} w_t.  Rebalance on the calendar the task names
  (month-end = last trading day of the month); between rebalances hold units fixed (weights drift) unless the task says
  "constant weights".
- Turnover_t = sum |w_t - w_{t-1}^{drifted}| (one-way) or half of it (two-way) as stated; costs = turnover x bps / 1e4
  charged on the rebalance day.
- Warm-up: the first weight needs a full trailing window; the backtest starts after it; "number of rebalances" counts
  only dates with a trade.
- Statistics: annualised return (mean x 252 or compound (prod(1 + r))^(252/n) - 1 as stated), annualised volatility
  = sd(ddof = 1) sqrt(252), Sharpe = (ann. return - r_f) / ann. vol, max drawdown = max over t of (peak - value) / peak
  (positive number), terminal wealth = prod(1 + r) x initial capital (or 1).  Median terminal wealth across simulated
  paths is the median of the paths' final values, in the same units as the initial capital (1.12 vs 1.12e6 is a units
  error).

## Invariants worth printing
- Weights sum / gross exposure exactly as the rule; regime fractions sum to one; no look-ahead (first weight date >=
  first date + window); turnover 0 on non-rebalance days; drawdown in [0, 1]; terminal wealth units consistent with the
  initial capital.
