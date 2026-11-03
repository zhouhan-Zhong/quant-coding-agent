---
title: The params.json / glossary.json / solution.json "intermediates" task family
category: meta
tags: params.json, glossary.json, solution.json, intermediates, checkpoints, results.json, verifier, production feed, planted anomalies, seed, estimation_window_days, kelly, garch, cta, regime, structured note, barrier var, risk parity, hmm, ipca, sentiment, alpha hedge
summary: Conventions of the multi-phase-verifier tasks (params + glossary + one data feed) whose single test demands every deliverable and every intermediate checkpoint within tolerance.
triggers: file:glossary.json, text:solution.json intermediates
---
## When this applies
Inputs are `params.json` (all run parameters incl. `seed`), `glossary.json` (vocabulary; not needed), one or two
"production feed" CSVs (returns/prices with planted NaNs, duplicates or outliers) and sometimes a sector map or
characteristics file. Deliverables are `results.json` (a flat dict of headline numbers) and `solution.json`
with an `"intermediates"` dict of named checkpoint values.

## What the single test demands
- Classification "PERFECT": every key of results.json within tolerance AND every checkpoint named in the task's
  solution.json template present in `solution.json["intermediates"]` and within tolerance. Missing or wrong
  checkpoint => fail. Extra intermediates are harmless, so when in doubt write more named intermediates.
- Checkpoint format: `{"name": {"value": x}}` (flat numbers also accepted). Tolerances are rtol OR atol (either
  suffices); an expected 0 needs |actual| <= atol; drawdown-like names are compared by absolute value.
- Alternative conventions (rf compounding, Sharpe with/without excess, drawdown sign) may be accepted via an
  alt-paths table, but matching the primary convention stated in the task is the safe route.

## Conventions that matched the references
- Cleaning: sort by date, drop duplicates, `dropna()` on any-NaN rows (the count of surviving rows is usually a
  reported intermediate), outliers only by the rule in the task.
- Windows: statistics on the LAST `estimation_window_days` valid rows (only if shorter than the sample); EWMA
  recursions seeded with the first row's outer product unless told otherwise; GARCH via `arch` on percent returns
  (`rescale=False`), parameters omega/alpha[1]/beta[1]; persistence = alpha + beta; long-run variance
  omega/(1 - persistence); report daily vol in return units (divide by 100 when fitted on percent).
- Risk-free: `rf_daily = annual / 252` (simple) unless the task says compound; excess means = mean - rf_daily.
- Covariance: `np.cov(X, rowvar=False, ddof=1)`; Kelly f* = Sigma^{-1} mu_excess (leverage allowed); parametric VaR
  = z_alpha * sqrt(f' Sigma f) (daily, positive, no mean); scaling factors = cap / VaR.
- Monte Carlo: `rng = np.random.default_rng(params["seed"])`; a FRESH draw per scheme/scenario in the order the
  task lists them (do not share one draw across schemes); wealth = W0 * cumprod(1 + port); drawdown on the path
  prefixed with W0; "median"/"p50" over paths = sorted[n // 2] to hit the reference exactly (np.median is close).
- Sharpe over pooled path-days: mean / std(ddof=1) * sqrt(252) (no rf unless stated).
- Regime/HMM tasks: use the stated number of states, the stated initialisation and seed, and report transition
  matrices row-stochastic; vol targeting: target / realised vol with leverage caps from params.
- Capital tasks (Basel-style): apply the stated multipliers/floors literally; report both the raw and capped numbers
  when the template names them.
- Barrier/GARCH VaR: filtered historical simulation with the stated GARCH spec and standardized residuals.

## Pitfalls
- Using all rows instead of the estimation window changes every number far beyond tolerance.
- `np.random.seed` + legacy `np.random.multivariate_normal` differs from `default_rng(...).multivariate_normal`.
- Adding a cash leg or excess-return Sharpe when the task did not ask.
- Forgetting an intermediate (write every name in the template, cast to float/int/bool, never NaN).
