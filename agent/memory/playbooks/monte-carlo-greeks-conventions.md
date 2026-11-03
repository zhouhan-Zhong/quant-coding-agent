---
title: Monte Carlo Greeks conventions: bump-and-revalue with common random numbers, pathwise (IPA) and likelihood-ratio estimators under GBM, mixed estimators for gamma, standard errors, discontinuous payoffs, path-dependent payoffs and checks against closed forms
category: numerics
tags: monte carlo greeks, delta, gamma, vega, rho, theta, bump and revalue, finite difference, central difference, common random numbers, crn, pathwise, ipa, infinitesimal perturbation, likelihood ratio, score function, digital option, barrier, asian option, standard error, confidence interval, antithetic, control variate, black scholes check, sobol, quasi monte carlo
summary: Bump-and-revalue must reuse the same random numbers for the base and bumped prices (central differences, relative bump 1e-4 to 1e-2 of S); pathwise delta of a call = e^{-rT} E[1{S_T > K} S_T / S_0] and vega = e^{-rT} E[1{S_T > K} S_T (ln(S_T/S_0) - (r - q + sigma^2/2) T) / sigma]; likelihood-ratio delta = e^{-rT} E[payoff Z / (S_0 sigma sqrt T)] and vega = e^{-rT} E[payoff ((Z^2 - 1)/sigma - Z sqrt T)] work for digitals; report standard errors from the per-path estimator samples; check vanillas against Black-Scholes within about 3 standard errors.
triggers: text:pathwise, text:likelihood ratio method, text:monte carlo greek, text:mc greek, text:greeks by monte carlo, text:common random numbers
---
## When this applies
"Estimate delta / gamma / vega by Monte Carlo", "compare finite-difference, pathwise and likelihood-ratio estimators",
"Greek surface over strikes and maturities", "standard error of the Greek estimate".

## Setting (GBM under the risk-neutral measure)
- S_T = S_0 exp((r - q - sigma^2/2) T + sigma sqrt(T) Z), Z ~ N(0, 1); discount e^{-rT}.  Use the stated seed and number
  of paths; antithetic pairs (Z, -Z) count as 2 paths unless the task says otherwise.

## Estimators
- Bump-and-revalue (finite differences): same Z for every bumped valuation (common random numbers).
  delta ~ (V(S+h) - V(S-h)) / (2h); gamma ~ (V(S+h) - 2 V(S) + V(S-h)) / h^2; vega ~ (V(sigma+dv) - V(sigma-dv)) / (2 dv).
  Bump size as stated (absolute or relative, e.g. h = 1e-2 S); for digitals and barriers small bumps explode the variance.
- Pathwise (differentiate the payoff along the path; needs a payoff that is continuous in the parameter):
  call delta = e^{-rT} E[1{S_T > K} S_T / S_0]; put delta = -e^{-rT} E[1{S_T < K} S_T / S_0];
  call vega = e^{-rT} E[1{S_T > K} S_T (ln(S_T / S_0) - (r - q + sigma^2/2) T) / sigma];
  rho (call) = e^{-rT} E[1{S_T > K} K T]; arithmetic Asian call delta = e^{-rT} E[1{A > K} A / S_0].
  Not valid for digital payoffs (derivative of the indicator is a delta function).
- Likelihood ratio (score function; payoff may be discontinuous):
  delta = e^{-rT} E[f(S_T) Z / (S_0 sigma sqrt T)];
  vega = e^{-rT} E[f(S_T) ((Z^2 - 1) / sigma - Z sqrt T)];
  gamma = e^{-rT} E[f(S_T) ((Z^2 - 1) / (S_0^2 sigma^2 T) - Z / (S_0^2 sigma sqrt T))].
  Higher variance than pathwise for smooth payoffs; for multi-step paths the score sums over the first step for delta.
- Mixed (gamma for a call = likelihood ratio applied to the pathwise delta, which depends on S_0 explicitly too):
  gamma = e^{-rT} E[1{S_T > K} (S_T / S_0^2) (Z / (sigma sqrt T) - 1)]; check it against the closed form.

## Errors and variance reduction
- Each estimator is a mean of per-path samples X_i: estimate = mean(X), SE = std(X, ddof=1) / sqrt(N); with antithetics
  average each pair first, then SE over the N/2 pair means.
- Control variates: payoff with known mean (S_T, or the geometric Asian for an arithmetic Asian), b = Cov(Y, C)/Var(C).
- Quasi-random: `scipy.stats.qmc.Sobol(d, scramble=True, seed=...)`, N a power of 2, Z = norm.ppf(u) (clip u away from 0, 1).

## Checks
- European call/put: compare with Black-Scholes delta N(d1) e^{-qT}, gamma e^{-qT} n(d1) / (S sigma sqrt T), vega
  S e^{-qT} n(d1) sqrt T; the MC value should be within ~3 SE.  Put-call parity: delta_call - delta_put = e^{-qT}.
- Digital call delta (LR) vs closed form e^{-rT} n(d2) / (S sigma sqrt T).
- Report units: vega per 1.00 of vol (or per 1 vol point = /100 if the task says so).
