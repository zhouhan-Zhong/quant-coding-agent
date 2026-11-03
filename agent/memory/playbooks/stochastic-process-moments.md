---
title: Mean-reverting processes: OU/Vasicek, Schwartz, jump extensions (moments, calibration, simulation)
category: numerics
tags: ornstein-uhlenbeck, ou process, vasicek, mean reversion, ar(1), ols calibration, mle, kappa, theta, half-life, exact discretisation, euler scheme, schwartz one-factor, geometric mean reversion, forward curve, commodity, mean reverting jump diffusion, conditional moments, stationary variance, simulation, jump detection, 3 sigma, poisson jumps, bernoulli jumps, log rates, lognormal moments, fred, treasury yield, dgs10, seasonal mean, temperature anomaly
summary: Exact OU transition and AR(1) calibration (kappa, theta, sigma), 3-sigma jump detection, conditional/stationary moments with jumps (levels and log-levels), Schwartz forward curve, half-life, and the exact/Euler simulation and draw-order conventions.
---
## When this applies
"Fit an OU / Vasicek process to rates or spreads", "estimate mean-reversion speed and half-life", "simulate the
process exactly / with Euler and compare", "conditional mean and variance at horizon T", "Schwartz one-factor
commodity model forward curve", "mean-reverting jump diffusion moments", "detect jumps in residuals", "geometric
(log) mean-reverting rate with jumps and its forward curve", seasonal-anomaly OU (temperature), pairs spreads.

## Method and formulas
OU: dX = kappa (theta - X) dt + sigma dW, kappa > 0 speed, theta long-run mean, sigma volatility (same units as X per sqrt(year)).
- Exact transition over dt: X_{t+dt} | X_t ~ Normal(theta + (X_t - theta) e^{-kappa dt}, sigma^2 (1 - e^{-2 kappa dt})/(2 kappa)).
  Conditional moments at horizon t: E = theta + (x0 - theta) e^{-kappa t}, Var = sigma^2 (1 - e^{-2 kappa t})/(2 kappa);
  stationary: mean theta, variance sigma^2/(2 kappa); half-life = ln 2 / kappa (same time unit as kappa^{-1}, years if dt in years).
- AR(1) link: X_{t+1} = a + b X_t + eps, b = e^{-kappa dt}, a = theta (1 - b), Var(eps) = sigma^2 (1 - b^2)/(2 kappa).
  Calibration (OLS on consecutive pairs, equals Gaussian MLE conditional on X_0): b, a = np.polyfit(X[:-1], X[1:], 1);
  kappa = -ln(b)/dt; theta = a/(1 - b); sigma = sd(eps) sqrt(2 kappa/(1 - b^2)) = sd(eps) sqrt(-2 ln b / ((1 - b^2) dt)).
  sd(eps): MLE uses ddof=0, OLS residual sd uses n-2; state which. Euler-style estimates kappa = (1 - b)/dt and
  sigma = sd(eps)/sqrt(dt) differ by O(kappa dt) - use them only if the task defines them that way.
  b >= 1 or b <= 0 -> no valid kappa (report NaN / flag not mean-reverting); statsmodels OLS gives the same a, b with
  standard errors; t-stat of (b - 1) is the Dickey-Fuller flavour of the mean-reversion test.
- Euler scheme: X_{t+dt} = X_t + kappa (theta - X_t) dt + sigma sqrt(dt) Z - biased variance (sigma^2 dt vs exact) and
  unstable for kappa dt > 2; exact scheme has no discretisation error at any dt. Both use the same Z for comparisons.
- Mean-reverting jump diffusion: dX = kappa (theta - X) dt + sigma dW + J dN, N ~ Poisson(lambda), J ~ Normal(mu_J, sigma_J^2):
  E[X_t | x0] = theta + (x0 - theta) e^{-kappa t} + (lambda mu_J / kappa)(1 - e^{-kappa t}),
  Var[X_t | x0] = (sigma^2 + lambda (mu_J^2 + sigma_J^2))(1 - e^{-2 kappa t})/(2 kappa); long-run mean theta + lambda mu_J/kappa.
  Simulation per step: exact OU update, then add sum of Poisson(lambda dt) jumps ~ n mu_J + sqrt(n) sigma_J Z.
- Jump calibration (reference recipe): AR(1) residuals eps; flag |eps| > 3 sd(eps, ddof=1); lambda = n_jumps/(n_pairs dt)
  per year; mu_J, sigma_J = mean, std(ddof=1) of the flagged eps; diffusion sigma = sd sqrt(2 kappa/(1 - b^2)) from ALL
  residuals or from the NON-jump residuals - references differ, follow the wording ("excluding detected jumps").
- Stationary (t -> inf) moments WITH jumps: mean theta + lambda mu_J/kappa, variance (sigma^2 + lambda(mu_J^2 + sigma_J^2))/(2 kappa).
  Tests like Var(t) <= stationary var need the jump-augmented version (jump variance can be ~half the total).
- Geometric (log-level) OU: X = ln(level) (e.g. ln(rate/100)); with conditional moments m, v of X:
  E[S_t] = exp(m + v/2), Var[S_t] = (e^v - 1) e^{2m + v}; model forward = E[S_t] (lambda = 0 for the pure-OU curve).
- Seasonal-anomaly OU (temperature): x = level - seasonal mean (day-of-month / day-of-year mean or Fourier fit); regress
  dx = a + b x on consecutive days only (no pairs across gaps/years): kappa = -b/dt, theta = -a/b, sigma = sd(eps, ddof=1)/sqrt(dt)
  (Euler-style inversion); see weather-derivatives.
- Schwartz (1997) one-factor: dS = kappa (mu - ln S) S dt + sigma S dW; X = ln S follows OU with long-run level
  alpha = mu - sigma^2/(2 kappa). Forward/futures: ln F(0, T) = e^{-kappa T} ln S0 + (1 - e^{-kappa T}) alpha* + sigma^2 (1 - e^{-2 kappa T})/(4 kappa),
  alpha* = alpha - lambda_risk/kappa (market price of risk lambda_risk; 0 if not given). F(0, T) -> exp(alpha* + sigma^2/(4 kappa)) as T -> inf.
  E[S_T] = exp(E[ln S_T] + Var[ln S_T]/2) with the OU moments of ln S; calibrate kappa, alpha, sigma on ln S by the AR(1) recipe.
- CIR (for comparison): exact transition via noncentral chi-square (see rates playbooks); Euler with full truncation
  max(v, 0) inside the sqrt.
- Simulation check: sample mean and variance at T over >= 1e5 paths must match the conditional moments within ~3 se
  (se_mean = sd/sqrt(n), se_var ~ var sqrt(2/(n-1))).

## Conventions and pitfalls (what graders check)
- dt: 1/252 for business-daily series, 1/365 (or 1/365.25) for calendar-daily data with weekends present (FRED DFF),
  1/12 monthly, 1/52 weekly. Mixing 252 vs 365 changes kappa and sigma by the same factor - follow the data frequency.
- Calibrate OU on LEVELS (rates, spreads, log prices), not on returns; drop NaN rows before forming pairs; use all n-1 pairs.
- Rates in percent vs decimals: convert before calibrating (sigma scales, kappa does not).
- Half-life in the same unit as 1/kappa; "half-life in days" = ln2/kappa / dt when kappa is annual.
- Seeds: default_rng(seed) vs np.random.seed as the task says; draw shape (n_paths, n_steps) in step order. Jump MC that
  matched references: legacy np.random.seed(42), per step draw Z (diffusion), then Poisson(lambda dt) counts (or
  uniforms < lambda dt for Bernoulli jumps), then jump sizes; Euler is fine (variance bias O(kappa dt)); record nested
  horizons from one run. Report MC-vs-analytic errors with the same definitions used for the validation flag.
- Units: keep the series in the units the checker's ranges imply (rates in percent when a range like 1 < mean < 8 is
  asserted); horizon tables are often checked by POSITION - write tau rows in the listed (ascending) order.
- Invariants asserted: kappa > 0, theta within the data range, sigma > 0, half_life = ln2/kappa to 1e-10, stationary
  variance = sigma^2/(2 kappa), exact-scheme moments match formulas, Euler bias shrinks with dt; Schwartz forward curve
  monotone toward the long-run level; F(0, 0) = S0.
- Report float() values; flags (e.g. mean_reverting) as bool; include n_obs and dt used.

## Tested snippet
```python
import math
import numpy as np

def ou_simulate(x0, kappa, theta, sigma, dt, n_steps, n_paths, seed=42, exact=True):
    """dX = kappa (theta - X) dt + sigma dW. Exact: X_{t+dt} = theta + (X_t - theta) e^{-k dt} + sigma sqrt((1 - e^{-2k dt})/(2k)) Z.
    Euler: X + kappa (theta - X) dt + sigma sqrt(dt) Z (biased, unstable if kappa dt > 2). Returns (n_paths, n_steps+1)."""
    rng = np.random.default_rng(seed); X = np.empty((n_paths, n_steps + 1)); X[:, 0] = x0
    e = math.exp(-kappa * dt); sd = sigma * math.sqrt((1 - e * e) / (2 * kappa)) if exact else sigma * math.sqrt(dt)
    for t in range(n_steps):
        Z = rng.standard_normal(n_paths)
        X[:, t + 1] = theta + (X[:, t] - theta) * e + sd * Z if exact else X[:, t] + kappa * (theta - X[:, t]) * dt + sd * Z
    return X

def ou_calibrate(x, dt, ddof=0):
    """AR(1) OLS = Gaussian MLE: X_{t+1} = a + b X_t + eps. kappa = -ln(b)/dt, theta = a/(1-b),
    sigma = sd(eps) sqrt(-2 ln(b) / ((1 - b^2) dt)). ddof=0 is the MLE variance; residual sd with n-2 is the OLS variant."""
    x = np.asarray(x, float); X, Y = x[:-1], x[1:]
    b, a = np.polyfit(X, Y, 1); eps = Y - (a + b * X)
    if not (0 < b < 1): return {"kappa": float("nan"), "theta": float("nan"), "sigma": float("nan"), "b": float(b)}
    kappa = -math.log(b) / dt; sd = math.sqrt(np.sum(eps ** 2) / (len(eps) - ddof))
    sigma = sd * math.sqrt(2 * kappa / (1 - b * b))
    return {"kappa": kappa, "theta": float(a / (1 - b)), "sigma": sigma, "half_life": math.log(2) / kappa, "b": float(b),
            "stationary_sd": sigma / math.sqrt(2 * kappa)}

def ou_moments(x0, kappa, theta, sigma, t):         # conditional mean / variance of X_t given X_0 = x0
    e = math.exp(-kappa * t); return theta + (x0 - theta) * e, sigma ** 2 * (1 - e * e) / (2 * kappa)

def mrjd_moments(x0, kappa, theta, sigma, lam, mJ, sJ, t):
    """dX = kappa (theta - X) dt + sigma dW + J dN, N ~ Poisson(lam), J ~ N(mJ, sJ^2). t = inf -> stationary moments."""
    e = math.exp(-kappa * t)
    return theta + (x0 - theta) * e + lam * mJ / kappa * (1 - e), (sigma ** 2 + lam * (mJ ** 2 + sJ ** 2)) * (1 - e * e) / (2 * kappa)

def ar1_jump_calibrate(x, dt, k_sd=3.0, sigma_from="nonjump"):    # AR(1) + |eps| > k_sd sd(eps) jump flags
    b, a = np.polyfit(x[:-1], x[1:], 1); eps = x[1:] - (a + b * x[:-1]); J = np.abs(eps) > k_sd * eps.std(ddof=1)
    kappa = -math.log(b) / dt; base = eps[~J] if sigma_from == "nonjump" else eps
    return {"kappa": kappa, "theta": a / (1 - b), "sigma": base.std(ddof=1) * math.sqrt(2 * kappa / (1 - b * b)),
            "lam": J.sum() / (len(eps) * dt), "mu_J": eps[J].mean(), "sigma_J": eps[J].std(ddof=1), "n_jumps": int(J.sum())}
lognormal_moments = lambda m, v: (math.exp(m + v / 2), (math.exp(v) - 1) * math.exp(2 * m + v))   # E[e^X], Var[e^X]

def schwartz_forward(S0, kappa, mu, sigma, T, lam_risk=0.0):
    """Schwartz (1997) one-factor: d ln S = kappa (alpha - ln S) dt + sigma dW, alpha = mu - sigma^2/(2 kappa).
    ln F(0,T) = e^{-kT} ln S0 + (1 - e^{-kT}) alpha* + sigma^2 (1 - e^{-2kT})/(4k), alpha* = alpha - lam_risk/kappa."""
    alpha = mu - sigma ** 2 / (2 * kappa); e = math.exp(-kappa * T)
    return math.exp(e * math.log(S0) + (1 - e) * (alpha - lam_risk / kappa) + sigma ** 2 * (1 - e * e) / (4 * kappa))
```
