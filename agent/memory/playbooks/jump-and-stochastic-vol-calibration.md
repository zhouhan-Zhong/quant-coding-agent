---
title: Merton jump-diffusion MLE and Heston/Bates characteristic-function pricing
category: derivatives-pricing
tags: merton jump diffusion, jump diffusion, poisson mixture, mle, maximum likelihood, logsumexp, l-bfgs-b, random restarts, merton series, heston, stochastic volatility, characteristic function, gauss-legendre, quad, lewis, carr-madan, bates, calibration, implied vol smile, vol of vol, feller condition, two-factor heston, double heston, multi-factor stochastic volatility, implied vol surface, integration comparison
summary: Exact Merton density/likelihood and series price (correct lambda' weights), (multi-factor) Heston characteristic function with P1/P2 integration by quad or cached vectorised Gauss-Legendre, parity/martingale checks, and Bates extension.
---
## When this applies
"Calibrate a Merton jump-diffusion to daily log returns by MLE", "price options under Merton / Heston / Bates",
"implied-vol smile from the model", "characteristic function pricing with numerical integration", calibration
to an option chain with bounded optimisation, Feller-condition flags, two-factor (double) Heston surfaces on a
strike x maturity grid, "compare quad vs Gauss-Legendre integration".

## Method and formulas
- Merton dynamics: dS/S = (mu - lambda kbar) dt + sigma dW + (Y - 1) dN, ln Y ~ Normal(mu_J, sigma_J^2),
  kbar = exp(mu_J + sigma_J^2/2) - 1. Log-return over dt given n jumps ~ Normal((mu - sigma^2/2 - lambda kbar) dt + n mu_J,
  sigma^2 dt + n sigma_J^2). Density p(x) = sum_{n=0}^{N_max} Pois(n; lambda dt) phi(x; mean_n, var_n), N_max 10-20
  (lambda dt is tiny for daily data). Log-likelihood via logsumexp over n of [log Pois + log phi]; log Pois =
  -lambda dt + n ln(lambda dt) - gammaln(n+1). Fix mu = mean(x)/dt or estimate it; parameters (sigma, lambda, mu_J, sigma_J).
- MLE: minimize(-LL, method="L-BFGS-B", bounds e.g. sigma [0.01, 1], lambda [0.1, 12] (per year), mu_J [-0.1, 0.1],
  sigma_J [0.005, 0.3]); moment-based start plus >= 5 seeded random restarts (default_rng(42)); keep the best LL; the
  lambda-sigma_J trade-off is weakly identified, so several maxima are normal. sigma_total^2 = sigma^2 + lambda (mu_J^2 + sigma_J^2)
  is the annualised total return variance (compare with std(x) sqrt(252)).
- Merton series price: C = sum_{n>=0} e^{-lambda' T} (lambda' T)^n / n! * BS(S, K, r_n, sigma_n, T), lambda' = lambda (1 + kbar),
  sigma_n^2 = sigma^2 + n sigma_J^2 / T, r_n = r - lambda kbar + n ln(1 + kbar) / T; truncate when the Poisson weight < 1e-16
  past the mode (>= 3 lambda' T + 10 terms). Verified identical to the characteristic-function price. Using plain lambda in
  the weights is ~1-2% off when kbar is not tiny - only do it if the task's formula says so.
- Merton CF of ln S_T: phi(u) = exp(i u (ln S + (r - q - sigma^2/2 - lambda kbar) T) - sigma^2 u^2 T/2 + lambda T (e^{i u mu_J - sigma_J^2 u^2/2} - 1)).
- Heston CF (Gatheral/"little trap", numerically stable for long T): xi = kappa - rho sigma i u, d = sqrt(xi^2 + sigma^2 (u^2 + i u)),
  g = (xi - d)/(xi + d), C = (r - q) i u T + (kappa theta / sigma^2)[(xi - d) T - 2 ln((1 - g e^{-dT})/(1 - g))],
  D = ((xi - d)/sigma^2)(1 - e^{-dT})/(1 - g e^{-dT}), phi(u) = exp(C + D v0 + i u ln S0). Parameters: kappa (mean reversion),
  theta (long-run variance), sigma (vol of vol), rho, v0 (initial variance). Checks: phi(0) = 1, phi(-i) = F = S e^{(r-q)T}.
- Pricing: C = S e^{-qT} P1 - K e^{-rT} P2, P_j = 1/2 + (1/pi) int_0^inf Re[e^{-i u ln K} phi_j(u) / (i u)] du with phi_2 = phi and
  phi_1(u) = phi(u - i)/phi(-i). Integrate with scipy quad(limit=500) on [1e-12, 200] or Gauss-Legendre (n ~ 200) after the
  substitution u = x/(1-x). Put by parity: P = C - S e^{-qT} + K e^{-rT}. Lewis form (alternative): C = S e^{-qT} -
  (sqrt(S K) e^{-(r+q)T/2}/pi) int_0^inf Re[e^{i u k} phi_X(u - i/2)]/(u^2 + 1/4) du, k = ln(S e^{(r-q)T}/K), phi_X = CF of ln(S_T/F).
- Bates = Heston x Merton jumps: phi_Bates(u) = phi_Heston(u) exp(lambda T (e^{i u mu_J - sigma_J^2 u^2/2} - 1) - i u lambda kbar T).
- Multi-factor Heston (independent variance factors, e.g. Christoffersen-Heston-Jacobs double Heston): the CF is the
  product exp(i u (ln S0 + (r - q) T)) prod_j exp(C_j + D_j v_j0), each (C_j, D_j) the little-trap terms WITHOUT the drift;
  P1/P2 as above; direct put = K e^{-rT}(1 - P2) - S e^{-qT}(1 - P1) (parity exact to 1e-14).
- Many strikes at once: Gauss-Legendre on [0, U] (U ~ 100-200, n ~ 1000-2000 nodes, u = (x + 1) U/2, w = U/2 weights),
  integrand matrix exp(-i outer(u, ln K)) * phi(u)/(i u) -> one matrix product per maturity. Cache leggauss (computing it
  per maturity costs ~1 s each). quad on a short range (e.g. [1e-8, 50]) loses the tail at short T (~5e-4 price errors,
  deep-OTM prices slightly negative): when asked to compare integrators, measure both against a wide-range high-n GL
  reference and choose the one with the smaller error; report the max abs difference.
- Sanity limits: Heston with sigma -> 0 and v0 = theta reproduces BS with vol sqrt(theta) (diff < 1e-5); Merton with lambda -> 0
  reproduces BS; implied-vol smile from Merton with mu_J < 0 is a skew (OTM puts richer).
- Calibration to prices: minimise sum of squared price (or IV) errors with L-BFGS-B bounds kappa [0.1, 20], theta [1e-4, 1],
  sigma [0.01, 3], rho [-0.99, 0.99], v0 [1e-4, 1]; Feller 2 kappa theta > sigma^2 is reported as a bool, not enforced.

## Conventions and pitfalls (what graders check)
- dt = 1/252 for daily business-day returns; lambda, mu_J, sigma_J are per-jump quantities, lambda per year.
- Never exponentiate the mixture then log it (underflow); logsumexp + gammaln; return a large penalty (1e12) when LL is
  non-finite so L-BFGS-B keeps going; clip random starts into the bounds.
- Report log_likelihood as the maximised value (positive numbers are normal for daily densities); n_returns = len(x).
- Invariants: Merton price >= 0; sigma_total within [0.5, 2] x std(x) sqrt(252); IV of ATM Merton price ~ sigma_total;
  "smile_present" = max IV - min IV across strikes > a few vol points; Heston call in (max(S e^{-qT} - K e^{-rT}, 0), S e^{-qT}).
- Complex arithmetic: use numpy complex128 throughout; take .real only at the end; avoid u = 0 exactly (start at 1e-12).
- Cast every reported parameter with float(); flags with bool(); the IV inversion for model prices should use the
  bounds-aware solver (model prices near intrinsic have no IV -> NaN, not a crash).

## Tested snippet
```python
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp, gammaln

def heston_cf(u, T, S0, r, q, factors):
    """CF of ln S_T, little-trap form (stable for long T); factors = [(kappa, theta, sigma, rho, v0), ...] independent
    variance factors (one tuple = Heston, two = double Heston)."""
    u = np.asarray(u, dtype=complex); out = 1j * u * (math.log(S0) + (r - q) * T)
    for k, th, s, rho, v0 in factors:
        xi = k - rho * s * 1j * u; d = np.sqrt(xi ** 2 + s ** 2 * (u ** 2 + 1j * u)); g = (xi - d) / (xi + d); e = np.exp(-d * T)
        out = out + k * th / s ** 2 * ((xi - d) * T - 2 * np.log((1 - g * e) / (1 - g))) + (xi - d) / s ** 2 * (1 - e) / (1 - g * e) * v0
    return np.exp(out)

def heston_call(S0, K, T, r, q, factors, umax=200.0):   # quad: P_j = 1/2 + 1/pi int Re[e^{-iu lnK} phi_j(u)/(iu)] du
    cf = lambda u: heston_cf(u, T, S0, r, q, factors); F = S0 * math.exp((r - q) * T); lnK = math.log(K)   # phi(-i) = F
    P1 = 0.5 + quad(lambda u: (np.exp(-1j * u * lnK) * cf(u - 1j) / (1j * u * F)).real, 1e-12, umax, limit=500)[0] / math.pi
    P2 = 0.5 + quad(lambda u: (np.exp(-1j * u * lnK) * cf(u) / (1j * u)).real, 1e-12, umax, limit=500)[0] / math.pi
    return S0 * math.exp(-q * T) * P1 - K * math.exp(-r * T) * P2

_GL = {}
def heston_calls_gl(S0, Ks, T, r, q, factors, n=1000, U=150.0):   # all strikes at once; nodes cached
    if n not in _GL: _GL[n] = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (_GL[n][0] + 1) * U; w = 0.5 * U * _GL[n][1]; F = S0 * math.exp((r - q) * T); Ks = np.asarray(Ks, float)
    ex = np.exp(-1j * np.outer(u, np.log(Ks))); cf = lambda v: heston_cf(v, T, S0, r, q, factors)
    P1 = 0.5 + w @ (ex * (cf(u - 1j) / (1j * u * F))[:, None]).real / math.pi
    P2 = 0.5 + w @ (ex * (cf(u) / (1j * u))[:, None]).real / math.pi
    return S0 * math.exp(-q * T) * P1 - Ks * math.exp(-r * T) * P2

def merton_loglik(theta, x, dt, n_max=15):
    """Log-likelihood of log returns x (step dt) under Merton jump-diffusion; theta = (mu, sigma, lam, mu_J, sigma_J).
    Poisson mixture with compensator: mean_n = (mu - sigma^2/2 - lam*kbar) dt + n mu_J, var_n = sigma^2 dt + n sigma_J^2."""
    mu, s, lam, mJ, sJ = theta
    n = np.arange(n_max + 1); kbar = math.exp(mJ + 0.5 * sJ ** 2) - 1
    log_pois = -lam * dt + n * math.log(lam * dt) - gammaln(n + 1)
    mean = (mu - 0.5 * s * s - lam * kbar) * dt + n * mJ; var = s * s * dt + n * sJ * sJ
    log_phi = -0.5 * (np.log(2 * np.pi * var)[None, :] + (x[:, None] - mean[None, :]) ** 2 / var[None, :])
    return float(logsumexp(log_pois[None, :] + log_phi, axis=1).sum())
```
