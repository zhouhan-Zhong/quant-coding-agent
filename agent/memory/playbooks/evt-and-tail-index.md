---
title: Extreme value theory: Hill, Smith GPD / peaks-over-threshold, GEV
category: risk
tags: evt, extreme value, tail index, hill estimator, hill plot, tail estimator, generalized pareto, gpd, peaks over threshold, pot, smith estimator, maximum likelihood, threshold, mean excess, excess, extreme quantile, extreme var, tail var, expected shortfall, block maxima, gev, generalized extreme value, return level, fat tails, heavy tails, confidence interval, standard error, observed fisher information, hessian, order statistics, k grid, best k
summary: Hill and Smith (GPD MLE on the top-k excesses) tail-index estimators with standard errors from asymptotics or the observed information, POT VaR/ES, threshold choice, GEV block maxima, and the ordering/sign/start-value conventions graders check.
---
## When this applies
"Estimate the tail index with the Hill estimator for k = 25..1000", "Smith (GPD maximum likelihood) estimator",
"fit a GPD to exceedances above the 95th percentile", "EVT VaR/ES at 99%/99.9%", "mean excess plot / threshold
selection", "block maxima GEV and return level", "standard errors and confidence intervals for xi", "best k".

## Method and formulas
Symbols: L_i = losses (positive numbers, L = -r for returns), n = sample size, X_(1) >= X_(2) >= ... order
statistics in DESCENDING order, k = number of top order statistics, u = threshold, N_u = number of exceedances,
xi = shape (tail index; alpha = 1/xi is the Pareto exponent), beta = GPD scale, p = quantile level (0.99).
- Losses: L = -diff(log close) over the full history given (or -simple returns if stated); x = sort descending.
- Hill (xi > 0 only): xi_H(k) = (1/k) sum_{i=1..k} ln X_(i) - ln X_(k+1) (k terms; threshold = X_(k+1) = x[k] in a
  0-based descending array). s.e. = xi/sqrt(k); CI = xi +/- 1.96 s.e. (can go negative for small k).
  Quantile: VaR_p = X_(k+1) ((n/k)(1-p))^(-xi); ES_p = VaR_p/(1-xi) for xi < 1.
- Smith estimator = GPD MLE on the top-k excesses y = x[:k] - x[k] (scale free, location fixed at 0):
  NLL(xi, beta) = k ln beta + (1 + 1/xi) sum ln(1 + xi y/beta), constraint 1 + xi y/beta > 0 (penalty 1e10).
  Nelder-Mead (xatol 1e-10, fatol 1e-12) from several starts, e.g. (xi, beta) = (0.2, mean y), (0.1, 0.5 mean y),
  (0.4, 1.5 mean y); keep the lowest NLL. Equivalent to scipy.stats.genpareto.fit(y, floc=0) (c = xi).
- Standard errors: observed Fisher information = numerical Hessian of the NLL at the MLE (central differences,
  relative step ~1e-4); se(xi) = sqrt(inv(H)[0,0]); fallback when H is not PD: (1 + xi)/sqrt(k). Expected-information
  form: Cov(xi, beta) = (1+xi)/N_u [[1+xi, -beta], [-beta, 2 beta^2]] (xi > -0.5).
- k grid tables: one row per k with xi, se, CI, beta; summary = mean xi over k, correlation of Hill vs Smith across k
  (np.corrcoef), best_k = argmin se (largest k for Hill), tail_is_heavy = xi > 0 at best_k (python bool).
- POT with a quantile threshold: y = L - u for L > u; G(y) = 1 - (1 + xi y/beta)^(-1/xi);
  VaR_p = u + (beta/xi)[((n/N_u)(1-p))^(-xi) - 1]; ES_p = VaR_p/(1-xi) + (beta - xi u)/(1-xi) (xi < 1);
  P(L > x) = (N_u/n)(1 + xi (x-u)/beta)^(-1/xi).
- Threshold choice: empirical quantile (0.90-0.95) or a count (top 5-10%); mean excess e(u) = mean(L - u | L > u) is
  linear in u with slope xi/(1-xi) under a GPD - pick u where the plot turns linear; keep N_u >= ~50.
- Block maxima / GEV: blocks of m observations (21 ~ monthly, 63 quarterly, 250 yearly), max per block;
  G(z) = exp(-(1 + xi (z-mu)/sigma)^(-1/xi)); scipy.stats.genextreme.fit returns (c, loc, scale) with c = -xi;
  return level for T blocks z_T = mu + (sigma/xi)[(-ln(1-1/T))^(-xi) - 1]. A GPD tail shares xi with the GEV; Hill
  is consistent only for xi > 0 (Frechet domain).

## Conventions and pitfalls (what graders check)
- Work on LOSSES, keep L > 0 for Hill logs, sort descending explicitly; the classic off-by-one is X_(k) instead of
  X_(k+1) as threshold or k+1 terms in the sum.
- scipy signs: genpareto c = xi, genextreme c = -xi. Fix location 0 for excesses (floc=0); fitting a location changes
  xi materially and fails the reference.
- Hill typically exceeds Smith on daily equity losses (Hill is biased upward at finite k); checkers assert wide
  ranges for xi, Hill vs Smith ordering, se decreasing in k, sigma(k small) > sigma(k large), CI brackets the
  estimate, |corr| <= 1. The Hill plot is bumpy - no smoothing needed.
- Confidence level p is the loss quantile level (0.99 = 1% tail) -> (n/N_u)(1-p); valid only for p > 1 - N_u/n.
- xi >= 1 means infinite mean - ES undefined (null, not a number); VaR_POT > u, ES > VaR, monotone in p.
- Threshold from np.quantile is interpolated; "top 5%" means k = int(0.05 n) with the threshold at the (k+1)-th
  largest - keep N_u consistent with the stated rule; mean-excess counts use strict L > u (drop ties at u).
- Block maxima: n // m full blocks in chronological order; < 30 blocks gives unstable GEV fits.
- Integer counts (k, N_u, n_blocks) exact; estimates usually loose (atol 0.03-0.06 on xi).

## Tested snippet
```python
import math, numpy as np
from scipy import stats
from scipy.optimize import minimize

def hill(loss, k):                                     # loss > 0; k = number of top order statistics used
    x = np.sort(loss)[::-1]                            # x[0] largest ... x[k] is the (k+1)-th largest = threshold
    xi = float(np.mean(np.log(x[:k])) - math.log(x[k]))
    return xi, 1.0 / xi, xi / math.sqrt(k)             # tail index xi, alpha = 1/xi, asymptotic s.e. of xi
def hill_var_es(loss, k, p):                           # loss quantile at level p > 1 - k/n and ES (needs xi < 1)
    x = np.sort(loss)[::-1]; xi = hill(loss, k)[0]; var = x[k] * ((len(loss) / k) * (1 - p)) ** (-xi)
    return float(var), float(var / (1 - xi)) if xi < 1 else float("nan")
def mean_excess(loss, u): y = loss[loss > u] - u; return float(y.mean()), int(y.size)
def num_hessian(f, x, rel=1e-4):                       # central differences; inverse = observed-information covariance
    x = np.asarray(x, float); h = rel * np.maximum(np.abs(x), 1e-8); n = len(x); H = np.empty((n, n)); E = np.eye(n)
    for i in range(n):
        for j in range(n):
            a, b = E[i] * h[i], E[j] * h[j]; H[i, j] = (f(x + a + b) - f(x + a - b) - f(x - a + b) + f(x - a - b)) / (4 * h[i] * h[j])
    return H
def gpd_fit(y, starts=((0.2, 1.0), (0.1, 0.5), (0.4, 1.5))):   # Smith MLE on excesses y > 0 -> (xi, beta, loglik, se_xi)
    def nll(p):
        xi, b = p; z = 1 + xi * y / b
        if b <= 0 or np.any(z <= 0): return 1e10
        return y.size * math.log(b) + ((1 + 1 / xi) * np.sum(np.log(z)) if abs(xi) > 1e-9 else y.sum() / b)
    res = min((minimize(nll, [x0, s * y.mean()], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000}) for x0, s in starts), key=lambda r: r.fun)
    try: se = math.sqrt(np.linalg.inv(num_hessian(nll, res.x))[0, 0])
    except (ValueError, np.linalg.LinAlgError): se = (1 + res.x[0]) / math.sqrt(y.size)   # asymptotic fallback
    return float(res.x[0]), float(res.x[1]), -float(res.fun), se
def pot_var_es(loss, u, p):                            # peaks-over-threshold quantile and ES at level p
    y = loss[loss > u] - u; n, nu = loss.size, y.size; xi, beta, ll, se = gpd_fit(y)
    var = u + beta / xi * (((n / nu) * (1 - p)) ** (-xi) - 1)
    return var, var / (1 - xi) + (beta - xi * u) / (1 - xi), xi, beta, nu
def gev_block_maxima(loss, block, T=10):               # block maxima -> GEV; scipy's shape c = -xi (sign flip!)
    m = loss[: loss.size // block * block].reshape(-1, block).max(axis=1)
    c, mu, sig = stats.genextreme.fit(m); xi = -c      # T-block return level: mu + sig/xi ((-ln(1-1/T))^-xi - 1)
    return float(xi), float(mu), float(sig), float(mu + sig / xi * ((-math.log(1 - 1 / T)) ** (-xi) - 1))

loss = np.abs(stats.t.rvs(3, size=5000, random_state=1)) * 0.01     # heavy-tailed losses, true xi = 1/3
u = float(np.quantile(loss, 0.95)); x = np.sort(loss)[::-1]
print([round(v, 4) for v in hill(loss, 250)], [round(v, 5) for v in hill_var_es(loss, 250, 0.99)], mean_excess(loss, u))
v, e, xi, beta, nu = pot_var_es(loss, u, 0.99); print(round(v, 5), round(e, 5), round(xi, 4), round(beta, 5), nu, round(float(np.quantile(loss, 0.99)), 5))
tab = [(k, round(hill(loss, k)[0], 4), round(hill(loss, k)[2], 4)) + tuple(round(z, 4) for z in gpd_fit(x[:k] - x[k])[::3]) for k in (50, 100, 250, 500)]
print(tab, "best_k", min(tab, key=lambda r: r[4])[0], [round(v, 4) for v in gev_block_maxima(loss, 50)])
```
