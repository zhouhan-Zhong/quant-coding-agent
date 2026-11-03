---
title: VaR and Expected Shortfall: estimators, scaling, capital and backtests
category: risk
tags: var, value at risk, expected shortfall, es, cvar, historical simulation, parametric, variance-covariance, normal, student-t, t-distribution, cornish-fisher, ewma, riskmetrics, age-weighted, filtered historical simulation, fhs, garch, dcc, monte carlo var, square root of time, 10-day, stressed var, basel, capital charge, multiplier, kupiec, christoffersen, conditional coverage, backtest, violations, exceedances, traffic light, stationary bootstrap, confidence level, quantile, sorted index
summary: Loss-positive VaR/ES formulas for HS, normal, Student-t, Cornish-Fisher, EWMA, age-weighted and filtered HS, Monte Carlo and horizon scaling, the quantile/sign variants references use, Basel-style capital, Kupiec/Christoffersen/traffic-light backtests and the stationary bootstrap.
---
## When this applies
"Compute 1-day and 10-day VaR and ES at 95%/99% with historical, parametric normal, Student-t, EWMA ... methods",
"backtest the VaR with Kupiec/Christoffersen", "count exceedances", "traffic-light zone", "portfolio VaR in
dollars on a notional", "most conservative method", "stressed VaR and the capital charge", "CVaR of a strategy".

## Method and formulas
Symbols: r_t returns (decimal, daily), L = -r loss, a = confidence level (0.95/0.99), q = quantile(r, 1-a),
z_a = Phi^-1(a), mu/sd = sample mean/std (ddof=1), N = notional. Report VaR and ES as POSITIVE losses.
- Historical simulation: VaR = -quantile(r, 1-a) (np.quantile default linear interpolation), ES = -mean(r[r <= q]).
- Parametric normal: VaR = -mu + z_a sd, ES = -mu + sd phi(z_a)/(1-a). Keep mu unless told "zero mean".
- Student-t (MLE: nu, loc, scale = scipy.stats.t.fit(r)): t_a = t.ppf(a, nu); VaR = -loc + scale t_a;
  ES = -loc + scale pdf_t(t_a)/(1-a) * (nu + t_a^2)/(nu-1). Moment-matched variant: scale = sd sqrt((nu-2)/nu).
- Cornish-Fisher: S = skew, K = excess kurtosis, z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36 with z = z_a;
  VaR = -mu + z_cf sd.
- EWMA/RiskMetrics: s2_t = lam s2_{t-1} + (1-lam) r_{t-1}^2 (lam 0.94 daily, 0.97 monthly), init = sample
  variance (or r_1^2); multivariate S_t = lam S_{t-1} + (1-lam) r r' then sd_p = sqrt(w'Sw). VaR = z_a sd_t (zero
  mean), ES = sd_t phi(z_a)/(1-a). Annualised vol = sd sqrt(252).
- Age-weighted HS (lam = 0.98): w_i = lam^(n-1-i)(1-lam)/(1-lam^n) (i = 0 oldest); sort returns ascending with their
  weights; VaR = -(first sorted return whose cumulative weight >= 1-a), no interpolation; ES = weighted mean of
  returns <= that quantile.
- Filtered HS: fit GARCH (returns*100), z_t = resid_t/sigma_t; VaR_{t+1} = -(mu + sigma_{t+1} quantile(z, 1-a)),
  ES uses the mean of z below that quantile times sigma_{t+1}. Multivariate (DCC) FHS and GARCH h-day variance
  sums: see garch-family.
- Monte Carlo: simulate from the fitted distribution with a fixed seed (rng = np.random.default_rng(seed)),
  VaR = -quantile(sim, 1-a), ES = -mean of the tail; report n_sims and seed.
- Quantile variants seen in references: (i) index rule VaR = ceil(aN)-th smallest loss (sorted ascending, no
  interpolation), clamped at 0 when every outcome is a gain, ES = mean of losses strictly above VaR (0 if none);
  (ii) loss-distribution form VaR = np.quantile(L, a), CVaR = mean(L[L >= VaR]) (credit losses); (iii) trailing
  strategy window VaR = -np.percentile(r[-W:], 1), CVaR = -mean(r[-W:][r[-W:] <= -VaR]); (iv) return-space
  reporting (VaR = 1% quantile of returns, CVaR = mean below it, both negative, so var_1pct > cvar_1pct).
- Horizon scaling: parametric VaR_h = VaR_1 sqrt(h) (zero drift; with drift VaR_h = -mu h + z sd sqrt(h)); HS/AWHS
  h-day: overlapping h-day sums np.convolve(r, ones(h), 'valid') then the same quantile rule; GARCH: z sqrt(sum of
  the h forecast variances).
- Capital (Basel IMA style): charge = m_c VaR_10d + m_s sVaR_10d, sVaR = max over rolling estimation windows (or the
  stated stress period); report each component; traffic-light plus-factor only if the task gives the table.
- Dollar VaR = VaR_return * N. Portfolio return = w'r (log or simple as the task states). Correlation = np.corrcoef
  of aligned returns.
- Backtest: VaR_t estimated from data up to t-1 (expanding window with minimum 250/252 obs, or rolling window),
  exceedance if L_t > VaR_t, i.e. r_t < -VaR_t (strict). Kupiec POF: LR = -2[(n-x)ln(1-p) + x ln p] + 2[(n-x)ln(1-x/n)
  + x ln(x/n)], p = 1-a, p-value = chi2.sf(LR, 1). Christoffersen independence: transition counts n00,n01,n10,n11,
  p01 = n01/(n00+n01), p11 = n11/(n10+n11), pi = (n01+n11)/(n-1), 0 ln 0 = 0, LR_ind = -2[ln L0 - ln L1] ~ chi2(1);
  LR_cc = LR_uc + LR_ind ~ chi2(2). Basel traffic light (250 days, 99%): green <= 4, yellow 5-9, red >= 10.
- Stationary bootstrap (Politis-Romano) for scenario/FHS resampling: block start ~ U{0..T-1}, length ~
  Geometric(1/b), wrap mod T, rng = default_rng(seed); report B and b.

## Conventions and pitfalls (what graders check)
- Sign: VaR, ES > 0; ES >= VaR (checker often asserts es >= 0.95*var); 10-day >= 1-day; VaR increases with a.
- Normal VaR must include the mean (dropping it moves the answer ~5%); EWMA VaR must NOT (conditional zero mean).
  Pure-normal outputs are checked through ES/VaR == phi(z)/((1-a) z) (rtol 1e-4).
- sd with ddof=1 everywhere; correlation on returns, not prices; n_returns = n_prices - 1.
- AWHS uses the discrete "first cumulative weight >= 1-a" rule; interpolation shifts the answer ~0.5%.
- EWMA via the plain recursion over the whole sample (pandas ewm(alpha=1-lam).var() differs in initialisation and
  bias correction); after a few hundred steps the initial value is irrelevant.
- Student-t with nu <= 2 has no ES; bound nu > 2 if fitting manually; scipy t.fit returns (df, loc, scale).
- Backtest count: n_test = n - window; exceedance uses the strict inequality; report rate = x/n_test, p-values from
  chi2.sf, reject flags = p < 0.05 consistent with the p written; num_violations is usually pinned exactly.
- Square-root-of-time applies to parametric methods; for HS the task specifies overlapping sums or scaling -
  follow it, and do not scale ES differently from VaR.
- Pinned Monte Carlo / bootstrap means depend on the exact draw stream (API, order, shapes); when the stream cannot
  be identified, stay with the documented convention and accept the tolerance.
- Write full-precision floats cast with float(); method names and alpha values exactly as the spec lists them;
  typical tolerances rtol 1-3% on VaR, 3-5% on ES, exact integers for counts.
- "Most conservative" = largest VaR at the stated level; ties broken by the order in the spec.

## Tested snippet
```python
import math, numpy as np
from scipy import stats

def var_es_hs(r, a):                                   # r = returns (loss = -r); VaR/ES returned positive
    q = np.quantile(r, 1 - a); return -q, -r[r <= q].mean()
def var_es_sorted(r, a):                               # index rule: ceil(aN)-th smallest loss, clamp 0; ES = mean of losses strictly above
    L = np.sort(-np.asarray(r)); v = max(float(L[math.ceil(a * len(L)) - 1]), 0.0); tail = L[L > v]
    return v, float(tail.mean()) if tail.size else 0.0
def var_es_normal(r, a, zero_mean=False):              # keep the mean unless the task says zero-mean
    mu, sd, z = (0.0 if zero_mean else r.mean()), r.std(ddof=1), stats.norm.ppf(a)
    return -mu + z * sd, -mu + sd * stats.norm.pdf(z) / (1 - a)
def var_es_t(r, a):                                    # MLE Student-t (nu, loc, scale)
    nu, loc, sc = stats.t.fit(r); tq = stats.t.ppf(a, nu)
    return -loc + sc * tq, -loc + sc * stats.t.pdf(tq, nu) / (1 - a) * (nu + tq * tq) / (nu - 1)
def ewma_var(r, a, lam=0.94):                          # RiskMetrics: zero mean, s2_t = lam s2_{t-1} + (1-lam) r_{t-1}^2
    s2 = r.var(ddof=1)
    for x in r: s2 = lam * s2 + (1 - lam) * x * x
    return stats.norm.ppf(a) * math.sqrt(s2), math.sqrt(s2)   # next-day VaR, current daily vol
def age_weighted_hs(r, a, lam=0.98):                   # first sorted obs whose cumulative weight >= 1-a (no interpolation)
    n = len(r); w = lam ** np.arange(n - 1, -1, -1); w /= w.sum()
    o = np.argsort(r); sr, sw = r[o], w[o]; cw = np.cumsum(sw)
    q = sr[int(np.searchsorted(cw, 1 - a))]; m = sr <= q
    return -q, -float(np.sum(sr[m] * sw[m]) / sw[m].sum())
def stationary_bootstrap_idx(T, B, mean_block, rng):   # Politis-Romano: start ~ U{0..T-1}, length ~ Geometric(1/b), wrap mod T
    idx, i = np.empty(B, int), 0
    while i < B:
        s, m = rng.integers(T), min(int(rng.geometric(1.0 / mean_block)), B - i); idx[i:i + m] = (s + np.arange(m)) % T; i += m
    return idx
def kupiec(x, n, p):                                   # x exceedances in n days, target prob p = 1 - a; returns LR, p-value
    ll0 = (n - x) * math.log(1 - p) + x * math.log(p)
    ll1 = (n - x) * math.log(1 - x / n) + (x * math.log(x / n) if x > 0 else 0.0)
    return -2 * (ll0 - ll1), float(stats.chi2.sf(-2 * (ll0 - ll1), 1))
def christoffersen_ind(hits):                          # independence LR on the 0/1 hit sequence; LR_cc = LR_uc + LR_ind ~ chi2(2)
    h = np.asarray(hits, int); n = [[int(np.sum((h[:-1] == i) & (h[1:] == j))) for j in (0, 1)] for i in (0, 1)]
    p01, p11, pi = n[0][1] / max(sum(n[0]), 1), n[1][1] / max(sum(n[1]), 1), (n[0][1] + n[1][1]) / (len(h) - 1)
    lg = lambda p, k: k * math.log(p) if k > 0 else 0.0
    lr = -2 * (lg(1 - pi, n[0][0] + n[1][0]) + lg(pi, n[0][1] + n[1][1]) - lg(1 - p01, n[0][0]) - lg(p01, n[0][1]) - lg(1 - p11, n[1][0]) - lg(p11, n[1][1]))
    return lr, float(stats.chi2.sf(lr, 1))

rng = np.random.default_rng(0); r = 0.0003 + 0.01 * rng.standard_t(5, 1500) / math.sqrt(5 / 3)
print([tuple(round(float(v), 5) for v in f(r, 0.99)) for f in (var_es_hs, var_es_sorted, var_es_normal, var_es_t, age_weighted_hs, ewma_var)])
hits = [-r[t] > -np.quantile(r[:t], 0.01) for t in range(250, len(r))]; uc, ind = kupiec(sum(hits), len(hits), 0.01), christoffersen_ind(hits)   # expanding HS
print(sum(hits), [round(float(v), 4) for v in uc + ind], round(uc[0] + ind[0], 4), round(float(stats.chi2.sf(uc[0] + ind[0], 2)), 4))
bs = r[stationary_bootstrap_idx(len(r), 1000, 21, np.random.default_rng(42))]; tail = r[-63:]; v63 = -np.percentile(tail, 1)
print(round(float(-np.percentile(bs, 1)), 5), round(float(v63), 5), round(float(-tail[tail <= -v63].mean()), 5))
```
