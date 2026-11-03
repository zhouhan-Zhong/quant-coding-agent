---
title: Lookback and Asian options (closed forms, approximations, MC checks)
category: derivatives-pricing
tags: lookback option, floating strike, fixed strike, goldman-sosin-gatto, conze-viswanathan, running maximum, asian option, arithmetic average, geometric average, levy approximation, curran approximation, turnbull-wakeman, moment matching, control variate, discrete monitoring, path dependent
summary: Exact lookback formulas (cost-of-carry form), discrete geometric Asian closed form, Levy and Curran arithmetic approximations, and the MC/control-variate conventions used to validate them.
---
## When this applies
Pricing fixed/floating-strike lookbacks, arithmetic/geometric average-price (Asian) options with n monitoring
dates, "compare Levy/Curran approximations with Monte Carlo", control-variate MC for Asians, tables over
(T, n, moneyness), and summary flags such as "geometric < arithmetic", "Curran better than Levy".

## Method and formulas
b = r - q (cost of carry), sq = sigma sqrt(T), N = normal cdf. Running extremes so far: Smax, Smin (= S0 at inception).
- Floating-strike lookback call (pays S_T - min): a1 = (ln(S/Smin) + (b + sigma^2/2)T)/sq, a2 = a1 - sq,
  C = S e^{(b-r)T} N(a1) - Smin e^{-rT} N(a2) + S e^{-rT} (sigma^2/(2b)) [ (S/Smin)^{-2b/sigma^2} N(-a1 + (2b/sigma) sqrt(T)) - e^{bT} N(-a1) ].
- Floating-strike put (pays max - S_T): b1 = (ln(S/Smax) + (b + sigma^2/2)T)/sq, b2 = b1 - sq,
  P = Smax e^{-rT} N(-b2) - S e^{(b-r)T} N(-b1) + S e^{-rT} (sigma^2/(2b)) [ -(S/Smax)^{-2b/sigma^2} N(b1 - (2b/sigma) sqrt(T)) + e^{bT} N(b1) ].
- Fixed-strike call (pays max(Smax_T - K, 0)): if K > Smax use d1 = (ln(S/K) + (b + sigma^2/2)T)/sq, d2 = d1 - sq,
  C = S e^{(b-r)T} N(d1) - K e^{-rT} N(d2) + S e^{-rT} (sigma^2/(2b)) [ -(S/K)^{-2b/sigma^2} N(d1 - (2b/sigma) sqrt(T)) + e^{bT} N(d1) ];
  if K <= Smax: C = e^{-rT}(Smax - K) + [same expression with K replaced by Smax].
- Fixed-strike put (pays max(K - Smin_T, 0)): if K < Smin, f1 = (ln(S/K) + (b + sigma^2/2)T)/sq, f2 = f1 - sq,
  P = -S e^{(b-r)T} N(-f1) + K e^{-rT} N(-f2) + S e^{-rT} (sigma^2/(2b)) [ (S/K)^{-2b/sigma^2} N(-f1 + (2b/sigma) sqrt(T)) - e^{bT} N(-f1) ];
  if K >= Smin: P = e^{-rT}(K - Smin) + [same with K -> Smin]. b = 0 needs the limit form (use b = 1e-8 in practice).
- Discrete geometric Asian, fixings t_i = i T/n (i = 1..n, S0 excluded): ln G ~ Normal(mu_G, sG^2) with
  mu_G = ln S0 + (r - q - sigma^2/2) T (n+1)/(2n), sG^2 = sigma^2 T (n+1)(2n+1)/(6 n^2) = sigma^2/n^2 sum_ij min(t_i, t_j).
  Call = e^{-rT}[e^{mu_G + sG^2/2} N(d1) - K N(d2)], d1 = (mu_G + sG^2 - ln K)/sG, d2 = d1 - sG; put by parity on G:
  Put = e^{-rT}[K N(-d2) - e^{mu_G + sG^2/2} N(-d1)]. Continuous limit: sigma_G = sigma/sqrt(3), b_G = (b - sigma^2/6)/2.
- Levy (lognormal moment matching) for the arithmetic average A: E[A] = S0/n sum e^{b t_i},
  E[A^2] = S0^2/n^2 sum_i sum_j exp(b (t_i + t_j) + sigma^2 min(t_i, t_j)); v = ln E[A^2] - 2 ln E[A],
  Call = e^{-rT}[E[A] N(d1) - K N(d2)], d1 = (ln(E[A]/K) + v/2)/sqrt(v), d2 = d1 - sqrt(v).
- Curran (condition on G): mu_i = (b - sigma^2/2) t_i, s_i^2 = sigma^2 t_i, sig_iG = sigma^2/n sum_j min(t_i, t_j),
  K_hat = 2K - S0/n sum_i exp(mu_i + sig_iG/sG^2 (ln K - mu_G) + (s_i^2 - sig_iG^2/sG^2)/2) (floor at 1e-8),
  a = (mu_G - ln K_hat)/sG, Call = e^{-rT}[ S0/n sum_i e^{mu_i + s_i^2/2} N(a + sig_iG/sG) - K N(a) ].
- MC: logS_i = ln S0 + cumsum((b - sigma^2/2) dt + sigma sqrt(dt) Z), A = mean(exp(logS)), G = exp(mean(logS));
  control variate: P_cv = P_A - beta (P_G - Geo_exact), beta = cov(P_A, P_G)/var(P_G) (beta = 1 also fine);
  se = std(P_cv, ddof=1)/sqrt(N). Variance drops 100-1000x.

## Conventions and pitfalls (what graders check)
- Fixing convention matters: n fixings at i T/n (exclude S0) vs n+1 including S0 - read the task; mu_G uses mean(t_i).
- Lookback closed forms assume continuous monitoring; daily MC is biased low for calls on the max (max under-sampled)
  by roughly 0.58 sigma sqrt(dt) relative; report MC honestly and expect "cf > mc"; use >= 100k paths and the
  discrete-monitoring correction only if the task asks.
- Invariants: geometric Asian < arithmetic Asian (Jensen); Asian < European with the same K and T; prices decrease
  with K; lookback floating call >= vanilla ATM call; fixed-strike lookback call >= vanilla call; all prices > 0.
- At inception set Smax = Smin = S0; for seasoned options pass the observed extremes.
- MC se = std(ddof=1)/sqrt(N) of discounted payoffs; "mc_validates" typically |cf - mc| < 2-3 se (plus a small floor).
- Seed exactly as instructed (np.random.seed vs default_rng produce different paths); draw the whole (N x n) matrix
  once and reuse it for all strikes of a (T, n) pair.

## Tested snippet
```python
import math
import numpy as np
from scipy.stats import norm

N = norm.cdf
def lookback_fixed_call(S, K, T, r, q, sig, Smax):
    """Conze-Viswanathan fixed-strike call max(Smax_T - K, 0); Smax = running max so far (S0 at inception). b = r - q != 0."""
    b = r - q; sq = sig * math.sqrt(T); c = sig ** 2 / (2 * b)
    if K > Smax:
        d1 = (math.log(S / K) + (b + 0.5 * sig ** 2) * T) / sq; d2 = d1 - sq
        return S * math.exp((b - r) * T) * N(d1) - K * math.exp(-r * T) * N(d2) + S * math.exp(-r * T) * c * (
            -(S / K) ** (-2 * b / sig ** 2) * N(d1 - 2 * b / sig * math.sqrt(T)) + math.exp(b * T) * N(d1))
    e1 = (math.log(S / Smax) + (b + 0.5 * sig ** 2) * T) / sq; e2 = e1 - sq
    return math.exp(-r * T) * (Smax - K) + S * math.exp((b - r) * T) * N(e1) - Smax * math.exp(-r * T) * N(e2) + \
        S * math.exp(-r * T) * c * (-(S / Smax) ** (-2 * b / sig ** 2) * N(e1 - 2 * b / sig * math.sqrt(T)) + math.exp(b * T) * N(e1))

def lookback_float_call(S, T, r, q, sig, Smin):
    """Goldman-Sosin-Gatto floating-strike call S_T - Smin_T; Smin = running min so far."""
    b = r - q; sq = sig * math.sqrt(T); c = sig ** 2 / (2 * b)
    a1 = (math.log(S / Smin) + (b + 0.5 * sig ** 2) * T) / sq; a2 = a1 - sq
    return S * math.exp((b - r) * T) * N(a1) - Smin * math.exp(-r * T) * N(a2) + S * math.exp(-r * T) * c * (
        (S / Smin) ** (-2 * b / sig ** 2) * N(-a1 + 2 * b / sig * math.sqrt(T)) - math.exp(b * T) * N(-a1))

def asian_geometric_call(S0, K, T, r, q, sig, n):
    """Discrete geometric-average call, fixings t_i = i T/n, i = 1..n (exact lognormal)."""
    t = np.arange(1, n + 1) * T / n
    mu_G = math.log(S0) + (r - q - 0.5 * sig ** 2) * t.mean()
    var_G = sig ** 2 / n ** 2 * np.minimum.outer(t, t).sum()          # = sig^2 T (n+1)(2n+1)/(6 n^2)
    sG = math.sqrt(var_G); d1 = (mu_G + var_G - math.log(K)) / sG
    return math.exp(-r * T) * (math.exp(mu_G + 0.5 * var_G) * N(d1) - K * N(d1 - sG))

def asian_levy_call(S0, K, T, r, q, sig, n):
    """Levy lognormal moment-matching for the discrete arithmetic average."""
    t = np.arange(1, n + 1) * T / n
    EA = S0 * np.exp((r - q) * t).mean()
    EA2 = S0 ** 2 / n ** 2 * np.exp((r - q) * np.add.outer(t, t) + sig ** 2 * np.minimum.outer(t, t)).sum()
    v = math.log(EA2) - 2 * math.log(EA); sA = math.sqrt(v); d1 = (math.log(EA / K) + 0.5 * v) / sA
    return math.exp(-r * T) * (EA * N(d1) - K * N(d1 - sA))
```
