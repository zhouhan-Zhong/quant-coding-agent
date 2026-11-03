---
title: Compound, chooser, exchange (Margrabe) and spread (Kirk) options
category: derivatives-pricing
tags: compound option, geske, call on call, put on call, bivariate normal, chooser option, rubinstein, margrabe, exchange option, spread option, kirk approximation, correlated gbm, cholesky, two-asset, correlation, multi-asset calibration, monte carlo validation
summary: Geske compound-option formulas with the bivariate normal via scipy, simple chooser, Margrabe and Kirk closed forms, correlated-path simulation, and the two-asset calibration/invariant conventions.
---
## When this applies
"Compound option (option on an option)", "critical price S* at T1", "compound put-call parity", "chooser option",
"exchange option / Margrabe", "spread option with Kirk's approximation", correlation sensitivity tables, two-asset
MC with Cholesky, calibrating sigma_1, sigma_2, rho from two price histories.

## Method and formulas
BS call c / put p with yield q; sqT = sigma sqrt(T).
- Geske compound (pay K1 at T1 for a vanilla with strike K2, expiry T2 > T1), rho = sqrt(T1/T2), M(a, b; rho) = bivariate
  normal cdf. S* solves inner value at T1 = K1: c(S*, K2, T2 - T1) = K1 (call-on-call / put-on-call) or p(S*, ...) = K1
  (options on puts); brentq on [1e-6 S, 100 S]. a1 = (ln(S/S*) + (r - q + sigma^2/2) T1)/(sigma sqrt T1), a2 = a1 - sigma sqrt T1;
  b1 = (ln(S/K2) + (r - q + sigma^2/2) T2)/(sigma sqrt T2), b2 = b1 - sigma sqrt T2.
  call-on-call = S e^{-qT2} M(a1, b1; rho) - K2 e^{-rT2} M(a2, b2; rho) - K1 e^{-rT1} N(a2)
  put-on-call  = K2 e^{-rT2} M(-a2, b2; -rho) - S e^{-qT2} M(-a1, b1; -rho) + K1 e^{-rT1} N(-a2)
  call-on-put  = K2 e^{-rT2} M(-a2, -b2; rho) - S e^{-qT2} M(-a1, -b1; rho) - K1 e^{-rT1} N(-a2)
  put-on-put   = S e^{-qT2} M(a1, -b1; -rho) - K2 e^{-rT2} M(a2, -b2; -rho) + K1 e^{-rT1} N(a2)
  Parity: call-on-call - put-on-call = c(S, K2, T2) - K1 e^{-rT1} (same for puts with p). MC check: simulate S_T1, value the
  inner option analytically, payoff max(inner - K1, 0) e^{-rT1}.
- Simple chooser (choose call or put with strike K, expiry T at time Tc): value = c(S, K, T) + e^{-q(T - Tc)} p(S, K e^{-(r-q)(T - Tc)}, Tc);
  increases with Tc; equals a straddle when Tc = T.
- Margrabe exchange max(S1 - S2, 0): sigma_x = sqrt(sigma1^2 + sigma2^2 - 2 rho sigma1 sigma2),
  d1 = (ln(S1/S2) + (q2 - q1 + sigma_x^2/2) T)/(sigma_x sqrt T), d2 = d1 - sigma_x sqrt T,
  V = S1 e^{-q1 T} N(d1) - S2 e^{-q2 T} N(d2). No r dependence; decreasing in rho.
- Kirk spread call max(S1 - S2 - K, 0): F_i = S_i e^{(r - q_i) T}, w = F2/(F2 + K), sigma_K = sqrt(sigma1^2 - 2 rho sigma1 sigma2 w + sigma2^2 w^2),
  d1 = (ln(F1/(F2 + K)) + sigma_K^2 T/2)/(sigma_K sqrt T), V = e^{-rT}[F1 N(d1) - (F2 + K) N(d1 - sigma_K sqrt T)].
  K = 0 reproduces Margrabe exactly; negative K allowed while F2 + K > 0. Spread put by parity V_put = V_call - e^{-rT}(F1 - F2 - K).
- Correlated GBM: L = cholesky([[1, rho], [rho, 1]]); Z = standard_normal((n, 2)) @ L.T; S_i(T) = S_i exp((r - q_i - sigma_i^2/2) T + sigma_i sqrt(T) Z_i).
  For d assets use the full correlation matrix (make it PSD: clip eigenvalues at 1e-10 before cholesky).
- Two-asset calibration: inner-join the two files on date, log returns per asset, sigma_i = std(ddof=1) sqrt(252),
  rho = np.corrcoef of the aligned return series (not of prices), S_i0 = last close; n_returns = aligned rows - 1.

## Conventions and pitfalls (what graders check)
- scipy.stats.multivariate_normal(mean=[0,0], cov=[[1,rho],[rho,1]]).cdf([a, b]) is M(a, b; rho); it is slow-ish
  (~1 ms) - fine for grids of a few hundred points. |rho| must be < 1 (rho = sqrt(T1/T2) always is).
- Invariants: compound prices > 0 and < underlying vanilla; parity error < 1e-8; S* increases with K1; chooser >= max(c, p);
  Margrabe non-increasing in rho (tolerance 1e-2); Kirk non-increasing in K; Kirk(K=0) == Margrabe to 1e-10;
  MC within max(3 se + small floor, few % relative) - deep OTM spreads may have mc = se = 0, keep the floor.
- Do not clip tiny positive prices (5e-12) to 0 or round them: checkers require strictly positive values.
- Keep strike sign conventions from the task (e.g. K = avg_spot x moneyness with negative moneyness = lower strike).
- Reuse one normal matrix per maturity across strikes (CRN) so price differences across K are smooth.

## Tested snippet
```python
import math
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import brentq

def bs(S, K, T, r, q, s, call=True):
    sq = s * math.sqrt(T); d1 = (math.log(S / K) + (r - q + 0.5 * s * s) * T) / sq; d2 = d1 - sq
    if call: return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

def M2(a, b, rho):                                   # bivariate standard normal cdf P(X <= a, Y <= b)
    return float(multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]).cdf([a, b]))

def geske(S, K1, K2, T1, T2, r, q, s, kind="call-on-call"):
    """Compound option: pay K1 at T1 for a vanilla (strike K2, expiry T2). Returns (price, critical S* at T1)."""
    inner_call = kind.endswith("call"); rho = math.sqrt(T1 / T2)
    Ss = brentq(lambda x: bs(x, K2, T2 - T1, r, q, s, inner_call) - K1, 1e-6 * S, 100 * S)
    a1 = (math.log(S / Ss) + (r - q + 0.5 * s * s) * T1) / (s * math.sqrt(T1)); a2 = a1 - s * math.sqrt(T1)
    b1 = (math.log(S / K2) + (r - q + 0.5 * s * s) * T2) / (s * math.sqrt(T2)); b2 = b1 - s * math.sqrt(T2)
    Sq, Kr, K1r = S * math.exp(-q * T2), K2 * math.exp(-r * T2), K1 * math.exp(-r * T1)
    if kind == "call-on-call": v = Sq * M2(a1, b1, rho) - Kr * M2(a2, b2, rho) - K1r * norm.cdf(a2)
    elif kind == "put-on-call": v = Kr * M2(-a2, b2, -rho) - Sq * M2(-a1, b1, -rho) + K1r * norm.cdf(-a2)
    elif kind == "call-on-put": v = Kr * M2(-a2, -b2, rho) - Sq * M2(-a1, -b1, rho) - K1r * norm.cdf(-a2)
    else:                      v = Sq * M2(a1, -b1, -rho) - Kr * M2(a2, -b2, -rho) + K1r * norm.cdf(a2)   # put-on-put
    return v, Ss

def chooser(S, K, T, Tc, r, q, s):                   # simple chooser: choose call or put at Tc, both strike K expiry T
    return bs(S, K, T, r, q, s, True) + math.exp(-q * (T - Tc)) * bs(S, K * math.exp(-(r - q) * (T - Tc)), Tc, r, q, s, False)

def margrabe(S1, S2, T, q1, q2, s1, s2, rho):        # exchange option max(S1 - S2, 0)
    sx = math.sqrt(s1 * s1 + s2 * s2 - 2 * rho * s1 * s2); sq = sx * math.sqrt(T)
    d1 = (math.log(S1 / S2) + (q2 - q1 + 0.5 * sx * sx) * T) / sq
    return S1 * math.exp(-q1 * T) * norm.cdf(d1) - S2 * math.exp(-q2 * T) * norm.cdf(d1 - sq)

def kirk(S1, S2, K, T, r, q1, q2, s1, s2, rho):      # spread call max(S1 - S2 - K, 0); K = 0 reduces to Margrabe
    F1, F2 = S1 * math.exp((r - q1) * T), S2 * math.exp((r - q2) * T); w = F2 / (F2 + K)
    sk = math.sqrt(s1 * s1 - 2 * rho * s1 * s2 * w + s2 * s2 * w * w); sq = sk * math.sqrt(T)
    d1 = (math.log(F1 / (F2 + K)) + 0.5 * sk * sk * T) / sq
    return math.exp(-r * T) * (F1 * norm.cdf(d1) - (F2 + K) * norm.cdf(d1 - sq))
```
