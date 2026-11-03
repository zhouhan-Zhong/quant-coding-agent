---
title: Merton structural credit model: asset value and volatility, distance to default, PD and credit spread
category: credit
tags: merton model, structural model, distance to default, dd, probability of default, pd, kmv, moody's kmv, edf, expected default frequency, asset value, asset volatility, equity as call option, leverage, debt face value, default point, default barrier, credit spread, risky debt, risky bond, two equations, fsolve, iterative, equity volatility, black-scholes, risk neutral pd, physical pd, asset drift, black-cox, first passage
summary: The two Merton equations (equity = call on assets, equity vol = N(d1) V sigma_V / E) solved for asset value and volatility, risk-neutral and physical distance to default and PD, risky-debt value and credit spread, and the leverage/vol invariants graders check.
---
## When this applies
"Treat equity as a call option on the firm's assets", "infer the asset value and asset volatility from equity value
and equity volatility", "distance to default / probability of default (Merton, KMV)", "credit spread implied by the
structural model", "risky debt value", tasks giving market cap, equity vol, debt face value, maturity and a rate.

## Method and formulas
Symbols: V asset value, sigma_V asset vol (annual), E equity value (market cap), sigma_E equity vol, D debt face value
(default point), T horizon in years (1 unless stated), r risk-free (continuous), mu asset drift (physical measure).
- d1 = (ln(V/D) + (r + sigma_V^2/2) T) / (sigma_V sqrt(T)), d2 = d1 - sigma_V sqrt(T).
- Equity = call on assets: E = V N(d1) - D e^{-rT} N(d2). Ito: sigma_E E = N(d1) sigma_V V. Two equations, two
  unknowns (V, sigma_V): solve with fsolve/root from V0 = E + D e^{-rT}, sigma_V0 = sigma_E E/(E + D); parametrise
  in logs so both stay positive; check both residuals (< 1e-8 relative).
- KMV-style iteration from an equity price series: start sigma_V = sigma_E; invert E_t = call(V_t) daily for V_t;
  sigma_V = std(diff ln V_t, ddof=1) sqrt(252); repeat until |change| < 1e-6. Default point (KMV) = short-term debt
  + 0.5 long-term debt when the task gives the balance-sheet split; otherwise D = total debt face.
- Risk-neutral PD = N(-d2) (probability V_T < D). Physical: DD = (ln(V/D) + (mu - sigma_V^2/2) T)/(sigma_V sqrt(T)),
  PD = N(-DD); KMV "simple" DD = (V - D)/(V sigma_V) when the task uses it (state which). EDF maps DD through an
  empirical table only when the task ships the table.
- Risky debt B = V - E = D e^{-rT} N(d2) + V N(-d1); yield y = -ln(B/D)/T; credit spread = y - r (continuous,
  decimal; x1e4 for bp). Expected recovery given default = E[V_T | V_T < D]/D = V e^{(r)T} N(-d1)/(D N(-d2)).
- Sensitivities: PD rises with leverage D/V, sigma_V and T (for d2 > 0); equity delta to assets N(d1); equity vol
  falls as V rises (leverage effect).
- Black-Cox / first-passage variant: default when V hits a barrier H before T; PD = N(-h1) + (H/V)^(2 nu/sigma^2) N(-h2)
  with nu = mu - sigma^2/2, h1 = (ln(V/H) + nu T)/(sigma sqrt(T)), h2 = (ln(V/H) - nu T)/(sigma sqrt(T)) - only when
  the task names a barrier model.

## Conventions and pitfalls (what graders check)
- Units: equity value and debt in the same currency unit (millions vs units), rates and vols as decimals, T in years.
  sigma_E from daily log returns: std(ddof=1 unless stated) * sqrt(252) (or the task's trading-day count).
- Invariants asserted: V > E, V > B > 0, E + B = V, 0 < sigma_V < sigma_E (leverage amplifies equity vol), PD in (0, 1),
  spread > 0, N(d1) in (0, 1), the two equations hold to tolerance at the solution, PD increasing in D/V and sigma_V.
- Risk-neutral vs physical: N(-d2) uses r; the task's "distance to default" may use mu (often the historical asset
  return or an assumed drift). Report the one named; when both are asked, label them separately.
- Do not iterate on E = V - D (book leverage): equity is the option value, not the difference.
- fsolve pitfalls: start from V0 = E + D e^{-rT}; if it returns the initial guess unchanged check `ier`; use
  xtol=1e-12 and verify residuals; for extreme leverage use log-parametrisation and bounds (scipy.optimize.least_squares).
- Credit spread from B/D uses the FACE value D and continuous compounding unless the task says annual yield.
- Default probability "over the next year" needs T = 1 even if the debt matures later, when the task frames it so.
- Outputs: PD as a probability (0.0123), not percent, unless the task says "in percent"; DD as a float; asset
  values with full precision; booleans like `default_likely` only from the task's threshold.

## Tested snippet
```python
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

def merton_equity(V, sv, D, r, T):                       # equity = call on assets with strike = debt face D
    d1 = (np.log(V / D) + (r + 0.5 * sv**2) * T) / (sv * np.sqrt(T)); d2 = d1 - sv * np.sqrt(T)
    return V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2), d1, d2

def solve_merton(E, sE, D, r, T):                        # unknowns: asset value V and asset vol sv (two equations)
    def eqs(x):
        V, sv = np.exp(x[0]), np.exp(x[1])               # log-parametrised to stay positive
        Em, d1, _ = merton_equity(V, sv, D, r, T)
        return [Em / E - 1.0, norm.cdf(d1) * V * sv / (E * sE) - 1.0]
    x, info, ier, msg = fsolve(eqs, [np.log(E + D * np.exp(-r * T)), np.log(sE * E / (E + D))], xtol=1e-12, full_output=True)
    assert ier == 1 and max(abs(np.array(eqs(x)))) < 1e-9, msg
    return float(np.exp(x[0])), float(np.exp(x[1]))

E, sE, D, r, T, mu = 60.0, 0.45, 80.0, 0.03, 1.0, 0.08    # equity value, equity vol, debt face, rate, horizon, asset drift
V, sv = solve_merton(E, sE, D, r, T)
_, d1, d2 = merton_equity(V, sv, D, r, T)
pd_rn = norm.cdf(-d2)                                                         # risk-neutral PD
dd = (np.log(V / D) + (mu - 0.5 * sv**2) * T) / (sv * np.sqrt(T)); pd_phys = norm.cdf(-dd)   # physical DD and PD
B = V - E                                                                     # risky debt value
spread = -np.log(B / D) / T - r                                               # continuously compounded credit spread
print(round(V, 4), round(sv, 6), round(pd_rn, 6), round(dd, 4), round(pd_phys, 6), round(spread * 1e4, 2), "bp")
print(V > E, sv < sE, abs(merton_equity(V, sv, D, r, T)[0] - E) < 1e-8, abs(norm.cdf(d1) * V * sv - sE * E) < 1e-8)
```
