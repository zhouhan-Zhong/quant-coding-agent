---
title: Nelson-Siegel / Svensson curve fitting and bond price, yield, accrued-interest conventions
category: rates-curves
tags: nelson-siegel, nelson siegel, svensson, nss, yield curve fit, parametric curve, beta0, beta1, beta2, tau, lambda, decay, level slope curvature, zero curve, par curve, spot rate, forward rate, treasury, bond price, dirty price, clean price, accrued interest, day count, act/act, 30/360, coupon frequency, semiannual, yield to maturity, ytm, duration, dv01, repricing error, rmse, least squares, grid search, curve_fit, smoothness, extrapolation
summary: Both Nelson-Siegel parametrisations (decay rate vs time scale) and Svensson, a robust fit (grid over tau + linear least squares for the betas, optional duration weights), zero/forward/par conversions, and the bond price/yield/accrued conventions that make repricing tests pass.
---
## When this applies
"Fit a Nelson-Siegel (Svensson) curve to Treasury yields or bond prices", "report the curve parameters and fitted yields
at a grid of maturities", "reprice the bonds from the fitted curve and report errors", "zero vs par vs forward curve",
"clean/dirty price and accrued interest", "yield to maturity, duration, DV01 of a coupon bond".

## Method and formulas
Symbols: t maturity in years, y(t) continuously compounded zero yield unless the task fixes another compounding.
- Nelson-Siegel with decay RATE lambda: y(t) = b0 + b1 (1 - e^{-lambda t})/(lambda t) + b2 ((1 - e^{-lambda t})/(lambda t) - e^{-lambda t}).
  With time SCALE tau: x = t/tau, f1 = (1 - e^{-x})/x, y(t) = b0 + b1 f1 + b2 (f1 - e^{-x}); tau = 1/lambda. Read
  which one the task names; the betas are identical, only the fourth parameter is inverted. Limits: y(0+) = b0 + b1
  (short rate), y(inf) = b0 (level); b1 = negative of the slope; b2 = curvature (hump at t ~ 1.8/lambda).
- Svensson adds b3 ((1 - e^{-x2})/x2 - e^{-x2}) with x2 = t/tau2 (second hump); needs tau1 != tau2.
- Instantaneous forward: f(t) = b0 + b1 e^{-x} + b2 x e^{-x} (NS). Discount factor DF(t) = e^{-y(t) t}; annual-compounded
  zero z_a = e^{y} - 1; par yield for frequency m with coupon dates t_1..t_n: c_par = m (1 - DF(t_n))/sum DF(t_i).
- Fitting: the model is LINEAR in (b0, b1, b2) given tau -> for each tau on a grid (0.2..10 years, ~300 points) solve
  least squares for the betas, keep the smallest SSE, optionally refine tau with minimize_scalar (bounded). Fit to
  yields (plain) or to prices with weights 1/duration (Treasury-style) when the task says "minimise price errors".
  Constraints usually imposed: b0 > 0, b0 + b1 > 0, tau > 0. curve_fit/least_squares with a good start (b0 = long
  yield, b1 = short - long, b2 = 0, tau = 2) also works; report RMSE in bp or price units as the task defines.
- Bond pricing from the curve: P_dirty = sum_i CF_i DF(t_i) with t_i from the settlement date in the stated day count;
  P_clean = P_dirty - AI; accrued interest AI = (c/m) * face * (days since last coupon)/(days in the coupon period)
  (ACT/ACT for Treasuries; 30/360 for many corporates: use the 30/360 day difference in both numerator and denominator).
- Yield to maturity: solve P_dirty = sum CF_i (1 + y/m)^{-m t_i} (brentq on [-0.05, 1]); for a bond on a coupon date
  t_i = k/m. Macaulay duration = sum t_i PV_i / P; modified = Macaulay/(1 + y/m); DV01 = modified x P x 1e-4 (per 100
  face; x notional/100 for a position) or the central finite difference (P(y - 1bp) - P(y + 1bp))/2 - say which.
- Zero from par (bootstrap) and par from zero are different curves: never feed par yields into e^{-y t} as zeros unless
  the task calls them zero/spot rates.

## Conventions and pitfalls (what graders check)
- Yields in input files are typically percent: /100 before fitting; write outputs in the input's unit (state it) and
  parameters as decimals (b0 = 0.045, not 4.5) unless the task shows percent in a template.
- Maturity grid: compute from the exact dates ((maturity - settlement).days/365 or /365.25 as stated), not from the
  label "10Y", when dates are given; keep the task's grid order for the output table.
- Invariants asserted: fitted curve finite and smooth, y(0+) = b0 + b1, y(inf) = b0, DF(0) = 1 and DF decreasing for
  positive yields, repricing errors within tolerance (e.g. max |error| < 10 bp), RMSE recomputed from the written
  table equals the reported RMSE, par bond (coupon = yield) prices at 100, price decreases when yield rises, DV01 > 0,
  modified duration x P x 1e-4 ~ finite-difference DV01 (0.1%).
- tau in years vs lambda per year: a wrong inversion gives a plausible-looking but failing fit; check the formula
  the task writes (the (1 - e^{-lambda t})/(lambda t) form means lambda is a rate).
- Day counts: ACT/ACT accrual uses actual days in the current coupon period (181-184); 30/360 clips day 31 to 30.
  Semiannual US Treasuries: m = 2, coupon c/2 per period, face 100.
- Clean vs dirty: quoted prices are clean; discounting gives dirty; AI is added/subtracted exactly once.
- Extrapolation beyond the last bond: NS is well-behaved (tends to b0); do not clamp unless told.
- Output hygiene: parameter names exactly as the template (beta0/beta1/beta2/tau or lambda), floats not numpy types,
  one row per maturity in ascending order, errors in the unit the task names (bp vs percent vs price).

## Tested snippet
```python
import numpy as np
from scipy.optimize import brentq

def ns_yield(t, b0, b1, b2, tau):                     # Nelson-Siegel zero yield, tau = decay TIME SCALE (years); lambda = 1/tau
    x = np.asarray(t, float) / tau; f1 = (1 - np.exp(-x)) / x
    return b0 + b1 * f1 + b2 * (f1 - np.exp(-x))

def fit_ns(t, y, taus=np.linspace(0.3, 10, 300)):       # grid over tau, linear least squares for the betas
    best = None
    for tau in taus:
        x = t / tau; f1 = (1 - np.exp(-x)) / x
        A = np.column_stack([np.ones_like(t), f1, f1 - np.exp(-x)])
        b, *_ = np.linalg.lstsq(A, y, rcond=None); sse = float(np.sum((A @ b - y) ** 2))
        if best is None or sse < best[0]: best = (sse, tau, b)
    sse, tau, b = best
    return dict(beta0=b[0], beta1=b[1], beta2=b[2], tau=tau, rmse=np.sqrt(sse / len(t)))

t = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]); y = np.array([0.052, 0.051, 0.048, 0.044, 0.042, 0.041, 0.0415, 0.043, 0.046, 0.047])
p = fit_ns(t, y); fitted = ns_yield(t, p["beta0"], p["beta1"], p["beta2"], p["tau"])
print({k: round(float(v), 6) for k, v in p.items()}, round(float(np.max(np.abs(fitted - y)) * 1e4), 3), "bp max error")
print(round(p["beta0"] + p["beta1"], 6), "= short-rate limit;", round(p["beta0"], 6), "= long-run level; DF(10) =", round(float(np.exp(-fitted[7] * 10)), 6))

def bond_price(y, c, T, freq=2, face=100.0):            # dirty price from a flat yield, whole coupon periods
    n = int(round(T * freq)); k = np.arange(1, n + 1); cf = np.full(n, c / freq * face); cf[-1] += face
    return float(np.sum(cf / (1 + y / freq) ** k))

def ytm(price, c, T, freq=2): return brentq(lambda y: bond_price(y, c, T, freq) - price, -0.05, 1.0, xtol=1e-14)
def accrued(c, days_since, days_in_period, freq=2, face=100.0): return face * c / freq * days_since / days_in_period
dirty = bond_price(0.045, 0.05, 5.0); ai = accrued(0.05, 61, 182)                 # ACT/ACT: actual days in the period
print(round(dirty, 6), round(ytm(dirty, 0.05, 5.0), 10), round(ai, 6), round(dirty - ai, 6), "= clean price")
p_up, p_dn = bond_price(0.045 + 1e-4, 0.05, 5.0), bond_price(0.045 - 1e-4, 0.05, 5.0)
print(round((p_dn - p_up) / 2, 6), "= DV01 per 100 face;", round(bond_price(0.05, 0.05, 5.0), 10), "= par bond at par")
```
