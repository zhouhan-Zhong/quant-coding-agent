---
title: Curve bootstrapping, discount factors and rate conventions
category: rates-curves
tags: bootstrap, bootstrapping, discount factor, discount curve, zero curve, zero rate, spot rate, swap curve, ois, deposit, fra, futures, irs, par swap rate, annuity, day count, act/360, act/365, 30/360, compounding, continuous, simple, log-linear interpolation, forward rate, pv01, dv01, repricing, residual, dual curve, libor, sofr, par curve, par bonds, coupon gaps, trial pillar, quote bump, projection curve, fra chain, spot date
summary: Exact formulas and conventions for turning deposit/FRA/futures/OIS/IRS/par-bond quotes into discount factors, zero and forward rates (gap coupons, two-stage interpolation, spot-date origin), and for repricing and quote-bump PV01.
---
## When this applies
Tasks that say "bootstrap a curve", "build the discount/zero/forward curve from quotes", "par swap rate",
"reprice the instruments", "PV01 / DV01 of a swap", "convert zero to par / continuous to annual", or that hand
you CSVs of deposits, FRAs, futures, OIS and swap quotes plus a conventions file (day counts, spot lag).

## Method and formulas
Symbols: t = time in years from the valuation date (for interpolation and zero rates), tau = accrual fraction
in the instrument's own day count, DF(t) = discount factor, r = quoted simple rate (decimal), z = zero rate.
- Day counts: ACT/360 tau = days/360; ACT/365 tau = days/365; ACT/365.25 for "years" if the task says so;
  30/360 US: D1 = min(D1,30), D2 = 30 if (D2 == 31 and D1 == 30); tau = (360 dY + 30 dM + (D2 - D1))/360.
- Compounding: DF = exp(-z_c t) = (1 + z_a)^(-t) = (1 + z_m/m)^(-m t) = 1/(1 + r tau) (simple).
  z_c = -ln DF / t; z_a = DF^(-1/t) - 1; z_c = ln(1 + z_a); z_m = m((1 + z_a)^(1/m) - 1).
- Deposit (single payment): DF(T) = 1/(1 + r tau(start, T)).  OIS <= 1Y: same formula with the OIS rate.
- FRA t1 x t2 at rate f: DF(t2) = DF(t1)/(1 + f tau(t1, t2)).  Futures: f = (100 - price)/100 minus a convexity
  adjustment (Ho-Lee: 0.5 sigma^2 t1 t2) if the task gives sigma, then the same recursion.
- Par swap / OIS > 1Y (fixed payments t_1..t_n, accruals tau_i, start t_0): float leg PV = DF(t_0) - DF(t_n)
  (single curve); annuity A = sum tau_i DF(t_i); par S = (DF(t_0) - DF(t_n))/A.
  Closed-form bootstrap when fixed and float dates coincide: DF(t_n) = (DF(t_0) - S sum_{i<n} tau_i DF(t_i))/(1 + S tau_n).
  Otherwise solve the par equation for the new pillar DF with brentq on [1e-6, 1] (xtol 1e-15), inserting the trial
  pillar into the sorted curve and interpolating the intermediate payment dates.
- Dual curve (OIS discounting, projection curve L): F_j = (DF_L(t_{j-1})/DF_L(t_j) - 1)/tau_j (float day count),
  par = sum tau_j F_j DF_OIS(t_j) / sum tau_i^fixed DF_OIS(t_i). Bootstrap OIS first, then the projection curve.
- Interpolation: log-linear in DF, ln DF(t) = (1-w) ln DF(t0) + w ln DF(t1), w = (t-t0)/(t1-t0) (= piecewise-constant
  forward). Flat continuous-zero extrapolation outside the pillars: DF(t) = exp(ln DF(t_edge)/t_edge * t).
- Forwards from DFs: simple F(t1,t2) = (DF(t1)/DF(t2) - 1)/tau(t1,t2); continuous f = -(ln DF(t2) - ln DF(t1))/(t2-t1).
- PV01 of a swap: bump every zero rate of the curve by +1bp (DF *= exp(-1e-4 t)) - or the input quotes, see below - reprice, PV01 = |NPV_bumped - NPV|
  (dual curve: bump only the curve the task names, usually the projection curve); analytic approx for a par swap =
  A * 1e-4 * notional. Pay-fixed NPV = float PV - fixed PV; receive-fixed is the negative.
- Repricing residual: for every input quote recompute its model rate from the finished curve;
  residual_bps = (model - quote) * 1e4; max |residual| must be ~1e-8 bp or smaller.
- Coupon dates falling in a gap between pillars (annual coupons, pillars 3 -> 5 -> 7 -> 10): inside the root-finder,
  interpolate them between the previous pillar and the TRIAL value of the new pillar (z linear in t, or -ln DF linear) -
  reprices to 1e-10. Extrapolating the last zero flat across the gap is a cruder variant (~1e-4 repricing errors).
- Two-stage curves: some references bootstrap with g(t) = -ln DF = z t LINEAR between knots (g(0) = 0) and then price
  with a not-a-knot CubicSpline through the same knots; pinned zeros match only if both stages are replicated, and
  "reprices inputs" flags must use the bootstrap's own (linear) interpolant. Bill-like knots (< 6M): DF = 1/(1 + y T);
  semi-annual par bonds (T >= 0.5): coupons 100 y/2 at T, T - 0.5, ... > 0.
- Annual-grid forwards from zeros: fwd(T) = z(T) T - z(T - 1)(T - 1) with z(T - 1) read off the interpolated curve; fwd(first) = z(first).
- Projection (IBOR) curve from a deposit + FRA chain: pseudo-DF Q(e) = Q(s)/(1 + f tau(s, e)), ln Q linear; forward
  L(a, b) = (Q(a)/Q(b) - 1)/tau(a, b) on accrual dates; quote adjustments (convexity/basis bps) only inside the build.
- Quote-level risk: PV01 per curve = PV(that curve's input QUOTES + 1bp, re-bootstrapped) - PV (bucketed: one quote at a
  time); a parallel zero-rate bump differs by ~1-2% and misses quote-bump pins.

## Conventions and pitfalls (what graders check)
- Time grid: t = days/365 (or 365.25 if stated) from the VALUATION date, not from spot, for interpolation and zero
  rates; accrual tau uses the instrument's day count from its own start (spot) date. Spot = valuation + spot lag
  business days (Mon-Fri unless a holiday list is given). Exception: desks/tasks that anchor curves at the SPOT date
  (DF(spot) = 1, t = days(spot, d)/basis of the currency, no val -> spot discount) - follow the curve file's year_frac.
- Zero-coupon-style OIS quotes (<= 1-2Y, or all quotes when the task treats them as single-period): DF = 1/(1 + r t).
- Quote files: keep the latest quote per key (sort by quote_time, groupby(key).tail(1)); fixings files may contain FUTURE
  dates - use only fixing_date <= valuation date. Integer maturities as JSON keys: str(int(T)) -> "5", not "5.0".
- Schedules: generate backward from maturity in calendar-month steps (6M fixed / 3M float typical USD), unadjusted,
  keep only dates > effective date (a short first stub). Clip month-ends when adding months (31 Jan + 1M = 29 Feb).
- Fixed leg accrual 30/360, float leg ACT/360 for USD; read the conventions file rather than assuming.
- Rates in percent in input files -> divide by 100; write outputs as decimals unless told "in percent"/"bps".
- Bootstrap strictly in maturity order; a later instrument must not change earlier pillars; reuse the same
  interpolation inside the bootstrap and in the final outputs or the residuals will not be zero.
- Invariants checkers assert: DF(0) = 1, 0 < DF <= 1, DF non-increasing for positive rates, zero rates finite,
  forward rates consistent with DFs (rtol 1e-6), residual_bps ~ 0 (< 1e-6 bp), par rate of the last pillar equals
  the input quote, PV01 > 0, NPV sign matches pay/receive.
- Output hygiene: exact column order, rows sorted ascending by pillar date (ISO strings sort correctly),
  full-precision floats (no rounding: cross-file equality is tested at 1e-12), cast with float().
- brentq default xtol (2e-12) can leave 1e-9 bp residuals on long swaps; use xtol=rtol=1e-15.
- Do not discount projection-curve forwards with the projection curve when the task is dual-curve.
- A pillar used both as deposit and swap maturity: keep separate curves (OIS vs projection) and separate pillars.

## Tested snippet
```python
import math, numpy as np
from datetime import date
from scipy.optimize import brentq

def yf(d1, d2, basis="ACT/360"):                       # year fraction between two dates
    if basis == "30/360":                                # US 30/360: D1 31->30; D2 31->30 only if D1 is 30
        D1, D2 = min(d1.day, 30), d2.day
        D2 = 30 if (D2 == 31 and D1 == 30) else D2
        return (360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (D2 - D1)) / 360.0
    return (d2 - d1).days / {"ACT/360": 360.0, "ACT/365": 365.0, "ACT/365.25": 365.25}[basis]

def df_at(curve, t):                                     # curve = sorted [(t, DF)]; log-linear DF, flat cc-zero extrapolation
    ts, dfs = zip(*curve)
    if t <= 0: return 1.0
    if t <= ts[0]: return math.exp(math.log(dfs[0]) / ts[0] * t)
    if t >= ts[-1]: return math.exp(math.log(dfs[-1]) / ts[-1] * t)
    return float(np.exp(np.interp(t, ts, np.log(dfs))))

zero_cc = lambda df, t: -math.log(df) / t                                 # continuous zero
fwd_simple = lambda c, t1, t2, tau: (df_at(c, t1) / df_at(c, t2) - 1.0) / tau   # simple forward over [t1, t2]

def swap_legs(curve, pay_t, taus, t0=0.0):               # single-curve: float PV = DF(t0) - DF(tn); annuity = sum tau DF
    ann = sum(tau * df_at(curve, t) for t, tau in zip(pay_t, taus))
    return df_at(curve, t0) - df_at(curve, pay_t[-1]), ann
par_rate = lambda curve, pay_t, taus: (lambda f, a: f / a)(*swap_legs(curve, pay_t, taus))

def bootstrap(dep, quotes):                              # 6M deposit then annual par swaps [(T, S)], sequential brentq;
    curve = [(0.5, 1.0 / (1.0 + dep * 0.5))]              # the TRIAL pillar is inserted, so gap dates (T = 4) interpolate to it
    for T, S in quotes:
        pay_t, taus = list(range(1, T + 1)), [1.0] * T
        f = lambda x: par_rate(sorted(curve + [(T, x)]), pay_t, taus) - S
        curve = sorted(curve + [(T, brentq(f, 1e-6, 1.0, xtol=1e-15, rtol=1e-15, maxiter=200))])
    return curve
quotes = [(1, 0.032), (2, 0.034), (3, 0.035), (5, 0.037)]                 # (maturity years, par rate)
curve = bootstrap(0.030, quotes)
resid_bp = max(abs(par_rate(curve, list(range(1, T + 1)), [1.0] * T) - S) * 1e4 for T, S in quotes)
npv = lambda c, K: (lambda f, a: f - K * a)(*swap_legs(c, [1, 2, 3, 4, 5], [1.0] * 5))   # pay-fixed K, receive float
pv01_zero = npv([(t, d * math.exp(-1e-4 * t)) for t, d in curve], 0.037) - npv(curve, 0.037)   # +1bp cc zero shift
pv01_quote = npv(bootstrap(0.030 + 1e-4, [(T, S + 1e-4) for T, S in quotes]), 0.037) - npv(curve, 0.037)  # quotes +1bp
print([round(d, 8) for _, d in curve], round(zero_cc(df_at(curve, 4.0), 4.0), 8), round(fwd_simple(curve, 1.0, 2.0, 1.0), 8))
print(round(resid_bp, 12), round(pv01_zero * 1e6, 2), round(pv01_quote * 1e6, 2), yf(date(2024, 1, 31), date(2024, 7, 31), "30/360"))
```
