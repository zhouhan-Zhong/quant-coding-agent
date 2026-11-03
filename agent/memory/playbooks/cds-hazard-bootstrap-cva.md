---
title: CDS pricing, hazard-rate bootstrap, survival curves and CVA
category: credit
tags: cds, credit default swap, par spread, running spread, upfront, points upfront, hazard rate, default intensity, piecewise constant hazard, survival probability, survival curve, default probability, cumulative pd, marginal pd, conditional pd, bootstrap, premium leg, protection leg, risky annuity, rpv01, recovery rate, lgd, accrual on default, isda standard model, mark to market, cva, dva, bcva, credit valuation adjustment, expected positive exposure, epe, counterparty credit risk, wrong-way risk, quarterly payments, act/360, risky bond, credit spread
summary: Premium/protection leg formulas, the sequential hazard bootstrap from par spreads (brentq per tenor), survival/PD conversions, upfront and MTM conventions, and unilateral CVA = LGD x sum DF x EPE x marginal PD with the invariants graders assert.
---
## When this applies
"Bootstrap a hazard-rate (default intensity) curve from CDS spreads", "survival probabilities at each tenor", "price a
CDS / par spread / upfront / mark-to-market of an existing contract", "credit valuation adjustment (CVA) of a swap or
portfolio given a counterparty hazard curve or CDS quotes", "marginal and cumulative default probabilities", "risky
annuity / RPV01", "price a risky bond from a hazard curve".

## Method and formulas
Symbols: t in years, DF(t) risk-free discount factor, lam_i hazard on (T_{i-1}, T_i], S(t) survival, R recovery,
LGD = 1 - R, N notional, s spread as a DECIMAL (100 bp = 0.01), payment dates t_j with accrual fractions delta_j.
- Survival with piecewise-constant hazard: S(t) = exp(-sum_i lam_i * length((T_{i-1}, T_i] intersect (0, t])), S(0) = 1.
  Flat hazard: S(t) = exp(-lam t). Cumulative PD(t) = 1 - S(t); marginal PD over (a, b] = S(a) - S(b); conditional
  (forward) PD = 1 - S(b)/S(a); hazard from a cumulative PD: lam = -ln(1 - PD)/T; credit triangle check: lam ~ s/LGD.
- Premium leg per unit notional: PV_prem = s * RPV01 with RPV01 = sum_j delta_j DF(t_j) S(t_j)
  + accrual on default sum_j (delta_j/2) DF(t_j) (S(t_{j-1}) - S(t_j)) (ISDA-style; drop the accrual term only when the
  task says "no accrued premium on default"). Standard schedule: quarterly payments, ACT/360 accruals (delta ~ 0.25).
- Protection leg: PV_prot = LGD * sum_k DF(u_k) (S(u_{k-1}) - S(u_k)) on a fine grid (monthly or finer); default paid
  at the grid midpoint u_k = (t_{k-1} + t_k)/2, or at the period end if the task says so. Continuous form:
  LGD * integral_0^T DF(u) lam(u) S(u) du.
- Par spread s_par = PV_prot / RPV01. Protection buyer MTM (long protection) = N (s_par - s_contract) RPV01;
  upfront for a standard coupon c (100 or 500 bp): U = N (s_par - c) RPV01 (the buyer PAYS U when s_par > c);
  points upfront = 100 U / N. Quoted "conventional spread" is the flat hazard that reproduces the upfront.
- Bootstrap: tenors increasing; for tenor i solve lam_i with brentq on [1e-8, 5] such that s_i RPV01_i = PV_prot_i with
  lam_1..lam_{i-1} fixed (both legs use the whole curve up to T_i). Reprice every quote afterwards: residual_bp =
  (s_model - s_quote) * 1e4 must be ~1e-8 or smaller.
- Unilateral CVA with an exposure profile: CVA = LGD * sum_k DF(t_k) EPE(t_k) (S(t_{k-1}) - S(t_k)),
  EPE(t) = E[max(V(t), 0)] from Monte Carlo paths of the underlying (or the task's analytic/tabulated profile); DVA uses
  the own survival curve and ENE = E[max(-V, 0)]; BCVA = CVA - DVA. Use DF at the bucket midpoint if the task says
  "default mid-period". Wrong-way risk: correlate the exposure driver and the default time (Gaussian copula on the
  default threshold) and report CVA(rho)/CVA(0). CVA as a running spread = CVA / RPV01 of the counterparty.
- Risky bond from a hazard curve: P = sum_j c DF(t_j) S(t_j) + DF(T) S(T) + R sum_k DF(u_k) (S(u_{k-1}) - S(u_k));
  credit spread = yield(P) - risk-free yield; bond-CDS basis = CDS spread - bond spread.

## Conventions and pitfalls (what graders check)
- Spreads in input files are usually in bp: divide by 1e4 (percent: /100). Write decimals unless the task says bp.
  Recovery 40% unless given. Risk-free: flat r -> DF = exp(-r t), or the task's discount curve (interpolate in ln DF).
- Invariants asserted: S(0) = 1, S strictly decreasing, 0 < S <= 1, lam > 0 at every tenor, legs balance at the par
  spread (|PV_prem - PV_prot| < 1e-8), input quotes reprice, CVA >= 0, CVA <= LGD * max EPE, CVA increases with the
  spread level, cumulative PD increasing, marginal PDs >= 0 and summing to 1 - S(T).
- Hazard is per year: survival to 5Y uses 5 years of hazard, never 20 quarters of a quarterly rate.
- Accrual-on-default and midpoint conventions move the par spread by ~1 bp: take the variant the task text names.
- Premium leg sums over payment dates only; the protection leg integrates over the fine grid, unless the task fixes
  "default observed at payment dates" (then both legs share the payment grid).
- MTM sign: the protection BUYER gains when spreads widen. For upfront quotes the contract coupon is the standard
  coupon, not the par spread; a legacy contract's running spread is its contractual spread.
- Hazards are continuous rates; an annual PD p converts with -ln(1 - p), not p itself.
- Off-grid maturities (4Y with 3Y/5Y quotes): hazard of the interval containing t (piecewise constant), not an
  interpolation of spreads; survival is still continuous in t.
- Output hygiene: tenors exactly as the task writes them ("5Y" vs 5 vs 5.0), survival as floats in [0, 1], hazards >= 0,
  no NaN when a brentq bracket fails (widen the bracket, then report the failure explicitly).

## Tested snippet
```python
import numpy as np
from scipy.optimize import brentq

def survival(t, knots, lam):            # piecewise-constant hazard lam[i] on (knots[i], knots[i+1]], knots[0] = 0
    t = np.atleast_1d(np.asarray(t, float)); H = np.zeros_like(t)
    for i in range(1, len(knots)):
        H += lam[i - 1] * np.clip(np.minimum(t, knots[i]) - knots[i - 1], 0, None)
    return np.exp(-H)

def cds_legs(T, knots, lam, r=0.03, R=0.4, freq=4, steps=12):
    """RPV01 (premium leg per unit spread, with accrual on default) and protection leg per unit notional."""
    pay = np.arange(1, int(round(T * freq)) + 1) / freq              # quarterly payment dates
    df = np.exp(-r * pay); S = survival(pay, knots, lam); Sprev = np.r_[1.0, S[:-1]]
    rpv01 = np.sum(df * S / freq) + np.sum(0.5 * df * (Sprev - S) / freq)   # accrued premium paid at default (mid-period)
    grid = np.linspace(0, T, int(round(T * steps)) + 1)              # monthly grid for the protection leg
    Sg = survival(grid, knots, lam); mid = 0.5 * (grid[1:] + grid[:-1])
    prot = (1 - R) * np.sum(np.exp(-r * mid) * (Sg[:-1] - Sg[1:]))
    return rpv01, prot

def bootstrap_hazard(tenors, spreads, **kw):                    # spreads as decimals, tenors increasing
    knots, lam = [0.0], []
    for T, s in zip(tenors, spreads):
        f = lambda h: (lambda a, p: s * a - p)(*cds_legs(T, knots + [T], lam + [h], **kw))
        lam.append(brentq(f, 1e-8, 5.0, xtol=1e-14)); knots.append(T)
    return np.array(knots), np.array(lam)

tenors, spreads = [1, 3, 5, 7, 10], [0.0060, 0.0085, 0.0110, 0.0125, 0.0140]
knots, lam = bootstrap_hazard(tenors, spreads)
S = survival(tenors, knots, lam)
par5 = (lambda a, p: p / a)(*cds_legs(5, knots, lam))            # reprices the 5Y quote
print(np.round(lam, 6), np.round(S, 6), round(par5 * 1e4, 8), "bp")
# CVA of a swap from an EPE profile: CVA = LGD * sum DF(t_k) EPE(t_k) [S(t_{k-1}) - S(t_k)]
t = np.arange(0.25, 5.01, 0.25); epe = 1e6 * np.sqrt(t) * (5 - t) / 5     # toy exposure profile, notional 1e6
St = survival(t, knots, lam); dPD = np.r_[1.0, St[:-1]] - St
cva = (1 - 0.4) * np.sum(np.exp(-0.03 * t) * epe * dPD)
print(round(cva, 2), bool(np.all(np.diff(St) < 0)), round(spreads[0] / (1 - 0.4), 5), "~", round(lam[0], 5), "(credit triangle)")
```
