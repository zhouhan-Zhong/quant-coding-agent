---
title: Cross-currency basis swaps: CIP forwards, basis curve, MtM-resetting legs, quote-bump risk
category: rates-curves
tags: cross-currency swap, xccy, cross currency basis, basis swap, mtm reset, mark-to-market resetting, notional reset, covered interest parity, cip, fx forward, forward points, pips, broken date, fx interpolation, ois discounting, collateral currency, libor projection, ibor, fra chain, dual curve, pv01, fx delta, par basis, par spread, stress scenarios, roll-down, accrued interest
summary: Spot-anchored OIS/IBOR curves from simple quotes, FX forward nodes from points with log-linear broken-date interpolation, implied CIP basis, MtM-resetting xccy leg valuation, par basis by linearity, PV01 by quote bumps (FX nodes rebuilt with basis fixed) and FX delta/stress conventions.
---
## When this applies
"Value a (mark-to-market resetting) cross-currency basis swap", "imply the cross-currency basis from FX forwards and OIS
curves", "build FX forward / basis curves with broken-date interpolation", "PV01 per curve, FX delta, par basis, stress
and roll-down scenarios", desks quoting deposits/OIS/FRAs per currency plus spot and forward points.

## Method and formulas
Notation: spot S = units of the quote (domestic, e.g. USD) currency per 1 base (e.g. GBP); DF_d, DF_b = OIS discount
factors of the two currencies; T_d, T_b = year fractions in each currency's day count (USD/EUR ACT/360, GBP ACT/365F).
- Quote hygiene: keep the latest quote per instrument key (sort by quote_time, groupby(key).tail(1); stale earlier rows
  differ by bp); rates pct/100; forward points pips/1e4 so F_node = S + pips/1e4; quote adjustments (e.g. FRA convexity
  bps) only where the task applies them (inside the projection build). Past fixings only for fixing_date <= valuation date.
- Time origin: often the SPOT date (DF(spot) = 1, T = days(spot, d)/basis per currency, no valuation->spot discount);
  curve files then report year_frac per currency from spot. Follow the task if it says valuation date.
- OIS: short zero-coupon-style quotes DF = 1/(1 + r T) (all quotes up to ~2Y when the task treats them as single-period);
  interpolate g = -ln DF linearly in T over (0, 0) + pillars. Projection (IBOR) curve: Q from the deposit, then chained
  FRAs Q(e) = Q(s)/(1 + f tau(s, e)), ln Q linear; coupon L(a, b) = (Q(a)/Q(b) - 1)/tau(a, b) on the accrual dates.
- CIP forward F_cip(T) = S DF_b(T)/DF_d(T). Implied basis b(T) = -ln(F_mkt(T)/F_cip(T))/T_d (bp = 1e4 b), so F_mkt =
  F_cip e^{-b T_d}; the reference divided by the DOMESTIC (USD ACT/360) year fraction - dividing by the ACT/365 fraction
  scales the basis by 365/360 and misses tight pins. Broken-date forwards: ln F linear in calendar days between
  (spot date, ln S) and the tenor nodes.
- MtM-resetting base leg (its notional resets so the domestic value stays dom_N): N_0 = dom_N/inception spot (or given);
  N_k = dom_N/F(reset_k) for resets after valuation; exchange at each reset of N_k - N_{k-1} on the reset date (the
  reference booked +(N_k - N_{k-1}) on the leg that receives the final notional; the 'economic' opposite sign changes PV by
  a lot - follow the task's cash-flow table); base coupons N_k (L_b + spread) tau_b; final principal N_last. Domestic leg:
  -dom_N L_d tau_d coupons, -dom_N principal (pay side). Base flows -> domestic at F(payment date) and discounted on the
  DOMESTIC (collateral) OIS curve; domestic flows discounted directly. Trade PV = PV_base + PV_dom.
- Par basis / par spread: PV is linear in the spread -> s* = s0 - PV(s0)/((PV(s0 + 1bp) - PV(s0))/1bp) (the all-in spread,
  applied to the already-fixed current coupon too if the task says "all-in").
- Risk: PV01_curve = PV(rebuilt with that curve's input QUOTES + 1bp) - PV (zero-rate bumps are ~1.5% off). OIS bumps also
  move FX forwards: rebuild the nodes as S DF_b/DF_d e^{-b T_d} with the basis b held fixed; projection bumps leave FX
  untouched. FX delta = PV(spot AND all forward nodes x 1.01) - PV; a "realized spot move" (spot x 1.01, points unchanged)
  is a different scenario. Basis widen: rebuild nodes with b + db ("widen" of a negative basis = more negative, db < 0).
- Accrued: domestic -dom_N L_d tau_d(start, val); base N_0 (L_b + s) tau_b(start, val) converted at F(current payment date).
- Roll-down: move valuation to R: spot rolls to F_base(R), DFs become DF(d)/DF(R) on frozen curves, nodes rebuilt from R
  with the frozen basis; fixings known by R are used.

## Conventions and pitfalls (what graders check)
- Solution checkpoints can be far tighter than results (trade PV rtol 1e-6 + ~$1, basis 1e-4 bp, broken-date FX rtol 1e-8,
  DFs rtol 1e-6): replicate day counts, time origin, interpolation (ln F in days; -ln DF in T) exactly.
- Identities asserted: scenario pnl = PV_stressed - PV_base; OIS +5bp pnl ~ 5 x PV01 (atol); spot_up pnl == fx_delta;
  "base-leg-only" basis widen == full basis widen to 1e-12 -> compute PV as PV_dom_base + PV_base_stressed from the SAME
  pieces so floats are identical; roll-down PV differs from base.
- Curves reprice their own quotes to 1e-12; FX nodes reprice (F_node from rebuilt CIP + basis equals the quote); DFs
  strictly decreasing; basis curve consistent with the forward curve; reset notionals x F(reset) == dom_N.
- Sorting: cash-flow schedule by payment date then a fixed leg/type order; quotes by curve, instrument, end date, tenor;
  forward points reported in pips; first row of a schedule is often the already-fixed current coupon.
- Flags/summary: real bools, exact key order when templated; past cash flows (payment < valuation) excluded.

## Tested snippet
```python
import math
import numpy as np, pandas as pd
D = pd.Timestamp

def yf(a, b, basis): return (D(b) - D(a)).days / basis                     # ACT/basis: USD/EUR 360, GBP 365

def simple_curve(o, quotes, basis, bump=0.0):        # zero-coupon simple quotes [(end, rate)] -> (T, g = -ln DF), g(0) = 0
    T = np.array([0.0] + [yf(o, d, basis) for d, _ in quotes])
    return T, np.concatenate([[0.0], np.log1p((np.array([r for _, r in quotes]) + bump) * T[1:])])
def projection_curve(o, dep, fras, basis):           # dep (end, r); FRAs [(start, end, f)] chained: Q(e) = Q(s)/(1 + f tau)
    T, G = [0.0, yf(o, dep[0], basis)], [0.0, math.log1p(dep[1] * yf(o, dep[0], basis))]
    for s, e, f in fras:
        g0 = np.interp(yf(o, s, basis), T, G); T.append(yf(o, e, basis)); G.append(g0 + math.log1p(f * yf(s, e, basis)))
    return np.array(T), np.array(G)
def DF(curve, o, d, basis): return math.exp(-np.interp(yf(o, d, basis), *curve))         # -ln DF (or -ln Q) linear in T
def fwd(curve, o, a, b, basis): return (DF(curve, o, a, basis) / DF(curve, o, b, basis) - 1) / yf(a, b, basis)

def fx_interp(o, spot, nodes):                       # broken dates: ln F linear in calendar days, spot node at origin
    x = [0] + [(D(d) - D(o)).days for d, _ in nodes]; y = [math.log(spot)] + [math.log(F) for _, F in nodes]
    return lambda d: math.exp(np.interp((D(d) - D(o)).days, x, y))
def cip_basis(o, spot, nodes, dom, base, bd, bb):    # spot = dom ccy per 1 base ccy; b = -ln(F_mkt/F_cip)/T_dom
    return [(d, -math.log(F / (spot * DF(base, o, d, bb) / DF(dom, o, d, bd))) / yf(o, d, bd)) for d, F in nodes]
def cip_nodes(o, spot, basis, dom, base, bd, bb):    # rebuild node forwards from (bumped) OIS curves, basis held fixed
    return [(d, spot * DF(base, o, d, bb) / DF(dom, o, d, bd) * math.exp(-b * yf(o, d, bd))) for d, b in basis]

def xccy_mtm_pv(o, dom_N, n0, periods, L_base, L_dom, spread, fx, dom, bd, bb):
    """Receive the resetting base leg, pay the domestic leg; periods = [(reset, start, pay)], period-0 notional n0 fixed,
    later N_k = dom_N / F(reset_k), reset exchange +(N_k - N_{k-1}) (reference sign), final +N_last; base flows converted
    at F(date); all discounted on the domestic OIS curve. Returns (PV_base_leg, PV_dom_leg) in domestic currency."""
    pv_b, pv_d, n_prev = 0.0, 0.0, n0
    for k, ((rs, a, p), Lb, Ld) in enumerate(zip(periods, L_base, L_dom)):
        n = n0 if k == 0 else dom_N / fx(rs)
        if k > 0: pv_b += (n - n_prev) * fx(rs) * DF(dom, o, rs, bd)
        pv_b += n * (Lb + spread) * yf(a, p, bb) * fx(p) * DF(dom, o, p, bd); n_prev = n
        pv_d -= dom_N * Ld * yf(a, p, bd) * DF(dom, o, p, bd)
    p = periods[-1][2]
    return pv_b + n_prev * fx(p) * DF(dom, o, p, bd), pv_d - dom_N * DF(dom, o, p, bd)
```
