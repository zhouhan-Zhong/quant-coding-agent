---
title: Curve bootstrapping and immunization conventions: discount factors, compounding, day counts, DV01/KRD, PV matching, units of notional
category: rates-curves
tags: bootstrap, discount factor, zero rate, continuous compounding, annual compounding, par swap, ois, libor, day count, act/360, act/365, 30/360, year fraction, dv01, pv01, duration, modified duration, macaulay, key rate duration, krd, immunization, liability matching, barbell, bullet, notional, units, dirty price, clean price, accrued interest, floater, frn, callable, residual, repricing check
summary: The conventions that decide whether a bootstrapped curve or an immunized portfolio "matches": which compounding a reported zero rate uses, which day count the year fractions use, DV01 as dirty price x modified duration x 1e-4 per 100 notional, KRD bumps of 1 bp applied one key rate at a time, PV of the hedge equal to the liability PV in the SAME currency units, and the repricing residual that must be ~0 after a correct bootstrap.
triggers: text:bootstrap the, text:immuniz, text:key rate duration, text:dv01
---
## When this applies
"Bootstrap the zero curve from deposits / futures / swaps / OIS", "report zero rates (continuous / annual)", "compute DV01,
duration, key-rate durations", "immunize the liability with bonds A and B", "stress the curve by +5 bp", "the bootstrap
residual must be below 1e-6".

## Conventions that must be read from the task text (write them down before coding)
- Compounding of the REPORTED zero rate: continuous r_c = -ln(DF)/t; annual r_a = DF^(-1/t) - 1; semi-annual
  r_s = 2 (DF^(-1/(2t)) - 1); simple (money market) r_m = (1/DF - 1)/t.  Converting at the end is cheap; mixing them
  mid-way is the classic error (a 4.17 % vs 3.93 % gap at 10 years is a compounding/day-count mismatch, not noise).
- Day count for each instrument class: deposits and OIS legs usually ACT/360, bond coupons 30/360 or ACT/ACT, swap fixed
  legs 30/360, floating legs ACT/360.  Year fraction tau_i = days_i / basis.  Use the task's calendar (business days
  or exact dates) when given.
- Par swap bootstrap: for maturity T_n with fixed coupon c_n, DF_n = (1 - c_n * sum_{i<n} tau_i DF_i) / (1 + c_n tau_n).
  OIS discounting: the same formula with OIS par rates; a LIBOR/IBOR projection curve then solves forward rates so that
  the swap reprices under OIS discount factors.  Interpolate missing pillars as the task says (log-linear DF or linear
  zero); never interpolate par rates unless told.
- Repricing residual: after bootstrapping, every input instrument re-priced with the curve must come back at par;
  report max |PV - par| in bp of notional.  A residual of tens of bp means a wrong formula, not a tolerance issue.

## Price, duration and risk measures (per 100 notional unless the task says otherwise)
- Dirty price = sum_i CF_i DF_i; clean = dirty - accrued; accrued = coupon x (days since last coupon / days in period)
  under the bond's day count.
- Macaulay duration D = sum_i t_i CF_i DF_i / dirty; modified duration D_mod = D / (1 + y/m) for yield y compounded m
  times a year; with continuous compounding D_mod = D.
- DV01 (PV01) = dirty x D_mod x 1e-4 per 100 notional - equivalently the price change for a 1 bp parallel shift,
  computed by finite difference [P(y - 1bp) - P(y + 1bp)] / 2 if the task says "bump".  State the sign convention:
  DV01 is usually reported POSITIVE for a long bond.
- Key-rate durations: bump ONE key rate by +1 bp (or +/-1 bp central), re-interpolate the curve, reprice:
  KRD_k = -(P_up - P_down) / (2 x 1e-4 x P).  The sum of KRDs approximately equals the modified duration - use this as
  an invariant.
- Floating-rate note between resets: value = (coupon fixed at the last reset + 100) discounted from the next reset
  date; it trades near par and ABOVE par when the fixed coupon exceeds the current forward.  A callable bond is capped
  by the call price near call dates: price(callable) <= price(non-callable).

## Immunization
- Match PV: sum_j units_j x dirty_j = PV(liability) in the SAME units (if prices are per 100 face, units = face/100).
  A hedge PV of 29,000 against a liability of 658,000 is a units error (per-100 price treated as per-1), not a model error.
- Match duration (or every key-rate duration): sum_j w_j D_j = D_L with weights w_j = PV_j / PV_total; with two bonds the
  system is 2 x 2 and solves exactly; with more bonds minimize sum of squared KRD gaps subject to PV match.
- Effectiveness after a shock: ratio = (PV_assets_after - PV_liability_after) / PV_liability_before, or the task's
  own definition; a correctly immunized portfolio stays within a few percent for parallel shifts of +/-100 bp.
- Report units as the task asks (face amounts, number of bonds, or weights); round only at the end.

## Invariants worth printing at the end of the script
- Every bootstrap input reprices at par (residual < 1e-8 of notional for exact bootstraps, < 1e-4 with interpolation).
- DF decreasing in maturity, zero rates finite, forward rates positive (or at least above the task's floor).
- sum(KRD) ~ modified duration (within 2-3 %); DV01 = dirty x D_mod x 1e-4 (within 1 %).
- Hedge PV == liability PV; callable <= straight bond; FRN near par.
