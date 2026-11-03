---
title: Mortgage pass-through, inflation-linked and convertible bond conventions: CPR/SMM/PSA prepayment, level-pay pool cash flows, WAL, index ratios with lagged CPI, deflation floor, breakeven inflation, conversion value and premium, binomial convertible valuation, callable bonds and yield to worst
category: rates-curves
tags: mbs, mortgage backed securities, pass-through, prepayment, cpr, smm, psa, wac, wam, servicing fee, scheduled principal, weighted average life, wal, oas, option adjusted spread, inflation linked, tips, linkers, cpi, index ratio, reference cpi, deflation floor, breakeven inflation, real yield, fisher, convertible bond, conversion ratio, conversion price, parity, conversion premium, bond floor, callable bond, yield to worst, yield to call
summary: SMM = 1 - (1 - CPR)^(1/12); PSA 100 = CPR rising 0.2 % per month to 6 % at month 30 then flat, scaled by PSA/100; level-pay pool: payment = B r / (1 - (1 + r)^-(n - t + 1)) with r = WAC/12, investor interest at the pass-through coupon/12, prepayment = SMM x (B - scheduled principal); WAL = sum t x principal_t / sum principal_t in years; TIPS index ratio = reference CPI(settle) / reference CPI(base) with the 3-month lag interpolated daily, principal floored at par at maturity; breakeven from (1 + n) = (1 + r)(1 + pi); convertible: conversion value = ratio x S, premium = price / conversion value - 1, tree node value = max(hold, convert) after the issuer call cap.
triggers: text:prepayment, text:psa speed, text:mortgage, text:pass-through, text:inflation-linked, text:inflation linked, text:treasury inflation, text:index ratio, text:convertible bond
---
## When this applies
"Project the pass-through cash flows at 150 PSA", "WAL and price at a yield", "TIPS coupon with the index ratio",
"breakeven inflation", "price the convertible on a binomial tree", "yield to worst of the callable bond".

## Mortgage pass-throughs
- Prepayment speeds: SMM_t = 1 - (1 - CPR_t)^(1/12).  PSA: CPR_t = (PSA/100) x 6 % x min(t, 30) / 30 for pool age t in
  months (t counts from 1; seasoned pools start at their age).
- Level-pay schedule per month t with balance B_{t-1}, mortgage rate r = WAC / 12 and remaining term n - t + 1:
  payment = B_{t-1} r / (1 - (1 + r)^-(n - t + 1)); scheduled principal = payment - B_{t-1} r;
  prepayment = SMM_t (B_{t-1} - scheduled principal); B_t = B_{t-1} - scheduled - prepayment.
- Investor cash flow = B_{t-1} x (pass-through coupon / 12) + scheduled principal + prepayment (servicing = WAC - coupon).
- WAL (years) = sum_t (t / 12) x total principal_t / sum_t total principal_t.  Price = sum CF_t / (1 + y/12)^t at the
  monthly-compounded yield (or the task's discounting); OAS = constant spread over simulated short-rate paths that matches
  the price (paths and prepayment model as given).
- Faster PSA -> shorter WAL, earlier principal; the balance reaches 0 at month n (or earlier).

## Inflation-linked bonds
- Reference CPI for day d of month M (US TIPS / most linkers): CPI(M-3) + (day-1)/(days in M) x (CPI(M-2) - CPI(M-3)).
- Index ratio = RefCPI(settlement or payment date) / RefCPI(base / dated date).  Coupon paid = real coupon / frequency x
  face x index ratio; principal paid at maturity = face x max(index ratio, 1) for TIPS (deflation floor; check the task).
- Real vs nominal: (1 + nominal) = (1 + real)(1 + inflation); breakeven ~ nominal yield - real yield (state which).
- Real yield pricing discounts real cash flows (no index ratio) at the real yield; the invoice price = real price x index
  ratio (+ accrued).

## Convertible bonds
- Conversion ratio CR shares per bond (conversion price = face / CR); conversion value (parity) = CR x S;
  conversion premium = bond price / conversion value - 1; bond floor = straight bond value at the issuer's credit spread.
- Binomial (CRR) valuation backward from maturity: at T value = max(face + final coupon, CR S_T); at each node
  continuation = discounted expected value (+ coupon on coupon dates); issuer call: value = min(continuation, call price)
  when callable; holder: value = max(value, CR S).  Credit risk: Tsiveriotis-Fernandes splits the value into an equity
  part (discounted at r) and a cash part (discounted at r + spread).
- Delta of the convertible = dV/dS (finite difference on the tree with the same nodes).

## Callable bonds
- Yield to call for each call date (price to call price at that date); yield to worst = min(YTM, all YTCs).
- Callable price = straight bond - call option (price both on the same short-rate tree when asked for OAS / effective
  duration).

## Invariants worth printing
- Sum of principal = original balance; SMM in [0, 1]; WAL falls as PSA rises; index ratio ~ 1 at the dated date;
  convertible value >= max(bond floor, conversion value) (minus call effects); YTW <= YTM.
