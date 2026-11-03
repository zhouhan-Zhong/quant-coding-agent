---
title: FX, commodities and crypto conventions: quote direction, forward points and pips, covered interest parity, carry and roll, basis and funding rates, perpetual-futures funding, settlement timestamps, contract roll schedules
category: fx-commodities-crypto
tags: fx, exchange rate, quote convention, base currency, quote currency, pips, forward points, outright forward, covered interest parity, cip, implied yield, carry trade, carry return, roll yield, roll cost, cross rate, triangular, ndf, deliverable, settlement, t+2, commodity futures, contango, backwardation, roll schedule, front month, continuous contract, basis, cash and carry, funding rate, perpetual, perp, 8-hour funding, mark price, index price, utc, timestamp, crypto, bitcoin, basis trade
summary: The conventions that decide FX / commodity / crypto numbers: a quote XXXYYY is units of YYY per 1 XXX (so a "USD return" of a long XXX position is dS/S of that quote), forward points are quoted in pips (1e-4 of the quote, 1e-2 for JPY pairs) and the outright is spot + points/1e4; CIP F = S (1 + r_quote tau)/(1 + r_base tau) with each currency's own day count; carry return per period = interest differential +/- spot move in the quote direction; futures roll yield = ln(F_front/F_next) scaled by the time between expiries; perpetual funding accrues every 8 hours at the funding rate (annualised x 3 x 365) and the basis trade return = basis - funding; timestamps in UTC with the funding/settlement times the task states; a sample end date "2023-12-31" on hourly data includes the whole day unless the task says end of day.
---
## When this applies
"Compute the implied yield / forward premium from spot and forward points", "carry strategy on G10 pairs", "roll cost
of the futures curve", "cash-and-carry basis between spot and futures / perpetuals", "funding-rate strategy",
"triangular arbitrage", NDF fixings, settlement conventions.

## FX
- Quote: EURUSD = 1.08 means 1.08 USD per EUR; base = first currency.  A long-EUR position earns (S_t/S_{t-1} - 1) in USD
  terms; for USDJPY the USD investor long JPY earns (S_{t-1}/S_t - 1).  State the direction for every pair before coding;
  the sign of carry and the ranking of pairs flip when it is wrong.
- Forward points: outright F = S + points / 10,000 (JPY, and other 2-decimal pairs: / 100).  Points can be negative.
  "1-month" = the task's tenor in days / 360 or / 365 per currency (USD, EUR, JPY, CHF: ACT/360; GBP, AUD, NZD, CAD:
  ACT/365).
- Covered interest parity: F = S (1 + r_q tau_q) / (1 + r_b tau_b); implied yield of the base currency from F and S:
  r_b = ((1 + r_q tau_q) S / F - 1) / tau_b.  Forward premium (annualised) = (F/S - 1) / tau.
- Carry trade: long high-yield / short low-yield; per-period excess return of long XXX funded in YYY = spot return +
  (r_XXX - r_YYY) tau, or equivalently (S_{t+1} - F_t)/S_t when forwards are used; transaction costs in pips as stated.
- Cross rate from two USD pairs: EURGBP = EURUSD / GBPUSD; EURJPY = EURUSD x USDJPY.  Triangular arbitrage profit
  = product of the three legs (with bid/ask on the correct side) - 1; zero within costs for consistent quotes.
- NDF: cash-settled on the fixing; P&L = notional x (fixing - contract rate) / fixing for the non-deliverable
  currency, converted as the task says.

## Commodities / futures curves
- Term structure: contango F_next > F_front; backwardation the reverse.  Roll yield (annualised) = ln(F_front / F_next)
  x 365 / days between the two expiries (sign: positive in backwardation).  Continuous series: roll on the task's
  rule (N days before expiry, first notice, volume-based); the "roll return" is the difference between the continuous
  and spot returns over the roll.
- Basis = F - S (or S - F, read it); cash-and-carry return = (F/S - 1) - carrying costs; annualise with ACT/365 unless
  the exchange convention says otherwise.
- Units: barrels / bushels / troy ounces and contract multipliers are stated in the task; never scale silently.

## Crypto
- Perpetual futures: funding paid every 8 hours (00:00, 08:00, 16:00 UTC on most venues) by longs to shorts when the
  rate is positive; annualised funding = rate x 3 x 365 (or 1095 x rate).  Basis trade (long spot, short perp) earns
  the funding received plus basis convergence minus fees.
- Timestamps are UTC; align funding timestamps with the price bars the task describes (the bar whose close is at the
  funding time, or the next bar); a daily series derived from hourly bars uses the task's cut-off hour (00:00 UTC unless
  stated).  Sample end dates include the full last day unless "before" is written.
- Cleaning: drop duplicated timestamps on (venue, symbol, timestamp); report counts per step (see the cleaning
  conventions playbook); a 16:00 vs 00:00 end-of-sample mismatch is a timezone error.
- Mean-reversion of the basis / funding (OU fit): fit on the cleaned series with the task's frequency; the number of
  observations in the fit = rows after cleaning minus one lag.

## Invariants worth printing
- CIP residual (F - F_cip) in pips near zero for consistent quotes; cross rates consistent with the two legs; roll yield
  sign matches contango / backwardation; funding annualisation = rate x 1095; row counts chain (raw -> deduped -> in
  window -> returns = n - 1).
