---
title: FX cross rates, triangular arbitrage with bid/ask, CIP deviations and currency hedging of portfolios
category: fx-commodities-crypto
tags: triangular arbitrage, cross rate, implied cross, arbitrage cycle, arbitrage detection, bid ask, rate matrix, currency triangle, profit per unit, round trip, direction, cycle direction, log rates, negative cycle, bellman-ford, cip deviation, covered interest parity, cip basis, forward points, pips, outright forward, fx hedge, hedge ratio, currency overlay, hedged return, unhedged return, forward hedge, rolling hedge, base currency, quote currency, price currency, inversion, exposure aggregation, net currency exposure, garman-kohlhagen
summary: Quote-convention-safe conversion (sell base at the bid, buy base at the ask), enumeration of ordered triangles with profit per unit and both directions, mid-only implied-cross deviations, CIP forwards/points/basis with per-currency day counts, and hedged vs unhedged return and exposure aggregation formulas graders recompute.
---
## When this applies
"Detect triangular arbitrage in an FX rate matrix (with or without bid/ask)", "implied cross rate vs quoted", "profit
per unit / in bp and the trading direction", "compute forwards from CIP and the deviation/basis", "hedged vs unhedged
return of a foreign asset, hedge ratio, forward hedge P&L", "net currency exposure of a multi-currency portfolio".

## Method and formulas
Convention: pair A/B (base/quote) = units of B per 1 unit of A (EUR/USD 1.08 = 1 EUR costs 1.08 USD). Bid < ask.
- Moving one unit of X into Y: if the market quotes X/Y you SELL the base X at the BID: Y = X x bid(X/Y); if it quotes
  Y/X you BUY the base Y at the ASK: Y = X / ask(Y/X). Inverted quotes: bid(B/A) = 1/ask(A/B), ask(B/A) = 1/bid(A/B).
  Mid = (bid + ask)/2; with mids only, rate(X -> Y) = mid(X/Y) or 1/mid(Y/X).
- Triangle A -> B -> C -> A: final = rate(A,B) x rate(B,C) x rate(C,A); profit per unit of A = final - 1 (x 1e4 = bp);
  each unordered triangle has TWO directions (A->B->C and A->C->B); report both or the profitable one as asked. Costs:
  multiply by (1 - c) per leg or widen the spread. Arbitrage exists iff final > 1 (strictly; add the tolerance the task
  gives, e.g. 1e-9 or 1 bp). With n currencies enumerate permutations of 3 (n(n-1)(n-2) ordered cycles, each cycle
  appearing 3 times with different start currencies - dedupe by rotation if the task wants unique cycles).
- Implied cross from mids: S(A/C)_implied = S(A/B) x S(B/C); deviation = S_quoted/S_implied - 1 (or in pips).
  Longer cycles: weights w = -ln(rate) on a directed graph, arbitrage iff a negative cycle exists (Bellman-Ford).
- CIP outright forward with simple money-market rates: F = S x (1 + r_quote tau_quote)/(1 + r_base tau_base), tau = days/
  basis per currency (USD, EUR, JPY ACT/360; GBP, AUD, CAD ACT/365); continuous version F = S e^{(r_quote - r_base) T}.
  Forward points = (F - S) x 1e4 (x 1e2 for JPY-quoted pairs); forward premium = F/S - 1; annualised = (F/S - 1)/tau.
- CIP deviation/basis: implied quote-currency rate r_q* = ((F_mkt/S)(1 + r_base tau_base) - 1)/tau_quote; basis_bp =
  (r_q* - r_quote) x 1e4; or implied base rate when the task says "implied foreign yield". Zero basis <-> CIP holds.
- Hedged return of a foreign-currency asset for a domestic investor (hedge ratio h, forward sold at F for the initial
  notional): r_unhedged = (1 + r_local) S_1/S_0 - 1; r_hedged = r_unhedged + h (F - S_1)/S_0; h = 1 gives
  r_hedged ~ r_local + (F/S_0 - 1) (locked forward premium) + r_local (S_1/S_0 - 1) (residual on the gain).
  Minimum-variance hedge ratio from history: h* = Cov(r_asset_dom, r_fx)/Var(r_fx) (state ddof); forward hedge P&L of
  a short forward = N (F_0 - S_T) in quote currency; rolling monthly hedges reset N to the current asset value.
- Exposure aggregation: convert each position to the base currency at SPOT (not forwards) unless the task says;
  net exposure per currency = sum of signed positions; hedged exposure = net x (1 - h). DV01/CS01 of foreign legs:
  convert the sensitivity at spot, state the currency.
- Garman-Kohlhagen (FX option): BS with q = r_foreign (base rate), r = r_domestic (quote rate); forward delta vs spot
  delta differ by e^{-r_f T}; put-call parity C - P = S e^{-r_f T} - K e^{-r_d T} = e^{-r_d T} (F - K).

## Conventions and pitfalls (what graders check)
- Invariants asserted: bid <= ask after every inversion, mid cycles ~ 1 within bid-ask, profit computed from the
  written rates reproduces the written profit, direction string consistent with the rates used, CIP forward reprices
  from S and the rates, hedged-return volatility < unhedged (not strictly asserted), exposures sum to the total.
- Which currency is the base decides everything: EUR/USD 1.08 means USD per EUR; a USD/EUR 0.9259 quote is its inverse.
  Never average a rate and its inverse; invert bid/ask by swapping.
- Profit per unit is quoted in the START currency; "profit in USD on 1,000,000 EUR" = profit x notional x S(EUR/USD)
  (converted at the end rate used in the cycle, or spot as the task says).
- Day counts and tenor days: use the task's day count per currency; months -> actual days from the dates when given,
  else 30/360-style approximations only when stated. 1Y = 360 or 365 days as the basis says.
- Pips: 4 decimals for most pairs, 2 for JPY pairs; forward points may be quoted in pips in input files (divide by 1e4
  or 1e2 before adding to spot; negative points for discount currencies).
- Hedged return: the forward is sold on the INITIAL notional unless the task hedges the expected ending value; the
  residual S_1 exposure on the asset's local gain stays unhedged with h = 1 (do not force r_hedged = r_local + premium
  unless the task's formula says so).
- Report both directions and all triangles when asked; sort by profit descending then by path string for ties; profits
  below the tolerance are not "opportunities" (count only strictly profitable ones).
- Output hygiene: currency codes upper-case ISO, paths with the task's separator ("EUR->USD->JPY->EUR" or lists),
  rates with full precision (no rounding to pips), booleans for `arbitrage_exists`.

## Tested snippet
```python
import numpy as np, pandas as pd
from itertools import permutations
quotes = {("EUR", "USD"): (1.0850, 1.0852), ("USD", "JPY"): (151.20, 151.24), ("EUR", "JPY"): (164.30, 164.36),  # BASE/QUOTE: (bid, ask)
          ("GBP", "USD"): (1.2700, 1.2703), ("EUR", "GBP"): (0.8530, 0.8533), ("GBP", "JPY"): (192.10, 192.18)}
def rate(frm, to):                       # units of `to` received per 1 unit of `frm`, hitting the market's bid/ask
    if (frm, to) in quotes: return quotes[(frm, to)][0]        # sell the base at the bid
    if (to, frm) in quotes: return 1.0 / quotes[(to, frm)][1]  # buy the base at the ask
    raise KeyError((frm, to))
ccys = sorted({c for p in quotes for c in p}); rows = []
for a, b, c in permutations(ccys, 3):                           # every ordered triangle a -> b -> c -> a (both directions)
    try: final = rate(a, b) * rate(b, c) * rate(c, a)
    except KeyError: continue
    rows.append(dict(start=a, path=f"{a}->{b}->{c}->{a}", final=final, profit_bps=(final - 1) * 1e4))
tri = pd.DataFrame(rows).sort_values(["profit_bps", "path"], ascending=[False, True]).reset_index(drop=True)
print(tri.head(3).round(4).to_string(index=False)); print("profitable cycles:", int((tri["profit_bps"] > 0).sum()), "of", len(tri))
mid = lambda p: sum(quotes[p]) / 2
implied = mid(("EUR", "USD")) * mid(("USD", "JPY"))              # implied EUR/JPY cross from mids
print(round(implied, 4), round(mid(("EUR", "JPY")) / implied - 1, 6), "= quoted/implied - 1 (mids)")
S, r_usd, r_eur, days = 1.0851, 0.0530, 0.0390, 90              # EUR/USD spot, USD (quote) rate, EUR (base) rate, tenor days
F = S * (1 + r_usd * days / 360) / (1 + r_eur * days / 360)     # CIP with simple rates, ACT/360 for both currencies here
points = (F - S) * 1e4                                          # pips (1e2 for JPY quotes)
F_mkt = 1.0890; r_usd_implied = ((F_mkt / S) * (1 + r_eur * days / 360) - 1) * 360 / days
print(round(F, 6), round(points, 2), round((r_usd_implied - r_usd) * 1e4, 2), "bp CIP basis (implied - actual USD rate)")
r_local, S1 = 0.03, 1.0700                                      # EUR asset return over the period, spot at the end
unhedged = (1 + r_local) * S1 / S - 1
hedged = unhedged + (F - S1) / S                                 # sold the initial notional forward at F (hedge ratio 1)
print(round(unhedged, 6), round(hedged, 6), round(r_local + (F / S - 1), 6), "~ local return + locked forward premium")
```
