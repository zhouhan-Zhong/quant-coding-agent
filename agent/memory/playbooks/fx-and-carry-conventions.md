---
title: FX quoting, settlement dates, forwards, implied yields, carry, FX options and perpetual funding
category: fx-commodities-crypto
tags: fx, foreign exchange, currency pair, base currency, quote currency, terms currency, pips, bid ask, mid, cross rate, triangulation, spot date, t+2, t+1, settlement, holiday calendar, value date, following, covered interest parity, cip, forward rate, forward points, outright, deposit rates, day count, act/360, act/365, implied yield, ndf, basis, fx swap, roll, carry trade, interest differential, forward hedge, hedged portfolio, garman-kohlhagen, fx option, 25 delta, risk reversal, butterfly, premium-adjusted delta, forward delta, mtm, pnl attribution, net delta, crypto, perpetual, perp, funding rate, basis carry, ornstein-uhlenbeck, half-life, adf, annualization
summary: Quote/pip conventions, holiday-aware spot and delivery dates, CIP forwards with per-currency day counts, implied yields and NDF basis, cross rates, forward MTM, daily carry backtests with forward-hedge overlays, Garman-Kohlhagen with spot/forward/premium-adjusted 25-delta strikes, and perpetual funding statistics with OU fits.
---
## When this applies
"Spot and delivery dates for EURUSD/USDCAD", "outright forwards and forward points from deposit rates", "implied
yields from forwards / NDF basis", "cross rates with bid/ask", "mark the forward book to market", "G10 carry
portfolio with a quarterly forward hedge", "25-delta strikes, Garman-Kohlhagen prices and greeks", "funding-rate
statistics, OU mean reversion and Monte Carlo carry of a perpetual".

## Method and formulas
Symbols: pair BASE/QUOTE (S = QUOTE per 1 BASE); r_b, r_q simple deposit rates; tau_c = days/B_c with the basis of
CURRENCY c (ACT/360: USD EUR JPY CHF ...; ACT/365: GBP AUD NZD CAD HKD SGD - follow the task table); F forward.
- Pips: 1e-4 (1e-2 for JPY quotes); forward points = (F - S) x pip multiplier; mid = (bid + ask)/2; bid >= ask = stale.
- Dates: good day = weekday and not a holiday of EITHER currency (USD holidays count for USD pairs); spot = trade +
  2 good days (T+1 for USD/CAD and others the task lists); tenor delivery = spot + DateOffset(months=n), then roll
  FOLLOWING to a good day (week tenors: + 7n days); days = delivery - spot.
- CIP (simple): F = S (1 + r_q tau_q)/(1 + r_b tau_b) with deposit MIDS unless sides are specified (F_bid = S_bid
  (1 + r_q,bid tau)/(1 + r_b,ask tau)); continuous F = S e^((r_q - r_b) T).
- Implied yield of currency c vs USD: ratio = F/S for USD/c, S/F for c/USD; r_c = (ratio (1 + r_usd d/360) - 1)/(d/B_c);
  CIP deviation (bp) = (implied - quoted) x 1e4. NDF: CIP outright from onshore and USD deposits with opposite sides
  (bid = S (1 + r_loc,bid t)/(1 + r_usd,offer t)); basis = market NDF - CIP (pips).
- FX swap roll: rolling a short-base position forward uses the bid points, a long-base position the offer points.
- Inversion: QUOTE/BASE bid = 1/ask, ask = 1/bid. Cross XXX/YYY = (XXX/USD)(USD/YYY): bid x bid, ask x ask, mid x mid.
- Forward MTM: pnl_quote = sign N_b (F_now - K) converted at SPOT mid; attribution spot = sign N_b (S_now - S_trade),
  carry = sign N_b (F_now - S_now), residual = total - spot - carry (sums exactly; checkers use atol ~1); net delta per
  currency (USD included) in USD at spot mid. Hedged foreign asset: V_t S_t + h V_0 (F_0 - S_t); unhedged return
  (1 + r_local)(S_1/S_0) - 1; carry log excess return ln(S_1/F_0).
- Carry backtest (daily): monthly deposits reindexed to spot dates and forward-filled; signal = r_c - r_usd on the LAST
  spot row of each month (plus the row before the start for the first basket), active from the NEXT row; daily
  currency return = (S_t/S_{t-1} - 1) x (+1 for c/USD, -1 for USD/c) + (r_c - r_usd)/100/252; long leg = mean(top k),
  short leg = -mean(bottom k), unhedged = long + short. Accrual /252 per row matched; calendar/360 differs slightly;
  no accrual fails. Monthly variant: S_1 (1 + r_b tau)/(S_0 (1 + r_q tau)) - 1.
- Forward hedge overlay: hedge dates (quarter ends), effective next row; basket = longs of the latest carry signal on or
  before the hedge date; sell forward at F (monthly mids, spot -> delivery days); per row overlay = mean over the basket
  of (-spot return + (F/S - 1)/window), window = actual rows until the next effective date (not busday_count).
- Garman-Kohlhagen (T = (expiry - trade)/365 for vol and both discounts; rd terms, rf base, continuous):
  F = S e^((rd-rf)T), price = w e^(-rd T)(F N(w d1) - K N(w d2)); spot delta = w e^(-rf T) N(w d1); premium-adjusted
  delta = delta - price/S; gamma = e^(-rf T) phi(d1)/(S vol sqrt T); vega = S e^(-rf T) phi(d1) sqrt T; base % =
  price/S x 100; base pips = terms pips / S. 25-delta strikes: spot delta d1 = w Phi^-1(0.25 e^(rf T)), forward delta
  d1 = w Phi^-1(0.25), K = F exp(-d1 vol sqrt T + 0.5 vol^2 T); premium-adjusted: solve w e^(-rf T)(K/F) N(w d2) = 0.25 w
  (brentq). Smile: vol_25c = ATM + BF + RR/2, vol_25p = ATM + BF - RR/2.
- Perpetual funding (8h): timestamps pd.to_datetime(ms, unit="ms", utc=True).dt.floor("s"), window [start, end+1d);
  longs pay notional x rate when the rate is positive (cumulative long PnL = -notional sum(rate)); APR = mean x 1095,
  APY = (1 + mean)^1095 - 1; basis = (F - S)/S, dated-future annualised basis = (F/S - 1) 365/days; cash-and-carry earns
  basis + funding received on the short perp. bps = rate x 1e4; ann vol = std x sqrt(1095); stats ddof=1, skew/kurtosis
  bias=False; autocorr = Series.autocorr(lag); ADF adfuller(maxlag, regression="c", autolag="AIC").
- OU on funding: OLS of diff(f) on [1, f_{t-1}] -> kappa = -b/dt, mu = -a/b, sigma = std(resid, ddof=1)/sqrt(dt),
  dt from params (1/1095); half-life = ln 2/(kappa dt) periods (/3 for days); stationary std = sigma/sqrt(2 kappa);
  regime fits re-difference inside each regime. MC: exact step f' = mu + (f - mu)e^(-k dt) + sigma sqrt((1 -
  e^(-2k dt))/(2k)) Z from the last observed rate; annual carry per path = sum of the 1095 simulated rates.

## Conventions and pitfalls (what graders check)
- Day count by currency, not pair; rates in percent -> /100; option *_rate_cc fields are already decimals.
- Keep bid < ask after every transformation; convert quote-currency PnL at SPOT mid; no discounting unless told.
- Spot-date tests hit holiday weekends (T+2 across a USD holiday) and T+1 pairs; delivery days count actual days.
- Return sign for USD/c pairs: -(pair return), not 1/S returns (close but different).
- Equity/benchmark correlations: inner-join on dates and drop NaN warm-up rows before checking [-1, 1].
- Funding: 8h rates are decimals per period; monthly/annual sums keep the sign of the position; stationary series
  checks (ADF stat < critical) and decaying autocorrelation are asserted; MC pins are seed-stream specific.
- Floats via float(); nested dicts pair -> tenor -> {bid, ask, mid}; round only when the spec says.

## Tested snippet
```python
import math, numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import brentq

BASIS = {"USD": 360, "EUR": 360, "JPY": 360, "CHF": 360, "CAD": 365, "GBP": 365, "AUD": 365, "NZD": 365}
HOL = {"USD": {pd.Timestamp("2024-05-27")}}                            # per-currency holiday sets (from the task data)
good = lambda d, b, q: d.weekday() < 5 and d not in HOL.get(b, set()) and d not in HOL.get(q, set())
def add_good_days(d, n, b, q):                                         # spot = trade + n good days of BOTH currencies (T+2; T+1 USD/CAD)
    while n > 0:
        d += pd.Timedelta(days=1); n -= good(d, b, q)
    return d
def delivery(spot, months, b, q):                                      # spot + calendar months, then roll FOLLOWING to a good day
    d = spot + pd.DateOffset(months=months)
    while not good(d, b, q): d += pd.Timedelta(days=1)
    return d
def cip_forward(S, r_b, r_q, days, b, q):                              # simple deposit rates in percent, day basis of EACH currency
    return S * (1 + r_q / 100 * days / BASIS[q]) / (1 + r_b / 100 * days / BASIS[b])
def implied_yield(F, S, r_usd, days, ccy, usd_is_base):               # non-USD yield (percent) implied by an outright vs USD deposits
    return ((F / S if usd_is_base else S / F) * (1 + r_usd / 100 * days / 360) - 1) / (days / BASIS[ccy]) * 100   # ratio F/S or S/F
def gk(S, K, T, rd, rf, vol, call=True):                               # Garman-Kohlhagen: terms ccy per 1 base, continuous rd (terms) / rf (base)
    F = S * math.exp((rd - rf) * T); sT = vol * math.sqrt(T); d1 = math.log(F / K) / sT + 0.5 * sT; w = 1 if call else -1
    return w * math.exp(-rd * T) * (F * stats.norm.cdf(w * d1) - K * stats.norm.cdf(w * (d1 - sT))), w * math.exp(-rf * T) * stats.norm.cdf(w * d1)
def strike_25d(S, T, rd, rf, vol, conv="spot", call=True):            # spot: e^{-rf T} N(w d1) = .25; forward: N(w d1) = .25; "pa": premium-adjusted
    F = S * math.exp((rd - rf) * T); sT = vol * math.sqrt(T); w = 1 if call else -1
    if conv != "pa": d1 = w * stats.norm.ppf(0.25 * (math.exp(rf * T) if conv == "spot" else 1.0)); return F * math.exp(-d1 * sT + 0.5 * sT * sT)
    f = lambda K: w * math.exp(-rf * T) * (K / F) * stats.norm.cdf(w * (math.log(F / K) / sT - 0.5 * sT)) - w * 0.25   # pa delta - target
    return brentq(f, F, 5 * F, xtol=1e-14) if call else brentq(f, 1e-3 * F, F, xtol=1e-14)
def ou_fit(x, dt):                                                     # OLS of dx on [1, x_{t-1}]: kappa = -b/dt, mu = -a/b, sigma = sd(resid)/sqrt(dt)
    X = np.column_stack([np.ones(len(x) - 1), x[:-1]]); (a, b), *_ = np.linalg.lstsq(X, np.diff(x), rcond=None)
    k, mu, sig = -b / dt, -a / b, float(np.std(np.diff(x) - X @ [a, b], ddof=1) / math.sqrt(dt))
    return k, mu, sig, math.log(2) / (k * dt), sig / math.sqrt(2 * k)    # + half-life in periods, stationary std
def ou_sum_paths(f0, k, mu, sig, dt, H, N, rng):                       # exact OU transition; per-path sum of the H simulated rates
    e, sd = math.exp(-k * dt), sig * math.sqrt((1 - math.exp(-2 * k * dt)) / (2 * k)); cur, cum = np.full(N, f0), np.zeros(N)
    for _ in range(H): cur = mu + (cur - mu) * e + sd * rng.standard_normal(N); cum += cur
    return cum

sp = add_good_days(pd.Timestamp("2024-05-24"), 2, "EUR", "USD"); dl = delivery(sp, 3, "EUR", "USD"); d = (dl - sp).days   # Fri + US holiday Mon
F = cip_forward(1.0849, 3.90, 5.30, d, "EUR", "USD"); print(sp.date(), dl.date(), d, round(F, 6), round((F - 1.0849) * 1e4, 2), round(implied_yield(F, 1.0849, 5.30, d, "EUR", False), 6))
p, delta = gk(1.25, 1.27, 0.25, 0.05, 0.04, 0.08); pp, _ = gk(1.25, 1.27, 0.25, 0.05, 0.04, 0.08, call=False)
Kc = strike_25d(1.25, 0.25, 0.05, 0.04, 0.08, "spot"); Kpa = strike_25d(145.0, 0.25, 0.001, 0.05, 0.12, "pa", call=False)
print(round(p, 8), round(delta, 6), round(p - pp - (1.25 * math.exp(-0.04 * 0.25) - 1.27 * math.exp(-0.05 * 0.25)), 14), round(gk(1.25, Kc, 0.25, 0.05, 0.04, 0.08)[1], 10), round(Kpa, 6), round(gk(145.0, Kpa, 0.25, 0.001, 0.05, 0.12, False)[1] - gk(145.0, Kpa, 0.25, 0.001, 0.05, 0.12, False)[0] / 145.0, 10))
rng = np.random.default_rng(42); dt = 1 / 1095; x = np.empty(1750); x[0] = 1e-4
for t in range(1, 1750): x[t] = x[t - 1] + 300 * (5e-5 - x[t - 1]) * dt + 0.002 * math.sqrt(dt) * rng.standard_normal()
k, mu, sig, hl, sstd = ou_fit(x, dt); cum = ou_sum_paths(x[-1], k, mu, sig, dt, 1095, 20000, rng)
print(round(k, 2), round(mu, 8), round(sig, 6), round(hl, 3), round(hl / 3, 3), round(sstd, 8), round(float(cum.mean()), 5), round(float(np.percentile(cum, 5)), 5), round(x.mean() * 1095, 5))
```
