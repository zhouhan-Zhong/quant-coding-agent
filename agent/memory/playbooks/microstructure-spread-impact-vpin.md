---
title: Spread estimators, price impact and informed trading: quoted/effective/realized spread, Roll, Kyle lambda, Amihud, Corwin-Schultz, VPIN
category: microstructure
tags: bid ask spread, quoted spread, effective spread, realized spread, price impact, midpoint, mid quote, half spread, roll model, roll estimator, serial covariance, kyle lambda, price impact regression, amihud illiquidity, corwin schultz, high low spread, lee ready, tick rule, trade classification, bulk volume classification, bvc, volume buckets, volume clock, vpin, pin, informed trading, toxicity, order imbalance, order flow imbalance, ofi, taq, tick data, quotes, trades, merge_asof, lob, limit order book, depth, microprice, time weighted, intraday intervals
summary: Exact definitions of quoted/effective/realized spread and impact (2 q (P - mid), 5-minute realized), Lee-Ready and tick classification, Roll's 2 sqrt(-cov), Kyle's lambda regression, Amihud, Corwin-Schultz, OFI and VPIN with BVC volume buckets, plus the quote-matching and bps conventions graders test.
---
## When this applies
"Estimate the effective/realized spread from trade-and-quote data", "Roll's model", "Kyle's lambda / price impact",
"Amihud illiquidity", "classify trades as buys/sells (Lee-Ready, tick rule)", "VPIN / order flow toxicity", "order
flow imbalance", "quoted spread by 15-minute interval", limit-order-book snapshots with bid/ask prices and sizes.

## Method and formulas
Symbols: P trade price, q = +1 buyer-initiated / -1 seller-initiated, mid = (bid + ask)/2 of the prevailing quote,
mid_{t+D} the mid D minutes later (D = 5 unless stated), r_d daily return, DVol_d dollar volume.
- Quoted spread = ask - bid (absolute), relative = (ask - bid)/mid, in bp x 1e4; time-weighted over an interval: weight
  each quote by its duration until the next quote update (clip at the interval end).
- Effective spread = 2 q (P - mid) = 2 |P - mid| for correctly signed trades (HALF spread = |P - mid|; follow the task).
  Realized spread = 2 q (P - mid_{t+D}); price impact = 2 q (mid_{t+D} - mid) = effective - realized. Share- or
  dollar-weighted averages per interval/stock as specified; relative versions divide by mid before averaging.
- Prevailing quote: the last quote with timestamp <= trade time (pd.merge_asof(..., direction="backward"); some tasks
  require strictly before: allow_exact_matches=False) within the same day/symbol (by="symbol").
- Lee-Ready: P > mid -> buy, P < mid -> sell, P == mid -> tick test (price up vs last different price -> buy, down ->
  sell, unchanged -> previous classification). Tick rule alone when no quotes exist. Signed volume = q x size.
- Roll (1984): spread = 2 sqrt(-Cov(dP_t, dP_{t-1})) using price CHANGES (or returns x price); Cov > 0 -> undefined
  (report NaN or 0 as the task says). State the covariance ddof (np.cov default ddof=1; population ddof=0).
- Kyle's lambda: regress mid (or price) changes per interval on signed volume (shares or $): dP = alpha + lambda SV;
  in bp per $1e6 when asked. OFI (Cont-Kukanov-Stoikov) per quote update: e_n = 1{b_n >= b_{n-1}} qb_n - 1{b_n <= b_{n-1}} qb_{n-1}
  - 1{a_n <= a_{n-1}} qa_n + 1{a_n >= a_{n-1}} qa_{n-1}; OFI_interval = sum e_n; regress mid changes on OFI.
- Amihud ILLIQ = mean_d(|r_d|/DVol_d) over the period, often reported x 1e6; exclude zero-volume days.
- Corwin-Schultz (daily high/low): beta = sum_{j=0}^{1} [ln(H_{t+j}/L_{t+j})]^2, gamma = [ln(max(H_t, H_{t+1})/min(L_t, L_{t+1}))]^2,
  alpha = (sqrt(2 beta) - sqrt(beta))/(3 - 2 sqrt(2)) - sqrt(gamma/(3 - 2 sqrt(2))), S = 2 (e^alpha - 1)/(1 + e^alpha);
  negative estimates set to 0 (or NaN as told); monthly = mean of daily S.
- VPIN (Easley, Lopez de Prado, O'Hara): bucket size V = (average daily volume)/50 (or the task's); fill buckets in
  time order, splitting a bar/trade across the boundary pro rata when the task says; per bar classify with BVC:
  V_buy = V_bar x Phi(dP/sigma_dP) (normal CDF of the standardised price change; Student-t variant if named), V_sell =
  V_bar - V_buy; per bucket OI_b = |V_sell - V_buy|; VPIN_b = sum_{last n buckets} OI / (n V), n = 50 typical;
  CDF(VPIN) for signals. Tick-rule classification per trade is the alternative when no bars are given.
- Microprice = (bid x ask_size + ask x bid_size)/(bid_size + ask_size); depth imbalance = (bid_size - ask_size)/(bid_size + ask_size).

## Conventions and pitfalls (what graders check)
- Invariants: bid <= ask (drop or flag crossed/locked quotes as the task says), effective spread >= 0 after signing,
  quoted >= 0, realized can be negative, impact + realized = effective exactly, VPIN in [0, 1], Roll spread >= 0 or
  NaN, Kyle lambda > 0 expected (not asserted strictly), Amihud > 0, buy + sell volume = total volume per bucket.
- Timestamps: parse with the exchange timezone as given; sort by time before merge_asof (also within the same second;
  keep the file order for ties); intervals via floor("15min") on the trade time; sessions 09:30-16:00 unless stated.
- Units: spreads in $ vs bp vs percent exactly as asked; prices in dollars (not cents); sizes in shares (round lots
  x 100 only if the task says).
- Weighting: "average effective spread" = share-weighted unless the task says simple mean; per-interval tables include
  the trade count; intervals with no trades -> NaN or 0 as specified (often dropped).
- VPIN warm-up: the first n - 1 buckets have no VPIN (NaN); bucket counts are integers; the last partial bucket is
  dropped unless told; bar-level BVC uses the std of price changes over the sample (ddof as stated).
- Roll on returns vs prices gives different units; use prices for a $ spread, returns for a relative spread.
- Lee-Ready needs the quote prevailing BEFORE the trade; same-timestamp quotes are ambiguous - follow the task's rule.
- Output hygiene: one row per (symbol, interval) sorted by symbol then time; ISO timestamps; floats not numpy types.

## Tested snippet
```python
import numpy as np, pandas as pd
from scipy.stats import norm
rng = np.random.default_rng(0)
q = pd.DataFrame({"ts": pd.date_range("2024-01-02 09:30", periods=2000, freq="s")}); mid = 100 + np.cumsum(rng.normal(0, 0.01, 2000))
q["bid"], q["ask"] = mid - 0.02, mid + 0.02
tr = pd.DataFrame({"ts": q["ts"].iloc[::4].to_numpy() + pd.Timedelta(milliseconds=500)})
tr["side"] = rng.choice([1, -1], len(tr)); tr["price"] = mid[::4] + tr["side"] * 0.02; tr["size"] = rng.integers(100, 1000, len(tr))
m = pd.merge_asof(tr.sort_values("ts"), q.sort_values("ts"), on="ts", direction="backward")   # last quote at or before the trade
m["mid"] = (m["bid"] + m["ask"]) / 2
m["quoted_bps"] = (m["ask"] - m["bid"]) / m["mid"] * 1e4
m["effective_bps"] = 2 * m["side"] * (m["price"] - m["mid"]) / m["mid"] * 1e4                # 2 q (P - mid)
later = pd.merge_asof(pd.DataFrame({"ts": m["ts"] + pd.Timedelta(minutes=5)}), q.assign(mid5=(q["bid"] + q["ask"]) / 2)[["ts", "mid5"]], on="ts", direction="backward")
m["realized_bps"] = 2 * m["side"] * (m["price"] - later["mid5"].to_numpy()) / m["mid"] * 1e4   # realized = effective - impact
m["impact_bps"] = m["effective_bps"] - m["realized_bps"]
print(m[["quoted_bps", "effective_bps", "realized_bps", "impact_bps"]].mean().round(3).to_dict(), "(synthetic quotes: no true impact)")
dp = m["price"].diff().dropna().to_numpy()
cov1 = np.cov(dp[1:], dp[:-1], ddof=0)[0, 1]                                                  # population serial covariance
roll = 2 * np.sqrt(-cov1) if cov1 < 0 else float("nan")                                       # Roll (1984)
lam_kyle = np.polyfit((m["side"] * m["size"]).to_numpy()[1:], dp, 1)[0]                        # price change per signed share
print(round(roll, 5), f"{lam_kyle:.3e}")
bars = pd.DataFrame({"vol": rng.integers(500, 1500, 600), "dp": rng.normal(0, 0.05, 600)})      # VPIN on volume bars
V, n = bars["vol"].sum() / 50, 20                                                               # 50 equal-volume buckets, window 20
z = norm.cdf(bars["dp"] / bars["dp"].std(ddof=1)); bars["buy"] = bars["vol"] * z; bars["sell"] = bars["vol"] - bars["buy"]   # BVC
bars["bucket"] = np.minimum((bars["vol"].cumsum() - 1e-9) // V, 49).astype(int)                 # whole bars; split pro rata if required
b = bars.groupby("bucket")[["buy", "sell"]].sum()
vpin = (b["buy"] - b["sell"]).abs().rolling(n).sum() / (n * V)
print(round(float(vpin.dropna().mean()), 4), bool(vpin.dropna().between(0, 1).all()), int(vpin.isna().sum()), "warm-up buckets")
```
