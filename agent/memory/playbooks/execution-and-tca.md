---
title: Execution scheduling and TCA: participation fills, implementation shortfall, intraday volume profiles, order-flow signals
category: microstructure
tags: tca, transaction cost analysis, implementation shortfall, participation, pov, vwap, twap, execution schedule, child order, bucket, arrival price, arrival mid, effective spread, realized spread, price impact, market vwap, fill rate, intraday volume profile, volume curve, walk-forward, ewma, winsorized, r-squared, largest remainder, order flow imbalance, ofi, limit order book, lob, information coefficient, rolling pca
summary: Bucketed participation (POV) schedules with quote-matched fills, IS/effective/realized spread and impact in bps, walk-forward intraday volume-profile models scored by R^2, integer bar schedules by largest remainder, and leakage-free OFI/PCA signals with ICs.
---
## When this applies
"Simulate a participation-capped (POV) or VWAP execution and compute TCA", "implementation shortfall, effective /
realized spread, temporary impact vs the arrival mid", "fit intraday volume profiles (mean / median / EWMA / winsorized)
walk-forward and pick the best by R^2", "schedule an order over the remaining 5-minute bars", "order-flow imbalance
from limit-order-book levels, rolling PCA + regression signal, information coefficient".

## Method and formulas
Participation schedule (epoch-millisecond clocks, quotes sorted by time):
- Buckets: for bs in range(start, end, bucket): be = min(bs + bucket, end); eligible market trades bs <= t < be
  (half-open; keep file order so first/last trade ids are defined). Market qty Q_b = sum qty, notional = sum price * qty.
- Child: child = min(remaining, cap * Q_b, max_child); remaining -= child only if child > 0; fill_qty = child (full fill
  unless the task models partial fills). A bucket with no trades gets child = 0.
- Fill price: the FIRST quote with timestamp >= bucket START (np.searchsorted(quote_t, bs, side="left")), ASK for a buy,
  BID for a sell; report that quote's ids/timestamps; leave the fields empty when there is no fill.
- Arrival mid M0 = mid of the first quote at/after the order start (even when the first bucket is empty); future mid
  M1 = first quote at/after end + horizon.
- Metrics in bps (buy side; multiply by -1 for a sell), all divided by M0: IS = 1e4 (avg_fill - M0)/M0; effective spread
  = 2e4 (avg_fill - M0)/M0; realized spread = 2e4 (avg_fill - M1)/M0; temporary impact = 1e4 (M1 - M0)/M0 (identity:
  effective = realized + 2 impact). avg_fill = sum(q p)/sum(q); market VWAP = sum notional/sum qty over eligible trades;
  fill_rate = executed/target; unfilled = target - executed; participation = executed/market volume.
Intraday volume profiles and scheduling:
- Keep regular-hours rows (09:30 <= time <= 16:00) BEFORE aggregating; bar = timestamp.floor("5min"); groupby(bar) with
  first/max/min/last/sum (not resample - it fabricates empty bins); keep bar starts 09:30..15:55 (78 bars); a day is
  complete iff it has all 78 bars (half-days are excluded and usually listed).
- Shares S[d, b] = V[d, b]/sum_b V[d, b]. Walk-forward: test day i >= W uses only days i-W..i-1 (W = 20 typical);
  num_test_days = n_complete_days - W.
- Models on the W x 78 window: mean; median; EWMA with w_L = exp(-ln2 L/H) (L = 0 most recent, half-life H, e.g. 20);
  winsorised mean (per-bar np.percentile 5 and 95, np.clip, then mean). Renormalise EVERY profile to sum 1.
- Score: daily R^2 = 1 - SSE/SST of predicted vs realised shares (SST = 0 -> 1 if identical else 0), averaged over test
  days; best = FIRST argmax in the task's model order (np.argmax keeps the first on ties).
- Order on day D at time t0 (D may lie after the sample): window = last W complete days before D; grid =
  pd.date_range(D 09:30, D 15:55, freq="5min"); keep bars strictly after t0; renormalise the profile on them; integer
  quantities = floor + largest remainder (ties -> earlier bar); cumulative = cumsum; total == order quantity.
Order-flow imbalance (LOB) signals:
- Per level i and side: net flow = limit_added - cancelled - market_executed (notional); weight 1/(1 + |distance_i|)
  (distances are signed: use abs); OFI_i = w_bid,i flow_bid,i - w_ask,i flow_ask,i. Target = forward log mid return
  ln(m_{t+1}/m_t) (NaN on the last row).
- Walk-forward step t: train on rows t-W..t-1 with a non-NaN target; z-score with the TRAINING mean/std (ddof=0,
  StandardScaler convention); sklearn PCA(1, random_state=0); flip the sign so corr(PC1, level-0 OFI) >= 0; OLS
  target ~ 1 + PC1; transform row t with the same scaler/PCA/sign and predict. IC = spearman / pearson of predictions
  vs realised targets over the steps that have a target.

## Conventions and pitfalls (what graders check)
- results.json may be compared with exact key ORDER and floats pre-rounded to 6 dp (round(x, 6), abs_tol 1e-6);
  timestamps, ids and counts as exact ints. Never rewrite the input spec/rules files (compared with literals).
- CSV rows compared as strings against a re-derivation: floats as f"{x:.6f}" (every float column, constants too), ints
  plain, missing fields "", booleans as the literal strings shown ("true"/"false"); exact column order. Keep raw floats
  for the totals; format only when writing.
- Counts: buckets_filled = fill_qty > 0; buckets_empty = zero eligible trades (they differ once remaining hits 0).
- The first bucket can precede the first quote/trade: it is still a row (empty), and "first quote at/after" still holds.
- Sells fill at the bid and flip the signs so that a cost is positive; when the task gives explicit formulas, use them.
- Volume profiles: filter hours before grouping; dates "YYYY-MM-DD"; schedule datetimes "%Y-%m-%d %H:%M:%S"; predicted
  shares sum to 1; scheduled quantities are ints summing exactly to the order; cumulative == cumsum.
- Signals: never include row t in its own training window; the scaler ddof moves PC1 by ~1% (enough to fail rtol 1e-2)
  but not the predictions; the last row gets a prediction but no IC pair.

## Tested snippet
```python
import math
import numpy as np, pandas as pd

def participation_fills(tr_t, tr_q, q_t, bid, ask, start, end, bucket, cap, max_child, target, side="buy"):
    """POV schedule: child = min(remaining, cap * bucket market qty, max_child); fill at the first quote at/after bucket start."""
    rows, rem = [], float(target)
    for bs in range(start, end, bucket):
        be = min(bs + bucket, end); m = (tr_t >= bs) & (tr_t < be); mq = float(tr_q[m].sum())
        child = min(rem, cap * mq, max_child); j = int(np.searchsorted(q_t, bs, side="left"))
        if child > 0: rem -= child
        rows.append({"bucket_start": bs, "bucket_end": be, "n_trades": int(m.sum()), "market_qty": mq, "fill_qty": child,
                     "fill_price": (ask[j] if side == "buy" else bid[j]) if child > 0 else np.nan, "remaining": rem})
    return pd.DataFrame(rows)

def tca_bps(avg_fill, m0, m1, side="buy"):        # normalised by the arrival mid m0; m1 = mid at end + horizon
    s = (1.0 if side == "buy" else -1.0) / m0
    return {"implementation_shortfall_bps": float(s * 1e4 * (avg_fill - m0)), "effective_spread_bps": float(s * 2e4 * (avg_fill - m0)),
            "realized_spread_bps": float(s * 2e4 * (avg_fill - m1)), "temporary_impact_bps": float(s * 1e4 * (m1 - m0))}

def volume_profiles(win, half_life=20.0):        # win: (W days x n_bars) shares, chronological rows
    ew = np.exp(-math.log(2) * np.arange(len(win)) / half_life)[::-1]                # most recent day has weight 1
    lo, hi = np.percentile(win, 5, axis=0), np.percentile(win, 95, axis=0)
    p = {"mean": win.mean(0), "median": np.median(win, 0), "ewma": ew @ win / ew.sum(), "winsorized": np.clip(win, lo, hi).mean(0)}
    return {k: v / v.sum() for k, v in p.items()}

def r2(y, yhat):
    sst = float(((y - y.mean()) ** 2).sum())
    return (1.0 if np.array_equal(y, yhat) else 0.0) if sst == 0 else 1.0 - float(((y - yhat) ** 2).sum()) / sst

def walk_forward_scores(S, W=20):                # S: (n_days x n_bars) share matrix -> mean daily R^2 per model
    sc = {}
    for i in range(W, len(S)):
        for k, p in volume_profiles(S[i - W:i]).items(): sc.setdefault(k, []).append(r2(S[i], p))
    return {k: float(np.mean(v)) for k, v in sc.items()}

def largest_remainder(total, w):                 # ints summing exactly to total; ties -> earlier bar
    raw = total * np.asarray(w, float) / np.sum(w); q = np.floor(raw).astype(int)
    q[np.lexsort((np.arange(len(q)), -(raw - q)))[: int(total - q.sum())]] += 1
    return q
```
