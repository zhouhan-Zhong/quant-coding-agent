---
title: Cross-venue arbitrage: fee-inclusive locked edges, capital allocation, daily equity
category: microstructure
tags: arbitrage, cross-venue, cross-exchange, dislocation, prediction markets, kalshi, polymarket, binary contracts, yes, no, locked edge, risk-free payoff, fees, bps, net edge, gross edge, entry cost, threshold, candidate snapshots, trade log, capital allocation, greedy allocation, tie-break, unaffordable, dedupe, source sequence, cutoff, resolution, synchronized snapshots, daily equity, reserved capital, funding cost, capacity utilization, event accounting
summary: Cleaning cascade with step counts, two-direction fee-inclusive locked edges, per-timestamp greedy allocation with deterministic tie-breaks and statuses, and daily equity with reserved capital and funding, including the float-order trap that decides ties.
triggers: text:prediction market, text:cross-venue, text:kalshi, text:polymarket
---
## When this applies
"The same event trades on two venues: detect dislocations where YES on one venue plus NO on the other costs < 1
after fees", "allocate limited capital to candidate signals ranked by net locked edge", "trade_log.csv,
candidate_snapshots.csv with statuses, daily_equity.csv with funding on reserved capital", and analogous
cross-exchange / ETF-vs-basket convergence trades with capital locked until settlement.

## Method and formulas
Symbols: pair p, snapshot time ts, y_V / n_V canonical YES / NO prices on venue V (already polarity-adjusted, in
(0,1)), fee_V in bps of the leg price, min_edge threshold, Q_max contract cap, C0 capital.
- Cleaning cascade, counting each step (counts are asserted exactly): keep approved pairs -> keep their snapshots ->
  duplicate groups on (pair, venue, ts): count groups with size > 1, keep the max source sequence
  (sort_values(key + [seq]).groupby(key).tail(1)) -> drop non-active status (counted after dedupe) -> drop rows with
  a NaN price or a price outside the OPEN interval (0,1) (between(0, 1, inclusive="neither")) -> drop ts >= cutoff =
  min(venue close times, resolution time) -> synchronized = (pair, ts) with both venues present, pivoted to
  yes/no x venue columns.
- Two directions: D1 = YES on V1 + NO on V2, D2 = YES on V2 + NO on V1; each pays exactly 1 per contract whatever the
  outcome. fee_leg = bps/1e4 x leg price; total_fee = fee_yes + fee_no; gross_cost = yes + no; gross_edge =
  1 - gross_cost; net_edge = 1 - gross_cost - total_fee; entry_cost = gross_cost + total_fee (per contract).
  Best direction = sort by (-net_edge, direction name); tie counted when |net_1 - net_2| <= 1e-12.
- Per timestamp group, chronologically: (1) release trades with exit <= ts (cash += qty x 1); (2) evaluate pairs not
  yet traded in sorted pair order; best net_edge < min_edge -> BELOW_THRESHOLD row with the group-start cash;
  (3) rank actionable by (-net_edge, entry_cost, pair_id); (4) qty = min(Q_max, floor(cash / entry_cost)); qty < 1 ->
  UNAFFORDABLE (cash unchanged); else OPENED: cash -= qty x entry_cost, one trade per pair, exit = resolution time.
  Trade-log money fields are qty-scaled (fees, gross cost, gross_pnl = qty gross_edge, net_pnl = qty net_edge);
  edges and prices are per unit. num_candidate_signals = distinct pairs that ever cleared the threshold.
- Daily equity: UTC calendar days from the first entry date to the last exit date; apply events with ts < day end
  (entries before exits at equal ts); reserved = sum qty x entry_cost of open trades; funding = reserved x rate / ann
  charged after the day's events (resolution day included); equity = cash + sum of open qty (the locked payout, so
  the edge is booked on the entry day); daily_pnl = equity.diff() (first 0); cumulative_return = equity / C0.
- Metrics: r = cumulative_return.pct_change().iloc[1:]; ann_ret = mean(r) x ann; ann_vol = std(r, ddof=1) sqrt(ann);
  Sharpe = ann_ret/ann_vol (0 if vol = 0); max drawdown on cumulative_return (positive); capacity utilisation =
  mean(reserved) / C0; total funding = last cumulative funding.

## Conventions and pitfalls (what graders check)
- The checker re-runs the whole pipeline from the inputs: trade set, entry times, qty, direction and cash columns
  must match exactly. Many pairs share identical edges and entry costs, so ranking ties are decided by float noise:
  write the spec's formulas in the spec's operation order (1 - gc - (fy + fn), not 1 - gc - fy - fn).
- Do not re-apply polarity flips to prices already called canonical; statuses exactly {OPENED, BELOW_THRESHOLD,
  UNAFFORDABLE}; each edge-case counter (duplicates, invalid, inactive, cutoff, below threshold, tie, unaffordable)
  is typically > 0 in the fixture, so a zero hints at a wrong step order.
- Timestamps: pd.to_datetime(..., utc=True); write ts.isoformat().replace("+00:00", "Z"); daily keys date().isoformat().
- Round floats (6 dp) only when writing; CSVs are compared with assert_frame_equal(atol 1e-6 / 1e-4) after sorting by
  (snapshot ts, pair) and with an exact column list - build every frame from an explicit column order.
- Payout per contract is 1 regardless of the outcome; the resolution outcome is reporting-only.

## Tested snippet
```python
import math, numpy as np, pandas as pd

FEE = {"A": 70.0, "B": 20.0}                                           # per-venue fee, bps of each leg's price
def directions(yes, no):                                               # yes/no: venue -> canonical price in (0, 1)
    out = []
    for yv, nv in (("A", "B"), ("B", "A")):                            # YES on yv + NO on nv pays exactly 1 at resolution
        fy, fn = FEE[yv] / 1e4 * yes[yv], FEE[nv] / 1e4 * no[nv]; gc, tf = yes[yv] + no[nv], fy + fn   # total fee FIRST (ties!)
        out.append(dict(direction=f"BUY_YES_{yv}_BUY_NO_{nv}", gross_cost=gc, entry_cost=gc + tf, gross_edge=1.0 - gc, net_edge=1.0 - gc - tf))
    return sorted(out, key=lambda d: (-d["net_edge"], d["direction"]))
def allocate(snaps, exits, cash, min_edge, max_q):                     # one trade per pair, capital locked until resolution
    opened, active, trades, log = set(), {}, [], []
    for ts, grp in snaps.groupby("ts", sort=True):
        for p in sorted(p for p, t in active.items() if t["exit"] <= ts): cash += active.pop(p)["qty"]   # release payouts first
        act = []
        for _, s in grp.sort_values("pair").iterrows():
            if s["pair"] in opened: continue
            d = directions(s["yes"], s["no"])
            if d[0]["net_edge"] < min_edge: log.append((ts, s["pair"], "BELOW_THRESHOLD", cash, cash)); continue
            act.append((s["pair"], d[0]))
        for p, d in sorted(act, key=lambda x: (-x[1]["net_edge"], x[1]["entry_cost"], x[0])):   # edge rank, cheaper, id
            q = min(max_q, math.floor(cash / d["entry_cost"]))
            if q < 1: log.append((ts, p, "UNAFFORDABLE", cash, cash)); continue
            b = cash; cash -= q * d["entry_cost"]; opened.add(p); active[p] = dict(qty=q, exit=exits[p]); log.append((ts, p, "OPENED", b, cash))
            trades.append(dict(pair=p, entry=ts, exit=exits[p], qty=q, cost=d["entry_cost"], net_pnl=q * d["net_edge"]))
    return pd.DataFrame(trades), log
def daily_equity(tr, cap0, rate, ann):                                 # UTC days; events with ts < day end, entries first
    ev = sorted([(t.entry, 0, i) for i, t in tr.iterrows()] + [(t.exit, 1, i) for i, t in tr.iterrows()])
    cash, op, k, cumf, rows = cap0, {}, 0, 0.0, []
    for d in pd.date_range(tr.entry.min().normalize(), tr.exit.max().normalize(), freq="D"):
        while k < len(ev) and ev[k][0] < d + pd.Timedelta(days=1):
            _, kind, i = ev[k]; t = tr.loc[i]; k += 1
            if kind == 0: cash -= t.qty * t.cost; op[i] = t
            else: cash += t.qty; op.pop(i)
        res = sum(t.qty * t.cost for t in op.values()); f = res * rate / ann; cash -= f; cumf += f   # funding after events
        rows.append(dict(date=d.date().isoformat(), equity=cash + sum(t.qty for t in op.values()), funding=f, cum_funding=cumf, reserved=res))
    de = pd.DataFrame(rows); de["cumulative_return"] = de["equity"] / cap0; return de

rng = np.random.default_rng(0); t0 = pd.Timestamp("2024-01-01", tz="UTC"); rows = []
for p, h in [(p, h) for p in range(6) for h in range(0, 48, 6)]:
    ya = rng.uniform(0.3, 0.7); yb = ya + rng.normal(0, 0.06)
    rows.append(dict(ts=t0 + pd.Timedelta(hours=h), pair=f"P{p}", yes={"A": ya, "B": yb}, no={"A": 1.01 - ya, "B": 1.01 - yb}))
exits = {f"P{p}": t0 + pd.Timedelta(days=2 + p % 3, hours=5) for p in range(6)}; tr, log = allocate(pd.DataFrame(rows), exits, 1000.0, 0.02, 500)
de = daily_equity(tr, 1000.0, 0.05, 365); r = de["cumulative_return"].pct_change().iloc[1:]      # metrics skip the first day
print(len(tr), round(float(tr.net_pnl.sum()), 6), sorted({x[2] for x in log}), round(float(de.cum_funding.iloc[-1]), 6), round(float(de.equity.iloc[-1]), 6))
print(round(float(r.mean() * 365), 6), round(float(r.std(ddof=1) * math.sqrt(365)), 6), round(float((de.reserved / 1000.0).mean()), 6), len(de))
```
