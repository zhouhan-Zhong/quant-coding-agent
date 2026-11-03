# memory-solution
# unit: t1-prediction-markets-cross-venue-dislocation | category: backtest-factors | tags: prediction-markets, kalshi, polymarket, cross-venue, locked-edge, arbitrage, dedupe, cutoff, capital-allocation, daily-equity, funding-cost, event-accounting
# deliverables: results.json, solution.json, trade_log.csv, candidate_snapshots.csv, daily_equity.csv
# Cleaning order (counts at every step): approved pairs -> keep their snapshots -> dedupe (pair,venue,ts) keep max source_sequence -> status=='active'
#   -> both prices in (0,1) exclusive -> ts < cutoff = min(kalshi_close, poly_close, resolution) -> synchronized = both venues present.
# Directions: A = buy YES Kalshi + NO Polymarket, B = buy YES Polymarket + NO Kalshi; fee = bps/1e4 * leg price; total_fee = fy + fn;
#   net = 1 - gross - total_fee, entry_cost = gross + total_fee (this exact float op order decides edge-rank ties between pairs!);
#   best = sort by (-net, direction string). Tie counted when |netA - netB| <= 1e-12 (before the threshold test).
# Per timestamp: release resolutions (exit <= ts), evaluate pairs without a trade (BELOW_THRESHOLD rows use start-of-group cash), rank actionable by
#   (-net, entry_cost, pair_id), qty = min(max_contracts, floor(cash/entry_cost)); qty < 1 -> UNAFFORDABLE. One trade per pair, exit = resolution_ts.
# Daily equity: calendar days first entry date..last exit date (UTC); apply events with ts < day end (entry before exit); funding = reserved*rate/ann
#   charged after events; equity = cash + unresolved qty; daily_pnl = diff (first 0); metrics from pct_change of cumulative_return[1:], ddof=1.
# Timestamps written as isoformat with 'Z'; floats rounded to 6 dp.
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
R6 = lambda x: round(float(x), 6)
iso = lambda ts: ts.isoformat().replace("+00:00", "Z")
DIR_A, DIR_B = "BUY_YES_KALSHI_BUY_NO_POLYMARKET", "BUY_YES_POLYMARKET_BUY_NO_KALSHI"

fees = json.load(open(os.path.join(DATA_DIR, "fees.json")))
FEE = {"KALSHI": float(fees["kalshi_fee_bps"]), "POLYMARKET": float(fees["polymarket_fee_bps"])}
min_edge, max_q, cap0 = float(fees["min_edge_bps"]) / 1e4, int(fees["max_contracts_per_trade"]), float(fees["capital_base"])
fund_rate, ann = float(fees.get("funding_rate", 0.0)), float(fees["annualization_factor"])

pairs = pd.read_csv(os.path.join(DATA_DIR, "market_pairs.csv"))
appr = pairs[pairs["review_status"] == "approved"].copy()
for c in ["kalshi_close_ts_utc", "polymarket_close_ts_utc", "resolution_ts_utc"]:
    appr[c] = pd.to_datetime(appr[c], utc=True)
appr["cutoff"] = appr[["kalshi_close_ts_utc", "polymarket_close_ts_utc", "resolution_ts_utc"]].min(axis=1)
appr["resolution_outcome"] = appr["resolution_outcome"].astype(str).str.upper()
meta = appr.set_index("pair_id")

px = pd.read_csv(os.path.join(DATA_DIR, "price_snapshots.csv"))
n_raw = len(px)
px["snapshot_ts_utc"] = pd.to_datetime(px["snapshot_ts_utc"], utc=True)
px["venue"] = px["venue"].astype(str).str.upper(); px["status"] = px["status"].astype(str).str.lower()
px["source_sequence"] = pd.to_numeric(px["source_sequence"], errors="coerce").fillna(-1)
for c in ["canon_yes_price", "canon_no_price"]:
    px[c] = pd.to_numeric(px[c], errors="coerce")
px = px[px["pair_id"].isin(set(appr["pair_id"]))]
key = ["pair_id", "venue", "snapshot_ts_utc"]
n_dup_groups = int((px.groupby(key).size() > 1).sum())
dedup = px.sort_values(key + ["source_sequence"]).groupby(key, as_index=False).tail(1)
n_inactive = int((dedup["status"] != "active").sum())
act = dedup[dedup["status"] == "active"]
valid = act["canon_yes_price"].between(0, 1, inclusive="neither") & act["canon_no_price"].between(0, 1, inclusive="neither")
n_invalid = int((~valid).sum())
clean = act[valid].merge(appr[["pair_id", "cutoff"]], on="pair_id")
n_cutoff = int((clean["snapshot_ts_utc"] >= clean["cutoff"]).sum())
clean = clean[clean["snapshot_ts_utc"] < clean["cutoff"]].drop(columns="cutoff")
nven = clean.groupby(["pair_id", "snapshot_ts_utc"])["venue"].transform("nunique")
sync = clean[nven == 2].pivot_table(index=["pair_id", "snapshot_ts_utc"], columns="venue",
                                    values=["canon_yes_price", "canon_no_price"], aggfunc="first")
sync.columns = [f"{m}_{v}" for m, v in sync.columns]
sync = sync.reset_index().sort_values(["snapshot_ts_utc", "pair_id"]).reset_index(drop=True)

def directions(s):
    out = []
    for d, yv, nv, yp, npx in [(DIR_A, "KALSHI", "POLYMARKET", s["canon_yes_price_KALSHI"], s["canon_no_price_POLYMARKET"]),
                               (DIR_B, "POLYMARKET", "KALSHI", s["canon_yes_price_POLYMARKET"], s["canon_no_price_KALSHI"])]:
        fy, fn = FEE[yv] / 1e4 * float(yp), FEE[nv] / 1e4 * float(npx)
        gc, tf = float(yp) + float(npx), fy + fn             # exact spec op order: total_fee first (ties are float-exact!)
        out.append(dict(direction=d, yes_venue=yv, no_venue=nv, yes_price=yp, no_price=npx, gross_cost=gc, fee_yes=fy, fee_no=fn,
                        entry_cost=gc + tf, gross_edge=1.0 - gc, net_edge=1.0 - gc - tf))
    return sorted(out, key=lambda x: (-x["net_edge"], x["direction"]))

def cand_row(s, ds, status, reason, before, after, qty):
    best, alt = ds[0], ds[1]; a = ds[0] if ds[0]["direction"] == DIR_A else ds[1]; b = ds[1] if a is ds[0] else ds[0]
    return dict(pair_id=s["pair_id"], snapshot_ts_utc=iso(s["snapshot_ts_utc"]), best_direction=best["direction"], candidate_status=status,
                selected_reason=reason, available_cash_before=R6(before), available_cash_after=R6(after), gross_cost_a=R6(a["gross_cost"]),
                net_edge_a=R6(a["net_edge"]), gross_cost_b=R6(b["gross_cost"]), net_edge_b=R6(b["net_edge"]), net_locked_edge=R6(best["net_edge"]),
                gross_locked_edge=R6(best["gross_edge"]), entry_cost_per_unit=R6(best["entry_cost"]), qty=int(qty), yes_price=R6(best["yes_price"]),
                no_price=R6(best["no_price"]), alt_direction=alt["direction"], alt_net_locked_edge=R6(alt["net_edge"]))

cash, opened, active, trades, acct, cands = cap0, set(), {}, [], [], []
n_below = n_tie = n_unaff = 0; signal_pairs = set()
for ts, grp in sync.groupby("snapshot_ts_utc", sort=True):
    for pid in sorted([p for p, t in active.items() if t["exit"] <= ts]):
        cash += active.pop(pid)["qty"]
    actionable = []
    for _, s in grp.sort_values("pair_id").iterrows():
        if s["pair_id"] in opened:
            continue
        ds = directions(s)
        if abs(ds[0]["net_edge"] - ds[1]["net_edge"]) <= 1e-12:
            n_tie += 1
        if ds[0]["net_edge"] < min_edge:
            n_below += 1; cands.append(cand_row(s, ds, "BELOW_THRESHOLD", "BELOW_MIN_EDGE", cash, cash, 0)); continue
        signal_pairs.add(s["pair_id"]); actionable.append((s, ds))
    actionable.sort(key=lambda x: (-x[1][0]["net_edge"], x[1][0]["entry_cost"], x[0]["pair_id"]))
    for s, ds in actionable:
        best, pid = ds[0], s["pair_id"]
        qty = min(max_q, math.floor(cash / best["entry_cost"]))
        if qty < 1:
            n_unaff += 1; cands.append(cand_row(s, ds, "UNAFFORDABLE", "INSUFFICIENT_CASH", cash, cash, 0)); continue
        before = cash; cash -= qty * best["entry_cost"]; opened.add(pid)
        exit_ts = meta.loc[pid, "resolution_ts_utc"]
        active[pid] = {"qty": qty, "exit": exit_ts}
        acct.append(dict(pair_id=pid, entry=ts, exit=exit_ts, qty=qty, cost=best["entry_cost"]))
        trades.append(dict(pair_id=pid, entry_ts_utc=iso(ts), exit_ts_utc=iso(exit_ts), direction=best["direction"], qty=int(qty),
                           buy_yes_venue=best["yes_venue"], buy_no_venue=best["no_venue"], yes_price=R6(best["yes_price"]), no_price=R6(best["no_price"]),
                           fee_yes=R6(qty * best["fee_yes"]), fee_no=R6(qty * best["fee_no"]), gross_cost=R6(qty * best["gross_cost"]),
                           entry_cost_per_unit=R6(best["entry_cost"]), gross_locked_edge=R6(best["gross_edge"]), net_locked_edge=R6(best["net_edge"]),
                           gross_pnl=R6(qty * best["gross_edge"]), net_pnl=R6(qty * best["net_edge"]), resolution_outcome=meta.loc[pid, "resolution_outcome"]))
        cands.append(cand_row(s, ds, "OPENED", "ALLOCATED_BY_EDGE_RANK", before, cash, qty))

tl = pd.DataFrame(trades).sort_values(["entry_ts_utc", "pair_id"]).reset_index(drop=True)
tl.to_csv(os.path.join(OUT_DIR, "trade_log.csv"), index=False)
pd.DataFrame(cands).sort_values(["snapshot_ts_utc", "pair_id"]).to_csv(os.path.join(OUT_DIR, "candidate_snapshots.csv"), index=False)

# daily equity from entry / resolution events
events = sorted([(t["entry"], 0, t["pair_id"]) for t in acct] + [(t["exit"], 1, t["pair_id"]) for t in acct])
by_pid = {t["pair_id"]: t for t in acct}
days = pd.date_range(min(t["entry"] for t in acct).date(), max(t["exit"] for t in acct).date(), freq="D", tz="UTC")
cash_t, open_t, ptr, cum_f, rows = cap0, {}, 0, 0.0, []
for d in days:
    end = d + pd.Timedelta(days=1)
    while ptr < len(events) and events[ptr][0] < end:
        _, kind, pid = events[ptr]; t = by_pid[pid]; ptr += 1
        if kind == 0: cash_t -= t["qty"] * t["cost"]; open_t[pid] = t
        else: cash_t += t["qty"]; open_t.pop(pid, None)
    recv = sum(t["qty"] for t in open_t.values()); reserved = sum(t["qty"] * t["cost"] for t in open_t.values())
    f = reserved * fund_rate / ann; cash_t -= f; cum_f += f
    rows.append(dict(date=d.date().isoformat(), equity=cash_t + recv, daily_funding_cost=f, cumulative_funding_cost=cum_f, reserved=reserved))
de = pd.DataFrame(rows)
de["daily_pnl"] = de["equity"].diff().fillna(0.0); de["cumulative_return"] = de["equity"] / cap0
cols = ["equity", "daily_pnl", "cumulative_return", "daily_funding_cost", "cumulative_funding_cost", "reserved"]
de[cols] = de[cols].round(6)
de[["date"] + cols[:-1]].to_csv(os.path.join(OUT_DIR, "daily_equity.csv"), index=False)

dr = de["cumulative_return"].pct_change().fillna(0.0).iloc[1:]
ann_ret = float(dr.mean() * ann); ann_vol = float(dr.std(ddof=1) * math.sqrt(ann)) if len(dr) > 1 else 0.0
runmax = de["cumulative_return"].cummax(); mdd = float(((runmax - de["cumulative_return"]) / runmax).max())
results = {"num_valid_pairs": int(len(appr)), "num_clean_snapshots": int(len(clean)), "num_synchronized_snapshots": int(len(sync)),
           "num_trades": int(len(tl)), "gross_edge_bps_mean": R6(tl["gross_locked_edge"].mean() * 1e4), "net_edge_bps_mean": R6(tl["net_locked_edge"].mean() * 1e4),
           "gross_locked_pnl": R6(tl["gross_pnl"].sum()), "net_locked_pnl": R6(tl["net_pnl"].sum()),
           "total_funding_cost": R6(de["cumulative_funding_cost"].iloc[-1]), "annualized_return": R6(ann_ret), "annualized_volatility": R6(ann_vol),
           "sharpe_ratio": R6(ann_ret / ann_vol if ann_vol > 0 else 0.0), "max_drawdown": R6(mdd), "capacity_utilization": R6((de["reserved"] / cap0).mean())}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
V = lambda x: {"value": x}
inter = {"num_raw_snapshots": V(int(n_raw)), "num_deduped_snapshots": V(int(len(dedup))), "num_active_snapshots": V(int(len(act))),
         "num_synchronized_snapshots": V(int(len(sync))), "num_rejected_pairs_filtered": V(int(len(pairs) - len(appr))),
         "num_duplicate_snapshot_groups": V(n_dup_groups), "num_invalid_price_rows_filtered": V(n_invalid), "num_inactive_rows_filtered": V(n_inactive),
         "num_cutoff_excluded_rows": V(n_cutoff), "num_candidate_signals": V(len(signal_pairs)), "num_below_threshold_synchronized_snapshots": V(n_below),
         "num_tie_break_candidates": V(n_tie), "num_unaffordable_candidate_signals": V(n_unaff), "num_trades": V(int(len(tl))),
         "mean_qty": V(R6(tl["qty"].mean())), "mean_gross_locked_edge": V(R6(tl["gross_locked_edge"].mean())),
         "mean_net_locked_edge": V(R6(tl["net_locked_edge"].mean())), "gross_locked_pnl": V(results["gross_locked_pnl"]),
         "net_locked_pnl": V(results["net_locked_pnl"]), "total_funding_cost": V(results["total_funding_cost"]),
         "mean_entry_cost_per_unit": V(R6(tl["entry_cost_per_unit"].mean()))}
json.dump({"intermediates": inter}, open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
