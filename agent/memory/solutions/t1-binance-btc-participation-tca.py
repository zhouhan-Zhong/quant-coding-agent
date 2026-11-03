# memory-solution
# unit: t1-binance-btc-participation-tca | category: microstructure | tags: tca, participation, vwap, implementation-shortfall, realized-spread, bucket-schedule, binance, futures, event-linked, fixed-decimals
# deliverables: results.json, fills.csv, solution.json
# Buckets: range(start, end, bucket_ms); bucket_end = min(start+bucket_ms, end); eligible trades start <= transact_time < end (file order).
# child = min(remaining, participation_cap * bucket_qty, max_child); remaining decremented only when child > 0; fill_qty == child.
# Fill benchmark = first quote with transaction_time >= bucket_start (searchsorted left) -> best ask price, bid/ask + update_id/tx/event links.
# Arrival = first quote at/after start_time_ms (mid); future = first quote at/after end_time_ms + realized_horizon_ms.
# All metric formulas read from tca_rules.json semantics: IS = 1e4*(avg-arr)/arr, eff = 2e4*(avg-arr)/arr, realized = 2e4*(avg-fut)/arr, temp = 1e4*(fut-arr)/arr.
# CSV: every float as f"{x:.6f}", ints/timestamps as plain digits, empty strings for missing fields, is_empty_bucket as true/false.
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

spec = json.load(open(os.path.join(DATA_DIR, "order_spec.json")))
rules = json.load(open(os.path.join(DATA_DIR, "tca_rules.json")))      # normative definitions (documented above)
q = pd.read_csv(os.path.join(DATA_DIR, "quotes.csv"))
t = pd.read_csv(os.path.join(DATA_DIR, "trades.csv"))
qt = q["transaction_time"].to_numpy(np.int64)
assert np.all(np.diff(qt) >= 0), "quotes must be sorted by transaction_time"
tt = t["transact_time"].to_numpy(np.int64)

start, end, bms = int(spec["start_time_ms"]), int(spec["end_time_ms"]), int(spec["bucket_ms"])
cap, max_child, target = float(spec["participation_cap"]), float(spec["max_child_quantity"]), float(spec["target_quantity"])
horizon = int(spec["realized_horizon_ms"])

def quote_at_or_after(ts):
    i = int(np.searchsorted(qt, ts, side="left"))
    r = q.iloc[i]
    return dict(update_id=int(r["update_id"]), bid=float(r["best_bid_price"]), ask=float(r["best_ask_price"]),
                tx=int(r["transaction_time"]), ev=int(r["event_time"]))

f6 = lambda x: f"{x:.6f}"
rows, remaining = [], target
for bi, bs in enumerate(range(start, end, bms)):
    be = min(bs + bms, end)
    m = (tt >= bs) & (tt < be)
    g = t.loc[m]
    n = int(m.sum())
    qty = float(g["quantity"].sum()); notional = float((g["price"] * g["quantity"]).sum())
    cap_qty = cap * qty
    child = min(remaining, cap_qty, max_child)
    fq = quote_at_or_after(bs)
    if child > 0:
        remaining -= child
    has = child > 0
    rows.append({
        "bucket_index": bi, "bucket_start_ms": bs, "bucket_end_ms": be,
        "eligible_trade_window_start_ms": bs, "eligible_trade_window_end_ms": be,
        "eligible_trade_count": n, "eligible_trade_qty": f6(qty), "eligible_trade_notional": f6(notional),
        "eligible_first_trade_id": int(g["agg_trade_id"].iloc[0]) if n else "",
        "eligible_last_trade_id": int(g["agg_trade_id"].iloc[-1]) if n else "",
        "eligible_first_trade_time_ms": int(g["transact_time"].iloc[0]) if n else "",
        "eligible_last_trade_time_ms": int(g["transact_time"].iloc[-1]) if n else "",
        "participation_cap_qty": f6(cap_qty), "max_child_qty": f6(max_child),
        "child_order_qty": f6(child), "fill_qty": f6(child),
        "fill_price": f6(fq["ask"]) if has else "", "reference_bid": f6(fq["bid"]) if has else "",
        "reference_ask": f6(fq["ask"]) if has else "",
        "fill_quote_update_id": fq["update_id"] if has else "",
        "fill_quote_transaction_time_ms": fq["tx"] if has else "",
        "fill_quote_event_time_ms": fq["ev"] if has else "",
        "remaining_qty": f6(remaining), "is_empty_bucket": "true" if n == 0 else "false",
        # raw numbers kept for the summary (dropped before writing)
        "_qty": qty, "_notional": notional, "_child": child, "_px": fq["ask"] if has else 0.0, "_tx": fq["tx"] if has else None,
    })

fills = pd.DataFrame(rows)
summary_cols = [c for c in fills.columns if not c.startswith("_")]
fills[summary_cols].to_csv(os.path.join(OUT_DIR, "fills.csv"), index=False)

arr, fut = quote_at_or_after(start), quote_at_or_after(end + horizon)
arr_mid, fut_mid = (arr["bid"] + arr["ask"]) / 2, (fut["bid"] + fut["ask"]) / 2
executed = float(fills["_child"].sum())
avg_px = float((fills["_child"] * fills["_px"]).sum() / executed)
mkt_vol, mkt_notional = float(fills["_qty"].sum()), float(fills["_notional"].sum())
filled = fills[fills["_child"] > 0]
R6 = lambda x: round(float(x), 6)
results = {
    "schema_version": "2.0.0", "task_slug": "binance-btc-participation-tca", "symbol": spec["symbol"], "side": spec["side"],
    "start_time_ms": start, "end_time_ms": end, "bucket_ms": bms,
    "target_qty": R6(target), "executed_qty": R6(executed), "unfilled_qty": R6(target - executed), "fill_rate": R6(executed / target),
    "arrival_mid": R6(arr_mid), "arrival_quote_update_id": arr["update_id"],
    "arrival_quote_transaction_time_ms": arr["tx"], "arrival_quote_event_time_ms": arr["ev"],
    "future_reference_time_ms": end + horizon, "future_mid": R6(fut_mid), "future_quote_update_id": fut["update_id"],
    "future_quote_transaction_time_ms": fut["tx"], "future_quote_event_time_ms": fut["ev"],
    "avg_fill_price": R6(avg_px), "market_vwap": R6(mkt_notional / mkt_vol),
    "implementation_shortfall_bps": R6(10000 * (avg_px - arr_mid) / arr_mid),
    "effective_spread_bps": R6(20000 * (avg_px - arr_mid) / arr_mid),
    "realized_spread_bps": R6(20000 * (avg_px - fut_mid) / arr_mid),
    "temporary_impact_bps": R6(10000 * (fut_mid - arr_mid) / arr_mid),
    "market_trade_count": int(fills["eligible_trade_count"].sum()), "market_volume": R6(mkt_vol),
    "bucket_count_total": int(len(fills)), "bucket_count_filled": int(len(filled)),
    "bucket_count_empty": int((fills["eligible_trade_count"] == 0).sum()),
}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

V = lambda x: {"value": x}
def checkpoint(i):
    r = rows[i]
    return {"bucket_start_ms": V(r["bucket_start_ms"]), "eligible_trade_count": V(r["eligible_trade_count"]),
            "eligible_trade_qty": V(R6(r["_qty"])), "child_order_qty": V(R6(r["_child"])),
            "remaining_qty": V(R6(float(r["remaining_qty"]))), "is_empty_bucket": V(r["eligible_trade_count"] == 0)}
solution = {
    "intermediates": {
        "bucket_count_total": V(results["bucket_count_total"]), "bucket_count_filled": V(results["bucket_count_filled"]),
        "bucket_count_empty": V(results["bucket_count_empty"]), "eligible_trade_count": V(results["market_trade_count"]),
        "eligible_trade_volume": V(R6(mkt_vol)), "eligible_trade_notional": V(R6(mkt_notional)),
        "first_fill_quote_transaction_time_ms": V(int(filled["_tx"].iloc[0])),
        "last_fill_quote_transaction_time_ms": V(int(filled["_tx"].iloc[-1])),
        "arrival_quote_transaction_time_ms": V(arr["tx"]), "arrival_quote_event_time_ms": V(arr["ev"]),
        "future_quote_transaction_time_ms": V(fut["tx"]), "future_quote_event_time_ms": V(fut["ev"]),
    },
    "checkpoints": {f"bucket_{i}": checkpoint(i) for i in (0, 4, 9) if i < len(rows)},
}
json.dump(solution, open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
