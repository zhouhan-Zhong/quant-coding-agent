# memory-solution
# unit: t1-option-put-call-parity-forward-audit | category: derivatives-pricing | tags: put-call-parity, synthetic-forward, implied-borrow, quote-cleaning, bid-ask, audit, act365, fixed-decimals
# deliverables: results.json, parity_audit.csv, violations.csv, solution.json
# Clean each quote (first matching reason): one_sided (bid<=0 or ask<=0) -> crossed (bid>=ask) -> stale (age > max_quote_age_sec).
# Pair clean call+put on (expiry_ts, strike); clean quotes with no clean opposite side -> missing_pair. T = ACT/365 seconds-based.
# DF = exp(-r T); F_bid = K + (C_bid - P_ask)/DF; F_ask = K + (C_ask - P_bid)/DF; F_th = S exp((r - q - borrow) T);
# implied_borrow = r - q - ln(F_mid/S)/T; violation = max(F_bid - F_th, F_th - F_ask, 0); flagged only if > threshold.
# Every number is formatted f"{x:.6f}" in CSVs; JSON summaries (max / medians) are computed from those 6-dp values so they reconcile exactly.
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

rules = json.load(open(os.path.join(DATA_DIR, "audit_rules.json")))
carry = json.load(open(os.path.join(DATA_DIR, "carry_inputs.json")))
spot = json.load(open(os.path.join(DATA_DIR, "spot.json")))
q = pd.read_csv(os.path.join(DATA_DIR, "option_quotes.csv"))
r, dq, b0 = float(carry["risk_free_rate"]), float(carry["dividend_yield"]), float(carry["reference_borrow_rate"])
S, thr, max_age = float(spot["spot_price"]), float(rules["violation_threshold"]), float(rules["max_quote_age_sec"])
P = int(rules.get("float_precision", 6))
fx = lambda x: f"{x:.{P}f}"                     # fixed-point string
rd = lambda x: round(float(x), P)
iso = lambda ts: ts.strftime("%Y-%m-%dT%H:%M:%SZ")
qts = pd.to_datetime(spot["quote_ts"], utc=True)

q["expiry"] = pd.to_datetime(q["expiry_ts"], utc=True)
q["option_type"] = q["option_type"].astype(str)
def reason(row):
    if row["bid"] <= 0 or row["ask"] <= 0: return "one_sided"
    if row["bid"] >= row["ask"]: return "crossed"
    if row["quote_age_sec"] > max_age: return "stale"
    return "clean"
q["reason"] = q.apply(reason, axis=1)
clean = q[q["reason"] == "clean"]
calls = clean[clean["option_type"] == "call"].set_index(["expiry", "strike"])
puts = clean[clean["option_type"] == "put"].set_index(["expiry", "strike"])
keys = sorted(set(calls.index) & set(puts.index))           # (expiry asc, strike asc)
missing = int(sum((e, k) not in set(keys) for e, k in zip(clean["expiry"], clean["strike"])))

rows = []
for exp, K in keys:
    c, p = calls.loc[(exp, K)], puts.loc[(exp, K)]
    T = (exp - qts).total_seconds() / (365.0 * 86400.0)
    df = math.exp(-r * T)
    fb, fa = K + (c["bid"] - p["ask"]) / df, K + (c["ask"] - p["bid"]) / df
    fm = (fb + fa) / 2
    fth = S * math.exp((r - dq - b0) * T)
    borrow = r - dq - math.log(fm / S) / T
    viol = max(fb - fth, fth - fa, 0.0)
    vtype = "none"
    if viol > thr:
        vtype = "forward_too_high" if fth < fb else ("forward_too_low" if fth > fa else "none")
    rows.append({"expiry_ts": iso(exp), "strike": fx(K), "call_quote_id": c["quote_id"], "put_quote_id": p["quote_id"],
                 "time_to_expiry_years": fx(T), "call_bid": fx(c["bid"]), "call_ask": fx(c["ask"]), "put_bid": fx(p["bid"]),
                 "put_ask": fx(p["ask"]), "synthetic_forward_bid": fx(fb), "synthetic_forward_ask": fx(fa),
                 "synthetic_forward_mid": fx(fm), "theoretical_forward": fx(fth), "implied_borrow_rate": fx(borrow),
                 "violation_amount": fx(viol), "violation_type": vtype, "_viol": viol})

audit = pd.DataFrame(rows)
audit_cols = [c for c in audit.columns if not c.startswith("_")]
audit[audit_cols].to_csv(os.path.join(OUT_DIR, "parity_audit.csv"), index=False)
viol_cols = ["expiry_ts", "strike", "call_quote_id", "put_quote_id", "violation_type", "violation_amount",
             "synthetic_forward_bid", "synthetic_forward_ask", "theoretical_forward"]
flag = audit[audit["_viol"] > thr]
flag[viol_cols].to_csv(os.path.join(OUT_DIR, "violations.csv"), index=False)

pid = (audit["expiry_ts"] + "|" + audit["strike"]).tolist()
va = audit["violation_amount"].astype(float)
counts = {k: int((q["reason"] == k).sum()) for k in ["stale", "crossed", "one_sided"]}
results = {
    "schema_version": "1.0.0", "task_slug": "option-put-call-parity-forward-audit", "quote_ts": iso(qts),
    "spot_price": rd(S), "matched_pair_count": int(len(audit)), "parity_break_count": int(len(flag)),
    "worst_abs_violation": rd(va.max()),
    "median_synthetic_forward": rd(np.median(audit["synthetic_forward_mid"].astype(float))),
    "median_implied_borrow_rate": rd(np.median(audit["implied_borrow_rate"].astype(float))),
    "excluded_stale_count": counts["stale"], "excluded_crossed_count": counts["crossed"],
    "excluded_one_sided_count": counts["one_sided"], "excluded_missing_pair_count": missing,
}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
solution = {"intermediates": {
    "exclusion_counts": {"value": {**counts, "missing_pair": missing}},
    "matched_pair_ids": {"value": pid},
    "checkpoint_forwards": {"value": {k: float(v) for k, v in zip(pid, audit["synthetic_forward_mid"])}},
    "worst_violation_pair": {"value": pid[int(va.values.argmax())]},
}}
json.dump(solution, open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
