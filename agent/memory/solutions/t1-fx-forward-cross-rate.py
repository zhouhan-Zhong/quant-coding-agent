# memory-solution
# unit: t1-fx-forward-cross-rate | category: fx-commodities-crypto | tags: fx-forward, covered-interest-parity, cross-rates, bid-ask, day-count, forward-points, mtm, pnl-attribution, net-delta
# deliverables: results.json
# CIP with simple money-market deposits: F = S * (1 + r_q*tau_q) / (1 + r_b*tau_b); tau = days/360 or days/365 per CURRENCY
#   (quote and base legs can have different day counts, e.g. EUR/GBP). Forward bid/ask = spot bid/ask * same factor.
# points = (F - S) * pip_multiplier (params.json). Cross XXX/YYY = XXX/USD * USD/YYY: mid*mid, bid*bid, ask*ask; a leg quoted
#   USD/ZZZ is inverted with bid/ask flipped (ZZZ/USD bid = 1/ask). Same recipe with tenor forwards for forward crosses.
# MTM: pnl_quote = sign*N_base*(F_tenor - K), sign=+1 buy_base/-1 sell_base; quote->USD at SPOT mid (not forward), no discounting.
# Attribution (exact sum): spot = sign*N*(S_now - S_hist), carry = sign*N*(F - S_now), residual = sign*N*(S_hist - K); hist cross = product of hist USD legs.
# Net delta: base leg = sign*N_base, quote leg = -sign*N_base*F_mid (current forward), each converted to USD at spot mid.
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

spot = pd.read_csv(os.path.join(DATA_DIR, "spot_rates.csv"))
spot = spot[spot["bid"] < spot["ask"]].drop_duplicates("pair")          # drop stale rows (bid >= ask)
dep = pd.read_csv(os.path.join(DATA_DIR, "deposit_rates.csv")).set_index("currency")
port = pd.read_csv(os.path.join(DATA_DIR, "portfolio.csv"))
hist = pd.read_csv(os.path.join(DATA_DIR, "historical_spots.csv")).set_index("pair")["mid"].to_dict()
params = json.load(open(os.path.join(DATA_DIR, "params.json")))
TENORS, tenor_days, pips = ["1M", "3M", "6M"], params["tenor_days"], params["pip_multipliers"]

S = {r.pair: {"bid": float(r.bid), "ask": float(r.ask), "mid": 0.5 * (float(r.bid) + float(r.ask))} for r in spot.itertuples()}
ccy_pairs = {p: p.split("/") for p in S}

def growth(ccy, tenor):                     # 1 + r * tau with the currency's own day count
    basis = 360.0 if dep.loc[ccy, "day_count"].strip().upper() == "ACT/360" else 365.0
    return 1.0 + dep.loc[ccy, f"tenor_{tenor.lower()}"] / 100.0 * tenor_days[tenor] / basis

def fwd(pair, tenor):                        # CIP: F = S * (1 + r_quote tau_q) / (1 + r_base tau_b), applied to bid/ask/mid
    b, q = ccy_pairs[pair]
    f = growth(q, tenor) / growth(b, tenor)
    return {k: S[pair][k] * f for k in ("bid", "ask", "mid")}

F = {p: {t: fwd(p, t) for t in TENORS} for p in S}
forward_rates = {p: {t: {k: round(v, 6) for k, v in F[p][t].items()} for t in TENORS} for p in S}
forward_points = {p: {t: {k: round((F[p][t][k] - S[p][k]) * pips[p], 2) for k in ("bid", "ask", "mid")} for t in TENORS} for p in S}

def leg_vs_usd(ccy, quotes, want_base):     # return CCY/USD (want_base=True) or USD/CCY bid/ask/mid from the available USD quote
    direct = f"{ccy}/USD" if want_base else f"USD/{ccy}"
    inverse = f"USD/{ccy}" if want_base else f"{ccy}/USD"
    if direct in quotes:
        return dict(quotes[direct])
    qq = quotes[inverse]
    return {"bid": 1.0 / qq["ask"], "ask": 1.0 / qq["bid"], "mid": 1.0 / qq["mid"]}

def cross(pair, quotes):                     # XXX/YYY = XXX/USD * USD/YYY, bid*bid / ask*ask / mid*mid
    x, y = pair.split("/")
    a, c = leg_vs_usd(x, quotes, True), leg_vs_usd(y, quotes, False)
    return {k: a[k] * c[k] for k in ("bid", "ask", "mid")}

cross_pairs = params["cross_rates_to_compute"]
X = {p: cross(p, S) for p in cross_pairs}
XF3 = {p: cross(p, {q: F[q]["3M"] for q in F}) for p in cross_pairs}
cross_rates = {p: {k: round(v, 4) for k, v in X[p].items()} for p in cross_pairs}
forward_cross_3m = {p: {"3M": {k: round(v, 4) for k, v in XF3[p].items()}} for p in cross_pairs}

# CIP sanity: implied quote-ccy deposit rate backed out of the published (rounded) 3M forward vs the actual deposit, in bps
cip_dev = {}
for p in S:
    b, q = ccy_pairs[p]
    tau_q = (growth(q, "3M") - 1.0) / (dep.loc[q, "tenor_3m"] / 100.0)
    r_impl = ((forward_rates[p]["3M"]["mid"] / S[p]["mid"]) * growth(b, "3M") - 1.0) / tau_q
    cip_dev[p] = round(float((r_impl - dep.loc[q, "tenor_3m"] / 100.0) * 1e4), 6)

def to_usd(ccy):                             # USD value of one unit of ccy at spot mid
    return 1.0 if ccy == "USD" else leg_vs_usd(ccy, S, True)["mid"]

def spot_mid(pair):
    return S[pair]["mid"] if pair in S else cross(pair, S)["mid"]

def fwd_mid(pair, tenor):
    return F[pair][tenor]["mid"] if pair in S else cross(pair, {q: F[q][tenor] for q in F})["mid"]

def hist_mid(pair):
    if pair in hist:
        return hist[pair]
    x, y = pair.split("/")
    hx = hist[f"{x}/USD"] if f"{x}/USD" in hist else 1.0 / hist[f"USD/{x}"]
    hy = hist[f"USD/{y}"] if f"USD/{y}" in hist else 1.0 / hist[f"{y}/USD"]
    return hx * hy

mtm, attrib, net = {}, {}, {}
for tr in port.itertuples():
    base, quote = tr.pair.split("/")
    sgn, N, K = (1.0 if tr.direction == "buy_base" else -1.0), float(tr.notional_base), float(tr.trade_rate)
    Fm, Sm, Sh, fx = fwd_mid(tr.pair, tr.tenor), spot_mid(tr.pair), hist_mid(tr.pair), to_usd(quote)
    pnl_q = sgn * N * (Fm - K)
    mtm[tr.trade_id] = {"current_forward_mid": round(Fm, 6), "pnl_quote_ccy": round(pnl_q, 2), "pnl_usd": round(pnl_q * fx, 2)}
    sd, cy = sgn * N * (Sm - Sh) * fx, sgn * N * (Fm - Sm) * fx
    tot = pnl_q * fx
    attrib[tr.trade_id] = {"spot_delta_usd": round(sd, 2), "carry_usd": round(cy, 2), "residual_usd": round(tot - sd - cy, 2), "total_usd": round(tot, 2)}
    net[base] = net.get(base, 0.0) + sgn * N * to_usd(base)
    net[quote] = net.get(quote, 0.0) - sgn * N * Fm * fx
net_delta = {c: round(v, 2) for c, v in sorted(net.items())}

results = {"forward_rates": forward_rates, "forward_points": forward_points, "cross_rates": cross_rates,
           "forward_cross_rates_3M": forward_cross_3m, "cip_deviations_bps": cip_dev, "portfolio_mtm": mtm,
           "pnl_attribution": attrib, "net_delta_usd": net_delta}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
