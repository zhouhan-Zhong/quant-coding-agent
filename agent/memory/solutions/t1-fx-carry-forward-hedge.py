# memory-solution
# unit: t1-fx-carry-forward-hedge | category: fx-commodities-crypto | tags: fx-carry, cip-forward, settlement-dates, bid-offer, implied-yield, ndf-basis, fx-swap-roll, garman-kohlhagen, smile-25d, premium-adjusted-delta, forward-hedge, risk-metrics
# deliverables: cross_rates.json, date_diagnostics.json, forward_rates.json, implied_yields.json, ndf_analysis.json, carry_signals.csv, roll_costs.csv, portfolio_returns.csv, smile_reconstruction.json, option_prices.json, greeks.json, risk_metrics.json, equity_correlation.csv
# Dates: good day = weekday and not a holiday of EITHER currency; spot = trade + lag good days (USD Labor Day pushes EURUSD T+2 to 09-06);
#   delivery = spot + n months, following adjustment. CIP simple deposits F = S(1 + r_terms t_terms)/(1 + r_base t_base), t = days/basis(ccy).
# Implied yield from market forward (spot mid): XXXUSD r_f = ((S/F)(1 + r_usd t_usd) - 1)/t_f, USDXXX r_f = ((F/S)(1 + r_usd t_usd) - 1)/t_f;
#   sides pinned by the instruction (bid: USDXXX fwd_bid+usd_offer, XXXUSD fwd_offer+usd_bid). Swap roll: side from far leg, points = far - near.
# Carry: daily signal = ffilled monthly 3M deposit (ccy - USD) pct; month-end signal (last spot row of month), active next row; long top3 / short bottom3.
#   Daily ccy return = pair return x (+1 XXXUSD / -1 USDXXX) + (r_ccy - r_usd)/100/252; legs are equal-weight means; Sharpe ~0.253.
# Hedge: signal = start date then month-ends of Mar/Jun/Sep/Dec, effective next row, basket = long ccys of latest carry signal <= hedge signal,
#   sell ccy forward 3M (CIP on monthly mids), hedge return F/S-1 (XXXUSD) or S/F-1 (USDXXX), amortised over rows in [effective, next effective).
# Options: 25d vols = atm + bf +- rr/2; T = (expiry - trade)/365 for vol AND discounting; strike from the delta convention
#   (spot: e^{-rf T}N(d1); forward: N(d1); spot premium-adjusted put: -e^{-rf T}(K/F)N(-d2) via brentq); GK prices in terms per base.
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
J = lambda n: json.load(open(os.path.join(DATA_DIR, n)))
dump = lambda obj, n: json.dump(obj, open(os.path.join(OUT_DIR, n), "w"), indent=2)
iso = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")

params, conv, basis, hol = J("params.json"), J("pair_conventions.json")["pairs"], J("day_counts.json"), J("holiday_calendar.json")["holidays"]
sbo, dbo, mf, ndf, opts = J("spot_bidoffer.json"), J("deposit_bidoffer.json"), J("market_forwards.json"), J("ndf_quotes.json"), J("option_params.json")
HOL = {c: {pd.Timestamp(d) for d in v} for c, v in hol.items()}
BAS = {c: 365.0 if v.upper().endswith("365") else 360.0 for c, v in basis.items()}
PAIRS = list(conv)
TD = pd.Timestamp(sbo["snapshot_date"])

def good(d, b, q): return d.weekday() < 5 and d not in HOL.get(b, set()) and d not in HOL.get(q, set())
def add_bd(d, n, b, q):
    while n > 0:
        d += pd.Timedelta(days=1)
        if good(d, b, q): n -= 1
    return d
def following(d, b, q):
    while not good(d, b, q): d += pd.Timedelta(days=1)
    return d
def spot_date(pair, trade):
    c = conv[pair]; return add_bd(trade, c["spot_lag_business_days"], c["base"], c["terms"])
def delivery(pair, sd, months): c = conv[pair]; return following(sd + pd.DateOffset(months=months), c["base"], c["terms"])
def cip(S, rb, rt, days, b, q): return S * (1 + rt / 100 * days / BAS[q]) / (1 + rb / 100 * days / BAS[b])
mid = lambda x: 0.5 * (x[0] + x[1])

# ---- cross rates (bid*bid / offer*offer, division swaps sides)
EJ = {"bid": sbo["EURUSD"][0] * sbo["USDJPY"][0], "offer": sbo["EURUSD"][1] * sbo["USDJPY"][1]}
CJ = {"bid": sbo["USDJPY"][0] / sbo["USDCHF"][1], "offer": sbo["USDJPY"][1] / sbo["USDCHF"][0]}
GC = {"bid": sbo["GBPUSD"][0] * sbo["USDCHF"][0], "offer": sbo["GBPUSD"][1] * sbo["USDCHF"][1]}
dump({k: {**v, "mid": 0.5 * (v["bid"] + v["offer"])} for k, v in {"EURJPY": EJ, "CHFJPY": CJ, "GBPCHF": GC}.items()}, "cross_rates.json")

# ---- forward diagnostics 9 pairs x 3 tenors
fwd = {}
for p in PAIRS:
    b, q = conv[p]["base"], conv[p]["terms"]; S = mid(sbo[p]); sd = spot_date(p, TD); fwd[p] = {}
    for ten, m in dbo["tenor_months"].items():
        dd = delivery(p, sd, m); days = (dd - sd).days; rb, rt = mid(dbo["rates"][b][ten]), mid(dbo["rates"][q][ten])
        F = cip(S, rb, rt, days, b, q)
        fwd[p][ten] = {"trade_date": iso(TD), "spot_date": iso(sd), "delivery_date": iso(dd), "forward_mid": F, "points": F - S,
                       "points_pips": (F - S) * conv[p]["point_multiplier"], "days_from_spot": days, "t_base": days / BAS[b],
                       "t_terms": days / BAS[q], "spot_mid": S, "r_base_mid_pct": rb, "r_terms_mid_pct": rt}
dump(fwd, "forward_rates.json")

# ---- date diagnostics: spot dates, broken date, swap rolls
bk = mf["broken_date_forwards"]["GBPUSD_6W"]; p = bk["pair"]; b, q = conv[p]["base"], conv[p]["terms"]
sd = spot_date(p, TD); dd = pd.Timestamp(bk["delivery_date"]); days = (dd - sd).days
broken = {"spot_date": iso(sd), "delivery_date": iso(dd), "days_from_spot": days, "t_base": days / BAS[b], "t_terms": days / BAS[q],
          "cip_forward_mid": cip(bk["spot_mid"], mid(dbo["rates"][b]["3M"]), mid(dbo["rates"][q]["3M"]), days, b, q)}
rolls = []
for rc in mf["roll_cases"]:
    side = "bid" if rc["position"] == "short_base_forward" else "offer"
    S = rc[f"spot_{side}"]; near, far = rc[f"near_points_{side}"], rc[f"far_points_{side}"]
    rolls.append({"case_id": rc["case_id"], "pair": rc["pair"], "side_used": side, "near_outright": S + near / rc["multiplier"],
                  "far_outright": S + far / rc["multiplier"], "swap_points": float(far - near)})
dump({"spot_dates": {p: {"trade_date": iso(TD), "spot_lag_business_days": conv[p]["spot_lag_business_days"], "spot_date": iso(spot_date(p, TD))} for p in PAIRS},
      "broken_dates": {"GBPUSD_6W": broken}, "roll_cases": rolls}, "date_diagnostics.json")

# ---- implied foreign yields from deliverable market forwards (spot mid, pinned sides)
def impl(p, S, F, r_usd, days, fccy):
    t_usd, t_f = days / 360.0, days / BAS[fccy]; ratio = (F / S) if p.startswith("USD") else (S / F)
    return (ratio * (1 + r_usd / 100 * t_usd) - 1) / t_f * 100
iy = {}
for p, d in mf["deliverable_forwards"].items():
    fccy = conv[p]["terms"] if p.startswith("USD") else conv[p]["base"]; sd = spot_date(p, TD); iy[p] = {}
    for ten in ("3M", "6M"):
        dd = pd.Timestamp(d[ten]["delivery_date"]); days = (dd - sd).days; ub, uo = dbo["rates"]["USD"][ten]
        if p.startswith("USD"): rb, ro = impl(p, d["spot_mid"], d[ten]["forward_bid"], uo, days, fccy), impl(p, d["spot_mid"], d[ten]["forward_offer"], ub, days, fccy)
        else: rb, ro = impl(p, d["spot_mid"], d[ten]["forward_offer"], ub, days, fccy), impl(p, d["spot_mid"], d[ten]["forward_bid"], uo, days, fccy)
        iy[p][ten] = {"spot_date": iso(sd), "delivery_date": iso(dd), "implied_foreign_rate_bid_pct": rb, "implied_foreign_rate_offer_pct": ro, "foreign_currency": fccy}
dump(iy, "implied_yields.json")

# ---- NDF: CIP with given year fractions vs market forward
c = ndf["cases"][0]; S, t = c["spot_mid"], c["t_usd"]
cb = S * (1 + c["local_rate_bid_pct"] / 100 * c["t_local"]) / (1 + c["usd_rate_offer_pct"] / 100 * t)
co = S * (1 + c["local_rate_offer_pct"] / 100 * c["t_local"]) / (1 + c["usd_rate_bid_pct"] / 100 * t)
il = lambda F, ru: ((F / S) * (1 + ru / 100 * t) - 1) / c["t_local"] * 100
dump({c["pair"]: {"trade_date": c["trade_date"], "fixing_date": c["fixing_date"], "settlement_date": c["settlement_date"], "local_currency": c["local_currency"],
                  "cip_forward_bid": cb, "cip_forward_offer": co, "market_forward_bid": c["market_forward_bid"], "market_forward_offer": c["market_forward_offer"],
                  "implied_local_rate_bid_pct": il(c["market_forward_bid"], c["usd_rate_offer_pct"]), "implied_local_rate_offer_pct": il(c["market_forward_offer"], c["usd_rate_bid_pct"]),
                  "market_basis_bid_pips": (c["market_forward_bid"] - cb) * c["point_multiplier"], "market_basis_offer_pips": (c["market_forward_offer"] - co) * c["point_multiplier"]}}, "ndf_analysis.json")

# ---- carry strategy
spot = pd.read_csv(os.path.join(DATA_DIR, "spot_rates.csv"), parse_dates=["date"]).set_index("date")
dep = pd.read_csv(os.path.join(DATA_DIR, "deposit_rates.csv"), parse_dates=["date"]).set_index("date")
CCY = ["EUR", "GBP", "JPY", "AUD", "NZD", "CHF", "CAD", "NOK", "SEK"]
PAIR = {(conv[p]["terms"] if p.startswith("USD") else conv[p]["base"]): p for p in PAIRS}          # foreign ccy -> pair
ct, hp, rp = params["carry_trade"], params["forward_hedge"], params["risk_metrics"]
dates = spot.index; depd = dep.reindex(dates.union(dep.index)).ffill().reindex(dates)
sig = depd[CCY].sub(depd["USD"], axis=0)
rng = dates[(dates >= ct["start_date"]) & (dates <= ct["end_date"])]
sig.loc[rng].reset_index().rename(columns={"index": "date"}).assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_csv(os.path.join(OUT_DIR, "carry_signals.csv"), index=False)
sgn = {cc: (1.0 if PAIR[cc].startswith(cc) else -1.0) for cc in CCY}
cret = pd.DataFrame({cc: (spot[PAIR[cc]] / spot[PAIR[cc]].shift(1) - 1) * sgn[cc] + (depd[cc] - depd["USD"]) / 100 / rp["annualization_factor"] for cc in CCY})
sret = pd.DataFrame({cc: (spot[PAIR[cc]] / spot[PAIR[cc]].shift(1) - 1) * sgn[cc] for cc in CCY})
me = dates.to_series().groupby(dates.to_period("M")).max()
sig_dates = me[(me < rng[0]) & (me >= dates[0])].tail(1).tolist() + me[(me >= rng[0]) & (me <= rng[-1])].tolist()
basket = {s: (list(sig.loc[s].sort_values(ascending=False).index[:ct["top_n"]]), list(sig.loc[s].sort_values().index[:ct["bottom_n"]])) for s in sig_dates}
pos = dates.get_indexer(sig_dates)
def active(t, sdates, lag):                               # latest signal whose lag has elapsed by row t
    i = dates.get_loc(t); cands = [s for s, k in zip(sdates, dates.get_indexer(sdates)) if k + lag <= i]
    return cands[-1] if cands else None

hs = [rng[0]] + [d for d in me[(me > rng[0]) & (me <= rng[-1])] if d.month in hp["roll_signal_months"]]
hs = [d for d in hs if dates.get_loc(d) + hp["signal_lag_business_days"] < len(dates)]
eff = {s: dates[dates.get_loc(s) + hp["signal_lag_business_days"]] for s in hs}
rows, hbask, hret = [], {}, {}
for k, s in enumerate(hs):
    cs = active(s, sig_dates, 0); longs = basket[cs][0]; hbask[s] = longs; hret[s] = {}
    nxt = eff[hs[k + 1]] if k + 1 < len(hs) else rng[-1] + pd.Timedelta(days=1)
    win = int(((dates >= eff[s]) & (dates < nxt)).sum()); td = eff[s]
    for cc in longs:
        p = PAIR[cc]; b, q = conv[p]["base"], conv[p]["terms"]; sd = spot_date(p, td); dd = delivery(p, sd, hp["roll_tenor_months"])
        S0 = float(spot.loc[td, p]); F = cip(S0, depd.loc[s, b], depd.loc[s, q], (dd - sd).days, b, q)
        hq = (F / S0 - 1) if p.startswith(cc) else (S0 / F - 1); hret[s][cc] = hq / win
        rows.append({"signal_date": iso(s), "effective_date": iso(td), "pair": p, "currency": cc, "trade_date": iso(td), "spot_date": iso(sd),
                     "delivery_date": iso(dd), "window_days": win, "spot_locked": S0, "forward_locked": F, "hedge_quote_return_pct": hq * 100,
                     "hedge_quote_return_bps": hq * 1e4, "daily_quote_return_bps": hq * 1e4 / win, "cumulative_cost_pct": np.nan})
rc = pd.DataFrame(rows); rc["cumulative_cost_pct"] = rc["hedge_quote_return_pct"].cumsum()
rc.to_csv(os.path.join(OUT_DIR, "roll_costs.csv"), index=False)

out = []
for t in rng:
    s = active(t, sig_dates, ct["signal_lag_business_days"]); L, Sh = basket[s]
    ll, ss = float(cret.loc[t, L].mean()), float(-cret.loc[t, Sh].mean())
    h = active(t, hs, hp["signal_lag_business_days"])
    ov = float(np.mean([-sret.loc[t, cc] + hret[h][cc] for cc in hbask[h]])) if h is not None else np.nan
    out.append({"date": iso(t), "signal_date": iso(s), "active_hedge_signal_date": iso(h) if h is not None else "", "long_currencies": ",".join(L),
                "short_currencies": ",".join(Sh), "active_hedge_currencies": ",".join(hbask[h]) if h is not None else "",
                "unhedged_return": ll + ss, "hedge_overlay_return": ov, "fwd_hedged_return": ll + ss + ov, "long_leg": ll, "short_leg": ss})
pr = pd.DataFrame(out)

# ---- options: smile, strikes, GK prices, greeks
smile, prices, greeks = [], [], []
for o in opts["options"]:
    S, rd, rf = o["spot"], o["domestic_rate_cc"], o["foreign_rate_cc"]
    T = (pd.Timestamp(o["expiry_date"]) - pd.Timestamp(o["trade_date"])).days / 365.0
    F, df, dfq, sT = S * math.exp((rd - rf) * T), math.exp(-rd * T), math.exp(-rf * T), None
    vc, vp = o["atm_vol"] + o["bf_25d"] + o["rr_25d"] / 2, o["atm_vol"] + o["bf_25d"] - o["rr_25d"] / 2
    is_call = o["quote_type"].endswith("call"); v = vc if is_call else vp; sT = v * math.sqrt(T); tgt = o["target_abs_delta"]
    d1f = lambda K: (math.log(F / K) + 0.5 * v * v * T) / sT
    if o["delta_convention"] == "spot_unadjusted": d1 = norm.ppf(tgt / dfq) if is_call else -norm.ppf(tgt / dfq); K = F * math.exp(0.5 * v * v * T - d1 * sT)
    elif o["delta_convention"] == "forward_unadjusted": d1 = norm.ppf(tgt) if is_call else -norm.ppf(tgt); K = F * math.exp(0.5 * v * v * T - d1 * sT)
    else:                                                  # spot premium-adjusted: dfq (K/F) N(d2) (call) / -dfq (K/F) N(-d2) (put)
        pa = lambda K: dfq * (K / F) * (norm.cdf(d1f(K) - sT) if is_call else -norm.cdf(-(d1f(K) - sT)))
        K = brentq(lambda K: pa(K) - (tgt if is_call else -tgt), 0.5 * F, 1.5 * F, xtol=1e-14)
    d1 = d1f(K); d2 = d1 - sT
    C = df * (F * norm.cdf(d1) - K * norm.cdf(d2)); P = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1)); sel = C if is_call else P
    delta = dfq * norm.cdf(d1) if is_call else -dfq * norm.cdf(-d1); dpa = delta - sel / S
    smile.append({"quote_id": o["quote_id"], "pair": o["pair"], "quote_type": o["quote_type"], "delta_convention": o["delta_convention"],
                  "premium_currency": o["premium_currency"], "atm_vol": o["atm_vol"], "rr_25d": o["rr_25d"], "bf_25d": o["bf_25d"], "vol_25d_call": vc,
                  "vol_25d_put": vp, "selected_vol": v, "target_delta": tgt if is_call else -tgt, "derived_strike": K})
    prices.append({"quote_id": o["quote_id"], "pair": o["pair"], "quote_type": o["quote_type"], "delta_convention": o["delta_convention"],
                   "premium_currency": o["premium_currency"], "spot": S, "strike": K, "forward": F, "T_years": T, "days_to_expiry": int(round(T * 365)),
                   "selected_vol": v, "call_price": C, "put_price": P, "selected_option_price": sel, "terms_pips": sel, "base_pips": sel / S,
                   "terms_pct": sel / K * 100, "base_pct": sel / S * 100, "pcp_error": abs(C - P - df * (F - K))})
    greeks.append({"quote_id": o["quote_id"], "pair": o["pair"], "quote_type": o["quote_type"], "delta_pips": delta, "delta_premium_included": dpa,
                   "gamma": dfq * norm.pdf(d1) / (S * sT), "vega": S * dfq * norm.pdf(d1) * math.sqrt(T), "vega_1pct": S * dfq * norm.pdf(d1) * math.sqrt(T) / 100,
                   "d1": d1, "d2": d2, "spot_hedge_amount_pips_delta": -delta * o["notional_base"], "spot_hedge_amount_pi_delta": -dpa * o["notional_base"]})
dump(smile, "smile_reconstruction.json"); dump(prices, "option_prices.json"); dump(greeks, "greeks.json")

# ---- risk metrics (option hedge = buy the three listed options, cost amortised over their average life)
cost = np.mean([p_["base_pct"] for p_ in prices]) / 100; life = np.mean([p_["days_to_expiry"] for p_ in prices])
pr["option_hedged_return"] = pr["unhedged_return"] - cost / (life / 365 * rp["annualization_factor"])
pr.to_csv(os.path.join(OUT_DIR, "portfolio_returns.csv"), index=False)
def metrics(r):
    r = r.dropna(); A = rp["annualization_factor"]; cum = (1 + r).cumprod(); var = float(r.quantile(1 - rp["var_confidence"]))
    return {"ann_return": float(r.mean() * A), "ann_vol": float(r.std() * math.sqrt(A)), "sharpe": float(r.mean() * A / (r.std() * math.sqrt(A))),
            "max_drawdown": float((cum / cum.cummax() - 1).min()), "var_1pct": var, "cvar_1pct": float(r[r <= var].mean()), "n_obs": int(len(r))}
dump({"unhedged": metrics(pr["unhedged_return"]), "fwd_hedged": metrics(pr["fwd_hedged_return"]), "option_hedged": metrics(pr["option_hedged_return"])}, "risk_metrics.json")

spx = pd.read_csv(os.path.join(DATA_DIR, "spx.csv"), parse_dates=["date"]).set_index("date")["close"].pct_change()
u = pr.set_index(pd.to_datetime(pr["date"]))["unhedged_return"]
both = pd.concat([u, spx.rename("spx")], axis=1, join="inner").dropna()
ec = both["unhedged_return"].rolling(rp["rolling_window_days"]).corr(both["spx"]).dropna()
pd.DataFrame({"date": ec.index.strftime("%Y-%m-%d"), "rolling_corr_63d": ec.values}).to_csv(os.path.join(OUT_DIR, "equity_correlation.csv"), index=False)
