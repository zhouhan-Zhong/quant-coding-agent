# memory-solution
# unit: t1-mtm-xccy-basis-desk | category: rates-curves | tags: cross-currency-swap, xccy-basis, mtm-reset, ois, fra, libor-projection, cip, fx-forward-interpolation, broken-date, dual-curve, pv01, fx-delta, stress-scenarios, par-basis, roll-down
# deliverables: cleaned_quotes.csv, usd_curve.csv, gbp_curve.csv, basis_curve.csv, fx_forward_curve.csv, cashflow_schedule.csv, results.json, stress_results.csv, reconciliation.json, solution.json, summary.json
# Clean: latest quote_time per key; pct/100 -> rate_decimal, pips/1e4 -> fx_forward_increment, spot -> fx_rate; FRA adjustment only in the curve build.
# Time T = days(SPOT, d)/basis (USD 360, GBP 365) from the SPOT date: DF(spot)=1, no val->spot discounting. OIS (deposits AND swaps) DF = 1/(1+rT),
#   interpolation linear in -lnDF over T. LIBOR Q: deposit 1/(1+rT), FRA Q(e) = Q(s)/(1+(f+adj)tau), linear in lnQ; coupon L = (Q(a)/Q(b)-1)/tau(a,b).
# FX nodes F = S + pips/1e4; broken dates linear in lnF vs calendar days from spot (spot node at the spot date). basis b = -ln(F/F_cip)/T_usd, F_cip = S DFg/DFu.
# MTM: N0 = usd_notional/inception_spot; N_k = USD_N/F(reset_k); mtm_reset CF on the reset date = N_k - N_{k-1} (sign the reference uses);
#   GBP flows -> USD at F(payment date), all discounted on USD OIS; fixings used when fixing_date <= valuation date; spread on GBP coupons only.
# Risk: bump the QUOTES (+1bp OIS par / deposit+FRA rates) and rebuild; OIS bumps rebuild FX nodes S DFg/DFu exp(-b T) with basis fixed; FX = spot & nodes x1.01.
# Par basis = all-in GBP spread (current fixed coupon included) with PV = 0 (PV is linear in the spread). Accrued GBP converted at F(current pay date).
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
P, O = (lambda n: os.path.join(DATA_DIR, n)), (lambda n: os.path.join(OUT_DIR, n))
ts = pd.Timestamp
tr = json.load(open(P("trade.json")))
VAL, SPOT = ts(tr["valuation_date"]), ts(tr["spot_date"])
BAS_DC = {c: 360.0 if tr[f"{c.lower()}_day_count"].endswith("360") else 365.0 for c in ("USD", "GBP")}
yf = lambda a, b, ccy: (ts(b) - ts(a)).days / BAS_DC[ccy]
NU, SP = float(tr["usd_notional"]), tr["contract_spread_bps"] / 1e4
N0 = NU / tr["inception_spot_fx"]

# ---- quote book
q = pd.read_csv(P("market_quotes.csv"))
KEY = ["curve", "instrument_type", "tenor", "start_date", "end_date", "ccy_or_pair"]
q = q.sort_values("quote_time", kind="stable").groupby(KEY, as_index=False, sort=False).tail(1)
UNIT = {"rate_pct": ("rate_decimal", 100.0), "forward_points_pips": ("fx_forward_increment", 1e4), "spot_rate": ("fx_rate", 1.0)}
q["normalized_unit"] = q["quote_unit"].map(lambda u: UNIT[u][0])
q["normalized_quote"] = q["quote_value"] / q["quote_unit"].map(lambda u: UNIT[u][1])
q = q.sort_values(["curve", "instrument_type", "end_date", "tenor"]).reset_index(drop=True)
q[["quote_time"] + KEY + ["normalized_unit", "normalized_quote"]].to_csv(O("cleaned_quotes.csv"), index=False)

def tenor_m(t):                                   # pillar maturity in months: 1M, 1Y, 18M, 3x6 -> 6
    t = t.split("x")[-1]; return int(t) if t.isdigit() else int(t[:-1]) * (12 if t[-1] == "Y" else 1)
def build(ccy, kind, bump=lambda m: 0.0):         # returns (T, g) with g = -lnDF (OIS) or -lnQ (LIBOR3M), g(0) = 0
    T, G = [0.0], [0.0]
    for _, x in q[q.curve == f"{ccy}_{kind}"].sort_values("end_date").iterrows():
        r = x.normalized_quote + (0.0 if pd.isna(x.quote_adjustment_bps) else x.quote_adjustment_bps / 1e4) + bump(tenor_m(x.tenor))
        g0 = np.interp(yf(SPOT, x.start_date, ccy), T, G)
        T.append(yf(SPOT, x.end_date, ccy)); G.append(g0 + math.log(1.0 + r * yf(x.start_date, x.end_date, ccy)))
    return np.array(T), np.array(G)
def curves(bumps={}):                             # bumps: {(ccy, kind): f(months) -> decimal}
    return {(c, k): build(c, k, bumps.get((c, k), lambda m: 0.0)) for c in ("USD", "GBP") for k in ("OIS", "LIBOR3M")}
df = lambda C, ccy, d: math.exp(-np.interp(yf(SPOT, d, ccy), *C[ccy, "OIS"]))
def fwd(C, ccy, a, b):
    T, G = C[ccy, "LIBOR3M"]; return (math.exp(np.interp(yf(SPOT, b, ccy), T, G) - np.interp(yf(SPOT, a, ccy), T, G)) - 1.0) / yf(a, b, ccy)
C0 = curves()

# ---- FX nodes, broken-date interpolation, implied basis
S0 = float(q[(q.curve == "GBPUSD") & (q.instrument_type == "spot")].normalized_quote.iloc[0])
fq = q[(q.curve == "GBPUSD") & (q.instrument_type == "forward")].sort_values("end_date")
NODES = [(ts(d), S0 + p) for d, p in zip(fq.end_date, fq.normalized_quote)]
def fx_fn(S, nodes, origin=SPOT):                 # linear in ln F vs days, spot node at origin
    x = [0] + [(d - origin).days for d, _ in nodes]; y = [math.log(S)] + [math.log(F) for _, F in nodes]
    return lambda d: math.exp(np.interp((ts(d) - origin).days, x, y))
cip = lambda C, S, d: S * df(C, "GBP", d) / df(C, "USD", d)
BAS = [(d, -math.log(F / cip(C0, S0, d)) / yf(SPOT, d, "USD")) for d, F in NODES]
def cip_nodes(C, S=S0, db=0.0):                   # rebuild tenor nodes from OIS curves with the basis held fixed
    return [(d, cip(C, S, d) * math.exp(-(b + db) * yf(SPOT, d, "USD"))) for d, b in BAS]
FX0 = fx_fn(S0, NODES)

# ---- cashflows and valuation
SCH = pd.read_csv(P("trade_schedule.csv"))
fx_ = pd.read_csv(P("libor_fixings.csv"))
FIX = {(ts(r.fixing_date), r.index_name[:3]): r.rate_pct / 100 for r in fx_.itertuples()}
ORDER = {("GBP", "coupon"): 0, ("GBP", "mtm_reset"): 1, ("USD", "coupon"): 2, ("GBP", "principal"): 3, ("USD", "principal"): 4}
def value(C=C0, fx=FX0, val=VAL, sp=SP, disc=None):
    disc = disc or (lambda d: df(C, "USD", d))
    rows, Nprev = [], N0
    def add(leg, typ, p, pay, N, yfr, rtype, rate, cf):
        F = fx(pay) if leg == "GBP" else 1.0; D = disc(pay)
        rows.append({"leg": leg, "cashflow_type": typ, "reset_date": p.reset_date, "fixing_date": p.fixing_date, "payment_date": str(pay.date()),
                     "accrual_start": p.accrual_start, "accrual_end": p.accrual_end, "year_frac": yfr, "notional": N, "rate_type": rtype,
                     "coupon_rate": rate, "cashflow_ccy": leg, "signed_cashflow_ccy": cf, "fx_forward": F, "discount_factor_usd": D, "pv_usd": cf * F * D})
    for _, p in SCH.iterrows():
        a, b, pay, rst, fxd = (ts(p[k]) for k in ("accrual_start", "accrual_end", "payment_date", "reset_date", "fixing_date"))
        hist = fxd <= val
        Lg, Lu = (FIX[fxd, "GBP"], FIX[fxd, "USD"]) if hist else (fwd(C, "GBP", a, b), fwd(C, "USD", a, b))
        N = N0 if rst <= VAL else NU / fx(rst)
        if rst > VAL: add("GBP", "mtm_reset", p, rst, N, np.nan, "fx_reset", np.nan, N - Nprev)
        Nprev, rt = N, ("historical_fixing" if hist else "projected")
        add("GBP", "coupon", p, pay, N, yf(a, b, "GBP"), rt, Lg + sp, N * (Lg + sp) * yf(a, b, "GBP"))
        add("USD", "coupon", p, pay, NU, yf(a, b, "USD"), rt, Lu, -NU * Lu * yf(a, b, "USD"))
    last = SCH.iloc[-1]
    add("GBP", "principal", last, ts(last.payment_date), Nprev, np.nan, "notional", np.nan, Nprev)
    add("USD", "principal", last, ts(last.payment_date), NU, np.nan, "notional", np.nan, -NU)
    cf = pd.DataFrame([r for r in rows if ts(r["payment_date"]) >= val])
    cf = cf.assign(o=[ORDER[k] for k in zip(cf.leg, cf.cashflow_type)]).sort_values(["payment_date", "o"], kind="stable").drop(columns="o")
    pu, pg = float(cf.pv_usd[cf.leg == "USD"].sum()), float(cf.pv_usd[cf.leg == "GBP"].sum())
    return cf.reset_index(drop=True), pu, pg, pu + pg

CF, PU, PG, PV = value()
CF.to_csv(O("cashflow_schedule.csv"), index=False)
pv_s = lambda s: value(sp=s)[3]
PAR = SP - PV / ((pv_s(SP + 1e-4) - PV) / 1e-4)                  # PV linear in the all-in spread
oisb = lambda ccy, h: curves({(ccy, "OIS"): lambda m: h})
def pv_ois(ccy, h): C = oisb(ccy, h); return value(C, fx_fn(S0, cip_nodes(C)))[3]
def pv_lib(ccy, h): return value(curves({(ccy, "LIBOR3M"): lambda m: h}))[3]
fx_up = fx_fn(S0 * 1.01, [(d, F * 1.01) for d, F in NODES])
risk = {"pv01_usd_ois": pv_ois("USD", 1e-4) - PV, "pv01_gbp_ois": pv_ois("GBP", 1e-4) - PV,
        "pv01_usd_libor3m": pv_lib("USD", 1e-4) - PV, "pv01_gbp_libor3m": pv_lib("GBP", 1e-4) - PV, "fx_delta_usd_per_1pct": value(fx=fx_up)[3] - PV}
cur = SCH[(SCH.accrual_start.map(ts) <= VAL) & (SCH.accrual_end.map(ts) > VAL)].iloc[0]
Lg0, Lu0, a0 = FIX[ts(cur.fixing_date), "GBP"], FIX[ts(cur.fixing_date), "USD"], cur.accrual_start
resets = CF[CF.cashflow_type == "mtm_reset"]
results = {"trade_pv_usd": PV, "usd_leg_pv_usd": PU, "gbp_leg_pv_usd": PG, "par_basis_bps": PAR * 1e4,
           "next_reset_notional_gbp": float(resets.notional.iloc[0]), "next_coupon_rate_usd": Lu0, "next_coupon_rate_gbp": Lg0 + SP,
           "accrued_usd": -NU * Lu0 * yf(a0, VAL, "USD"), "accrued_gbp_usd": N0 * (Lg0 + SP) * yf(a0, VAL, "GBP") * FX0(cur.payment_date),
           **{k: float(v) for k, v in risk.items()}, "num_clean_quotes": int(len(q)), "num_future_resets": int(len(resets))}
json.dump(results, open(O("results.json"), "w"), indent=2)

# ---- stresses
twist = {k: (lambda m: 5e-4 if m <= 12 else -5e-4) for k in [(c, kk) for c in ("USD", "GBP") for kk in ("OIS", "LIBOR3M")]}
Ct = curves(twist)
R = ts("2020-05-15"); SR = FX0(R)                                # roll: spot rolls along the base forward
rnodes = [(d, SR * (df(C0, "GBP", d) / df(C0, "GBP", R)) / (df(C0, "USD", d) / df(C0, "USD", R)) * math.exp(-b * yf(R, d, "USD"))) for d, b in BAS if d > R]
fxr_after = fx_fn(SR, rnodes, origin=R)
fx_roll = lambda d: FX0(d) if ts(d) <= R else fxr_after(d)
fx_bw = fx_fn(S0, cip_nodes(C0, db=-2e-4))
st = [("usd_ois_up_5bp", pv_ois("USD", 5e-4)), ("gbp_ois_up_5bp", pv_ois("GBP", 5e-4)), ("basis_widen_2bp", value(fx=fx_bw)[3]),
      ("gbp_leg_basis_widen_2bp", PU + value(fx=fx_bw)[2]), ("spot_up_1pct", value(fx=fx_up)[3]),
      ("spot_realized_1d_up_1pct", value(fx=fx_fn(S0 * 1.01, NODES))[3]), ("curve_twist", value(Ct, fx_fn(S0, cip_nodes(Ct)))[3]),
      ("roll_down_to_2020_05_15", value(fx=fx_roll, val=R, disc=lambda d: df(C0, "USD", d) / df(C0, "USD", R))[3])]
pd.DataFrame([{"scenario": s, "trade_pv_usd": v, "pnl_usd": v - PV} for s, v in st]).to_csv(O("stress_results.csv"), index=False)

# ---- curve / FX / basis files
for ccy in ("USD", "GBP"):
    r = q[q.curve == f"{ccy}_OIS"].sort_values("end_date"); T, G = C0[ccy, "OIS"]
    dfs = np.exp(-G)
    pd.DataFrame({"curve_type": f"{ccy}_OIS", "tenor": r.tenor.values, "pillar_date": r.end_date.values, "year_frac": T[1:], "discount_factor": dfs[1:],
                  "zero_rate": G[1:] / T[1:], "forward_rate": (dfs[:-1] / dfs[1:] - 1) / np.diff(T)}).to_csv(O(f"{ccy.lower()}_curve.csv"), index=False)
bc = pd.DataFrame({"tenor": fq.tenor.values, "maturity_date": fq.end_date.values, "year_frac": [yf(SPOT, d, "USD") for d, _ in NODES], "spot_fx": S0,
                   "quoted_forward_fx": [F for _, F in NODES], "cip_forward_no_basis": [cip(C0, S0, d) for d, _ in NODES], "basis_bps": [b * 1e4 for _, b in BAS]})
bc.to_csv(O("basis_curve.csv"), index=False)
fc = pd.DataFrame({"tenor": bc.tenor, "maturity_date": bc.maturity_date, "spot_fx": S0, "forward_fx": bc.quoted_forward_fx,
                   "forward_points": fq.normalized_quote.values * 1e4, "implied_basis_bps": bc.basis_bps})
fc.to_csv(O("fx_forward_curve.csv"), index=False)

# ---- reconciliation, checkpoints, summary
def reprices(ccy, kind, C=C0, tol=1e-12):         # every pillar quote recomputed from the built curve
    T, G = C[ccy, kind]; ok = True
    for _, x in q[q.curve == f"{ccy}_{kind}"].iterrows():
        adj = 0.0 if pd.isna(x.quote_adjustment_bps) else x.quote_adjustment_bps / 1e4
        ok &= abs((fwd(C, ccy, x.start_date, x.end_date) if kind == "LIBOR3M" else (1 / df(C, ccy, x.end_date) - 1) / yf(SPOT, x.end_date, ccy)) - x.normalized_quote - adj) < tol
    return bool(ok)
fe = max(abs(F - (S0 + p)) * 1e4 for (_, F), p in zip(NODES, fq.normalized_quote))
be = float(np.max(np.abs(bc.basis_bps.values - fc.implied_basis_bps.values)))
rebuilt_ok = all(abs(a[1] / F - 1) < 1e-12 for a, (_, F) in zip(cip_nodes(C0), NODES))
gp = CF[(CF.leg == "GBP") & (CF.cashflow_type == "principal")].iloc[0]
recon = {"cleaned_quote_count": int(len(q)), "max_forward_node_abs_error_pips": float(fe), "max_basis_node_abs_error_bps": be,
         "first_broken_date_forward": float(CF.fx_forward.iloc[0]), "final_payment_forward": float(gp.fx_forward),
         "usd_curve_reprices_inputs": reprices("USD", "OIS"), "gbp_curve_reprices_inputs": reprices("GBP", "OIS"),
         "fx_forward_nodes_reprice": bool(fe < 1e-9 and rebuilt_ok), "basis_curve_matches_fx_forwards": bool(np.allclose(bc.quoted_forward_fx, fc.forward_fx, atol=1e-12, rtol=0) and be < 1e-12)}
recon["all_reconciliation_checks_passed"] = all(v for k, v in recon.items() if isinstance(v, bool))
json.dump(recon, open(O("reconciliation.json"), "w"), indent=2)

fra = lambda ccy, ten: next(fwd(C0, ccy, x.start_date, x.end_date) for _, x in q[(q.curve == f"{ccy}_LIBOR3M") & (q.tenor == ten)].iterrows())
inter = {"usd_df_18m": df(C0, "USD", "2021-08-12"), "gbp_df_2y": df(C0, "GBP", "2022-02-14"), "usd_fra_9x12": fra("USD", "9x12"),
         "gbp_fra_15x18": fra("GBP", "15x18"), "gbp_fra_18x21_effective": fra("GBP", "18x21"), "fx_forward_18m": NODES[4][1],
         "implied_basis_1y_bps": BAS[3][1] * 1e4, "derived_initial_gbp_notional": N0, "broken_date_fx_2020_02_17": FX0("2020-02-17"),
         "current_usd_coupon": Lu0, "current_gbp_coupon": Lg0 + SP, "next_reset_notional_gbp": results["next_reset_notional_gbp"],
         "final_reset_notional_gbp": float(resets.notional.iloc[-1]), "usd_leg_pv_usd": PU, "gbp_leg_pv_usd": PG, "trade_pv_usd": PV, "par_basis_bps": PAR * 1e4}
json.dump({"intermediates": {k: {"value": float(v)} for k, v in inter.items()}}, open(O("solution.json"), "w"), indent=2)

stres = dict((s, v - PV) for s, v in st)
summary = {"curves_reprice_inputs": all(reprices(c, k) for c in ("USD", "GBP") for k in ("OIS", "LIBOR3M")),
           "fx_forwards_consistent": recon["all_reconciliation_checks_passed"],
           "past_cashflows_excluded": bool((CF.payment_date.map(ts) >= VAL).all()),
           "fixed_coupon_uses_historical_fixing": bool(CF.rate_type.iloc[0] == "historical_fixing" and abs(CF.coupon_rate.iloc[0] - Lg0 - SP) < 1e-15),
           "mtm_reset_sequence_consistent": bool(len(resets) == len(SCH[SCH.reset_date.map(ts) > VAL]) and resets.notional.nunique() == len(resets)
                                                 and abs(gp.notional - resets.notional.iloc[-1]) < 1e-9 and np.allclose(resets.notional * resets.fx_forward, NU)),
           "risk_signs_consistent": bool(risk["pv01_usd_libor3m"] < 0 < risk["pv01_gbp_libor3m"] and abs(stres["usd_ois_up_5bp"] - 5 * risk["pv01_usd_ois"]) < 15
                                         and abs(stres["gbp_ois_up_5bp"] - 5 * risk["pv01_gbp_ois"]) < 15 and stres["spot_up_1pct"] == risk["fx_delta_usd_per_1pct"])}
summary["all_checks_passed"] = all(summary.values())
json.dump(summary, open(O("summary.json"), "w"), indent=2)
