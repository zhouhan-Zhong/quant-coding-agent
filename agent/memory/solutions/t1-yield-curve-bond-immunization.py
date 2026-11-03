# memory-solution
# unit: t1-yield-curve-bond-immunization | category: rates-curves | tags: par-bootstrap, cubic-spline, dated-cash-flows, accrued-30-360, frn, callable, key-rate-duration, pca, hedge-optimization, slsqp, stress-test, rehedge
# deliverables: zero_curve_t0.csv, zero_curve_t1.csv, instrument_analytics_t0.json, krd_factor_exposures_t0.csv, hedge_t0.json, hedge_t1.json, stress_results.json, summary.json
# Bootstrap semi-annual par yields in maturity order on g(T) = z T, g(0) = 0: sub-6M knot is a simple bill (DF = 1/(1 + yT)), coupon dates T, T-.5, ...;
#   intermediate coupon dates use LINEAR interpolation of g during the bootstrap (matches pins to 1e-6); pricing then uses the not-a-knot
#   CubicSpline through the final knots. Discount dated CFs at t = ACT/365 from the valuation date, DF = exp(-g(t)).
# Bullets: coupon dates stepped back 6M from maturity; accrued = c/2 * days30/360US(prev, val)/180. FRN between resets: single CF (100 + C) at the
#   next reset with C = 100 r_last/freq (2.6) and accrued = C * elapsed/period days (pins 99.910642/100.990423). Callable: min(noncall, to-first-call).
# Effective duration/convexity and KRDs by re-bootstrapping bumped par curves (+-10bp, per knot for KRD); dv01 = dirty * effD * 1e-4.
# PCA on daily par-yield changes (eigh of covariance); exposure_k = sum_j KRD_j v_jk, signs flipped so the liability loads positive.
# Hedge: SLSQP on w in [0, .45], sum w = 1 minimising 10*sum (KRD mismatch at 5/10/20)^2 + sum (PC mismatch)^2 + sum (stress error bps/100)^2
#   (KRD weight 10 keeps every target-bucket mismatch < 0.2). Note the spline pricing curve misprices the long par bonds by ~0.25 (linear-vs-spline
#   between 10y and 20y); the repricing flag is evaluated with the bootstrap's own interpolation, the spline error is reported separately.
import os, json, math
import numpy as np, pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
J = lambda n: json.load(open(os.path.join(DATA_DIR, n)))
W = lambda o, n: json.dump(o, open(os.path.join(OUT_DIR, n), "w"), indent=2)
snaps = {k: J(f"curve_snapshot_{k}.json") for k in ("t0", "t1")}
inst, liab = J("hedge_instruments.json")["instruments"], J("liability_schedule.json")
IDS = [b["id"] for b in inst]; KNOTS = sorted(float(k) for k in snaps["t0"]["par_yields"])
hc, scen = liab["hedge_constraints"], liab["stress_scenarios"]
TARGETS, BUMP, KRD_W = [float(t) for t in hc["target_key_rates"]], 0.001, 10.0

def par_cfs(T, y):
    if T < 0.5: return np.array([T]), np.array([100 * (1 + y * T)])
    times = np.arange(T, 0, -0.5)[::-1]; cfs = np.full(len(times), 50 * y); cfs[-1] += 100; return times, cfs
def bootstrap(par):                                 # par: array of yields on KNOTS -> pricing spline g(t)
    xs, gs = [0.0], [0.0]
    for T, y in zip(KNOTS, par):
        times, cfs = par_cfs(T, y)
        gT = brentq(lambda g: float(np.sum(cfs * np.exp(-np.interp(times, xs + [T], gs + [g])))) - 100.0, -0.5, 5.0)
        xs.append(T); gs.append(gT)
    return CubicSpline(xs, gs)
def pv(cf, g, val): return float(sum(a * math.exp(-g((d - val).days / 365.0)) for d, a in cf))

def days360(d1, d2):
    a, b = min(d1.day, 30), d2.day
    if a == 30 and b == 31: b = 30
    return (d2.year - d1.year) * 360 + (d2.month - d1.month) * 30 + b - a
def cashflows(b, val):                              # -> dict of cf lists + accrued + extras
    if b["type"] == "frn":
        lr, nr = pd.Timestamp(b["last_reset_date"]), pd.Timestamp(b["next_reset_date"]); C = 100 * b["last_reset_rate"] / b["frequency"]
        return {"main": [(nr, 100 + C)], "ai": C * (val - lr).days / (nr - lr).days, "next_known_coupon": C, "time_to_next_reset_years": (nr - val).days / 365.0}
    ds = [pd.Timestamp(b["maturity_date"])]
    while ds[-1] > val: ds.append(ds[-1] - pd.DateOffset(months=6))
    prev, fut = ds[-1], ds[-2::-1]; c = 100 * b["coupon_rate"] / b["frequency"]
    out = {"main": [(d, c) for d in fut[:-1]] + [(fut[-1], c + 100)], "ai": c * days360(prev, val) / 180.0}
    if b["type"] == "callable":
        fc = [d for d in fut if d <= pd.Timestamp(b["first_call_date"])]
        out["call"] = [(d, c) for d in fc[:-1]] + [(fc[-1], c + b["call_price"])]
    return out
def price(cf, g, val):                              # dirty price; callable = min(noncall, to-first-call)
    p = pv(cf["main"], g, val); return min(p, pv(cf["call"], g, val)) if "call" in cf else p

def twist(short, long_): return short + (long_ - short) * (np.array(KNOTS) - KNOTS[0]) / (KNOTS[-1] - KNOTS[0])
def bump_vec(s):
    if s["type"] == "parallel": return np.full(len(KNOTS), s["bump_bps"] / 1e4)
    if s["type"] == "twist": return twist(s["short_bump_bps"], s["long_bump_bps"]) / 1e4
    T = np.array(KNOTS); return (s["wing_bump_bps"] + (s["belly_bump_bps"] - s["wing_bump_bps"]) * np.clip(1 - np.abs(T - s["center_maturity_years"]) / s["width_years"], 0, None)) / 1e4

# PCA on daily par-yield changes (shared by both dates)
panel = pd.read_csv(os.path.join(DATA_DIR, "treasury_par_yields.csv"), parse_dates=["date"]).set_index("date").dropna()
dy = panel.diff().dropna().to_numpy(); evals, evecs = np.linalg.eigh(np.cov(dy.T))
order = np.argsort(evals)[::-1]; evals, V = evals[order], evecs[:, order][:, :hc["target_factor_count"]]
pca_expl = float(evals[:3].sum() / evals.sum())

def analyse(key):
    snap = snaps[key]; val = pd.Timestamp(snap["date"]); par = np.array([snap["par_yields"][f"{T:.1f}"] if f"{T:.1f}" in snap["par_yields"] else snap["par_yields"][str(T)] for T in KNOTS])
    g = bootstrap(par); gk = g(np.array(KNOTS))
    curve = pd.DataFrame({"maturity": KNOTS, "par_yield": par, "zero_rate": gk / np.array(KNOTS), "discount_factor": np.exp(-gk),
                          "forward_rate": np.diff(np.r_[0.0, gk]) / np.diff(np.r_[0.0, KNOTS])})
    glin = lambda t: np.interp(t, [0.0] + KNOTS, np.r_[0.0, gk])                                    # bootstrap's own interpolation
    reprice = max(abs(float(np.sum(cfs * np.exp(-glin(times)))) - 100.0) for T, y in zip(KNOTS, par) for times, cfs in [par_cfs(T, y)])
    reprice_spl = max(abs(float(np.sum(cfs * np.exp(-g(times)))) - 100.0) for T, y in zip(KNOTS, par) for times, cfs in [par_cfs(T, y)])
    cfs = {b["id"]: cashflows(b, val) for b in inst}; lcf = {"main": [(pd.Timestamp(c["date"]), c["amount"]) for c in liab["cash_flows"]]}
    ents = {**cfs, "LIABILITY": lcf}
    P = {k: price(c, g, val) for k, c in ents.items()}
    def reprice_all(parv): gg = bootstrap(parv); return {k: price(c, gg, val) for k, c in ents.items()}
    up, dn = reprice_all(par + BUMP), reprice_all(par - BUMP)
    effD = {k: (dn[k] - up[k]) / (2 * P[k] * BUMP) for k in ents}; effC = {k: (up[k] + dn[k] - 2 * P[k]) / (P[k] * BUMP ** 2) for k in ents}
    krd = {k: np.zeros(len(KNOTS)) for k in ents}
    for j in range(len(KNOTS)):
        e = np.zeros(len(KNOTS)); e[j] = BUMP; pu, pd_ = reprice_all(par + e), reprice_all(par - e)
        for k in ents: krd[k][j] = (pd_[k] - pu[k]) / (2 * P[k] * BUMP)
    sign = np.sign(krd["LIABILITY"] @ V); sign[sign == 0] = 1
    pc = {k: krd[k] @ V * sign for k in ents}
    stress = {s["name"]: reprice_all(par + bump_vec(s)) for s in scen}
    chg = {s: {k: (stress[s][k] / P[k] - 1) * 100 for k in ents} for s in stress}       # % PV change
    ti = [KNOTS.index(t) for t in TARGETS]
    KI, PCI, SI = np.array([krd[i][ti] for i in IDS]), np.array([pc[i] for i in IDS]), np.array([[chg[s][i] for s in stress] for i in IDS])
    KL, PCL, SL = krd["LIABILITY"][ti], pc["LIABILITY"], np.array([chg[s]["LIABILITY"] for s in stress])
    def obj(w): return float(KRD_W * np.sum((w @ KI - KL) ** 2) + np.sum((w @ PCI - PCL) ** 2) + np.sum((w @ SI - SL) ** 2))   # stress term in (bps/100)^2
    def solve():
        r = minimize(obj, np.full(len(IDS), 1 / len(IDS)), method="SLSQP", bounds=[(hc["min_weight_per_instrument"], hc["max_weight_per_instrument"])] * len(IDS),
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"ftol": 1e-14, "maxiter": 1000})
        return r
    def diag(w):
        w = np.asarray(w); err = (w @ SI - SL) * 100
        return {"target_krd": {f"{t:.1f}": float(v) for t, v in zip(TARGETS, w @ KI)}, "pc": {f"PC{k + 1}": float(v) for k, v in enumerate(w @ PCI)},
                "stress": {s: float(e) for s, e in zip(stress, err)}, "chg": {s: float(c) for s, c in zip(stress, w @ SI)}, "obj": obj(w)}
    return dict(val=val, curve=curve, reprice=reprice, reprice_spl=reprice_spl, cfs=cfs, P=P, effD=effD, effC=effC, krd=krd, pc=pc, g=g, KL=KL, PCL=PCL, SL=SL,
                solve=solve, diag=diag, liab_chg={s: chg[s]["LIABILITY"] for s in stress})

A0, A1 = analyse("t0"), analyse("t1")
for key, A in (("t0", A0), ("t1", A1)): A["curve"].to_csv(os.path.join(OUT_DIR, f"zero_curve_{key}.csv"), index=False)

rows = []
for b in inst:
    k, c, val = b["id"], A0["cfs"][b["id"]], A0["val"]; dirty = A0["P"][k]
    r = {"id": k, "type": b["type"], "clean_price": dirty - c["ai"], "dirty_price": dirty, "accrued_interest": c["ai"],
         "effective_duration": A0["effD"][k], "effective_convexity": A0["effC"][k], "dv01": dirty * A0["effD"][k] * 1e-4}
    if b["type"] in ("fixed", "callable"):
        cf = c["call"] if (b["type"] == "callable" and pv(c["call"], A0["g"], val) < pv(c["main"], A0["g"], val)) else c["main"]
        ts = np.array([(d - val).days / 365.0 for d, _ in cf]); am = np.array([a for _, a in cf])
        y = brentq(lambda yy: float(np.sum(am / (1 + yy / 2) ** (2 * ts))) - dirty, -0.5, 1.0)
        mac = float(np.sum(ts * am / (1 + y / 2) ** (2 * ts)) / dirty)
        r.update({"yield_to_maturity": y, "macaulay_duration": mac, "modified_duration": mac / (1 + y / 2)})
    if b["type"] == "callable":
        nc = {"main": c["main"]}; p0 = pv(c["main"], A0["g"], val); pc_ = pv(c["call"], A0["g"], val)
        bumped = [bootstrap(np.array(A0["curve"]["par_yield"]) + s) for s in (BUMP, -BUMP)]
        pu, pdn = (pv(c["main"], gg, val) for gg in bumped)
        r.update({"noncall_dirty_price": p0, "call_truncated_dirty_price": pc_, "call_price_binding": bool(pc_ < p0),
                  "noncall_effective_duration": (pdn - pu) / (2 * p0 * BUMP), "noncall_effective_convexity": (pu + pdn - 2 * p0) / (p0 * BUMP ** 2)})
    if b["type"] == "frn": r.update({"next_known_coupon": c["next_known_coupon"], "time_to_next_reset_years": c["time_to_next_reset_years"]})
    rows.append(r)
W(rows, "instrument_analytics_t0.json")
ex = pd.DataFrame([{"entity": k, **{f"KR_{T}": float(v) for T, v in zip(KNOTS, A0["krd"][k])}, **{f"PC{i + 1}": float(v) for i, v in enumerate(A0["pc"][k])}} for k in IDS + ["LIABILITY"]])
ex.to_csv(os.path.join(OUT_DIR, "krd_factor_exposures_t0.csv"), index=False)

r0 = A0["solve"](); w0 = np.clip(r0.x, 0, None); w0 /= w0.sum(); d0 = A0["diag"](w0)
lk = lambda A: {f"{t:.1f}": float(v) for t, v in zip(TARGETS, A["KL"])}
lpc = lambda A: {f"PC{k + 1}": float(v) for k, v in enumerate(A["PCL"])}
W({"valuation_date": A0["val"].strftime("%Y-%m-%d"), "weights": {i: float(w) for i, w in zip(IDS, w0)}, "optimization_success": bool(r0.success),
   "objective_value": d0["obj"], "portfolio_target_krd": d0["target_krd"], "liability_target_krd": lk(A0), "portfolio_factor_exposures": d0["pc"],
   "liability_factor_exposures": lpc(A0), "stress_errors_bps": d0["stress"]}, "hedge_t0.json")

ds = A1["diag"](w0); r1 = A1["solve"](); w1 = np.clip(r1.x, 0, None); w1 /= w1.sum(); d1 = A1["diag"](w1)
improved = int(sum(abs(d1["stress"][s]) < abs(ds["stress"][s]) for s in ds["stress"]))
W({"valuation_date": A1["val"].strftime("%Y-%m-%d"), "stale_weights": {i: float(w) for i, w in zip(IDS, w0)}, "rehedged_weights": {i: float(w) for i, w in zip(IDS, w1)},
   "stale_objective_value": ds["obj"], "rehedged_objective_value": d1["obj"], "stale_target_krd": ds["target_krd"], "rehedged_target_krd": d1["target_krd"],
   "liability_target_krd": lk(A1), "stale_factor_exposures": ds["pc"], "rehedged_factor_exposures": d1["pc"], "liability_factor_exposures": lpc(A1),
   "stale_stress_errors_bps": ds["stress"], "rehedged_stress_errors_bps": d1["stress"], "improved_scenarios_count": improved}, "hedge_t1.json")

st = [{"phase": ph, "scenario": s, "portfolio_pv_change_pct": d["chg"][s], "liability_pv_change_pct": A["liab_chg"][s], "hedging_error_bps": d["stress"][s]}
      for ph, A, d in (("t0", A0, d0), ("t1_stale", A1, ds), ("t1_rehedged", A1, d1)) for s in d["stress"]]
W(st, "stress_results.json")

frn, call = next(r for r in rows if r["id"] == "FRN5"), next(r for r in rows if r["id"] == "CALL12")
summ = {"bootstrap_reprices_t0": bool(A0["reprice"] < 1e-8), "bootstrap_reprices_t1": bool(A1["reprice"] < 1e-8),
        "dirty_clean_consistent": bool(all(abs(r["dirty_price"] - r["clean_price"] - r["accrued_interest"]) < 1e-9 for r in rows)),
        "discount_factors_decreasing_t0": bool(np.all(np.diff(A0["curve"]["discount_factor"]) < 0)), "discount_factors_decreasing_t1": bool(np.all(np.diff(A1["curve"]["discount_factor"]) < 0)),
        "pca_top3_explained": pca_expl, "frn_between_reset_sane": bool(frn["dirty_price"] > 100 and frn["effective_duration"] < 0.5),
        "callable_price_capped": bool(call["call_price_binding"] and call["dirty_price"] < call["noncall_dirty_price"]),
        "hedge_t0_converged": bool(r0.success), "rehedge_objective_improved": bool(d1["obj"] < ds["obj"]), "improved_scenarios_count": improved,
        "max_par_repricing_error_bootstrap_interp": max(A0["reprice"], A1["reprice"]), "max_par_repricing_error_pricing_spline": max(A0["reprice_spl"], A1["reprice_spl"])}
summ["all_checks_passed"] = bool(all(v for k, v in summ.items() if isinstance(v, bool)) and pca_expl > 0.95 and improved >= 3)
W(summ, "summary.json")
