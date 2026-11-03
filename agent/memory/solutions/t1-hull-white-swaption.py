# memory-solution
# unit: t1-hull-white-swaption | category: rates-curves | tags: hull-white, short-rate, caplet, calibration, l-bfgs-b, jamshidian, swaption, bermudan, trinomial-tree, arrow-debreu, dv01
# deliverables: calibration.json, swaption_prices.json, yield_curve.csv, tree_diagnostics.csv, calibration_fit.png, bermudan_prices.json, summary.json
# Curve: log-linear interpolation of market DFs (piecewise-flat forwards, flat extrapolation past 10y) -> market caplets
#   are reproduced exactly at (a, sigma) = (0.05, 0.01); f(0,t) = right-limit forward keeps the tree alpha_n and A(t,T) consistent.
# HW ZCB: B = (1-e^{-a(T-t)})/a, lnA = ln(P_T/P_t) + B f(0,t) - s^2/(4a)(1-e^{-2at})B^2; P = A e^{-Br}.
# ZBP(T,S,X) = X P_T N(-h+sp) - P_S N(-h), sp = s sqrt((1-e^{-2aT})/(2a)) B(T,S), h = ln(P_S/(P_T X))/sp + sp/2;
#   caplet = (1+K tau) ZBP(T1, T2, 1/(1+K tau)); calibrate SSE in bps^2 with L-BFGS-B from (0.10, 0.015).
# Jamshidian: coupons c_i = K d (last + 1), r* solves sum c_i P_HW(T0,T_i,r*) = 1, payer = sum c_i ZBP(T0,T_i,P_HW(T0,T_i,r*)).
# Tree: dx = s sqrt(3dt), jmax = ceil(0.184/(a dt)), M = e^{-a dt}-1, Hull branching probs, Arrow-Debreu alpha_n closed form;
#   exercise value at node = max(1 - sum_{T_i>t} c_i P_HW(t,T_i,alpha_n + j dx), 0); Bermudan = backward induction with max().
# DV01: bump DFs by e^{-1bp*T}, rebuild tree, reprice European (same strike) -> (bumped - base)*1e4.
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

md = json.load(open(os.path.join(DATA_DIR, "market_data.json")))
mT = np.array([float(k) for k in md["discount_factors"]]); mP = np.array(list(md["discount_factors"].values()), float)
o = np.argsort(mT); mT, mP = mT[o], mP[o]
cp = md["caplets"]; K_c, DELTA = float(cp["strike"]), float(cp["frequency"])
T2 = np.array(cp["expiries_T2"], float); T1 = T2 - DELTA; mkt_caps = np.array(cp["market_prices"], float)
DT = 1.0 / int(md["tree"]["steps_per_year"]); N_STEPS = int(round(float(md["tree"]["max_maturity"]) / DT))


def make_curve(T, P):                       # log-linear DF interpolation == piecewise-flat forwards
    Tk, lnP = np.r_[0.0, T], np.r_[0.0, np.log(P)]
    fwd = -np.diff(lnP) / np.diff(Tk)
    def disc(t):
        t = np.asarray(t, float)
        return np.exp(np.where(t <= Tk[-1], np.interp(t, Tk, lnP), lnP[-1] - fwd[-1] * (t - Tk[-1])))
    def f0(t):                              # right-limit instantaneous forward f(0,t)
        return fwd[np.clip(np.searchsorted(Tk, t, side="right") - 1, 0, len(fwd) - 1)]
    return disc, f0, fwd


def Bf(a, t, T): return (1.0 - np.exp(-a * (T - t))) / a


def P_hw(disc, f0, a, s, t, T, r):          # T array (m,), r array (n,) -> (m, n)
    T = np.atleast_1d(np.asarray(T, float)); r = np.atleast_1d(np.asarray(r, float))
    b = Bf(a, t, T)
    lnA = np.log(disc(T) / disc(t)) + b * f0(t) - s ** 2 / (4 * a) * (1 - math.exp(-2 * a * t)) * b ** 2
    return np.exp(lnA[:, None] - b[:, None] * r[None, :])


def zbp(disc, a, s, T, S, X):
    sp = s * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a)) * Bf(a, T, S)
    h = np.log(disc(S) / (disc(T) * X)) / sp + sp / 2
    return X * disc(T) * norm.cdf(-h + sp) - disc(S) * norm.cdf(-h)


def caplets(disc, a, s): return (1 + K_c * DELTA) * zbp(disc, a, s, T1, T2, 1.0 / (1 + K_c * DELTA))


disc, f0, fwd = make_curve(mT, mP)
res = minimize(lambda p: float(np.sum(((caplets(disc, *p) - mkt_caps) * 1e4) ** 2)), [0.10, 0.015],
               method="L-BFGS-B", bounds=[(0.001, 1.0), (0.001, 0.10)])
a, sig = float(res.x[0]), float(res.x[1])
model_caps = caplets(disc, a, sig)
rmse_bps = float(np.sqrt(np.mean((model_caps - mkt_caps) ** 2)) * 1e4)
json.dump({"a": a, "sigma": sig, "rmse_bps": rmse_bps,
           "caplets": [{"expiry": float(t), "market_price": float(m), "model_price": float(v), "diff_bps": float((v - m) * 1e4)}
                       for t, m, v in zip(T2, mkt_caps, model_caps)]}, open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)
plt.figure(figsize=(6, 4)); plt.plot(T2, mkt_caps * 1e4, "o-", label="market"); plt.plot(T2, model_caps * 1e4, "s--", label="Hull-White")
plt.xlabel("caplet expiry T2 (years)"); plt.ylabel("price (bps)"); plt.legend(); plt.title(f"Caplet fit: a={a:.4f}, sigma={sig:.4f}")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "calibration_fit.png"), dpi=100); plt.close()

pd.DataFrame({"maturity": mT, "discount_factor": mP, "zero_rate": -np.log(mP) / mT, "forward_rate": fwd}).to_csv(
    os.path.join(OUT_DIR, "yield_curve.csv"), index=False)


def build_tree(disc, a, s):                 # Hull-White trinomial tree with Arrow-Debreu calibration
    dx = s * math.sqrt(3 * DT); jmax = max(1, math.ceil(0.184 / (a * DT)))
    js = np.arange(-jmax, jmax + 1); nn = len(js); M = math.exp(-a * DT) - 1.0
    jM = js * M; jM2 = jM ** 2
    probs = np.stack([1 / 6 + (jM2 + jM) / 2, 2 / 3 - jM2, 1 / 6 + (jM2 - jM) / 2], 1)
    tgt = np.stack([js + 1, js, js - 1], 1)
    t_, b_ = jM[-1], jM[0]
    probs[-1] = [7 / 6 + (t_ ** 2 + 3 * t_) / 2, -1 / 3 - t_ ** 2 - 2 * t_, 1 / 6 + (t_ ** 2 + t_) / 2]; tgt[-1] = [jmax, jmax - 1, jmax - 2]
    probs[0] = [1 / 6 + (b_ ** 2 - b_) / 2, -1 / 3 - b_ ** 2 + 2 * b_, 7 / 6 + (b_ ** 2 - 3 * b_) / 2]; tgt[0] = [-jmax + 2, -jmax + 1, -jmax]
    idx = tgt + jmax
    Q = np.zeros(nn); Q[jmax] = 1.0; Qs = [Q.copy()]; alpha = np.zeros(N_STEPS)
    for n in range(N_STEPS):
        alpha[n] = (math.log(np.sum(Q * np.exp(-js * dx * DT))) - math.log(float(disc((n + 1) * DT)))) / DT
        d = np.exp(-(alpha[n] + js * dx) * DT)
        Qn = np.zeros(nn); np.add.at(Qn, idx.ravel(), ((Q * d)[:, None] * probs).ravel())
        Q = Qn; Qs.append(Q.copy())
    return dict(dx=dx, jmax=jmax, js=js, probs=probs, idx=idx, alpha=alpha, Qs=np.array(Qs))


def tree_price(tr, disc, f0, a, s, pay, c, ex_times):   # backward induction with optimal stopping at ex_times
    js, jmax, dx, alpha = tr["js"], tr["jmax"], tr["dx"], tr["alpha"]
    steps = {int(round(t / DT)): t for t in ex_times}; n_last = max(steps)
    V = np.zeros(len(js)); boundary = []
    for n in range(n_last, -1, -1):
        cont = np.zeros(len(js)) if n == n_last else np.exp(-(alpha[n] + js * dx) * DT) * np.sum(tr["probs"] * V[tr["idx"]], 1)
        if n in steps:
            t = steps[n]; r = alpha[n] + js * dx; m = pay > t + 1e-9
            ex = np.maximum(1.0 - np.sum(c[m][:, None] * P_hw(disc, f0, a, s, t, pay[m], r), 0), 0.0)
            pref = (ex > cont) & (np.abs(js) <= min(n, jmax))
            jst = int(js[pref].min()) if pref.any() else int(jmax)
            boundary.append({"date": float(t), "j_star": jst, "r_star": float(alpha[n] + jst * dx)})
            V = np.maximum(ex, cont)
        else:
            V = cont
    return float(V[jmax]), boundary[::-1]


tree = build_tree(disc, a, sig)
diag = pd.DataFrame({"maturity": mT, "zcb_tree": [float(tree["Qs"][int(round(T / DT))].sum()) for T in mT], "zcb_market": mP})
diag["abs_diff"] = (diag["zcb_tree"] - diag["zcb_market"]).abs()
diag.to_csv(os.path.join(OUT_DIR, "tree_diagnostics.csv"), index=False)

bP = mP * np.exp(-1e-4 * mT)                             # +1bp parallel bump of zero rates
disc_b, f0_b, _ = make_curve(mT, bP); tree_b = build_tree(disc_b, a, sig)

sw_out, berm_out = {}, {}
for sw in md["swaptions"]:
    T0, tenor, fq, key = float(sw["expiry"]), float(sw["tenor"]), float(sw["frequency"]), sw["description"]
    pay = T0 + fq * np.arange(1, int(round(tenor / fq)) + 1)
    K = float((disc(T0) - disc(pay[-1])) / (fq * disc(pay).sum()))
    c = np.full(len(pay), K * fq); c[-1] += 1.0
    g = lambda r: float(np.sum(c * P_hw(disc, f0, a, sig, T0, pay, r)[:, 0])) - 1.0
    r_star = brentq(g, -1.0, 2.0)
    Ki = P_hw(disc, f0, a, sig, T0, pay, r_star)[:, 0]
    analytic = float(np.sum(c * zbp(disc, a, sig, T0, pay, Ki)))
    eu, _ = tree_price(tree, disc, f0, a, sig, pay, c, [T0])
    eu_b, _ = tree_price(tree_b, disc_b, f0_b, a, sig, pay, c, [T0])
    berm, bnd = tree_price(tree, disc, f0, a, sig, pay, c, list(T0 + fq * np.arange(len(pay))))
    sw_out[key] = {"expiry": T0, "tenor": tenor, "atm_rate": K, "r_star": float(r_star), "analytical_price": analytic,
                   "analytical_bps": analytic * 1e4, "tree_price": eu, "tree_bps": eu * 1e4,
                   "tree_analytical_diff_bps": abs(eu - analytic) * 1e4, "dv01_bps_per_bp": (eu_b - eu) * 1e4,
                   "payment_dates": [float(x) for x in pay]}
    berm_out[key] = {"european_tree_bps": eu * 1e4, "bermudan_bps": berm * 1e4, "premium_bps": (berm - eu) * 1e4,
                     "premium_pct": (berm - eu) / eu * 100.0, "exercise_boundary": bnd}
json.dump(sw_out, open(os.path.join(OUT_DIR, "swaption_prices.json"), "w"), indent=2)
json.dump(berm_out, open(os.path.join(OUT_DIR, "bermudan_prices.json"), "w"), indent=2)

flags = {"calibration_rmse_bps": rmse_bps, "calibration_rmse_ok": bool(rmse_bps < 5.0),
         "params_in_range": bool(0.01 < a < 0.5 and 0.005 < sig < 0.03), "a": a, "sigma": sig,
         "tree_zcb_match_market": bool(diag["abs_diff"].max() < 1e-6),
         "tree_vs_jamshidian_ok": bool(all(v["tree_analytical_diff_bps"] < 3.0 for v in sw_out.values())),
         "dv01_positive": bool(all(v["dv01_bps_per_bp"] > 0 for v in sw_out.values())),
         "bermudan_geq_european": bool(all(v["bermudan_bps"] >= v["european_tree_bps"] - 1e-9 for v in berm_out.values())),
         "bermudan_premium_positive": bool(all(v["premium_bps"] > 0 for v in berm_out.values()))}
flags["all_checks_passed"] = bool(all(v for k, v in flags.items() if isinstance(v, bool)))
json.dump(flags, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
