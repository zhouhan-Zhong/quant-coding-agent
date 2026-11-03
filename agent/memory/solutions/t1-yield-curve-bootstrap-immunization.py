# memory-solution
# unit: t1-yield-curve-bootstrap-immunization | category: rates-curves | tags: bootstrap, par-yields, discount-factors, spot-forward, ytm, duration, convexity, key-rate-duration, z-spread, immunization, slsqp, svensson, l-bfgs-b, stress-test
# deliverables: discount_factors.csv, bond_analytics.csv, liability_analytics.json, immunization_portfolio.json, nelson_siegel.json, stress_test.csv
# Annual par bonds: d(T) = (1 - y_T sum_{k<T} d(k)) / (1 + y_T); spot r = -ln d/T; fwd f(T) = -ln(d(T)/d(T-1)), f(1) = r(1).
# Bond: P = F(c sum d + d_M); YTM brentq on [1e-6, .3] (annual comp.); Mac D = sum k CF d/P; ModD = D/(1+y);
#   convexity = sum k(k+1) CF d / (P (1+y)^2); KRD_K = -(P(r_K + 1bp) - P)/(P 1e-4), only tenor K bumped (no interpolation);
#   z-spread brentq on [-.05, .5] of sum CF exp(-(r_k + z) k) = P.
# Immunization: weights w >= 0 via SLSQP, equalities sum w = 1, sum w krd2 = krdL2, sum w krd5 = krdL5; objective = squared
#   Macaulay-duration mismatch (makes the 1-dim feasible family unique and the parallel-shift hedge good). x = w * PV_L.
# Svensson (lambda = time scale): r = b0 + b1 f1 + b2 (f1 - e^{-T/l1}) + b3 (f2 - e^{-T/l2}), f = (1-e^{-T/l})/(T/l);
#   L-BFGS-B ftol 1e-14 gtol 1e-10 maxiter 5000, bounds b0,l1,l2 > 0, 3 starts, keep lowest SSE -> b0 ~ 0.028, l2 ~ 4.7.
# Stress: shift spot rates (parallel +100bp; steepener -50bp@1 -> +50bp@10 linear; butterfly interp over (2,+30),(5,-30),(10,+30)).
import os, json, math
import numpy as np, pandas as pd
from scipy.optimize import brentq, minimize

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

par = pd.read_csv(os.path.join(DATA_DIR, "par_yields.csv")).sort_values("maturity_years")
bonds = pd.read_csv(os.path.join(DATA_DIR, "bonds.csv"))
liab = json.load(open(os.path.join(DATA_DIR, "liability.json")))["cash_flows"]
T = np.arange(1, 11); y = par["par_yield"].to_numpy(float)

d = np.zeros(10)
for i in range(10):
    d[i] = (1.0 - y[i] * d[:i].sum()) / (1.0 + y[i])
r = -np.log(d) / T
fwd = np.r_[r[0], -np.log(d[1:] / d[:-1])]
pd.DataFrame({"maturity_years": T, "discount_factor": d, "spot_rate": r, "forward_rate": fwd}).to_csv(
    os.path.join(OUT_DIR, "discount_factors.csv"), index=False)

def cfs(F, c, M):                                      # cash-flow vector on the 1..10 grid
    cf = np.zeros(10); cf[:M] = F * c; cf[M - 1] += F; return cf
def pv(cf, rates): return float(np.sum(cf * np.exp(-rates * T)))
def krd(cf, K):
    rb = r.copy(); rb[K - 1] += 1e-4
    return -(pv(cf, rb) - pv(cf, r)) / (pv(cf, r) * 1e-4)

rows, bond_cf = [], {}
for _, b in bonds.iterrows():
    F, c, M = float(b["face_value"]), float(b["coupon_rate"]), int(b["maturity_years"])
    cf = cfs(F, c, M); bond_cf[b["bond_id"]] = cf
    P = pv(cf, r)
    ytm = brentq(lambda yy: np.sum(cf / (1 + yy) ** T) - P, 1e-6, 0.30)
    mac = float(np.sum(T * cf * d) / P)
    conv = float(np.sum(T * (T + 1) * cf * d) / (P * (1 + ytm) ** 2))
    z = brentq(lambda zz: np.sum(cf * np.exp(-(r + zz) * T)) - P, -0.05, 0.50)
    rows.append({"bond_id": b["bond_id"], "price": P, "ytm": ytm, "macaulay_duration": mac,
                 "modified_duration": mac / (1 + ytm), "convexity": conv,
                 "krd_2": krd(cf, 2), "krd_5": krd(cf, 5), "krd_10": krd(cf, 10), "z_spread": z})
ba = pd.DataFrame(rows)
ba.to_csv(os.path.join(OUT_DIR, "bond_analytics.csv"), index=False)

lcf = np.zeros(10)
for c_ in liab:
    lcf[int(c_["years"]) - 1] += float(c_["amount"])
PV_L = pv(lcf, r); D_L = float(np.sum(T * lcf * d) / PV_L)
kL = {K: krd(lcf, K) for K in (2, 5, 10)}
json.dump({"pv_liability": round(PV_L, 4), "macaulay_duration": round(D_L, 6),
           "krd_2": round(kL[2], 6), "krd_5": round(kL[5], 6), "krd_10": round(kL[10], 6)},
          open(os.path.join(OUT_DIR, "liability_analytics.json"), "w"), indent=2)

ids = list(ba["bond_id"]); K2, K5, K10 = (ba[f"krd_{k}"].to_numpy() for k in (2, 5, 10))
Dv = ba["macaulay_duration"].to_numpy(); price = ba["price"].to_numpy()
cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "eq", "fun": lambda w: w @ K2 - kL[2]},
        {"type": "eq", "fun": lambda w: w @ K5 - kL[5]}]
w0 = np.full(4, 0.25); res = None
for ftol in (1e-10, 1e-8):
    res = minimize(lambda w: (w @ Dv - D_L) ** 2, w0, method="SLSQP", bounds=[(0, None)] * 4,
                   constraints=cons, options={"ftol": ftol, "maxiter": 1000})
    if res.success: break
w = np.clip(res.x, 0, None); x = w * PV_L; units = x / price
port = {f"bond_{bid}": {"units": round(float(u), 6), "dollar_allocation": round(float(a), 4)} for bid, u, a in zip(ids, units, x)}
port.update({"portfolio_pv": round(float(x.sum()), 4), "portfolio_krd_2": round(float(w @ K2), 6),
             "portfolio_krd_5": round(float(w @ K5), 6), "portfolio_krd_10": round(float(w @ K10), 6)})
json.dump(port, open(os.path.join(OUT_DIR, "immunization_portfolio.json"), "w"), indent=2)

def svensson(p, t):
    b0, b1, b2, b3, l1, l2 = p
    x1, x2 = t / l1, t / l2
    f1, f2 = (1 - np.exp(-x1)) / x1, (1 - np.exp(-x2)) / x2
    return b0 + b1 * f1 + b2 * (f1 - np.exp(-x1)) + b3 * (f2 - np.exp(-x2))
sse = lambda p: float(np.sum((svensson(p, T) - r) ** 2))
starts = [[r[-1], r[0] - r[-1], 0.5, -0.5, 1.0, 5.0], [0.03, -0.01, 0.1, -0.1, 2.0, 3.0], [r.mean(), 0, 0, 0, 0.5, 2.0]]
bnds = [(1e-6, None), (None, None), (None, None), (None, None), (1e-6, None), (1e-6, None)]
best = None
for s0 in starts:
    rs = minimize(sse, s0, method="L-BFGS-B", bounds=bnds, options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 5000})
    if best is None or rs.fun < best.fun: best = rs
b0, b1, b2, b3, l1, l2 = [float(v) for v in best.x]
json.dump({"beta_0": round(b0, 8), "beta_1": round(b1, 8), "beta_2": round(b2, 8), "beta_3": round(b3, 8),
           "lambda_1": round(l1, 8), "lambda_2": round(l2, 8), "rmse": round(math.sqrt(best.fun / 10), 10)},
          open(os.path.join(OUT_DIR, "nelson_siegel.json"), "w"), indent=2)

shifts = {"parallel_up_100": np.full(10, 0.01),
          "steepener": -0.005 + 0.01 * (T - 1) / 9.0,
          "butterfly": np.interp(T, [2, 5, 10], [0.003, -0.003, 0.003])}
st = []
for name, s in shifts.items():
    ppv = float(sum(u * pv(bond_cf[bid], r + s) for bid, u in zip(ids, units)))
    lpv = pv(lcf, r + s)
    st.append({"scenario": name, "portfolio_pv": ppv, "liability_pv": lpv,
               "portfolio_pv_change": ppv - float(x.sum()), "liability_pv_change": lpv - PV_L})
pd.DataFrame(st).to_csv(os.path.join(OUT_DIR, "stress_test.csv"), index=False)
