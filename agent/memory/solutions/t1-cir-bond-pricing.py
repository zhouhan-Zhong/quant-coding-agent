# memory-solution
# unit: t1-cir-bond-pricing | category: rates-curves | tags: cir, short-rate, mle, ncx2, l-bfgs-b, random-restarts, zero-coupon-bond, riccati, yield-curve, forward-rates, fred
# deliverables: calibration.json, bond_prices.csv, forward_rates.csv, summary.json
# DFF is a calendar-daily series (weekends included) -> dt = 1/365; filter date >= 2016-01-01, rates /100.
# CIR exact transition: c = 2k/(s^2 (1-e^{-k dt})), 2c r_{t+dt} ~ ncx2(df = 4 k theta/s^2, nc = 2c r_t e^{-k dt})
#   -> log p(r_{t+dt}|r_t) = ncx2.logpdf(2c r_{t+dt}, df, nc) + log(2c); use logpdf (nc ~ 1e5-1e7 overflows pdf).
# MLE: L-BFGS-B on -LL with bounds k[0.01,1] theta[0.001,0.1] s[0.001,0.2]; grid start + 5 seeded restarts, keep best.
# Riccati ZCB: gamma = sqrt(k^2 + 2 s^2), B = 2(e^{g tau}-1)/((g+k)(e^{g tau}-1)+2g), A = [2g e^{(k+g)tau/2}/(...)]^{2k theta/s^2}
#   P = A exp(-B r_T) with r_T the last DFF; y = -ln P/tau; forward f(t1,t2) = -(ln P2 - ln P1)/(t2 - t1).
# Market yields: last-row DGS1/2/5/10 mapped to tau 1,2,5,10 (DGS30 unused); other taus NaN; errors over non-NaN only.
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import ncx2
from scipy.optimize import minimize

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "fred_rates.csv"), parse_dates=["date"]).sort_values("date")
df = df[df["date"] >= "2016-01-01"].reset_index(drop=True)
r = (df["DFF"].to_numpy(float) / 100.0)
r = r[np.isfinite(r)]
dt = 1.0 / 365.0                                   # calendar-daily observations
r0, r1 = r[:-1], r[1:]

def loglik(p):
    k, th, s = p
    e = math.exp(-k * dt)
    c = 2.0 * k / (s * s * (1.0 - e))
    dof = 4.0 * k * th / (s * s)
    nc = 2.0 * c * r0 * e
    lp = ncx2.logpdf(2.0 * c * r1, dof, nc) + math.log(2.0 * c)
    return float(np.sum(lp)) if np.all(np.isfinite(lp)) else -1e12

bounds = [(0.01, 1.0), (0.001, 0.10), (0.001, 0.20)]
rng = np.random.default_rng(42)
starts = [[0.3, float(r.mean()), float(np.std(np.diff(r)) * math.sqrt(365.0 / max(r.mean(), 1e-4)))]]
for _ in range(5):
    starts.append([rng.uniform(0.02, 0.9), rng.uniform(0.005, 0.08), rng.uniform(0.005, 0.15)])
best = None
for s0 in starts:
    s0 = [min(max(v, lo), hi) for v, (lo, hi) in zip(s0, bounds)]
    res = minimize(lambda p: -loglik(p), s0, method="L-BFGS-B", bounds=bounds)
    if np.isfinite(res.fun) and (best is None or -res.fun > best[0]):
        best = (-res.fun, res.x)
LL, (kappa, theta, sigma) = best[0], [float(v) for v in best[1]]

feller_ratio = 2.0 * kappa * theta / sigma ** 2
feller = bool(feller_ratio >= 1.0)
r_cur = float(r[-1])
json.dump({"n_obs": int(len(r)), "kappa": kappa, "theta": theta, "sigma": sigma, "log_likelihood": LL,
           "feller_condition_satisfied": feller, "feller_ratio": float(feller_ratio), "r_current": r_cur,
           "r_mean": float(r.mean()), "r_std": float(r.std(ddof=1)), "r_min": float(r.min()), "r_max": float(r.max())},
          open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)

def zcb(tau):
    g = math.sqrt(kappa ** 2 + 2.0 * sigma ** 2)
    eg = math.exp(g * tau)
    den = (g + kappa) * (eg - 1.0) + 2.0 * g
    B = 2.0 * (eg - 1.0) / den
    A = (2.0 * g * math.exp((kappa + g) * tau / 2.0) / den) ** (2.0 * kappa * theta / sigma ** 2)
    return A * math.exp(-B * r_cur)

taus = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
last = df.iloc[-1]
mkt = {1.0: "DGS1", 2.0: "DGS2", 5.0: "DGS5", 10.0: "DGS10"}
rows = []
for tau in taus:
    P = zcb(tau); y = -math.log(P) / tau
    my = float(last[mkt[tau]]) / 100.0 if tau in mkt and pd.notna(last[mkt[tau]]) else float("nan")
    rows.append({"tau": tau, "zcb_price": P, "zero_rate": -math.log(P), "model_yield": y,
                 "market_yield": my, "yield_diff": y - my})
bp = pd.DataFrame(rows).sort_values("tau")
bp.to_csv(os.path.join(OUT_DIR, "bond_prices.csv"), index=False)

fw = pd.DataFrame({"tau1": taus[:-1], "tau2": taus[1:]})
lnP = np.log(bp["zcb_price"].to_numpy())
fw["forward_rate"] = -(lnP[1:] - lnP[:-1]) / (fw["tau2"] - fw["tau1"]).to_numpy()
fw.to_csv(os.path.join(OUT_DIR, "forward_rates.csv"), index=False)

err = bp["yield_diff"].abs().dropna()
json.dump({"kappa": kappa, "theta": theta, "sigma": sigma, "r_current": r_cur, "feller_satisfied": feller,
           "mean_yield_error": float(err.mean()), "max_yield_error": float(err.max()),
           "yield_curve_upward_sloping": bool(bp["model_yield"].iloc[-1] > bp["model_yield"].iloc[0])},
          open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
