# memory-solution
# unit: t1-lookback-options | category: derivatives-pricing | tags: lookback-options, conze-viswanathan, goldman-sosin-gatto, running-maximum, monte-carlo, spy-calibration
# deliverables: calibration.json, fixed_strike_lookback.csv, floating_strike_lookback.csv, summary.json
# closed forms (q = 0, continuous monitoring, M = running max so far, m = running min so far; at inception M = m = S0):
#   floating call: S N(a1) - S (s^2/2r) N(-a1) - m e^{-rT}[N(a2) - (s^2/2r) e^{Y1} N(-a3)],
#       a1 = [ln(S/m)+(r+s^2/2)T]/(s√T), a2 = a1 - s√T, a3 = [ln(S/m)+(-r+s^2/2)T]/(s√T), Y1 = -2(r-s^2/2) ln(S/m)/s^2
#   floating put : M e^{-rT}[N(b1) - (s^2/2r) e^{Y2} N(-b3)] + S (s^2/2r) N(-b2) - S N(b2),
#       b1 = [ln(M/S)+(-r+s^2/2)T]/(s√T), b2 = b1 - s√T, b3 = [ln(M/S)+(r-s^2/2)T]/(s√T), Y2 = 2(r-s^2/2) ln(M/S)/s^2
#   fixed call (K > M): S N(d1) - K e^{-rT} N(d2) + S e^{-rT}(s^2/2r)[-(S/K)^{-2r/s^2} N(d1 - 2r√T/s) + e^{rT} N(d1)]
#   fixed call (K <= M): e^{-rT}(M-K) + same expression with K -> M
# MC with daily steps is biased low vs continuous monitoring (Broadie-Glasserman): report it honestly.
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

px = pd.read_csv(os.path.join(DATA_DIR, "spy_daily.csv"))
px.columns = [c.lower() for c in px.columns]
px["date"] = pd.to_datetime(px["date"]); px = px.sort_values("date")
close = px["close"].to_numpy(float); logret = np.diff(np.log(close))
S0 = float(close[-1]); ret_mean, ret_std = float(logret.mean()), float(logret.std(ddof=1))
sigma = ret_std * math.sqrt(252.0); r = 0.05
json.dump({"n_returns": int(len(logret)), "return_mean": ret_mean, "return_std": ret_std, "sigma": sigma, "S0": S0, "r": r, "D": 0.0},
          open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)

N = norm.cdf
def fixed_call(S, K, T, M):
    s2r = sigma ** 2 / (2 * r); sq = sigma * math.sqrt(T)
    if K > M:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq; d2 = d1 - sq
        return S * N(d1) - K * math.exp(-r * T) * N(d2) + S * math.exp(-r * T) * s2r * (
            -(S / K) ** (-2 * r / sigma ** 2) * N(d1 - 2 * r * math.sqrt(T) / sigma) + math.exp(r * T) * N(d1))
    e1 = (math.log(S / M) + (r + 0.5 * sigma ** 2) * T) / sq; e2 = e1 - sq
    return math.exp(-r * T) * (M - K) + S * N(e1) - M * math.exp(-r * T) * N(e2) + S * math.exp(-r * T) * s2r * (
        -(S / M) ** (-2 * r / sigma ** 2) * N(e1 - 2 * r * math.sqrt(T) / sigma) + math.exp(r * T) * N(e1))

def floating_call(S, T, m):
    s2r = sigma ** 2 / (2 * r); sq = sigma * math.sqrt(T)
    a1 = (math.log(S / m) + (r + 0.5 * sigma ** 2) * T) / sq; a2 = a1 - sq
    a3 = (math.log(S / m) + (-r + 0.5 * sigma ** 2) * T) / sq
    Y1 = -2 * (r - 0.5 * sigma ** 2) * math.log(S / m) / sigma ** 2
    return S * N(a1) - S * s2r * N(-a1) - m * math.exp(-r * T) * (N(a2) - s2r * math.exp(Y1) * N(-a3))

def floating_put(S, T, M):
    s2r = sigma ** 2 / (2 * r); sq = sigma * math.sqrt(T)
    b1 = (math.log(M / S) + (-r + 0.5 * sigma ** 2) * T) / sq; b2 = b1 - sq
    b3 = (math.log(M / S) + (r - 0.5 * sigma ** 2) * T) / sq
    Y2 = 2 * (r - 0.5 * sigma ** 2) * math.log(M / S) / sigma ** 2
    return M * math.exp(-r * T) * (N(b1) - s2r * math.exp(Y2) * N(-b3)) + S * s2r * N(-b2) - S * N(b2)

# ---- Monte Carlo: one set of 200k daily paths over 1y; record at T = 0.25, 0.5, 1.0
np.random.seed(42)
n_paths, steps_per_year = 200_000, 252
maturities = [0.25, 0.5, 1.0]
dt = 1.0 / steps_per_year
S = np.full(n_paths, S0); Mx = S.copy(); Mn = S.copy()
snap = {}
for step in range(1, steps_per_year + 1):
    z = np.random.standard_normal(n_paths)
    S = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z)
    Mx = np.maximum(Mx, S); Mn = np.minimum(Mn, S)
    for T in maturities:
        if step == int(round(T * steps_per_year)):
            snap[T] = (S.copy(), Mx.copy(), Mn.copy())

def mc_stats(payoff, T):
    disc = math.exp(-r * T)
    return float(disc * payoff.mean()), float(disc * payoff.std(ddof=1) / math.sqrt(len(payoff)))

rows = []
for T in maturities:
    ST, MT, mT = snap[T]
    for m in [0.90, 0.95, 1.00, 1.05, 1.10]:
        K = S0 * m
        mc, se = mc_stats(np.maximum(MT - K, 0.0), T)
        rows.append({"T": T, "K": K, "moneyness": m, "cf_price": fixed_call(S0, K, T, S0), "mc_price": mc, "mc_std_err": se})
fixed = pd.DataFrame(rows); fixed.to_csv(os.path.join(OUT_DIR, "fixed_strike_lookback.csv"), index=False)

rows = []
for T in maturities:
    ST, MT, mT = snap[T]; mc, se = mc_stats(ST - mT, T)
    rows.append({"T": T, "type": "call", "cf_price": floating_call(S0, T, S0), "mc_price": mc, "mc_std_err": se})
for T in maturities:
    ST, MT, mT = snap[T]; mc, se = mc_stats(MT - ST, T)
    rows.append({"T": T, "type": "put", "cf_price": floating_put(S0, T, S0), "mc_price": mc, "mc_std_err": se})
flt = pd.DataFrame(rows); flt.to_csv(os.path.join(OUT_DIR, "floating_strike_lookback.csv"), index=False)

err_f = (fixed["cf_price"] - fixed["mc_price"]).abs(); err_fl = (flt["cf_price"] - flt["mc_price"]).abs()
def atm(T): return float(fixed[(np.isclose(fixed["T"], T)) & (np.isclose(fixed["moneyness"], 1.0))]["cf_price"].iloc[0])
summary = {
    "sigma": sigma, "S0": S0, "n_fixed_strike": int(len(fixed)), "n_floating_strike": int(len(flt)),
    "max_mc_error_fixed": float(err_f.max()), "max_mc_error_floating": float(err_fl.max()),
    "mc_validates": bool((err_f < 2.0 * fixed["mc_std_err"]).all() and (err_fl < 2.0 * flt["mc_std_err"]).all()),
    "atm_fixed_T025": atm(0.25), "atm_fixed_T050": atm(0.5), "atm_fixed_T100": atm(1.0),
    "float_call_T050": float(flt[(flt["type"] == "call") & np.isclose(flt["T"], 0.5)]["cf_price"].iloc[0]),
    "float_put_T050": float(flt[(flt["type"] == "put") & np.isclose(flt["T"], 0.5)]["cf_price"].iloc[0]),
}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
