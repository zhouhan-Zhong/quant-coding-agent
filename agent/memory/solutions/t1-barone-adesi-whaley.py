# memory-solution
# unit: t1-barone-adesi-whaley | category: derivatives-pricing | tags: american-options, baw-quadratic-approximation, critical-price, crr-binomial, early-exercise-premium, dividend-yield, spy-calibration
# deliverables: calibration.json, baw_prices.csv, critical_prices.csv, binomial_comparison.csv, dividend_sensitivity.csv, summary.json
# BAW (1987): M = 2r/s^2, N = 2(r-q)/s^2, K_T = 1 - e^{-rT}; q2 = (-(N-1) + sqrt((N-1)^2 + 4M/K_T))/2 (call), q1 with "-" (put)
#   call: S* solves S*-K = c(S*) + (1 - e^{-qT} N(d1(S*))) S*/q2 ;  C = c(S) + A2 (S/S*)^q2 for S<S*, else S-K, A2 = (S*/q2)(1-e^{-qT}N(d1(S*)))
#   put : S** solves K-S** = p(S**) - (1 - e^{-qT} N(-d1(S**))) S**/q1 ; P = p(S) + A1 (S/S**)^q1 for S>S**, else K-S, A1 = -(S**/q1)(1-e^{-qT}N(-d1(S**)))
#   q = 0  -> American call = European call (no finite S*).  CRR with dividend yield: p = (e^{(r-q)dt} - d)/(u - d).
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)

px = pd.read_csv(os.path.join(DATA_DIR, "spy_daily.csv"), parse_dates=["date"]).sort_values("date")
close = px["close"].to_numpy(float); logret = np.diff(np.log(close))
S0 = float(close[-1]); sigma = float(logret.std(ddof=1) * np.sqrt(252.0))
r, D0 = 0.05, 0.013
json.dump({"S0": S0, "sigma": sigma, "r": r, "D": D0, "n_returns": int(len(logret))},
          open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)

def bs(S, K, T, q, s=sigma):
    sq = s * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * s * s) * T) / sq; d2 = d1 - sq
    c = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    p = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    return c, p, d1

def baw(S, K, T, q, s=sigma):
    """Return (amer_call, amer_put, S_star_call, S_star_put)."""
    c, p, _ = bs(S, K, T, q, s)
    M = 2 * r / (s * s); N = 2 * (r - q) / (s * s); KT = 1.0 - math.exp(-r * T)
    root = math.sqrt((N - 1) ** 2 + 4 * M / KT)
    q2 = (-(N - 1) + root) / 2; q1 = (-(N - 1) - root) / 2
    # ---- call
    if q <= 0:
        amer_c, s_star_c = c, float("inf")
    else:
        def fc(Ss):
            cc, _, d1 = bs(Ss, K, T, q, s)
            return Ss - K - cc - (1 - math.exp(-q * T) * norm.cdf(d1)) * Ss / q2
        hi = K
        while fc(hi) < 0: hi *= 2
        s_star_c = brentq(fc, K * (1 + 1e-9), hi, xtol=1e-10, maxiter=200)
        _, _, d1s = bs(s_star_c, K, T, q, s)
        A2 = (s_star_c / q2) * (1 - math.exp(-q * T) * norm.cdf(d1s))
        amer_c = c + A2 * (S / s_star_c) ** q2 if S < s_star_c else S - K
    # ---- put
    def fp(Ss):
        _, pp, d1 = bs(Ss, K, T, q, s)
        return K - Ss - pp + (1 - math.exp(-q * T) * norm.cdf(-d1)) * Ss / q1
    lo = K * 1e-3
    while fp(lo) < 0: lo *= 0.5
    s_star_p = brentq(fp, lo, K * (1 - 1e-9), xtol=1e-10, maxiter=200)
    _, _, d1p = bs(s_star_p, K, T, q, s)
    A1 = -(s_star_p / q1) * (1 - math.exp(-q * T) * norm.cdf(-d1p))
    amer_p = p + A1 * (S / s_star_p) ** q1 if S > s_star_p else K - S
    return max(amer_c, c), max(amer_p, p), s_star_c, s_star_p

def crr_american(S, K, T, q, n=1000, call=True, s=sigma):
    dt = T / n; u = math.exp(s * math.sqrt(dt)); d = 1 / u
    pu = (math.exp((r - q) * dt) - d) / (u - d); disc = math.exp(-r * dt)
    j = np.arange(n + 1)
    ST = S * u ** j * d ** (n - j)
    V = np.maximum(ST - K, 0.0) if call else np.maximum(K - ST, 0.0)
    for step in range(n - 1, -1, -1):
        j = np.arange(step + 1)
        Snode = S * u ** j * d ** (step - j)
        V = disc * (pu * V[1:] + (1 - pu) * V[:-1])
        ex = np.maximum(Snode - K, 0.0) if call else np.maximum(K - Snode, 0.0)
        V = np.maximum(V, ex)
    return float(V[0])

rows = []
for m in [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]:
    for T in [0.25, 0.50, 1.00]:
        K = S0 / m
        c, p, _ = bs(S0, K, T, D0); ac, ap, ssc, ssp = baw(S0, K, T, D0)
        rows.append({"moneyness": m, "T": T, "K": K, "bs_call": c, "bs_put": p, "baw_call": ac, "baw_put": ap,
                     "premium_call": ac - c, "premium_put": ap - p, "S_star_call": ssc, "S_star_put": ssp})
bawdf = pd.DataFrame(rows); bawdf.to_csv(os.path.join(OUT_DIR, "baw_prices.csv"), index=False)

rows = []
for T in [0.01, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00]:
    _, _, ssc, ssp = baw(S0, S0, T, D0)
    rows.append({"T": T, "S_star_call": ssc, "S_star_put": ssp, "S_star_call_ratio": ssc / S0, "S_star_put_ratio": ssp / S0})
crit = pd.DataFrame(rows); crit.to_csv(os.path.join(OUT_DIR, "critical_prices.csv"), index=False)

rows = []
for T in [0.25, 0.50, 1.00]:
    ac, ap, _, _ = baw(S0, S0, T, D0)
    cc, cp = crr_american(S0, S0, T, D0, call=True), crr_american(S0, S0, T, D0, call=False)
    rows.append({"T": T, "baw_call": ac, "baw_put": ap, "crr_call": cc, "crr_put": cp, "call_diff": ac - cc, "put_diff": ap - cp,
                 "call_reldiff": abs(ac - cc) / cc, "put_reldiff": abs(ap - cp) / cp})
binom = pd.DataFrame(rows); binom.to_csv(os.path.join(OUT_DIR, "binomial_comparison.csv"), index=False)

rows = []
for Dv in [0.00, 0.01, 0.02, 0.03, 0.05]:
    c, p, _ = bs(S0, S0, 1.0, Dv); ac, ap, _, _ = baw(S0, S0, 1.0, Dv)
    rows.append({"D": Dv, "baw_call": ac, "baw_put": ap, "bs_call": c, "bs_put": p, "premium_call": ac - c, "premium_put": ap - p})
divs = pd.DataFrame(rows); divs.to_csv(os.path.join(OUT_DIR, "dividend_sensitivity.csv"), index=False)

atm1 = bawdf[(np.isclose(bawdf["moneyness"], 1.0)) & (np.isclose(bawdf["T"], 1.0))].iloc[0]
summary = {
    "baw_call_atm_T1": float(atm1["baw_call"]), "baw_put_atm_T1": float(atm1["baw_put"]),
    "premium_call_atm_T1": float(atm1["premium_call"]), "premium_put_atm_T1": float(atm1["premium_put"]),
    "S_star_call_T1": float(atm1["S_star_call"]), "S_star_put_T1": float(atm1["S_star_put"]),
    "max_crr_reldiff_call": float(binom["call_reldiff"].max()), "max_crr_reldiff_put": float(binom["put_reldiff"].max()),
    "call_premium_increases_with_D": bool(np.all(np.diff(divs["premium_call"].values) >= -1e-10)),
    "put_premium_positive": bool((bawdf["premium_put"] > -1e-10).all()),
    "call_geq_euro": bool((bawdf["baw_call"] >= bawdf["bs_call"] - 1e-10).all()),
    "put_geq_euro": bool((bawdf["baw_put"] >= bawdf["bs_put"] - 1e-10).all()),
    "S_star_call_increases_with_T": bool(np.all(np.diff(crit["S_star_call"].values) >= -1e-6)),
    "S_star_put_decreases_with_T": bool(np.all(np.diff(crit["S_star_put"].values) <= 1e-6)),
    "zero_D_call_premium_negligible": bool(abs(float(divs[np.isclose(divs["D"], 0.0)]["premium_call"].iloc[0])) < 0.01),
}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
