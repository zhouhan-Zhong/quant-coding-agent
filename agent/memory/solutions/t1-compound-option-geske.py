# memory-solution
# unit: t1-compound-option-geske | category: derivatives-pricing | tags: compound-options, geske, bivariate-normal, chooser-option, put-call-parity, monte-carlo, brentq, spy-calibration
# deliverables: calibration.json, compound_prices.csv, parity_check.csv, chooser_prices.csv, summary.json
# Geske (dividend yield q, rho = sqrt(T1/T2), S* solves inner BS value at T1 == K1):
#   a1 = [ln(S/S*) + (r-q+s^2/2)T1]/(s√T1), a2 = a1 - s√T1 ; b1 = [ln(S/K2) + (r-q+s^2/2)T2]/(s√T2), b2 = b1 - s√T2
#   call-on-call = S e^{-qT2} M(a1,b1;rho) - K2 e^{-rT2} M(a2,b2;rho) - K1 e^{-rT1} N(a2)
#   put-on-call  = K2 e^{-rT2} M(-a2,b2;-rho) - S e^{-qT2} M(-a1,b1;-rho) + K1 e^{-rT1} N(-a2)
#   put-on-put   = K2 e^{-rT2} M(-a2,-b2;rho) - S e^{-qT2} M(-a1,-b1;rho) - K1 e^{-rT1} N(-a2)
#   parity: Cc - Pc = C_BS(K2,T2) - K1 e^{-rT1}.  simple chooser = c(K,T) + e^{-q(T-Tc)} p(K e^{-(r-q)(T-Tc)}, Tc)
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm, multivariate_normal
from scipy.optimize import brentq

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

px = pd.read_csv(os.path.join(DATA_DIR, "spy_daily.csv"), parse_dates=["date"]).sort_values("date")
close = px["close"].to_numpy(float); logret = np.diff(np.log(close))
S0 = float(close[-1]); ret_mean, ret_std = float(logret.mean()), float(logret.std(ddof=1))
sigma = ret_std * math.sqrt(252.0); r, q = 0.05, 0.01
json.dump({"n_returns": int(len(logret)), "return_mean": ret_mean, "return_std": ret_std, "sigma": sigma, "S0": S0, "r": r, "D": q},
          open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)

def bs_call(S, K, T):
    sq = sigma * math.sqrt(T); d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sq
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d1 - sq)

def bs_put(S, K, T):
    sq = sigma * math.sqrt(T); d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sq
    return K * math.exp(-r * T) * norm.cdf(-(d1 - sq)) - S * math.exp(-q * T) * norm.cdf(-d1)

def M(a, b, rho):
    return float(multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([a, b]))

def geske(S, K1, K2, T1, T2, kind):
    rho = math.sqrt(T1 / T2)
    if kind in ("call-on-call", "put-on-call"):
        s_star = brentq(lambda x: bs_call(x, K2, T2 - T1) - K1, 1.0, 5 * S0)
    else:
        s_star = brentq(lambda x: bs_put(x, K2, T2 - T1) - K1, 1.0, 5 * S0)
    a1 = (math.log(S / s_star) + (r - q + 0.5 * sigma ** 2) * T1) / (sigma * math.sqrt(T1)); a2 = a1 - sigma * math.sqrt(T1)
    b1 = (math.log(S / K2) + (r - q + 0.5 * sigma ** 2) * T2) / (sigma * math.sqrt(T2)); b2 = b1 - sigma * math.sqrt(T2)
    if kind == "call-on-call":
        price = S * math.exp(-q * T2) * M(a1, b1, rho) - K2 * math.exp(-r * T2) * M(a2, b2, rho) - K1 * math.exp(-r * T1) * norm.cdf(a2)
    elif kind == "put-on-call":
        price = K2 * math.exp(-r * T2) * M(-a2, b2, -rho) - S * math.exp(-q * T2) * M(-a1, b1, -rho) + K1 * math.exp(-r * T1) * norm.cdf(-a2)
    else:  # put-on-put
        price = K2 * math.exp(-r * T2) * M(-a2, -b2, rho) - S * math.exp(-q * T2) * M(-a1, -b1, rho) - K1 * math.exp(-r * T1) * norm.cdf(-a2)
    return price, s_star

T1, T2 = 0.25, 1.0
np.random.seed(42)
n_mc = 300_000
z = np.random.standard_normal(n_mc)
S_T1 = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T1 + sigma * math.sqrt(T1) * z)

def inner_values(kind, K2):
    tau = T2 - T1; sq = sigma * math.sqrt(tau)
    d1 = (np.log(S_T1 / K2) + (r - q + 0.5 * sigma ** 2) * tau) / sq; d2 = d1 - sq
    if kind == "call-on-call":
        return S_T1 * math.exp(-q * tau) * norm.cdf(d1) - K2 * math.exp(-r * tau) * norm.cdf(d2)
    return K2 * math.exp(-r * tau) * norm.cdf(-d2) - S_T1 * math.exp(-q * tau) * norm.cdf(-d1)

rows, parity_rows = [], []
for kind in ("call-on-call", "put-on-put"):
    for K1 in [5, 10, 15, 20, 25]:
        for m in [0.95, 1.00, 1.05]:
            K2 = S0 * m
            price, s_star = geske(S0, K1, K2, T1, T2, kind)
            payoff = np.maximum(inner_values(kind, K2) - K1, 0.0) * math.exp(-r * T1)
            rows.append({"type": kind, "K1": K1, "K2": K2, "K2_moneyness": m, "T1": T1, "T2": T2, "S_star": s_star,
                         "compound_price": price, "mc_price": float(payoff.mean()), "mc_std_err": float(payoff.std(ddof=1) / math.sqrt(n_mc))})
            if kind == "call-on-call":
                pc, _ = geske(S0, K1, K2, T1, T2, "put-on-call")
                bsc = bs_call(S0, K2, T2); k1pv = K1 * math.exp(-r * T1)
                parity_rows.append({"K1": K1, "K2": K2, "K2_moneyness": m, "cc_price": price, "pc_price": pc, "bs_call": bsc, "K1_pv": k1pv,
                                    "parity_lhs": price - pc, "parity_rhs": bsc - k1pv, "parity_error": abs(price - pc - (bsc - k1pv))})
comp = pd.DataFrame(rows); comp.to_csv(os.path.join(OUT_DIR, "compound_prices.csv"), index=False)
par = pd.DataFrame(parity_rows); par.to_csv(os.path.join(OUT_DIR, "parity_check.csv"), index=False)

ch_rows = []
for Tc in [0.1, 0.25]:
    K, T = S0, 1.0
    c, p = bs_call(S0, K, T), bs_put(S0, K, T)
    chooser = c + math.exp(-q * (T - Tc)) * bs_put(S0, K * math.exp(-(r - q) * (T - Tc)), Tc)
    ch_rows.append({"T_choose": Tc, "K": K, "T_underlying": T, "chooser_price": chooser, "call_price": c, "put_price": p})
ch = pd.DataFrame(ch_rows); ch.to_csv(os.path.join(OUT_DIR, "chooser_prices.csv"), index=False)

mc_err = (comp["compound_price"] - comp["mc_price"]).abs()
summary = {
    "sigma": sigma, "S0": S0, "n_compound_prices": int(len(comp)),
    "max_parity_error": float(par["parity_error"].max()), "parity_holds": bool(par["parity_error"].max() < 0.01),
    "max_mc_error": float(mc_err.max()), "mc_validates": bool((mc_err <= 4 * comp["mc_std_err"] + 0.02).all()),
    "mean_cc_price": float(comp[comp["type"] == "call-on-call"]["compound_price"].mean()),
    "mean_pp_price": float(comp[comp["type"] == "put-on-put"]["compound_price"].mean()),
    "chooser_price_T010": float(ch[np.isclose(ch["T_choose"], 0.1)]["chooser_price"].iloc[0]),
    "chooser_price_T025": float(ch[np.isclose(ch["T_choose"], 0.25)]["chooser_price"].iloc[0]),
}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
