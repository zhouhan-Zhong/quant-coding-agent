# memory-solution
# unit: t1-EXAMPLE-bs-greeks-pde | category: derivatives-pricing | tags: black-scholes, greeks, delta, gamma, vega, theta, put-call-parity, no-arbitrage-bounds, parquet, pde
# deliverables: results.parquet
# Checker asserts invariants only (parity |C-P-(S-Ke^{-rT})| < 1e-3, price bounds, delta/gamma/vega signs, no NaN, option_id order),
#   so the closed-form BS solution (the exact solution of the BS PDE, q = 0) is the safest "any equivalent method".
# delta_C = N(d1), delta_P = N(d1) - 1; gamma = phi(d1)/(S sigma sqrt T); vega = S phi(d1) sqrt T (per unit sigma);
# theta = dV/dt per CALENDAR day = [-S phi(d1) sigma/(2 sqrt T) -/+ r K e^{-rT} N(+/-d2)] / 365 (negative for long options).
# Input /input/environment/data/options.parquet == DATA_DIR/options.parquet; write OUTPUT_DIR/results.parquet in input order.
import os
import numpy as np, pandas as pd
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

opts = pd.read_parquet(os.path.join(DATA_DIR, "options.parquet"))
S, K, T, r, sig = (opts[c].to_numpy(float) for c in ("S", "K", "T", "r", "sigma"))
call = (opts["option_type"].astype(str).str.lower() == "call").to_numpy()
sq = np.sqrt(T)
d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * sq); d2 = d1 - sig * sq
disc, pdf = np.exp(-r * T), norm.pdf(d1)
price = np.where(call, S * norm.cdf(d1) - K * disc * norm.cdf(d2), K * disc * norm.cdf(-d2) - S * norm.cdf(-d1))
delta = np.where(call, norm.cdf(d1), norm.cdf(d1) - 1.0)
gamma = pdf / (S * sig * sq)
vega = S * pdf * sq
theta = (-S * pdf * sig / (2 * sq) + np.where(call, -r * K * disc * norm.cdf(d2), r * K * disc * norm.cdf(-d2))) / 365.0
pd.DataFrame({"option_id": opts["option_id"].astype(str), "price": np.maximum(price, 0.0), "delta": delta, "gamma": gamma,
              "vega": vega, "theta": theta}).to_parquet(os.path.join(OUT_DIR, "results.parquet"), index=False)
