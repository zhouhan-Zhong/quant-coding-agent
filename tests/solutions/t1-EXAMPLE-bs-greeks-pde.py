# DELIVERABLES: results.parquet
import numpy as np, pandas as pd
from scipy.stats import norm
opts = pd.read_parquet("/input/environment/data/options.parquet")
S = opts["S"].to_numpy(float); K = opts["K"].to_numpy(float); T = opts["T"].to_numpy(float)
r = opts["r"].to_numpy(float); sig = opts["sigma"].to_numpy(float)
call = (opts["option_type"].astype(str) == "call").to_numpy()
sq = np.sqrt(T); d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * sq); d2 = d1 - sig * sq
disc = np.exp(-r * T); pdf = norm.pdf(d1)
price = np.where(call, S * norm.cdf(d1) - K * disc * norm.cdf(d2), K * disc * norm.cdf(-d2) - S * norm.cdf(-d1))
delta = np.where(call, norm.cdf(d1), norm.cdf(d1) - 1.0)
gamma = pdf / (S * sig * sq); vega = S * pdf * sq
theta = np.where(call, -S * pdf * sig / (2 * sq) - r * K * disc * norm.cdf(d2), -S * pdf * sig / (2 * sq) + r * K * disc * norm.cdf(-d2)) / 365.0
pd.DataFrame({"option_id": opts["option_id"], "price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta}).to_parquet("/output/results.parquet", index=False)
