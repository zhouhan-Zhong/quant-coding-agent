# memory-solution
# unit: t1-variance-swap-replication | category: volatility | tags: variance-swap, log-contract, carr-madan, demeterfi-derman, option-chain-cleaning, otm-selection, implied-vol, brentq, trapezoidal-weights
# deliverables: results.json
# Keep only target expiry (90d); OTM side per strike: put if K < F else call (F = supplied forward, never recomputed).
# Filters on the OTM side: volume >= 10, otm_bid > 0, (ask - bid)/mid <= 0.5 -> 27 of 41 strikes kept (15 puts, 12 calls).
# IV per kept strike: brentq on BS(S, K, T, r, q) - otm_mid with xtol 1e-12 (residual << 1e-8); ATM IV = linear interpolation
# in strike between the kept strikes straddling F (K_lo < F <= K_hi).
# Fair variance: K_var2 = (2 e^{rT}/T) sum_i dK_i / K_i^2 * otm_mid_i, trapezoid dK on the non-uniform kept grid
# (one-sided at the ends, (K_{i+1} - K_{i-1})/2 inside). VRP = K_var2 - iv_atm^2; PnL = 10000 (sigma_RV^2 - K_var2).
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

P = json.load(open(os.path.join(DATA_DIR, "params.json")))
S, r, q = float(P["spot"]), float(P["risk_free_rate"]), float(P["dividend_yield"])
exp_id = P["target_expiry"]; T = float(P["expiries"][exp_id]); F = float(P["forward_prices"][exp_id])
flt = P["filters"]

ch = pd.read_csv(os.path.join(DATA_DIR, "option_chain.csv"))
ch = ch[ch["expiry"] == exp_id].sort_values("strike").reset_index(drop=True)
n_input = int(len(ch))
ch["call_mid"] = 0.5 * (ch["call_bid"] + ch["call_ask"]); ch["put_mid"] = 0.5 * (ch["put_bid"] + ch["put_ask"])
is_put = ch["strike"] < F
ch["otm_is_put"] = is_put
ch["otm_bid"] = np.where(is_put, ch["put_bid"], ch["call_bid"])
ch["otm_ask"] = np.where(is_put, ch["put_ask"], ch["call_ask"])
ch["otm_mid"] = np.where(is_put, ch["put_mid"], ch["call_mid"])
keep = (ch["volume"] >= flt["min_volume"]) & (ch["otm_mid"] > 0)
if flt.get("exclude_zero_bid", True):
    keep &= ch["otm_bid"] > 0
keep &= (ch["otm_ask"] - ch["otm_bid"]) / ch["otm_mid"] <= flt["max_relative_spread"]
kept = ch[keep].sort_values("strike").reset_index(drop=True)

def bs_price(K, vol, call):
    sq = vol * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / sq; d2 = d1 - sq
    if call:
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

def implied_vol(price, K, call):
    f = lambda v: bs_price(K, v, call) - price
    return brentq(f, 1e-6, 5.0, xtol=1e-12, rtol=1e-12, maxiter=500)

Ks = kept["strike"].to_numpy(float); px = kept["otm_mid"].to_numpy(float); puts = kept["otm_is_put"].to_numpy(bool)
ivs = np.array([implied_vol(p, k, not ip) for p, k, ip in zip(px, Ks, puts)])

# ATM implied vol at the forward: linear interpolation between the straddling kept strikes
hi = int(np.searchsorted(Ks, F, side="left"))          # first kept strike >= F
lo = hi - 1
w = (F - Ks[lo]) / (Ks[hi] - Ks[lo])
iv_atm = float((1 - w) * ivs[lo] + w * ivs[hi]); atm_var = iv_atm ** 2

# Log-contract replication with trapezoidal weights on the non-uniform grid
n = len(Ks); dK = np.empty(n)
dK[0] = Ks[1] - Ks[0]; dK[-1] = Ks[-1] - Ks[-2]
if n > 2:
    dK[1:-1] = 0.5 * (Ks[2:] - Ks[:-2])
K_var2 = float(2.0 * math.exp(r * T) / T * np.sum(dK / Ks ** 2 * px))
fair_vol = math.sqrt(K_var2)

N_var = 10_000.0
scen = [{"name": nm, "sigma_RV": s_rv, "realized_variance": s_rv ** 2, "pnl": N_var * (s_rv ** 2 - K_var2)}
        for nm, s_rv in [("low", 0.18), ("mid", 0.25), ("high", 0.35)]]

out = {"target_expiry": exp_id, "forward_price": F, "n_strikes_input": n_input, "n_strikes_kept": int(n),
       "n_otm_puts_kept": int(puts.sum()), "n_otm_calls_kept": int((~puts).sum()),
       "fair_variance_strike": K_var2, "fair_vol": fair_vol, "iv_atm_forward": iv_atm, "atm_var": atm_var,
       "variance_risk_premium": K_var2 - atm_var, "fair_vol_minus_atm_iv": fair_vol - iv_atm, "scenarios": scen}
json.dump(out, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
