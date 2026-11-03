# memory-solution
# unit: t1-structured-note-risk | category: derivatives-pricing | tags: barrier-option, down-and-out-call, reiner-rubinstein, garch, arch, delta-normal-var, expected-shortfall, structured-note, breakeven-vol, params-json-family
# deliverables: results.json, solution.json
# GARCH(1,1): arch_model(100*log_return, mean='Zero', vol='GARCH', dist='normal', rescale=False); omega/1e4 back to return units
# persistence = alpha+beta; LRV = omega/(1-persistence); garch_vol = conditional_volatility[-1]/100*sqrt(252) (last in-sample, not forecast)
# note = zero bond (notional*exp(-rT)) + n_options down-and-out calls, n_options = notional*participation/S0
# DOC priced analytically (Reiner-Rubinstein, B<K, q=0); delta = central finite difference with bump h = 1% of S0
# fair_value = bond + n*price; markup_pct = (notional - fair_value)/fair_value*100; breakeven vol: brentq on fair_value(sigma)=notional
# delta-normal 1-day risk: VaR99 = note_delta*S0*(garch_vol/sqrt(252))*z99; ES99 = VaR99*phi(z99)/0.01 (reference convention)
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from arch import arch_model

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

P = json.load(open(os.path.join(DATA_DIR, "params.json")))
df = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"))
df["log_return"] = pd.to_numeric(df["log_return"], errors="coerce")
r_ = df.dropna().drop_duplicates("date").sort_values("date")["log_return"].to_numpy(float)

# --- GARCH(1,1) on percent returns, zero mean ---
res = arch_model(100.0 * r_, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
omega = float(res.params["omega"]) / 1e4
alpha, beta = float(res.params["alpha[1]"]), float(res.params["beta[1]"])
pers = alpha + beta
lrv = omega / (1.0 - pers)
sigma = float(np.asarray(res.conditional_volatility)[-1]) / 100.0 * math.sqrt(252.0)   # annualised current vol

S0, K, B, r, T, part, notional = (P["S0"], P["K"], P["B"], P["r"], P["T"], P["participation"], P["notional"])


def doc_price(S, s):
    """Down-and-out call (B < K), Reiner-Rubinstein closed form, no dividends."""
    st = s * math.sqrt(T)
    mu = (r - 0.5 * s * s) / (s * s)
    x1 = math.log(S / K) / st + (1.0 + mu) * st
    y1 = math.log(B * B / (S * K)) / st + (1.0 + mu) * st
    return (S * norm.cdf(x1) - K * math.exp(-r * T) * norm.cdf(x1 - st)
            - S * (B / S) ** (2 * mu + 2) * norm.cdf(y1) + K * math.exp(-r * T) * (B / S) ** (2 * mu) * norm.cdf(y1 - st))


bond_pv = notional * math.exp(-r * T)
n_opt = notional * part / S0
unit = doc_price(S0, sigma)
opt_val = n_opt * unit
fair = bond_pv + opt_val
markup = (notional - fair) / fair * 100.0
h = 0.01 * S0
delta = (doc_price(S0 + h, sigma) - doc_price(S0 - h, sigma)) / (2 * h)
note_delta = n_opt * delta
be_vol = brentq(lambda s: bond_pv + n_opt * doc_price(S0, s) - notional, 1e-4, 3.0)

z = norm.ppf(0.99)
sd_daily = sigma / math.sqrt(252.0)
var99 = note_delta * S0 * sd_daily * z
es99 = var99 * norm.pdf(z) / 0.01

results = {"fair_value": fair, "markup_pct": markup, "note_var_99": var99, "note_es_99": es99,
           "breakeven_vol": be_vol, "garch_persistence": pers, "long_run_variance": lrv, "garch_vol": sigma,
           "bond_pv": bond_pv, "option_unit_price": unit, "option_value": opt_val, "n_options": n_opt,
           "option_delta": delta, "note_delta": note_delta}
results = {k: float(v) for k, v in results.items()}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

inter = {"garch_omega": omega, "garch_alpha": alpha, "garch_beta": beta, "garch_persistence": pers,
         "long_run_variance": lrv, "garch_vol": sigma, "bond_pv": bond_pv, "option_unit_price": unit,
         "option_value": opt_val, "n_options": n_opt, "option_delta": delta, "note_delta": note_delta}
solution = {"intermediates": {k: {"value": float(v)} for k, v in inter.items()}, "deliverables": results}
json.dump(solution, open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
