# memory-solution
# unit: t1-barrier-garch-var | category: derivatives-pricing | tags: barrier-option, down-and-out-call, reiner-rubinstein, garch, arch, finite-difference-greeks, delta-gamma-vega-var, variance-decomposition, params-json-family
# deliverables: results.json, solution.json
# GARCH(1,1): arch_model(100*r, mean='Zero', dist='normal', rescale=False); omega/1e4; garch_vol = cond_vol[-1]/100*sqrt(252)
# DOC call priced with the Reiner-Rubinstein closed form (B < K, q = 0); Greeks by central FD: dS = fd_dS_pct*S0, dsigma = fd_dsigma
# 1-day components: delta_c = n*delta*S0*sd*z, gamma_c = 0.5*n*gamma*(S0*sd*z)^2, vega_c = n*vega*vov*z
# sd = garch_vol/sqrt(252); vov = std(diff(annualised cond vol series), ddof=1) (annualised then de-annualised = same number)
# portfolio_var = sqrt(delta_c^2 + gamma_c^2 + vega_c^2); *_var_pct = 100 * component^2 / portfolio_var^2 (variance shares)
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm
from arch import arch_model
import warnings; warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

P = json.load(open(os.path.join(DATA_DIR, "params.json")))
df = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"))
df["log_return"] = pd.to_numeric(df["log_return"], errors="coerce")
r_ = df.dropna().drop_duplicates("date").sort_values("date")["log_return"].to_numpy(float)

res = arch_model(100.0 * r_, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
omega = float(res.params["omega"]) / 1e4
alpha, beta = float(res.params["alpha[1]"]), float(res.params["beta[1]"])
pers = alpha + beta
lrv = omega / (1.0 - pers)
ann_cv = np.asarray(res.conditional_volatility) / 100.0 * math.sqrt(252.0)   # in-sample annualised conditional vol
sigma = float(ann_cv[-1])
vov = float(np.std(np.diff(ann_cv), ddof=1))                                  # daily vol-of-vol (ann. std / sqrt(252))

S0, K, B, r, T = P["S0"], P["K"], P["B"], P["r"], P["T"]


def doc_price(S, s):
    """Down-and-out call, B < K, no dividends (Reiner-Rubinstein)."""
    st = s * math.sqrt(T); mu = (r - 0.5 * s * s) / (s * s)
    x1 = math.log(S / K) / st + (1.0 + mu) * st
    y1 = math.log(B * B / (S * K)) / st + (1.0 + mu) * st
    return (S * norm.cdf(x1) - K * math.exp(-r * T) * norm.cdf(x1 - st)
            - S * (B / S) ** (2 * mu + 2) * norm.cdf(y1) + K * math.exp(-r * T) * (B / S) ** (2 * mu) * norm.cdf(y1 - st))


price = doc_price(S0, sigma)
h, dv = P["fd_dS_pct"] * S0, P["fd_dsigma"]
up, dn = doc_price(S0 + h, sigma), doc_price(S0 - h, sigma)
delta = (up - dn) / (2 * h)
gamma = (up - 2 * price + dn) / h ** 2
vega = (doc_price(S0, sigma + dv) - doc_price(S0, sigma - dv)) / (2 * dv)

z = norm.ppf(P["confidence_level"]); n = P["n_contracts"]
sd = sigma / math.sqrt(252.0)
dS_var = S0 * sd * z                                                           # 1-day underlying move at the VaR quantile
comp = {"delta": n * delta * dS_var, "gamma": 0.5 * n * gamma * dS_var ** 2, "vega": n * vega * vov * z}
tot_var = sum(c * c for c in comp.values())
port_var = math.sqrt(tot_var)

results = {"portfolio_var": port_var, **{f"{k}_var_pct": 100.0 * c * c / tot_var for k, c in comp.items()}}
results = {k: float(v) for k, v in results.items()}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

inter = {"garch_omega": omega, "garch_alpha": alpha, "garch_beta": beta, "garch_persistence": pers,
         "long_run_variance": lrv, "garch_vol": sigma, "analytical_price": price, "delta": delta, "gamma": gamma,
         "vega": vega, "vol_of_vol_daily": vov, "delta_var": comp["delta"], "gamma_var": comp["gamma"], "vega_var": comp["vega"]}
solution = {"intermediates": {k: {"value": float(v)} for k, v in inter.items()}, "deliverables": results}
json.dump(solution, open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
