# memory-solution
# unit: t1-ewma-portfolio-risk-decomposition | category: risk | tags: ewma, riskmetrics, covariance, parametric-var, euler-decomposition, marginal-risk, component-risk, debug-template
# deliverables: results.json, solution.json
# Fixed template bugs: (1) no demeaning of returns before the EWMA recursion (RiskMetrics assumes zero mean);
# (2) rank-1 update is np.outer(r, r), not np.dot; (3) z = norm.ppf(conf_level), not 2.33;
# (4) params.json has estimation_window_days = 63 -> use only the LAST 63 return rows;
# (5) marginal risk = (Sigma w) / sigma_p (not * sigma_p); component = w * marginal; pct = component / sigma_p.
# EWMA: S_0 = r_0 r_0', S_t = lam S_{t-1} + (1-lam) r_t r_t'. Oracle tolerance rtol 1e-6 so the recursion must be exact.
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

p = json.load(open(os.path.join(DATA_DIR, "params.json")))
assets, lam, conf = p["assets"], float(p["lambda"]), float(p["confidence_level"])
w = np.asarray(p["weights"], float); ann = float(p["annualization_factor"])
R = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"))[assets].to_numpy(float)
win = p.get("estimation_window_days")
if win is not None and win < len(R):
    R = R[-int(win):]                                  # most recent window only

S = np.outer(R[0], R[0])                               # raw (not demeaned) returns
for r in R[1:]:
    S = lam * S + (1 - lam) * np.outer(r, r)

port_var = float(w @ S @ w)
port_vol = math.sqrt(port_var)
marginal = (S @ w) / port_vol                          # d sigma_p / d w_i
component = w * marginal                               # sums to sigma_p (Euler)
pct = component / port_vol

json.dump({"portfolio_daily_vol": port_vol, "var_99_parametric": float(norm.ppf(conf) * port_vol),
           "annualized_vol": port_vol * math.sqrt(ann)}, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
json.dump({"intermediates": {
    "ewma_cov_0_0": float(S[0, 0]), "ewma_cov_0_1": float(S[0, 1]), "ewma_cov_1_1": float(S[1, 1]),
    "ewma_cov_2_3": float(S[2, 3]), "ewma_cov_3_4": float(S[3, 4]), "portfolio_variance": port_var,
    "marginal_risk": {a: float(marginal[i]) for i, a in enumerate(assets)},
    "component_risk": {a: float(component[i]) for i, a in enumerate(assets)},
    "pct_risk_contribution": {a: float(pct[i]) for i, a in enumerate(assets)}}},
    open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
