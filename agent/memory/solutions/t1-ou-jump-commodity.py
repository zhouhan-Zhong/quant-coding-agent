# memory-solution
# unit: t1-ou-jump-commodity | category: fx-commodities-crypto | tags: ornstein-uhlenbeck, jump-diffusion, ar1-ols, jump-detection, conditional-moments, stationary-distribution, euler-maruyama, bernoulli-jumps, fred-dgs10
# deliverables: calibration.json, conditional_moments.csv, stationary_distribution.json, summary.json
# S = DGS10 level in percent (dropna), dt = 1/252. OLS S_{t+1} = a + b S_t + eps -> kappa = -ln b/dt, theta = a/(1-b).
# Jumps: |eps| > 3 sd(eps) (67 of 6563); lambda = n_jumps/(n dt); mu_J, sigma_J = mean/std of flagged eps.
# sigma_ou re-estimated from NON-jump residuals: sd_nj * sqrt(2 kappa/(1-b^2)) (exact AR(1) inversion).
# OU: E = theta + (S0-theta)e^{-kt}, Var = sigma^2(1-e^{-2kt})/(2k). Jumps add lam mu_J (1-e^{-kt})/k to the mean and
#   lam(mu_J^2+sigma_J^2)(1-e^{-2kt})/(2k) to the variance; stationary = t -> inf limits. mu_J > 0 here so jump mean >= OU mean.
# MC: np.random.seed(42), 100000 paths, 252 Euler steps, Bernoulli(lam dt) jump indicator, read off horizons 21/63/126/252.
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "fred_rates.csv"), parse_dates=["date"]).sort_values("date")
S = df["DGS10"].dropna().to_numpy(float)
dS = np.diff(S)
dt = 1.0 / 252.0
n = len(dS)

b, a = np.polyfit(S[:-1], S[1:], 1)
eps = S[1:] - (a + b * S[:-1])
kappa = -math.log(b) / dt
theta = a / (1.0 - b)
sd_all = float(eps.std(ddof=1))
jmask = np.abs(eps) > 3.0 * sd_all
n_jumps = int(jmask.sum())
lam = n_jumps / (n * dt)
mu_J = float(eps[jmask].mean()) if n_jumps > 0 else 0.0
sigma_J = float(eps[jmask].std(ddof=1)) if n_jumps > 1 else 0.0
sd_nj = float(eps[~jmask].std(ddof=1))
sigma = sd_nj * math.sqrt(2.0 * kappa / (1.0 - b * b))
S0 = float(S[-1])

json.dump({"n_observations": int(n), "mean_level": float(S.mean()), "std_level": float(S.std(ddof=1)),
           "mean_daily_change": float(dS.mean()), "std_daily_change": float(dS.std(ddof=1)),
           "kappa": float(kappa), "theta": float(theta), "sigma_ou": float(sigma), "lambda_jump": float(lam),
           "mu_J": mu_J, "sigma_J": sigma_J, "n_jumps_detected": n_jumps, "S0": S0},
          open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)

def moments(t):
    e = math.exp(-kappa * t)
    E_ou = theta + (S0 - theta) * e
    V_ou = sigma ** 2 * (1.0 - e * e) / (2.0 * kappa)
    E_j = E_ou + lam * mu_J / kappa * (1.0 - e)
    V_j = V_ou + lam * (mu_J ** 2 + sigma_J ** 2) * (1.0 - e * e) / (2.0 * kappa)
    return E_ou, V_ou, E_j, V_j

# --- Monte Carlo: Euler + Bernoulli jumps over 1 year, nested horizons ---
taus = [1.0 / 12.0, 0.25, 0.5, 1.0]
steps = {tau: int(round(tau * 252)) for tau in taus}
np.random.seed(42)
N = 100000
X = np.full(N, S0)
mc = {}
for k in range(1, 253):
    Z = np.random.standard_normal(N)
    jump_on = np.random.rand(N) < lam * dt
    J = np.random.normal(mu_J, sigma_J, N)
    X = X + kappa * (theta - X) * dt + sigma * math.sqrt(dt) * Z + jump_on * J
    for tau, s in steps.items():
        if s == k:
            mc[tau] = (float(X.mean()), float(X.var(ddof=1)))

rows = []
for tau in taus:
    E_ou, V_ou, E_j, V_j = moments(tau)
    rows.append({"tau": tau, "E_ST_ou": E_ou, "Var_ST_ou": V_ou, "E_ST_jump": E_j, "Var_ST_jump": V_j,
                 "mc_mean": mc[tau][0], "mc_var": mc[tau][1]})
cm = pd.DataFrame(rows)
cm.to_csv(os.path.join(OUT_DIR, "conditional_moments.csv"), index=False)

stat_mean_ou, stat_var_ou = theta, sigma ** 2 / (2.0 * kappa)
stat_mean_j = theta + lam * mu_J / kappa
stat_var_j = stat_var_ou + lam * (mu_J ** 2 + sigma_J ** 2) / (2.0 * kappa)
json.dump({"stationary_mean_ou": float(stat_mean_ou), "stationary_var_ou": float(stat_var_ou),
           "stationary_mean_jump": float(stat_mean_j), "stationary_var_jump": float(stat_var_j),
           "empirical_mean": float(S.mean()), "empirical_var": float(S.var(ddof=1))},
          open(os.path.join(OUT_DIR, "stationary_distribution.json"), "w"), indent=2)

mean_err = float((cm["E_ST_jump"] - cm["mc_mean"]).abs().max())
var_err_pct = float(((cm["Var_ST_jump"] - cm["mc_var"]).abs() / cm["Var_ST_jump"]).max() * 100.0)
json.dump({"kappa": float(kappa), "theta": float(theta), "sigma_ou": float(sigma), "lambda_jump": float(lam),
           "mu_J": mu_J, "sigma_J": sigma_J, "half_life_days": float(math.log(2.0) / kappa * 252.0),
           "n_jumps_detected": n_jumps, "mc_validates": bool(mean_err < 0.05 and var_err_pct < 20.0),
           "max_mc_mean_error": mean_err, "max_mc_var_error_pct": var_err_pct},
          open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
