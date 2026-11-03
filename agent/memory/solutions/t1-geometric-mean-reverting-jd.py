# memory-solution
# unit: t1-geometric-mean-reverting-jd | category: rates-curves | tags: ornstein-uhlenbeck, jump-diffusion, ar1-ols, jump-detection, conditional-moments, euler-maruyama, compound-poisson, lognormal-forward, fred-dgs10
# deliverables: calibration.json, conditional_moments.csv, forward_curve.csv, summary.json
# X = log(DGS10/100) on NaN-dropped rows, dt = 1/252. AR(1) OLS X_{t+1} = a + b X_t + eps -> kappa = -ln b/dt,
#   theta = a/(1-b), sigma = sd(eps) * sqrt(2 kappa / (1 - b^2)) (exact-discretisation inversion; sd with ddof=1).
# Jumps: |eps/sd(eps)| > 3 -> lambda = n_jumps/(n_changes dt), mu_J/sigma_J = mean/std of flagged eps.
# Moments: E = theta + (x0-theta)e^{-k t} + lam mu_J (1-e^{-k t})/k ; Var = (sigma^2 + lam(mu_J^2+sigma_J^2))(1-e^{-2k t})/(2k)
#   stationary var uses the same jump-augmented variance so Var(X_tau) <= stationary var holds by construction.
# E[S] = exp(E + Var/2), Var[S] = (exp(Var)-1) exp(2E + Var); forward F = exp(E + Var/2), OU forward = same with lam = 0.
# MC: one Euler run of 10000 paths over 5y with Poisson(lam dt) jump counts per step; read off at the nested horizons.
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "fred_rates.csv"), parse_dates=["date"]).sort_values("date")
rate = (df["DGS10"].dropna().to_numpy(float) / 100.0)
rate = rate[rate > 0]
X = np.log(rate)
dt = 1.0 / 252.0
n_obs, n_changes = len(X), len(X) - 1

# --- OU calibration by AR(1) regression ---
b, a = np.polyfit(X[:-1], X[1:], 1)
eps = X[1:] - (a + b * X[:-1])
kappa = -math.log(b) / dt
theta = a / (1.0 - b)
sd_eps = float(eps.std(ddof=1))
sigma = sd_eps * math.sqrt(2.0 * kappa / (1.0 - b * b))

# --- jump detection ---
z = eps / sd_eps
jmask = np.abs(z) > 3.0
n_jumps = int(jmask.sum())
lam = n_jumps / (n_changes * dt)
mu_J = float(eps[jmask].mean()) if n_jumps > 0 else 0.0
sigma_J = float(eps[jmask].std(ddof=1)) if n_jumps > 1 else 0.0

x0 = float(X[-1]); S0 = math.exp(x0)
v_tot = sigma ** 2 + lam * (mu_J ** 2 + sigma_J ** 2)
stat_mean = theta + lam * mu_J / kappa
stat_var = v_tot / (2.0 * kappa)

def moments(t, lam_):
    e = math.exp(-kappa * t)
    EX = theta + (x0 - theta) * e + lam_ * mu_J / kappa * (1.0 - e)
    VX = (sigma ** 2 + lam_ * (mu_J ** 2 + sigma_J ** 2)) * (1.0 - e * e) / (2.0 * kappa)
    return EX, VX

json.dump({"n_obs": int(n_obs), "n_changes": int(n_changes), "rate_mean": float(rate.mean()), "rate_std": float(rate.std(ddof=1)),
           "log_rate_mean": float(X.mean()), "log_rate_std": float(X.std(ddof=1)), "kappa": float(kappa), "theta": float(theta),
           "sigma": float(sigma), "lambda": float(lam), "mu_J": mu_J, "sigma_J": sigma_J, "n_jumps": n_jumps,
           "x0": x0, "S0": S0, "stationary_mean_X": float(stat_mean), "stationary_var_X": float(stat_var)},
          open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)

# --- Monte Carlo (Euler-Maruyama + compound Poisson), nested horizons in one run ---
taus = [0.25, 0.50, 1.00, 2.00, 5.00]
steps = {tau: int(round(tau / dt)) for tau in taus}
np.random.seed(42)
N = 10000
Xp = np.full(N, x0)
mc = {}
for k in range(1, max(steps.values()) + 1):
    Zd = np.random.standard_normal(N)
    nj = np.random.poisson(lam * dt, N)
    Zj = np.random.standard_normal(N)
    Xp = Xp + kappa * (theta - Xp) * dt + sigma * math.sqrt(dt) * Zd + nj * mu_J + np.sqrt(nj) * sigma_J * Zj
    for tau, s in steps.items():
        if s == k:
            mc[tau] = (float(Xp.mean()), float(Xp.var(ddof=1)))

rows, fwd = [], []
for tau in taus:
    EX, VX = moments(tau, lam)
    EX0, VX0 = moments(tau, 0.0)
    ES = math.exp(EX + 0.5 * VX)
    VS = (math.exp(VX) - 1.0) * math.exp(2.0 * EX + VX)
    rows.append({"tau": tau, "E_X": EX, "Var_X": VX, "E_S": ES, "Var_S": VS, "E_X_mc": mc[tau][0], "Var_X_mc": mc[tau][1]})
    fwd.append({"tau": tau, "forward_rate_jd": ES, "forward_rate_ou": math.exp(EX0 + 0.5 * VX0)})
cm = pd.DataFrame(rows); fc = pd.DataFrame(fwd)
cm.to_csv(os.path.join(OUT_DIR, "conditional_moments.csv"), index=False)
fc.to_csv(os.path.join(OUT_DIR, "forward_curve.csv"), index=False)

half_life = math.log(2.0) / kappa
mean_rel = float((np.abs(cm["E_X"] - cm["E_X_mc"]) / np.abs(cm["E_X"])).max())
var_rel = float((np.abs(cm["Var_X"] - cm["Var_X_mc"]) / cm["Var_X"]).max())
f1 = fc[np.isclose(fc["tau"], 1.0)].iloc[0]
json.dump({"kappa": float(kappa), "theta": float(theta), "sigma": float(sigma), "lambda_annual": float(lam),
           "half_life_years": float(half_life), "stationary_vol_X": float(math.sqrt(stat_var)),
           "feller_ratio": float(2.0 * kappa * theta / sigma ** 2), "max_mc_mean_relerr": mean_rel, "max_mc_var_relerr": var_rel,
           "jump_impact_on_forward": float(abs(f1["forward_rate_jd"] - f1["forward_rate_ou"]) / f1["forward_rate_ou"]),
           "mean_reversion_confirmed": bool(kappa > 0 and half_life < 20.0), "jumps_detected": bool(n_jumps > 0)},
          open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
