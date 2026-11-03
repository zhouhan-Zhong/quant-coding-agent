# memory-solution
# unit: t1-first-passage-time | category: derivatives-pricing / stochastic-processes | tags: reflection-principle, running-maximum, running-minimum, first-passage-time, inverse-gaussian, monte-carlo, spy-calibration
# deliverables: calibration.json, running_max_probs.csv, running_min_probs.csv, first_passage_time.csv, summary.json
# formulas for X_t = mu t + sigma W_t (mu = r - D - sigma^2/2), b = ln(H/S0) > 0, a = ln(L/S0) < 0:
#   P(max X <= T >= b) = N((-b + mu T)/(s√T)) + exp(2 mu b / s^2) N((-b - mu T)/(s√T))
#   P(min X <= a)      = N((a - mu T)/(s√T))  + exp(2 mu a / s^2) N((a + mu T)/(s√T))
#   FPT cdf = same expressions with T -> t ; E[tau_b] = b / mu (mu > 0)
#   P(X_T <= x, max <= b) = N((x - mu T)/(s√T)) - exp(2 mu b / s^2) N((x - 2b - mu T)/(s√T))   (x <= b)
import os, json
import numpy as np, pandas as pd
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

px = pd.read_csv(os.path.join(DATA_DIR, "spy_daily.csv"), parse_dates=["date"]).sort_values("date")
close = px["close"].to_numpy(float); logret = np.diff(np.log(close))
S0 = float(close[-1]); sigma = float(logret.std(ddof=1) * np.sqrt(252.0))
r, D = 0.05, 0.013
mu = r - D - 0.5 * sigma ** 2
json.dump({"S0": S0, "sigma": sigma, "r": r, "D": D, "mu_rn": float(mu), "n_returns": int(len(logret))},
          open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)

def p_max_ge(b, T):
    sq = sigma * np.sqrt(T)
    return norm.cdf((-b + mu * T) / sq) + np.exp(2 * mu * b / sigma ** 2) * norm.cdf((-b - mu * T) / sq)

def p_min_le(a, T):
    sq = sigma * np.sqrt(T)
    return norm.cdf((a - mu * T) / sq) + np.exp(2 * mu * a / sigma ** 2) * norm.cdf((a + mu * T) / sq)

def p_joint_le(x, b, T):  # P(X_T <= x, max <= b)
    sq = sigma * np.sqrt(T)
    return norm.cdf((x - mu * T) / sq) - np.exp(2 * mu * b / sigma ** 2) * norm.cdf((x - 2 * b - mu * T) / sq)

# ---- Monte Carlo (daily steps, seed 42)
np.random.seed(42)
n_paths, n_steps, T_mc = 50000, 252, 1.0
dt = T_mc / n_steps
z = np.random.standard_normal((n_paths, n_steps))
X = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(mu * dt + sigma * np.sqrt(dt) * z, axis=1)], axis=1)  # log(S_t/S0)
run_max = np.maximum.accumulate(X, axis=1)
run_min = np.minimum.accumulate(X, axis=1)
tgrid = np.arange(n_steps + 1) * dt

def mc_prob_max(b, T):
    idx = int(round(T / dt)); return float((run_max[:, idx] >= b).mean())

def mc_prob_min(a, T):
    idx = int(round(T / dt)); return float((run_min[:, idx] <= a).mean())

def first_passage(level, up=True):
    hit = (X >= level) if up else (X <= level)
    any_hit = hit.any(axis=1)
    first_idx = np.where(any_hit, hit.argmax(axis=1), n_steps + 1)
    tau = np.where(any_hit, first_idx * dt, np.inf)
    return tau

maturities = [0.25, 0.50, 1.00]
rows = []
for T in maturities:
    for hr in [1.05, 1.10, 1.15, 1.20, 1.30]:
        b = np.log(hr)
        rows.append({"T": T, "H_ratio": hr, "H": hr * S0, "b": b, "prob_analytical": float(p_max_ge(b, T)), "prob_mc": mc_prob_max(b, T)})
rmax = pd.DataFrame(rows); rmax.to_csv(os.path.join(OUT_DIR, "running_max_probs.csv"), index=False)
rows = []
for T in maturities:
    for lr in [0.70, 0.80, 0.85, 0.90, 0.95]:
        a = np.log(lr)
        rows.append({"T": T, "L_ratio": lr, "L": lr * S0, "a": a, "prob_analytical": float(p_min_le(a, T)), "prob_mc": mc_prob_min(a, T)})
rmin = pd.DataFrame(rows); rmin.to_csv(os.path.join(OUT_DIR, "running_min_probs.csv"), index=False)

b_up, a_dn = np.log(1.10), np.log(0.90)
tau_up, tau_dn = first_passage(b_up, True), first_passage(a_dn, False)
rows = []
for t in np.arange(1, 11) / 10.0:
    rows.append({"barrier_type": "up_110", "t": float(t), "cdf_analytical": float(p_max_ge(b_up, t)), "cdf_mc": float((tau_up <= t + 1e-12).mean())})
for t in np.arange(1, 11) / 10.0:
    rows.append({"barrier_type": "down_090", "t": float(t), "cdf_analytical": float(p_min_le(a_dn, t)), "cdf_mc": float((tau_dn <= t + 1e-12).mean())})
fpt = pd.DataFrame(rows); fpt.to_csv(os.path.join(OUT_DIR, "first_passage_time.csv"), index=False)

def relerr(df):
    sub = df[df["prob_analytical"] > 0.05]
    return float((abs(sub["prob_analytical"] - sub["prob_mc"]) / sub["prob_analytical"]).max())

hit_up = np.isfinite(tau_up)
joint_an = float(p_joint_le(0.0, b_up, 1.0))
joint_mc = float(((X[:, -1] <= 0.0) & (run_max[:, -1] <= b_up)).mean())
summary = {
    "n_mc_paths": n_paths,
    "max_prob_up_relerr": relerr(rmax), "max_prob_down_relerr": relerr(rmin),
    "max_fpt_cdf_abserr": float((fpt["cdf_analytical"] - fpt["cdf_mc"]).abs().max()),
    "E_tau_up": float(b_up / mu) if mu > 0 else None,
    "E_tau_up_mc": float(tau_up[hit_up].mean()) if hit_up.any() else None,   # conditional on hitting within 1y (discrete, truncated)
    "prob_hit_110_T1": float(p_max_ge(b_up, 1.0)), "prob_hit_090_T1": float(p_min_le(a_dn, 1.0)),
    "joint_prob_analytical": joint_an, "joint_prob_mc": joint_mc,
    "max_decreases_with_barrier": bool(all(np.all(np.diff(rmax[np.isclose(rmax["T"], T)].sort_values("H_ratio")["prob_analytical"].values) <= 1e-12) for T in maturities)),
    "min_increases_with_barrier": bool(all(np.all(np.diff(rmin[np.isclose(rmin["T"], T)].sort_values("L_ratio")["prob_analytical"].values) >= -1e-12) for T in maturities)),
    "prob_increases_with_T": bool(all(np.all(np.diff(rmax[np.isclose(rmax["H_ratio"], hr)].sort_values("T")["prob_analytical"].values) >= -1e-12) for hr in [1.05, 1.10, 1.15, 1.20, 1.30])),
}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
