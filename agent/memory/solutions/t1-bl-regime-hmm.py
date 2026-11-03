# memory-solution
# unit: t1-bl-regime-hmm | category: portfolio | tags: black-litterman, hmm, baum-welch, regime-switching, reverse-optimization, absolute-views, gross-exposure, sharpe, max-drawdown, params-json-family
# deliverables: results.json, solution.json
# HMM on market-cap portfolio return (simple returns): 2-state Gaussian, Baum-Welch in log space, init mu/sigma(ddof=1)
# from first-half/second-half split, A and pi from params; stop when max|mu_new-mu_old| < tol; iterations = EM passes done
# bull = state with higher mu; regime_prob_bull = smoothed gamma at the last day; hard assignment gamma_bull > 0.5
# regime covariance = np.cov(ddof=1) of assigned rows (fallback: all rows if < 10); use the CURRENT regime's covariance
# BL (daily units): pi = delta*Sigma*w_mkt; absolute views: P rows e_i, Q = annual/252, Omega = diag(conf^2/252)
# mu_BL = [(tau S)^-1 + P'O^-1 P]^-1 [(tau S)^-1 pi + P'O^-1 Q]; w = (delta S)^-1 mu_BL, normalised by sum|w|
# performance on full history: ann_ret = mean*252, ann_vol = std(ddof=1)*sqrt252, Sharpe = ret/vol, mdd on cumprod(1+r)
import os, json, math
import numpy as np, pandas as pd
from scipy.special import logsumexp

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

P = json.load(open(os.path.join(DATA_DIR, "params.json")))
df = pd.read_csv(os.path.join(DATA_DIR, "returns.csv")).drop_duplicates("date").sort_values("date").dropna()
assets = [c for c in df.columns if c != "date"]
R = df[assets].to_numpy(float)
N, n = R.shape
AF = float(P["annualization_factor"])
w_mkt = np.asarray(P["market_cap_weights"], float)
x = R @ w_mkt                                                        # market portfolio return series

# ---------- 2-state Gaussian HMM, Baum-Welch (log space) ----------
K = int(P["n_regimes"])
h = N // 2
mu = np.array([x[:h].mean(), x[h:].mean()])
sd = np.array([x[:h].std(ddof=1), x[h:].std(ddof=1)])
A = np.asarray(P["init_trans_matrix"], float)
pi = np.asarray(P["init_pi"], float)
n_iter = 0
for it in range(int(P["hmm_max_iter"])):
    logB = -0.5 * np.log(2 * np.pi * sd ** 2)[None, :] - 0.5 * ((x[:, None] - mu[None, :]) / sd[None, :]) ** 2
    logA = np.log(A)
    la = np.zeros((N, K)); lb = np.zeros((N, K))
    la[0] = np.log(pi) + logB[0]
    for t in range(1, N):                                            # forward
        la[t] = logsumexp(la[t - 1][:, None] + logA, axis=0) + logB[t]
    for t in range(N - 2, -1, -1):                                   # backward
        lb[t] = logsumexp(logA + (logB[t + 1] + lb[t + 1])[None, :], axis=1)
    ll = logsumexp(la[-1])
    lg = la + lb - ll
    g = np.exp(lg)                                                   # smoothed posteriors gamma_t(k)
    lxi = la[:-1, :, None] + logA[None, :, :] + (logB[1:] + lb[1:])[:, None, :] - ll
    xi = np.exp(lxi).sum(axis=0)                                     # expected transition counts
    n_iter = it + 1
    pi = g[0] / g[0].sum()
    A = xi / xi.sum(axis=1, keepdims=True)
    mu_new = (g * x[:, None]).sum(axis=0) / g.sum(axis=0)
    sd = np.sqrt((g * (x[:, None] - mu_new[None, :]) ** 2).sum(axis=0) / g.sum(axis=0))
    converged = np.max(np.abs(mu_new - mu)) < P["hmm_tol"]
    mu = mu_new
    if converged:
        break

bull = int(np.argmax(mu)); bear = 1 - bull
prob_bull = float(g[-1, bull])
current = "bull" if prob_bull > 0.5 else "bear"
assign_bull = g[:, bull] > 0.5
n_bull = int(assign_bull.sum())
Sig_all = np.cov(R, rowvar=False, ddof=1)
cov_bull = np.cov(R[assign_bull], rowvar=False, ddof=1) if n_bull >= 10 else Sig_all
cov_bear = np.cov(R[~assign_bull], rowvar=False, ddof=1) if (N - n_bull) >= 10 else Sig_all
Sigma = cov_bull if current == "bull" else cov_bear

# ---------- Black-Litterman (daily units) ----------
delta, tau = float(P["risk_aversion"]), float(P["tau"])
pi_eq = delta * Sigma @ w_mkt
views = P["views"]
Pm = np.zeros((len(views), n)); Q = np.zeros(len(views)); Om = np.zeros((len(views), len(views)))
for j, v in enumerate(views):
    Pm[j, int(v["asset"])] = 1.0
    Q[j] = v["return"] / AF
    Om[j, j] = v["confidence"] ** 2 / AF
tS_inv = np.linalg.inv(tau * Sigma); Om_inv = np.linalg.inv(Om)
mu_bl = np.linalg.solve(tS_inv + Pm.T @ Om_inv @ Pm, tS_inv @ pi_eq + Pm.T @ Om_inv @ Q)
w = np.linalg.solve(delta * Sigma, mu_bl)
w = w / np.abs(w).sum()                                              # gross exposure = 1

port = R @ w
ann_ret = port.mean() * AF
ann_vol = port.std(ddof=1) * math.sqrt(AF)
wealth = np.cumprod(1 + port)
mdd = float((1 - wealth / np.maximum.accumulate(wealth)).max())

results = {"current_regime": current, "regime_prob_bull": prob_bull, "annualized_return": float(ann_ret),
           "annualized_volatility": float(ann_vol), "sharpe_ratio": float(ann_ret / ann_vol), "max_drawdown": mdd}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

inter = {"hmm_iterations": n_iter, "hmm_mu_bull": float(mu[bull]), "hmm_sigma_bear": float(sd[bear]),
         "regime_prob_bull": prob_bull, "implied_return_asset1": float(pi_eq[0] * AF), "bl_weight_asset1": float(w[0]),
         "n_bull_days": n_bull, "cov_bull_diag_asset1": float(cov_bull[0, 0]), "annualized_volatility": float(ann_vol),
         "max_drawdown": mdd}
solution = {"intermediates": {k: {"value": v} for k, v in inter.items()}, "deliverables": results,
            "bl_weights": [float(v) for v in w]}
json.dump(solution, open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
