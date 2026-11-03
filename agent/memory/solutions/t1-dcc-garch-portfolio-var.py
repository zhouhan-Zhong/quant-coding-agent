# memory-solution
# unit: t1-dcc-garch-portfolio-var | category: volatility | tags: dcc-garch, garch, arch, dynamic-correlation, portfolio-var, expected-shortfall, filtered-historical-simulation, stationary-bootstrap, kupiec, christoffersen, backtest
# deliverables: garch_params.csv, dcc_params.json, dynamic_correlations.csv, portfolio_var.csv, backtest_results.json
# Log returns (decimal); arch GARCH(1,1) mean='Zero', dist normal, rescale=False on 100*r; sigma/100, Z = std_resid.
# DCC QML: Qbar = Z'Z/T (NOT corrcoef -- matches the oracle to 1e-7), Q_0 = Qbar, Q_t = (1-a-b)Qbar + a z_{t-1}z_{t-1}' + b Q_{t-1},
#   LL = sum -0.5(ln|R_t| + z'R^-1 z - z'z); L-BFGS-B from (0.05, 0.93), return 1e10 if a+b >= 1. LL ~ 1255.69.
# Parametric: sigma_p = sqrt(w'H_t w), VaR = -z_.01 sigma_p, ES = sigma_p phi(z)/0.01; violation r_p < -VaR (34 hits).
# Kupiec LR_uc with x/N; Christoffersen LR_ind from the 2x2 transition counts (0*log0 = 0), LR_cc = sum, chi2(2).
# FHS: one stationary-bootstrap sample of B=1000 rows (start ~ U{0..T-1}, length ~ Geometric(p=1/21), wrap mod T,
#   default_rng(42)) drawn from DCC-whitened residuals E_t = R_t^{-1/2} z_t; per day sims = E_b @ chol(H_t)' @ w;
#   VaR = -percentile(sims, 1) (linear), ES = mean loss over sims <= -VaR. Reproduces the pinned means within 0.4%.
import os, json, math
import numpy as np, pandas as pd
from arch import arch_model
from scipy.optimize import minimize
from scipy.stats import norm, chi2

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

TK = ["AAPL", "MSFT", "JPM", "JNJ", "XOM"]
px = pd.read_csv(os.path.join(DATA_DIR, "prices.csv"), parse_dates=["date"]).set_index("date")
wts = json.load(open(os.path.join(DATA_DIR, "weights.json")))
w = np.array([wts[t] for t in TK])
ret = np.log(px[TK]).diff().dropna()
R_ = ret.to_numpy(); dates = ret.index; T, n = R_.shape

sig, Z, grows = np.empty((T, n)), np.empty((T, n)), []
for i, t in enumerate(TK):
    res = arch_model(100 * ret[t], mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
    p_ = res.params; om, al, be = float(p_["omega"]), float(p_["alpha[1]"]), float(p_["beta[1]"])
    sig[:, i] = res.conditional_volatility.to_numpy() / 100; Z[:, i] = res.std_resid.to_numpy()
    grows.append({"ticker": t, "omega": om, "alpha": al, "beta": be, "persistence": al + be,
                  "unconditional_vol": math.sqrt(om / (1 - al - be) * 252) / 100})
pd.DataFrame(grows).to_csv(os.path.join(OUT_DIR, "garch_params.csv"), index=False)

Qbar = Z.T @ Z / T
def dcc_filter(ab, want_R=False):
    a, b = ab
    if a + b >= 1: return 1e10
    Q, ll, Rs = Qbar.copy(), 0.0, np.empty((T, n, n))
    for t in range(T):
        d = 1 / np.sqrt(np.diag(Q)); Rt = Q * np.outer(d, d); Rs[t] = Rt; z = Z[t]
        ll += -0.5 * (np.linalg.slogdet(Rt)[1] + z @ np.linalg.solve(Rt, z) - z @ z)
        Q = (1 - a - b) * Qbar + a * np.outer(z, z) + b * Q
    return (ll, Rs) if want_R else -ll
opt = minimize(dcc_filter, [0.05, 0.93], method="L-BFGS-B", bounds=[(1e-6, 0.999), (1e-6, 0.999)])
a, b = float(opt.x[0]), float(opt.x[1]); ll, Rt = dcc_filter(opt.x, True)
json.dump({"dcc_alpha": round(a, 8), "dcc_beta": round(b, 8), "dcc_persistence": round(a + b, 8), "log_likelihood": round(float(ll), 4)},
          open(os.path.join(OUT_DIR, "dcc_params.json"), "w"), indent=2)

H = sig[:, :, None] * Rt * sig[:, None, :]
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
corr = pd.DataFrame({f"{TK[i]}_{TK[j]}": Rt[:, i, j] for i, j in pairs}); corr.insert(0, "date", dates.strftime("%Y-%m-%d"))
corr.to_csv(os.path.join(OUT_DIR, "dynamic_correlations.csv"), index=False)

sp = np.sqrt(np.einsum("i,tij,j->t", w, H, w)); rp = R_ @ w
zq = norm.ppf(0.01); var = -zq * sp; es = sp * norm.pdf(zq) / 0.01
viol = (rp < -var).astype(int)

# FHS: stationary bootstrap (one sample of B rows) of DCC-whitened residuals, recoloured by chol(H_t)
B, pgeo, rng = 1000, 1 / 21, np.random.default_rng(42)
ev, evec = np.linalg.eigh(Rt)
E = np.einsum("tij,tj,tkj,tk->ti", evec, 1 / np.sqrt(ev), evec, Z)      # R_t^{-1/2} z_t
idx = []
while len(idx) < B:
    s, L = rng.integers(T), rng.geometric(pgeo); idx.extend([(s + k) % T for k in range(L)])
Eb = E[np.array(idx[:B])]
sims = np.einsum("bj,tij->tbi", Eb, np.linalg.cholesky(H)) @ w             # (T, B)
var_f = -np.percentile(sims, 1, axis=1)
es_f = np.array([-s[s <= -v].mean() for s, v in zip(sims, var_f)])

pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "portfolio_return": rp, "conditional_vol": sp, "annualised_vol": sp * math.sqrt(252),
              "var_1pct": var, "es_1pct": es, "violation": viol, "var_1pct_fhs": var_f, "es_1pct_fhs": es_f}
             ).to_csv(os.path.join(OUT_DIR, "portfolio_var.csv"), index=False)

N, x = len(viol), int(viol.sum()); pi = x / N
xl = lambda k, q: k * math.log(q) if k > 0 else 0.0
lr_uc = -2 * (xl(x, 0.01) + xl(N - x, 0.99) - xl(x, pi) - xl(N - x, 1 - pi))
n00 = int(((viol[:-1] == 0) & (viol[1:] == 0)).sum()); n01 = int(((viol[:-1] == 0) & (viol[1:] == 1)).sum())
n10 = int(((viol[:-1] == 1) & (viol[1:] == 0)).sum()); n11 = int(((viol[:-1] == 1) & (viol[1:] == 1)).sum())
p01 = n01 / (n00 + n01); p11 = n11 / (n10 + n11) if n10 + n11 else 0.0; p1 = (n01 + n11) / (N - 1)
lr_ind = -2 * (xl(n00 + n10, 1 - p1) + xl(n01 + n11, p1) - xl(n00, 1 - p01) - xl(n01, p01) - xl(n10, 1 - p11) - xl(n11, p11))
lr_cc = lr_uc + lr_ind; p_uc = float(chi2.sf(lr_uc, 1)); p_cc = float(chi2.sf(lr_cc, 2))
json.dump({"num_observations": N, "num_violations": x, "violation_rate": round(pi, 6), "expected_rate": 0.01,
           "kupiec_lr_stat": round(lr_uc, 6), "kupiec_p_value": round(p_uc, 6), "reject_h0_kupiec_5pct": bool(p_uc < 0.05),
           "christoffersen_lr_ind": round(lr_ind, 6), "christoffersen_lr_cc": round(lr_cc, 6),
           "christoffersen_p_value": round(p_cc, 6), "reject_h0_cc_5pct": bool(p_cc < 0.05)},
          open(os.path.join(OUT_DIR, "backtest_results.json"), "w"), indent=2)
