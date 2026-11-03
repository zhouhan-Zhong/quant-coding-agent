# memory-solution
# unit: t1-evt-pot-var | category: risk | tags: evt, peaks-over-threshold, gpd, hill-estimator, var, expected-shortfall, garch, arch, rolling-backtest, kupiec, christoffersen
# deliverables: results.json, solution.json
# Losses L = -log returns; EVERY quantile uses np.quantile(..., method="lower"); std ddof=1; exceedances/violations strict >.
# POT: u = Q_lower(L, 0.95); y = L[L>u]-u; genpareto.fit(y, floc=0) -> (xi, 0, sigma);
#   VaR_p = u + s/xi [ (N/N_u (1-p))^{-xi} - 1 ], ES_p = VaR/(1-xi) + (s - xi u)/(1-xi). Hill: k = floor(sqrt N), mean ln L_(1..k) - ln L_(k+1).
# Historical ES = mean(L[L >= VaR]); Normal ES = mu + sd phi(z)/(1-p).
# GARCH-EVT: arch_model(100 r, mean Zero, GARCH(1,1), Normal); omega/1e4, vol/100; z = r/vol, l = -z, POT at 0.90, scale by last vol
#   (rolling: one-step forecast vol sqrt(forecast variance)/100). Backtest: window 500 losses -> VaR for the next day, 1006 days.
# Kupiec/Christoffersen with 0 ln 0 = 0; p-values chi2.sf (df 1, 1, 2).
import os, json, math, warnings
import numpy as np, pandas as pd
from scipy.stats import norm, genpareto, chi2
from arch import arch_model
warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

P = json.load(open(os.path.join(DATA_DIR, "params.json")))
px = pd.read_csv(os.path.join(DATA_DIR, "sp500.csv"))["close"].to_numpy(float)
r = np.diff(np.log(px)); L = -r; N = len(r)
LEVELS, TQ, ZQ, W, BL = P["confidence_levels"], P["threshold_quantile"], P["garch_evt_threshold_quantile"], int(P["backtest_window"]), P["backtest_level"]
q_lower = lambda x, q: float(np.quantile(x, q, method="lower"))


def pot_fit(x, q):                                   # threshold, exceedance count, GPD (xi, sigma)
    u = q_lower(x, q); y = x[x > u] - u
    xi, _, s = genpareto.fit(y, floc=0)
    return u, len(y), float(xi), float(s), y


def pot_var_es(u, n_tot, n_u, xi, s, p):
    v = u + s / xi * ((n_tot / n_u * (1 - p)) ** (-xi) - 1)
    return v, v / (1 - xi) + (s - xi * u) / (1 - xi)


def garch_fit(ret):                                  # returns result on percentage returns
    return arch_model(100 * ret, mean="Zero", vol="GARCH", p=1, q=1, dist="Normal").fit(disp="off")


def garch_evt_var(ret, p, vol_scale, res):           # standardised-loss POT scaled by a volatility
    vol = res.conditional_volatility.to_numpy() / 100 if hasattr(res.conditional_volatility, "to_numpy") else np.asarray(res.conditional_volatility) / 100
    lz = -(ret / vol)
    uz, nuz, xiz, sz, _ = pot_fit(lz, ZQ)
    v, e = pot_var_es(uz, len(lz), nuz, xiz, sz, p)
    return vol_scale * v, vol_scale * e


# ---- full sample ----
mu, sd = float(L.mean()), float(L.std(ddof=1))
u, n_u, xi, sig, y = pot_fit(L, TQ)
k = int(math.floor(math.sqrt(N))); Ls = np.sort(L)[::-1]
hill = float(np.mean(np.log(Ls[:k])) - math.log(Ls[k]))
res_full = garch_fit(r)
om, al, be = [float(v) for v in res_full.params.values[:3]]
vol_full = np.asarray(res_full.conditional_volatility) / 100
out = {}
for p in LEVELS:
    key = str(p)[2:]
    v_h = q_lower(L, p); out[f"var_{key}_historical"], out[f"es_{key}_historical"] = v_h, float(L[L >= v_h].mean())
    z = norm.ppf(p); out[f"var_{key}_normal"], out[f"es_{key}_normal"] = mu + sd * z, mu + sd * norm.pdf(z) / (1 - p)
    out[f"var_{key}_evt"], out[f"es_{key}_evt"] = pot_var_es(u, N, n_u, xi, sig, p)
    out[f"var_{key}_garch_evt"], out[f"es_{key}_garch_evt"] = garch_evt_var(r, p, float(vol_full[-1]), res_full)

# ---- rolling one-day-ahead backtest ----
alpha = 1 - BL; z_bl = norm.ppf(BL)
viol = {m: [] for m in ["historical", "normal", "evt", "garch_evt"]}
for t in range(W, N):
    win, rw, actual = L[t - W:t], r[t - W:t], L[t]
    viol["historical"].append(actual > q_lower(win, BL))
    viol["normal"].append(actual > win.mean() + win.std(ddof=1) * z_bl)
    uw, nuw, xiw, sw, _ = pot_fit(win, TQ)
    viol["evt"].append(actual > pot_var_es(uw, W, nuw, xiw, sw, BL)[0])
    res = garch_fit(rw)
    vol_f = math.sqrt(float(res.forecast(horizon=1).variance.values[-1, 0])) / 100
    viol["garch_evt"].append(actual > garch_evt_var(rw, BL, vol_f, res)[0])


def xlnx(a, b):                                      # a * ln(b) with 0 ln 0 = 0
    return 0.0 if a == 0 else a * math.log(b)


out["backtest_num_days"] = int(N - W)
for m, v in viol.items():
    v = np.asarray(v, bool); T = len(v); x = int(v.sum()); pi = x / T
    lr_uc = -2 * (xlnx(x, alpha) + xlnx(T - x, 1 - alpha) - xlnx(x, pi) - xlnx(T - x, 1 - pi))
    n00 = int(np.sum(~v[:-1] & ~v[1:])); n01 = int(np.sum(~v[:-1] & v[1:]))
    n10 = int(np.sum(v[:-1] & ~v[1:])); n11 = int(np.sum(v[:-1] & v[1:]))
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0; p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pp = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlnx(n00 + n10, 1 - pp) + xlnx(n01 + n11, pp)
    ll1 = xlnx(n00, 1 - p01) + xlnx(n01, p01) + xlnx(n10, 1 - p11) + xlnx(n11, p11)
    lr_ind = -2 * (ll0 - ll1); lr_cc = lr_uc + lr_ind
    out.update({f"backtest_{m}_violations": x, f"backtest_{m}_kupiec_lr": float(lr_uc),
                f"backtest_{m}_kupiec_pvalue": float(chi2.sf(lr_uc, 1)),
                f"backtest_{m}_christoffersen_ind_lr": float(lr_ind), f"backtest_{m}_christoffersen_ind_pvalue": float(chi2.sf(lr_ind, 1)),
                f"backtest_{m}_cc_lr": float(lr_cc), f"backtest_{m}_cc_pvalue": float(chi2.sf(lr_cc, 2))})
json.dump({k: (float(v) if not isinstance(v, int) else v) for k, v in out.items()},
          open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

inter = {"num_returns": N, "mean_log_return": float(r.mean()), "std_log_return": float(r.std(ddof=1)),
         "threshold_value": u, "num_exceedances": n_u, "mean_excess": float(y.mean()), "gpd_shape_xi": xi,
         "gpd_scale_sigma": sig, "hill_xi": hill, "garch_omega": om / 1e4, "garch_alpha": al, "garch_beta": be,
         "garch_persistence": al + be, "garch_last_vol": float(vol_full[-1])}
json.dump({"intermediates": {k: {"value": v} for k, v in inter.items()}},
          open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
