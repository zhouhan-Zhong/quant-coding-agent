# memory-solution
# unit: t1-merton-jump-diffusion | category: derivatives-pricing / calibration | tags: merton-jump-diffusion, mle, poisson-mixture, l-bfgs-b, random-restarts, merton-series, implied-vol-smile, newton-raphson, spy-calibration
# deliverables: calibration.json, merton_prices.csv, summary.json
# daily log-return density (dt = 1/252, N_max = 15, mu = mean(x)/dt fixed):
#   p(x) = sum_n Pois(n; lambda dt) * phi(x; (mu - s^2/2 - lambda kbar) dt + n mu_J, s^2 dt + n sigma_J^2),  kbar = exp(mu_J + sigma_J^2/2) - 1
#   -> use logsumexp over n for numerical safety; L-BFGS-B with bounds s[0.01,1] lambda[0.1,12] mu_J[-0.1,0.1] sigma_J[0.005,0.3], >=5 restarts keep best LL
# Merton series: C = sum_n e^{-lambda tau} (lambda tau)^n / n! * C_BS(S0, K, r_n, sigma_n, tau)
#   sigma_n^2 = s^2 + n sigma_J^2 / tau,  r_n = r - lambda kbar + n ln(1 + kbar) / tau,  N_series >= 3 lambda max(T) (use >= 50)
# BS comparison with sigma_total = sqrt(s^2 + lambda (mu_J^2 + sigma_J^2)); IV by Newton from sigma_total (fallback brentq)
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize, brentq
from scipy.special import logsumexp, gammaln

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

px = pd.read_csv(os.path.join(DATA_DIR, "spy_daily.csv"), parse_dates=["date"]).sort_values("date")
close = px["close"].to_numpy(float)
x = np.diff(np.log(close))
S0 = float(close[-1]); dt = 1.0 / 252.0; r = 0.05
mu = float(x.mean()) / dt
N_MAX = 15
n = np.arange(N_MAX + 1)
log_fact = gammaln(n + 1.0)

def loglik(theta):
    s, lam, mJ, sJ = theta
    kbar = math.exp(mJ + 0.5 * sJ * sJ) - 1.0
    log_pois = -lam * dt + n * math.log(lam * dt) - log_fact                      # (N_MAX+1,)
    means = (mu - 0.5 * s * s - lam * kbar) * dt + n * mJ
    var = s * s * dt + n * sJ * sJ
    z = (x[:, None] - means[None, :]) ** 2 / var[None, :]
    log_phi = -0.5 * (np.log(2 * np.pi * var)[None, :] + z)
    return float(logsumexp(log_pois[None, :] + log_phi, axis=1).sum())

bounds = [(0.01, 1.0), (0.1, 12.0), (-0.10, 0.10), (0.005, 0.30)]
rng = np.random.default_rng(42)
best = None
starts = [[x.std(ddof=1) * math.sqrt(252) * 0.8, 2.0, -0.01, 0.03]]
for _ in range(7):
    starts.append([rng.uniform(0.05, 0.3), rng.uniform(0.5, 8.0), rng.uniform(-0.05, 0.05), rng.uniform(0.01, 0.1)])
for s0 in starts:
    res = minimize(lambda th: -loglik(th), s0, method="L-BFGS-B", bounds=bounds)
    if best is None or -res.fun > best[0]:
        best = (-res.fun, res.x)
LL, (sigma, lam, mu_J, sigma_J) = best[0], [float(v) for v in best[1]]

json.dump({"n_returns": int(len(x)), "return_mean": float(x.mean()), "return_std": float(x.std(ddof=1)),
           "return_skewness": float(skew(x)), "return_excess_kurtosis": float(kurtosis(x, fisher=True)),
           "sigma": sigma, "lambda": lam, "mu_J": mu_J, "sigma_J": sigma_J, "log_likelihood": LL, "S0": S0},
          open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)

def bs_call(S, K, rr, vol, tau):
    sq = vol * math.sqrt(tau)
    d1 = (math.log(S / K) + (rr + 0.5 * vol * vol) * tau) / sq
    return S * norm.cdf(d1) - K * math.exp(-rr * tau) * norm.cdf(d1 - sq)

kbar = math.exp(mu_J + 0.5 * sigma_J ** 2) - 1.0
sigma_total = math.sqrt(sigma ** 2 + lam * (mu_J ** 2 + sigma_J ** 2))
maturities, moneyness = [0.25, 0.5, 1.0], [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
N_SERIES = max(50, int(math.ceil(3 * lam * max(maturities))) + 10)

def merton_call(K, tau):
    tot, lt = 0.0, lam * tau
    for k in range(N_SERIES + 1):
        w = math.exp(-lt + k * math.log(lt) - math.lgamma(k + 1)) if lt > 0 else (1.0 if k == 0 else 0.0)
        if w < 1e-16 and k > lt:
            break
        sig_n = math.sqrt(sigma ** 2 + k * sigma_J ** 2 / tau)
        r_n = r - lam * kbar + k * math.log1p(kbar) / tau
        tot += w * bs_call(S0, K, r_n, sig_n, tau)
    return tot

def implied_vol(price, K, tau):
    v = sigma_total
    for _ in range(100):
        sq = v * math.sqrt(tau); d1 = (math.log(S0 / K) + (r + 0.5 * v * v) * tau) / sq
        diff = bs_call(S0, K, r, v, tau) - price
        vega = S0 * norm.pdf(d1) * math.sqrt(tau)
        if abs(diff) < 1e-10:
            return v
        if vega < 1e-12:
            break
        v_new = v - diff / vega
        if not (1e-4 < v_new < 5.0):
            break
        if abs(v_new - v) < 1e-12:
            return v_new
        v = v_new
    try:
        return brentq(lambda vv: bs_call(S0, K, r, vv, tau) - price, 1e-4, 5.0)
    except ValueError:
        return v

rows = []
for T in maturities:
    for m in moneyness:
        K = S0 * m
        mp = merton_call(K, T); bp = bs_call(S0, K, r, sigma_total, T)
        rows.append({"T": T, "K": K, "moneyness": m, "merton_price": mp, "bs_price": bp, "merton_implied_vol": implied_vol(mp, K, T)})
df = pd.DataFrame(rows).sort_values(["T", "K"]).reset_index(drop=True)
df.to_csv(os.path.join(OUT_DIR, "merton_prices.csv"), index=False)

spreads = {T: float(g["merton_implied_vol"].max() - g["merton_implied_vol"].min()) for T, g in df.groupby("T")}
def atm(T): return float(df[np.isclose(df["T"], T) & np.isclose(df["moneyness"], 1.0)]["merton_price"].iloc[0])
summary = {"sigma_total": sigma_total, "n_maturities": 3, "n_strikes": 7, "n_prices": int(len(df)),
           "smile_present": bool(max(spreads.values()) > 0.005),
           "atm_merton_price_T025": atm(0.25), "atm_merton_price_T050": atm(0.5), "atm_merton_price_T100": atm(1.0),
           "max_iv_spread": float(max(spreads.values())),
           "mean_price_ratio_merton_over_bs": float((df["merton_price"] / df["bs_price"]).mean())}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
