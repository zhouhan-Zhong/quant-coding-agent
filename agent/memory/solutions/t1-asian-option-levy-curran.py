# memory-solution
# unit: t1-asian-option-levy-curran | category: derivatives-pricing | tags: asian-options, geometric-average, levy-moment-matching, curran, discrete-monitoring, monte-carlo, spy-calibration
# deliverables: calibration.json, asian_prices.csv, summary.json
# Discrete monitoring t_i = i T/n (i=1..n), GBM drift r:
#   geometric: ln G ~ N(mu_G, sG^2), mu_G = ln S0 + (r - s^2/2) T (n+1)/(2n), sG^2 = s^2 T (n+1)(2n+1)/(6 n^2)
#              C = e^{-rT}[ e^{mu_G + sG^2/2} N(d1) - K N(d2) ],  d1 = (mu_G + sG^2 - ln K)/sG, d2 = d1 - sG
#   Levy: match E[A] = S0/n sum e^{r t_i}, E[A^2] = S0^2/n^2 sum_i sum_j e^{r(t_i+t_j) + s^2 min(t_i,t_j)} to a lognormal
#   Curran: condition on G; sigma_iG = s^2/n sum_j min(t_i,t_j); K_hat = 2K - 1/n sum_i exp(mu_i + sigma_iG/sG^2 (ln K - mu_G) + (s_i^2 - sigma_iG^2/sG^2)/2)
#           C = e^{-rT}[ sum_i S0/n e^{mu_i' + s_i^2/2} N((mu_G - ln K_hat)/sG + sigma_iG/sG) - K N((mu_G - ln K_hat)/sG) ],  mu_i' = (r - s^2/2) t_i
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

px = pd.read_csv(os.path.join(DATA_DIR, "spy_daily.csv"), parse_dates=["date"]).sort_values("date")
close = px["close"].to_numpy(float); logret = np.diff(np.log(close))
S0 = float(close[-1]); ret_mean, ret_std = float(logret.mean()), float(logret.std(ddof=1))
sigma = ret_std * math.sqrt(252.0); r = 0.05
json.dump({"n_prices": int(len(close)), "sigma": sigma, "S0": S0, "return_mean": ret_mean, "return_std": ret_std},
          open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)

def geo_exact(K, T, n):
    t = np.arange(1, n + 1) * T / n
    mu_G = math.log(S0) + (r - 0.5 * sigma ** 2) * t.mean()
    var_G = sigma ** 2 / n ** 2 * np.minimum.outer(t, t).sum()
    sG = math.sqrt(var_G)
    d1 = (mu_G + var_G - math.log(K)) / sG; d2 = d1 - sG
    return math.exp(-r * T) * (math.exp(mu_G + 0.5 * var_G) * norm.cdf(d1) - K * norm.cdf(d2))

def levy(K, T, n):
    t = np.arange(1, n + 1) * T / n
    EA = S0 * np.exp(r * t).mean()
    EA2 = S0 ** 2 / n ** 2 * np.exp(r * np.add.outer(t, t) + sigma ** 2 * np.minimum.outer(t, t)).sum()
    var = math.log(EA2) - 2 * math.log(EA); sA = math.sqrt(var); muA = math.log(EA) - 0.5 * var
    d1 = (muA + var - math.log(K)) / sA; d2 = d1 - sA
    return math.exp(-r * T) * (EA * norm.cdf(d1) - K * norm.cdf(d2))

def curran(K, T, n):
    t = np.arange(1, n + 1) * T / n
    mu_i = (r - 0.5 * sigma ** 2) * t                 # drift of ln(S_i/S0)
    s2_i = sigma ** 2 * t
    mu_G = math.log(S0) + mu_i.mean()
    sG2 = sigma ** 2 / n ** 2 * np.minimum.outer(t, t).sum(); sG = math.sqrt(sG2)
    sig_iG = sigma ** 2 / n * np.minimum.outer(t, t).sum(axis=1)
    lnK = math.log(K)
    K_hat = 2 * K - (S0 / n) * np.exp(mu_i + sig_iG / sG2 * (lnK - mu_G) + 0.5 * (s2_i - sig_iG ** 2 / sG2)).sum()
    K_hat = max(K_hat, 1e-8)
    a = (mu_G - math.log(K_hat)) / sG
    return math.exp(-r * T) * ((S0 / n) * (np.exp(mu_i + 0.5 * s2_i) * norm.cdf(a + sig_iG / sG)).sum() - K * norm.cdf(a))

np.random.seed(42)
N_MC = 100_000
rows = []
for T in [0.5, 1.0]:
    for n in [4, 12, 52]:
        dt = T / n
        z = np.random.standard_normal((N_MC, n))
        logS = math.log(S0) + np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z, axis=1)
        S = np.exp(logS)
        A = S.mean(axis=1); G = np.exp(logS.mean(axis=1))
        for m in [0.95, 1.00, 1.05]:
            K = S0 * m
            pa = math.exp(-r * T) * np.maximum(A - K, 0.0); pg = math.exp(-r * T) * np.maximum(G - K, 0.0)
            rows.append({"T": T, "n_monitoring": n, "K": K, "moneyness": m, "geo_exact": geo_exact(K, T, n),
                         "levy_arith": levy(K, T, n), "curran_arith": curran(K, T, n),
                         "mc_arith": float(pa.mean()), "mc_geo": float(pg.mean()), "mc_std_err": float(pa.std(ddof=1) / math.sqrt(N_MC))})
df = pd.DataFrame(rows).sort_values(["T", "n_monitoring", "K"])
df.to_csv(os.path.join(OUT_DIR, "asian_prices.csv"), index=False)
levy_err = float((df["levy_arith"] - df["mc_arith"]).abs().max()); curran_err = float((df["curran_arith"] - df["mc_arith"]).abs().max())
json.dump({"n_rows": int(len(df)), "geo_less_than_arith_pct": float(100.0 * (df["geo_exact"] < df["mc_arith"]).mean()),
           "levy_vs_mc_max_error": levy_err, "curran_vs_mc_max_error": curran_err, "curran_better_than_levy": bool(curran_err < levy_err)},
          open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
