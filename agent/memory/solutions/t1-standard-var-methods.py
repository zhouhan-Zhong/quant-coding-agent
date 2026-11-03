# memory-solution
# unit: t1-standard-var-methods | category: risk | tags: var, expected-shortfall, historical-simulation, variance-covariance, student-t-mle, age-weighted-hs, ewma, kupiec-backtest, sqrt-time-scaling
# deliverables: data_summary.json, var_1day.csv, var_10day.csv, backtest.json, summary.json
# portfolio r_p = mean of the two log-returns; VaR/ES reported as positive dollar losses on $1M.
# HS: VaR = -np.quantile(r, 1-a) (linear interpolation); ES = -mean(r[r <= q]).
# Normal: VaR = -(mu + z sd) with sd ddof=1 (mean kept); ES = -(mu - sd phi(z)/(1-a)).
# StudentT: scipy.stats.t.fit (df, loc, scale) MLE; VaR = -t.ppf(1-a); ES closed form with (nu + t^2)/(nu-1).
# AgeWeightedHS: w_i ∝ 0.98^(age) normalised, sort returns, VaR = first return whose cumulative weight >= 1-a (no interpolation).
# EWMA: S_t = 0.94 S_{t-1} + 0.06 r_t r_t' over the asset returns (init sample cov), zero mean -> VaR = -z sqrt(w'Sw).
# 10-day: parametric x sqrt(10); HS/AWHS on overlapping 10-day rolling sums. Backtest: expanding window (min 252), loss > VaR, Kupiec POF LR ~ chi2(1).
import os, json, math
import numpy as np, pandas as pd
from scipy import stats

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

NOTIONAL = 1_000_000.0
ALPHAS = [0.95, 0.99]
raw = pd.read_csv(os.path.join(DATA_DIR, "pair_aapl_jpm_daily.csv"), parse_dates=["date"])
px = raw.pivot(index="date", columns="symbol", values="close").dropna().sort_index()
lr = np.log(px).diff().dropna()                       # asset log-returns, aligned
R = lr[["AAPL", "JPM"]].to_numpy()
port = R.mean(axis=1)                                 # equal-weight portfolio return
w = np.array([0.5, 0.5])

json.dump({"n_aapl": int((raw["symbol"] == "AAPL").sum()), "n_jpm": int((raw["symbol"] == "JPM").sum()),
           "n_aligned": int(len(px)), "n_returns": int(len(port)),
           "portfolio_mean": float(port.mean()), "portfolio_std": float(port.std(ddof=1)),
           "correlation": float(np.corrcoef(R[:, 0], R[:, 1])[0, 1])},
          open(os.path.join(OUT_DIR, "data_summary.json"), "w"), indent=2)


def hs(r, a):
    q = np.quantile(r, 1 - a)
    return -q, -r[r <= q].mean()


def normal(mu, sd, a):
    z = stats.norm.ppf(1 - a)
    return -(mu + z * sd), -(mu - sd * stats.norm.pdf(z) / (1 - a))


def student_t(r, a):
    nu, loc, sc = stats.t.fit(r)
    tq = stats.t.ppf(1 - a, nu)
    es_std = -stats.t.pdf(tq, nu) / (1 - a) * (nu + tq ** 2) / (nu - 1)
    return -(loc + sc * tq), -(loc + sc * es_std)


def age_weighted_hs(r, a, lam=0.98):
    n = len(r)
    wts = lam ** np.arange(n - 1, -1, -1)
    wts /= wts.sum()                                  # most recent observation has the largest weight
    order = np.argsort(r)
    sr, sw = r[order], wts[order]
    cw = np.cumsum(sw)
    idx = int(np.searchsorted(cw, 1 - a))             # first sorted return with cumulative weight >= 1-a
    q = sr[idx]
    m = sr <= q
    return -q, -float(np.sum(sr[m] * sw[m]) / np.sum(sw[m]))


def ewma_port_sd(R, lam=0.94):
    S = np.cov(R.T, ddof=1)
    for row in R:
        S = lam * S + (1 - lam) * np.outer(row, row)
    return math.sqrt(w @ S @ w)


mu, sd = port.mean(), port.std(ddof=1)
sd_ewma = ewma_port_sd(R)
port10 = np.convolve(port, np.ones(10), mode="valid")  # overlapping 10-day returns

rows1, rows10 = [], []
for a in ALPHAS:
    v, e = hs(port, a); rows1.append(("HS", a, v, e))
    v10, e10 = hs(port10, a); rows10.append(("HS", a, v10, e10))
    v, e = normal(mu, sd, a); rows1.append(("Normal", a, v, e)); rows10.append(("Normal", a, v * math.sqrt(10), e * math.sqrt(10)))
    v, e = student_t(port, a); rows1.append(("StudentT", a, v, e)); rows10.append(("StudentT", a, v * math.sqrt(10), e * math.sqrt(10)))
    v, e = age_weighted_hs(port, a); rows1.append(("AgeWeightedHS", a, v, e))
    v10, e10 = age_weighted_hs(port10, a); rows10.append(("AgeWeightedHS", a, v10, e10))
    v, e = normal(0.0, sd_ewma, a); rows1.append(("EWMA", a, v, e)); rows10.append(("EWMA", a, v * math.sqrt(10), e * math.sqrt(10)))

cols = ["method", "alpha", "var_dollar", "es_dollar"]
df1 = pd.DataFrame(rows1, columns=cols); df10 = pd.DataFrame(rows10, columns=cols)
for d in (df1, df10):
    d["var_dollar"] = d["var_dollar"].astype(float) * NOTIONAL
    d["es_dollar"] = d["es_dollar"].astype(float) * NOTIONAL
df1.to_csv(os.path.join(OUT_DIR, "var_1day.csv"), index=False)
df10.to_csv(os.path.join(OUT_DIR, "var_10day.csv"), index=False)

# Backtest: expanding-window HS 99% 1-day VaR, min 252 observations
MIN_OBS, P = 252, 0.01
exceed = 0
for t in range(MIN_OBS, len(port)):
    var_t = -np.quantile(port[:t], P)
    exceed += int(-port[t] > var_t)
n_test = len(port) - MIN_OBS
x = exceed
ll0 = (n_test - x) * math.log(1 - P) + x * math.log(P)
ll1 = (n_test - x) * math.log(1 - x / n_test) + (x * math.log(x / n_test) if x > 0 else 0.0)
LR = -2 * (ll0 - ll1)
json.dump({"n_test_days": int(n_test), "n_exceedances": int(x), "exceedance_rate": float(x / n_test),
           "expected_rate": P, "kupiec_pvalue": float(stats.chi2.sf(LR, 1)), "kupiec_lr": float(LR)},
          open(os.path.join(OUT_DIR, "backtest.json"), "w"), indent=2)

v99 = df1[np.isclose(df1["alpha"], 0.99)].set_index("method")["var_dollar"]
json.dump({"most_conservative_method": str(v99.idxmax()), "least_conservative_method": str(v99.idxmin()),
           "var_range_99": [float(v99.min()), float(v99.max())],
           "ewma_current_vol": float(sd_ewma * math.sqrt(252))},
          open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
