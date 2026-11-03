# memory-solution
# unit: t1-copula-equity-fitting | category: risk | tags: copula, gaussian, student-t, clayton, gumbel, kendall-tau, pseudo-observations, profile-likelihood, aic, bic, tail-dependence, equity-pairs
# deliverables: data_summary.json, copula_fits.csv, best_copulas.json, tail_dependence.csv, summary.json
# Log-returns of closes (pivot by date, dropna) -> 692 rows; pairs = itertools.combinations([AAPL, JPM, GS, CVX, MSFT], 2) in that order.
# Pseudo-obs u = rankdata/(n+1); tau = scipy kendalltau (tau-b). Tau inversion: rho = sin(pi tau/2) (Gaussian and t), Clayton 2tau/(1-tau),
#   Gumbel 1/(1-tau). Negative tau: Clayton floored at 1e-6 and Gumbel at 1 (independence) so the loglik/AIC stay finite.
# t: rho fixed from tau, nu by bounded profile MLE on (2.01, 49.9); k = 2 for t, 1 otherwise; AIC = 2k - 2LL, BIC = k ln n - 2LL; best = min AIC.
# Empirical tails at q = 0.95: upper = #(u>q & v>q)/#(u>q), lower = #(u<1-q & v<1-q)/#(u<1-q).
import os, json, math, itertools
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
STOCKS, Q = ["AAPL", "JPM", "GS", "CVX", "MSFT"], 0.95

px = pd.read_csv(os.path.join(DATA_DIR, "dj30_constituents_daily.csv"), parse_dates=["date"])
px = px[px.symbol.isin(STOCKS)].pivot(index="date", columns="symbol", values="close").sort_index()[STOCKS]
R = np.log(px).diff().dropna()
n = len(R)

def gauss_ll(u, v, rho):
    x, y = stats.norm.ppf(u), stats.norm.ppf(v); r2 = 1 - rho * rho
    return float(np.sum(-0.5 * math.log(r2) - (rho * rho * (x * x + y * y) - 2 * rho * x * y) / (2 * r2)))
def t_ll(u, v, rho, nu):
    x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu); r2 = 1 - rho * rho
    joint = (math.lgamma((nu + 2) / 2) - math.lgamma(nu / 2) - math.log(nu * math.pi) - 0.5 * math.log(r2)
             - (nu + 2) / 2 * np.log1p((x * x - 2 * rho * x * y + y * y) / (nu * r2)))
    return float(np.sum(joint - stats.t.logpdf(x, nu) - stats.t.logpdf(y, nu)))
def clayton_ll(u, v, th):
    return float(np.sum(math.log1p(th) - (1 + th) * (np.log(u) + np.log(v)) - (2 + 1 / th) * np.log(u ** -th + v ** -th - 1)))
def gumbel_ll(u, v, th):
    x, y = -np.log(u), -np.log(v); A = (x ** th + y ** th) ** (1 / th)
    return float(np.sum(-A - np.log(u * v) + (th - 1) * (np.log(x) + np.log(y)) + (1 - 2 * th) * np.log(A) + np.log(A + th - 1)))

fits, tails, best = [], [], {}
for s1, s2 in itertools.combinations(STOCKS, 2):
    u, v = stats.rankdata(R[s1]) / (n + 1), stats.rankdata(R[s2]) / (n + 1)
    tau = float(stats.kendalltau(R[s1], R[s2])[0])
    rho = math.sin(math.pi * tau / 2)
    nu = float(minimize_scalar(lambda nu: -t_ll(u, v, rho, nu), bounds=(2.01, 49.9), method="bounded", options={"xatol": 1e-6}).x)
    th_c, th_g = max(2 * tau / (1 - tau), 1e-6), max(1 / (1 - tau), 1.0)
    rows = [("gaussian", rho, np.nan, gauss_ll(u, v, rho), 1), ("student_t", rho, nu, t_ll(u, v, rho, nu), 2),
            ("clayton", th_c, np.nan, clayton_ll(u, v, th_c), 1), ("gumbel", th_g, np.nan, gumbel_ll(u, v, th_g), 1)]
    for cop, p1, p2, ll, k in rows:
        fits.append({"stock1": s1, "stock2": s2, "copula": cop, "param1": p1, "param2": p2, "loglik": ll,
                     "aic": 2 * k - 2 * ll, "bic": k * math.log(n) - 2 * ll})
    best[f"{s1}-{s2}"] = min(rows, key=lambda r: 2 * r[4] - 2 * r[3])[0]
    up, lo = (u > Q) & (v > Q), (u < 1 - Q) & (v < 1 - Q)
    tails.append({"stock1": s1, "stock2": s2, "lambda_upper": up.sum() / max((u > Q).sum(), 1),
                  "lambda_lower": lo.sum() / max((u < 1 - Q).sum(), 1), "kendall_tau": tau})

pd.DataFrame(fits).to_csv(os.path.join(OUT_DIR, "copula_fits.csv"), index=False)
td = pd.DataFrame(tails); td.to_csv(os.path.join(OUT_DIR, "tail_dependence.csv"), index=False)
dump = lambda o, f: json.dump(o, open(os.path.join(OUT_DIR, f), "w"), indent=2)
dump({"n_stocks": len(STOCKS), "n_returns": int(n), "stock_list": STOCKS, "n_pairs": len(best)}, "data_summary.json")
dump(best, "best_copulas.json")
dump({"n_pairs": len(best), "most_common_best_copula": pd.Series(list(best.values())).value_counts().index[0],
      "mean_kendall_tau": float(td.kendall_tau.mean()),
      "has_tail_dependence": bool(((td.lambda_upper > 0.1) | (td.lambda_lower > 0.1)).any())}, "summary.json")
