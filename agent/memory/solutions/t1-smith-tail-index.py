# memory-solution
# unit: t1-smith-tail-index | category: risk | tags: evt, hill-estimator, smith-gpd-mle, tail-index, order-statistics, observed-fisher-information, confidence-interval, hill-plot, sp500
# deliverables: hill_estimates.csv, smith_estimates.csv, comparison.json, summary.json
# Losses L = -ln(P_t/P_{t-1}) sorted DESCENDING: x[0] largest, x[k] = (k+1)-th largest = threshold u.
# Hill: xi = mean(ln x[:k]) - ln x[k]; se = xi/sqrt(k); CI = xi -/+ 1.96 se.
# Smith: GPD on excesses y = x[:k] - u, NLL = k ln s + (1 + 1/xi) sum ln(1 + xi y/s), Nelder-Mead with positivity guard;
#   se(xi) = sqrt([H^-1]_00) with H the numerically differentiated (central differences) observed Fisher information at the MLE.
# best_k = argmin se_xi (largest k here); correlation = Pearson across the 10 k values; tail_is_heavy = mean xi_smith > 0.
import os, json, math
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

px = pd.read_csv(os.path.join(DATA_DIR, "sp500_daily.csv"), parse_dates=["date"]).sort_values("date")
loss = -np.diff(np.log(px["close"].to_numpy(float)))
x = np.sort(loss)[::-1]                                   # descending order statistics
K_VALUES = [25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
z = norm.ppf(0.975)

def gpd_nll(p, y):
    xi, s = p
    if s <= 0:
        return 1e10
    w = 1.0 + xi * y / s
    if np.any(w <= 0):
        return 1e10
    if abs(xi) < 1e-9:
        return len(y) * math.log(s) + y.sum() / s
    return len(y) * math.log(s) + (1.0 + 1.0 / xi) * np.sum(np.log(w))

def num_hessian(f, p, h=1e-4):
    p = np.asarray(p, float); n = len(p); H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i = np.zeros(n); e_j = np.zeros(n)
            e_i[i] = h * max(1.0, abs(p[i])); e_j[j] = h * max(1.0, abs(p[j]))
            H[i, j] = (f(p + e_i + e_j) - f(p + e_i - e_j) - f(p - e_i + e_j) + f(p - e_i - e_j)) / (4 * e_i[i] * e_j[j])
    return H

hill_rows, smith_rows = [], []
for k in K_VALUES:
    u = x[k]
    xi_h = float(np.mean(np.log(x[:k])) - math.log(u))
    se_h = xi_h / math.sqrt(k)
    hill_rows.append({"k": k, "xi_hill": xi_h, "se_hill": se_h, "ci_lower": xi_h - z * se_h, "ci_upper": xi_h + z * se_h})

    y = x[:k] - u
    best = None
    for x0 in ([0.2, y.mean()], [0.1, 0.5 * y.mean()], [0.4, 1.5 * y.mean()]):
        res = minimize(gpd_nll, x0, args=(y,), method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000})
        if best is None or res.fun < best.fun:
            best = res
    xi_s, sig_s = float(best.x[0]), float(best.x[1])
    H = num_hessian(lambda p: gpd_nll(p, y), best.x)      # observed Fisher information (NLL Hessian)
    try:
        cov = np.linalg.inv(H)
        se_xi = math.sqrt(cov[0, 0]) if cov[0, 0] > 0 else float("nan")
    except np.linalg.LinAlgError:
        se_xi = float("nan")
    if not np.isfinite(se_xi):
        se_xi = (1.0 + xi_s) / math.sqrt(k)               # asymptotic fallback
    smith_rows.append({"k": k, "xi_smith": xi_s, "sigma_smith": sig_s, "se_xi": se_xi,
                       "ci_lower": xi_s - z * se_xi, "ci_upper": xi_s + z * se_xi})

hill = pd.DataFrame(hill_rows); smith = pd.DataFrame(smith_rows)
hill.to_csv(os.path.join(OUT_DIR, "hill_estimates.csv"), index=False)
smith.to_csv(os.path.join(OUT_DIR, "smith_estimates.csv"), index=False)

mean_h, mean_s = float(hill["xi_hill"].mean()), float(smith["xi_smith"].mean())
corr = float(np.corrcoef(hill["xi_hill"], smith["xi_smith"])[0, 1])
json.dump({"mean_xi_hill": mean_h, "mean_xi_smith": mean_s, "correlation": corr},
          open(os.path.join(OUT_DIR, "comparison.json"), "w"), indent=2)

ib = int(smith["se_xi"].idxmin())
best_k = int(smith.loc[ib, "k"])
json.dump({"best_k": best_k, "xi_hill_at_best_k": float(hill.loc[hill["k"] == best_k, "xi_hill"].iloc[0]),
           "xi_smith_at_best_k": float(smith.loc[ib, "xi_smith"]), "tail_is_heavy": bool(mean_s > 0)},
          open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
