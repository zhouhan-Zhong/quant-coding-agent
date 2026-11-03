# memory-solution
# unit: t1-var-es-estimation | category: risk | tags: var, expected-shortfall, historical-simulation, parametric-normal, student-t-mle, gaussian-kde, silverman, mixture-distribution, rmse-comparison, randomstate
# deliverables: true_values.json, estimates.csv, comparison.json, summary.json
# Losses (not returns): VaR_a = alpha-quantile (upper tail), ES_a = E[X | X >= VaR_a].
# RNG: ONE RandomState(42); order: standard_normal(N), standard_t(5,N), uniform(N), normal(0,1,N), normal(0,3,N), np.where(u<0.95,...).
# True values: Normal ES = phi(z)/(1-a); t ES = pdf(tq)/(1-a) * (nu+tq^2)/(nu-1); mixture VaR by brentq on 0.95 Phi(x)+0.05 Phi(x/3)=a, ES by quad.
# Historical: np.quantile + mean(x >= q). Normal fit: mean / std(ddof=0). t fit: scipy t.fit, ES by quad of x f(x) on [VaR, inf).
# KDE: gaussian_kde(bw='silverman'), CDF on fine grid (cumulative trapezoid) -> interp alpha; ES = quad(x kde(x), VaR, inf)/(1-a).
# RMSE per method over the 6 (dist, alpha) pairs; best per distribution = min of sqrt(mse_var)+sqrt(mse_es) over its 2 alphas.
import os, json, math
import numpy as np, pandas as pd
from scipy import stats, integrate, optimize

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

N, SEED, ALPHAS, NU = 10_000, 42, [0.95, 0.99], 5
DISTS = ["normal", "student_t", "contaminated_normal"]
METHODS = ["historical", "parametric_normal", "parametric_t", "kernel_density"]

rs = np.random.RandomState(SEED)
samples = {"normal": rs.standard_normal(N), "student_t": rs.standard_t(NU, N)}
u = rs.uniform(size=N)
comp1, comp2 = rs.normal(0.0, 1.0, N), rs.normal(0.0, 3.0, N)
samples["contaminated_normal"] = np.where(u < 0.95, comp1, comp2)

# ---- true risk measures ----
def normal_var_es(a, mu=0.0, sd=1.0):
    z = stats.norm.ppf(a)
    return mu + sd * z, mu + sd * stats.norm.pdf(z) / (1 - a)

def t_var_es_true(a, nu):
    tq = stats.t.ppf(a, nu)
    return tq, stats.t.pdf(tq, nu) / (1 - a) * (nu + tq ** 2) / (nu - 1)

mix_cdf = lambda x: 0.95 * stats.norm.cdf(x) + 0.05 * stats.norm.cdf(x / 3.0)
mix_pdf = lambda x: 0.95 * stats.norm.pdf(x) + 0.05 * stats.norm.pdf(x / 3.0) / 3.0
def mix_var_es(a):
    v = optimize.brentq(lambda x: mix_cdf(x) - a, -20, 50)
    es = integrate.quad(lambda x: x * mix_pdf(x), v, np.inf)[0] / (1 - a)
    return v, es

true = {d: {} for d in DISTS}
for a in ALPHAS:
    for d, (v, e) in zip(DISTS, [normal_var_es(a), t_var_es_true(a, NU), mix_var_es(a)]):
        true[d][str(a)] = {"var": float(v), "es": float(e)}
json.dump(true, open(os.path.join(OUT_DIR, "true_values.json"), "w"), indent=2)

# ---- estimators ----
def est_historical(x, a):
    q = np.quantile(x, a)
    return q, x[x >= q].mean()

def est_normal(x, a):
    return normal_var_es(a, x.mean(), x.std(ddof=0))

def est_t(x, a):
    nu, loc, sc = stats.t.fit(x)
    v = stats.t.ppf(a, nu, loc, sc)
    es = integrate.quad(lambda z: z * stats.t.pdf(z, nu, loc, sc), v, np.inf)[0] / (1 - a)
    return v, es

def est_kde(x, a):
    kde = stats.gaussian_kde(x, bw_method="silverman")
    grid = np.linspace(x.min() - 3 * kde.factor * x.std(), x.max() + 3 * kde.factor * x.std(), 20001)
    pdf = kde(grid)
    cdf = integrate.cumulative_trapezoid(pdf, grid, initial=0.0)
    cdf /= cdf[-1]
    v = float(np.interp(a, cdf, grid))
    es = integrate.quad(lambda z: z * kde(z)[0], v, grid[-1] + 5.0)[0] / (1 - a)
    return v, es

ESTIMATORS = dict(zip(METHODS, [est_historical, est_normal, est_t, est_kde]))
rows = []
for d in DISTS:
    for m in METHODS:
        for a in ALPHAS:
            v, e = ESTIMATORS[m](samples[d], a)
            rows.append({"distribution": d, "method": m, "alpha": a, "var_estimate": float(v), "es_estimate": float(e),
                         "true_var": true[d][str(a)]["var"], "true_es": true[d][str(a)]["es"]})
est = pd.DataFrame(rows)
est[["distribution", "method", "alpha", "var_estimate", "es_estimate"]].to_csv(os.path.join(OUT_DIR, "estimates.csv"), index=False)

# ---- comparison ----
est["se_var"] = (est["var_estimate"] - est["true_var"]) ** 2
est["se_es"] = (est["es_estimate"] - est["true_es"]) ** 2
method_rmse = {m: {"var_rmse": float(math.sqrt(g["se_var"].mean())), "es_rmse": float(math.sqrt(g["se_es"].mean()))}
               for m, g in est.groupby("method")}
best_per_dist = {}
for d, g in est.groupby("distribution"):
    comb = g.groupby("method").apply(lambda h: math.sqrt(h["se_var"].mean()) + math.sqrt(h["se_es"].mean()), include_groups=False)
    best_per_dist[d] = str(comb.idxmin())
json.dump({"method_rmse": method_rmse, "best_method_per_distribution": best_per_dist},
          open(os.path.join(OUT_DIR, "comparison.json"), "w"), indent=2)

overall = min(METHODS, key=lambda m: method_rmse[m]["var_rmse"] + method_rmse[m]["es_rmse"])
json.dump({"n_samples": N, "seed": SEED, "alpha_levels": ALPHAS, "distributions": DISTS, "methods": METHODS,
           "overall_best_method": overall}, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
