# memory-solution
# unit: t1-fft-compound-poisson | category: risk | tags: compound-poisson, aggregate-loss, fft, characteristic-function, panjer-alternative, var, expected-shortfall, gamma, lognormal, pareto, monte-carlo
# deliverables: fft_results.json, mc_results.json, comparison.json, summary.json
# Severity discretised on x_k = k h (k = 0..n-1, n = 2^16) by the rounding method p_k = F((k+1/2)h) - F((k-1/2)h), p_0 = F(h/2).
# Aggregate CF: phi_S = exp(lambda (fft(p) - 1)); f_S = ifft(phi_S).real (clip >= 0, renormalise); CDF = cumsum.
# Grid: n h = 2 x (MC 99.99th percentile) so the grid covers the tail with margin (h ~ 5 / 30 / 2 for gamma/lognormal/pareto).
# VaR_a = smallest grid x with CDF >= a; ES_a = sum_{x > VaR} x f / sum_{x > VaR} f.  MC: np.random.seed(42), 500k sims,
#   N ~ Poisson(50) then one vector of sum(N) severities split with np.add.reduceat; VaR = order statistic ceil(a n)-1, ES = mean(S > VaR).
# Pareto(alpha, x_m) sampled as x_m * U^{-1/alpha}; theoretical means: lambda*alpha*beta, lambda*exp(mu + s^2/2), lambda*alpha x_m/(alpha-1).
import os, json, math
import numpy as np
from scipy.stats import gamma as gamma_dist, lognorm, pareto as pareto_dist

OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

LAM, N_GRID, N_SIM = 50, 2 ** 16, 500_000
ALPHAS = [0.95, 0.99, 0.995]
SEV = {
    "gamma": dict(cdf=lambda x: gamma_dist.cdf(x, a=2.0, scale=1000.0),
                  sample=lambda m: np.random.gamma(2.0, 1000.0, m), mean=2.0 * 1000.0),
    "lognormal": dict(cdf=lambda x: lognorm.cdf(x, s=1.5, scale=math.exp(7.0)),
                      sample=lambda m: np.random.lognormal(7.0, 1.5, m), mean=math.exp(7.0 + 1.5 ** 2 / 2)),
    "pareto": dict(cdf=lambda x: pareto_dist.cdf(x, b=3.0, scale=500.0),
                   sample=lambda m: 500.0 * np.random.uniform(size=m) ** (-1.0 / 3.0), mean=3.0 * 500.0 / 2.0),
}

def mc_aggregate(sample):
    counts = np.random.poisson(LAM, N_SIM)
    sev = sample(int(counts.sum()))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    S = np.add.reduceat(sev, starts)
    S[counts == 0] = 0.0                    # reduceat returns sev[start] for empty segments
    return S

def risk_from_samples(S):
    Ss = np.sort(S); n = len(Ss); out = {}
    for a in ALPHAS:
        var = float(Ss[min(int(math.ceil(a * n)) - 1, n - 1)])
        tail = Ss[Ss > var]
        out[str(a)] = {"var": var, "es": float(tail.mean()) if len(tail) else var}
    return out

def fft_aggregate(cdf, x_max):
    h = x_max / N_GRID
    k = np.arange(N_GRID)
    edges = cdf((k + 0.5) * h)
    p = np.diff(np.concatenate([[0.0], edges]))          # p_0 = F(h/2), p_k = F((k+1/2)h) - F((k-1/2)h)
    phi_X = np.fft.fft(p)
    f = np.fft.ifft(np.exp(LAM * (phi_X - 1.0))).real
    f = np.clip(f, 0.0, None); f /= f.sum()
    x = k * h
    F = np.cumsum(f)
    out = {}
    for a in ALPHAS:
        idx = int(np.searchsorted(F, a))                 # smallest x with CDF >= a
        var = float(x[idx])
        tail = x > var
        out[str(a)] = {"var": var, "es": float((x[tail] * f[tail]).sum() / f[tail].sum())}
    return out

np.random.seed(42)
fft_res, mc_res, comp, means = {}, {}, {}, {}
for name, spec in SEV.items():
    S = mc_aggregate(spec["sample"])
    mc_res[name] = risk_from_samples(S)
    x_max = 2.0 * float(np.quantile(S, 0.9999))
    fft_res[name] = fft_aggregate(spec["cdf"], x_max)
    comp[name] = {str(a): {"var_rel_error": abs(fft_res[name][str(a)]["var"] - mc_res[name][str(a)]["var"]) / mc_res[name][str(a)]["var"],
                           "es_rel_error": abs(fft_res[name][str(a)]["es"] - mc_res[name][str(a)]["es"]) / mc_res[name][str(a)]["es"]}
                  for a in ALPHAS}
    means[name] = LAM * spec["mean"]

json.dump(fft_res, open(os.path.join(OUT_DIR, "fft_results.json"), "w"), indent=2)
json.dump(mc_res, open(os.path.join(OUT_DIR, "mc_results.json"), "w"), indent=2)
json.dump(comp, open(os.path.join(OUT_DIR, "comparison.json"), "w"), indent=2)
json.dump({"n_grid": N_GRID, "n_sim": N_SIM, "lambda": LAM, "severities": list(SEV), "alphas": ALPHAS,
           "max_relative_error_var": float(max(v[str(a)]["var_rel_error"] for v in comp.values() for a in ALPHAS)),
           "max_relative_error_es": float(max(v[str(a)]["es_rel_error"] for v in comp.values() for a in ALPHAS)),
           "mean_loss_gamma": float(means["gamma"]), "mean_loss_lognormal": float(means["lognormal"]),
           "mean_loss_pareto": float(means["pareto"])},
          open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
