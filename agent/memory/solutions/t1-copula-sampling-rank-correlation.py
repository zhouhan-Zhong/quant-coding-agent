# memory-solution
# unit: t1-copula-sampling-rank-correlation | category: risk | tags: copula, gaussian, student-t, clayton, gumbel, frank, kendall-tau, spearman, tail-dependence, marshall-olkin, monte-carlo
# deliverables: parameters.json, tail_dependence.json, samples.csv, summary.json
# Kendall-tau inversions: Gaussian/t rho = sin(pi tau/2); Clayton theta = 2tau/(1-tau); Gumbel theta = 1/(1-tau);
#   Frank: tau = 1 - 4/theta (1 - D1(theta)), D1 = (1/theta) int_0^theta t/(e^t-1) dt, solved with brentq (theta ~ 5.736).
# Sampling: Gaussian U = Phi(Z); t: T = Z/sqrt(chi2_nu/nu), U = t.cdf(T, nu); Clayton Marshall-Olkin with V ~ Gamma(1/theta);
#   Gumbel Marshall-Olkin with positive-stable S(alpha = 1/theta) via Chambers-Mallows-Stuck; Frank conditional inversion.
# Empirical tails: lambda_U(q) = #(u1>q & u2>q)/#(u1>q), lambda_L(q) = #(u1<1-q & u2<1-q)/#(u1<1-q). Single rng(42).
import os, json, math
import numpy as np, pandas as pd
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

N, SEED, TAU, NU = 100_000, 42, 0.5, 4
rng = np.random.default_rng(SEED)

rho = math.sin(math.pi * TAU / 2)
th_clayton = 2 * TAU / (1 - TAU)
th_gumbel = 1 / (1 - TAU)
def frank_tau(th):
    d1 = quad(lambda t: t / math.expm1(t), 0, th)[0] / th
    return 1 - 4 / th * (1 - d1)
th_frank = brentq(lambda th: frank_tau(th) - TAU, 0.01, 50)


def sample_gaussian(n):
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return stats.norm.cdf(z)


def sample_student_t(n):
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    w = rng.chisquare(NU, size=n) / NU
    return stats.t.cdf(z / np.sqrt(w)[:, None], NU)


def sample_clayton(n, th):                               # Marshall-Olkin with gamma frailty
    v = rng.gamma(1 / th, 1.0, size=n)
    e = rng.exponential(size=(n, 2))
    return (1 + e / v[:, None]) ** (-1 / th)


def sample_gumbel(n, th):                                # Marshall-Olkin with positive stable frailty
    a = 1 / th
    w = rng.uniform(0, math.pi, size=n); e0 = rng.exponential(size=n)
    s = (np.sin(a * w) / np.sin(w) ** (1 / a)) * (np.sin((1 - a) * w) / e0) ** ((1 - a) / a)
    e = rng.exponential(size=(n, 2))
    return np.exp(-(e / s[:, None]) ** a)


def sample_frank(n, th):                                 # conditional inversion
    u1, p = rng.uniform(size=n), rng.uniform(size=n)
    u2 = -np.log1p(p * (1 - math.exp(-th)) / (p * (np.exp(-th * u1) - 1) - np.exp(-th * u1))) / th
    return np.column_stack([u1, u2])


samplers = {"gaussian": (rho, lambda: sample_gaussian(N)), "student_t": (rho, lambda: sample_student_t(N)),
            "clayton": (th_clayton, lambda: sample_clayton(N, th_clayton)),
            "gumbel": (th_gumbel, lambda: sample_gumbel(N, th_gumbel)), "frank": (th_frank, lambda: sample_frank(N, th_frank))}

params, tails, rows = {}, {}, []
for name, (par, fn) in samplers.items():
    U = np.clip(fn(), 0.0, 1.0); u1, u2 = U[:, 0], U[:, 1]
    params[name] = {"native_param": float(par), "kendall_tau_theoretical": TAU,
                    "kendall_tau_empirical": float(stats.kendalltau(u1, u2)[0]),
                    "spearman_rho_empirical": float(stats.spearmanr(u1, u2)[0])}
    t = {}
    for q in (0.95, 0.99):
        k = str(q)[2:]
        hi = u1 > q; lo = u1 < 1 - q
        t[f"lambda_upper_{k}"] = float(np.sum(hi & (u2 > q)) / hi.sum())
        t[f"lambda_lower_{k}"] = float(np.sum(lo & (u2 < 1 - q)) / lo.sum())
    tails[name] = t
    rows.append(pd.DataFrame({"copula": name, "u1": u1[:1000], "u2": u2[:1000]}))

json.dump(params, open(os.path.join(OUT_DIR, "parameters.json"), "w"), indent=2)
json.dump(tails, open(os.path.join(OUT_DIR, "tail_dependence.json"), "w"), indent=2)
pd.concat(rows, ignore_index=True).to_csv(os.path.join(OUT_DIR, "samples.csv"), index=False)

taus = {k: v["kendall_tau_empirical"] for k, v in params.items()}
json.dump({"n_samples": N, "seed": SEED, "mean_kendall_tau_empirical": float(np.mean(list(taus.values()))),
           "max_tau_deviation": float(max(abs(v - TAU) for v in taus.values())),
           "copula_with_strongest_upper_tail": max(tails, key=lambda k: tails[k]["lambda_upper_95"]),
           "copula_with_strongest_lower_tail": max(tails, key=lambda k: tails[k]["lambda_lower_95"]),
           "frank_has_no_tail_dependence": bool(tails["frank"]["lambda_upper_95"] < 0.10 and tails["frank"]["lambda_lower_95"] < 0.10)},
          open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
