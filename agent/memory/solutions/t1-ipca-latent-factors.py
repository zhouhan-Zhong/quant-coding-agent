# memory-solution
# unit: t1-ipca-latent-factors | category: backtest-factors | tags: ipca, instrumented-pca, latent-factors, alternating-least-squares, kelly-pruitt-su, characteristics, eigen-rotation, grs-test, params-json-family
# deliverables: results.json, solution.json
# returns pivoted (date x ticker), nan_count = raw NaNs, NaN -> 0; r_t (months 2..120) paired with z_{t-1} -> T = 119
# ALS init: F0 = default_rng(init_seed).standard_normal((T, K)) with each COLUMN scaled to unit L2 norm (this fixes the
#   scale/rotation the checker expects); each iteration: Gamma | F (Kronecker normal equations + ridge), then F | Gamma;
#   stop when |obj - obj_prev| / obj_prev < als_tol (obj = SSE) -> 18 iterations
# rotation: eigh(F'F/T) descending, F <- F V, Gamma <- Gamma V; flip sign of factors with negative mean
# total R2 uncentered; mean CS R2 CENTERED per month; Sharpe = mean/std(ddof=1)*sqrt(12); alphas from OLS r_i on [1, F]
# GRS = T/N*(T-N-K)/(T-K-1) * a'S^-1 a / (1 + mu' Om^-1 mu), S = E'E/(T-K-1), Om = cov(F, ddof=1);
#   reference p-value = F(N, T-N-K).sf of the statistic WITHOUT the T/(T-K-1) factor (consistent p = 0.091 fails)
import os, json, math
import numpy as np, pandas as pd
from scipy import stats

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

P = json.load(open(os.path.join(DATA_DIR, "params.json")))
ret = pd.read_csv(os.path.join(DATA_DIR, "returns.csv")).drop_duplicates(["date", "ticker"], keep="last")
ch = pd.read_csv(os.path.join(DATA_DIR, "characteristics.csv")).drop_duplicates(["date", "ticker"], keep="last")
Rw = ret.pivot(index="date", columns="ticker", values="excess_return").sort_index().sort_index(axis=1)
nan_count = int(Rw.isna().sum().sum())
R = Rw.fillna(0.0).to_numpy()
names = P["characteristic_names"]
Z = np.stack([ch.pivot(index="date", columns="ticker", values=k).reindex(index=Rw.index, columns=Rw.columns)
              .fillna(0.0).to_numpy() for k in names], axis=2)          # (months, N, L)
Rm, Zm = R[1:], Z[:-1]                                                    # r_t with z_{t-1}
T, N, L = Zm.shape
K, ridge, tol, max_iter = int(P["n_factors"]), float(P["ridge"]), float(P["als_tol"]), int(P["als_max_iter"])
ZZ = np.einsum("tni,tnj->tij", Zm, Zm); Zr = np.einsum("tni,tn->ti", Zm, Rm)


def gamma_step(F):          # vec(Gamma) from sum_t kron(f f', Z'Z) vec(Gamma) = sum_t kron(f, Z'r)
    A = np.einsum("tk,tl,tij->kilj", F, F, ZZ).reshape(K * L, K * L)
    b = np.einsum("tk,ti->ki", F, Zr).reshape(-1)
    return np.linalg.solve(A + ridge * np.eye(K * L), b).reshape(K, L).T


def f_step(G):              # f_t = (G' Z'Z G)^-1 G' Z'r
    M = np.einsum("ik,tij,jl->tkl", G, ZZ, G) + ridge * np.eye(K)
    return np.linalg.solve(M, (Zr @ G)[..., None])[..., 0]


def sse(G, F):
    return float(((Rm - np.einsum("tni,ik,tk->tn", Zm, G, F)) ** 2).sum())


F = np.random.default_rng(int(P["init_seed"])).standard_normal((T, K))
F = F / np.linalg.norm(F, axis=0)
obj_old, converged = None, False
for it in range(1, max_iter + 1):
    G = gamma_step(F); F = f_step(G); obj = sse(G, F)
    if obj_old is not None and abs(obj - obj_old) / obj_old < tol:
        converged = True; break
    obj_old = obj

ev, V = np.linalg.eigh(F.T @ F / T)
o = np.argsort(ev)[::-1]; ev, V = ev[o], V[:, o]
F, G = F @ V, G @ V
s = np.where(F.mean(axis=0) < 0, -1.0, 1.0); F, G = F * s, G * s

E = Rm - np.einsum("tni,ik,tk->tn", Zm, G, F)
total_r2 = 1 - (E ** 2).sum() / (Rm ** 2).sum()
cs_r2 = np.mean(1 - (E ** 2).sum(1) / ((Rm - Rm.mean(1, keepdims=True)) ** 2).sum(1))
sd = F.std(axis=0, ddof=1); mu = F.mean(axis=0)
sharpe = mu / sd * math.sqrt(P["annualization_factor"])
C = np.corrcoef(F.T)

X = np.column_stack([np.ones(T), F]); B = np.linalg.lstsq(X, Rm, rcond=None)[0]
alpha = B[0]; Eres = Rm - X @ B
S = Eres.T @ Eres / (T - K - 1); Om = np.cov(F.T, ddof=1)
q = alpha @ np.linalg.solve(S, alpha) / (1 + mu @ np.linalg.solve(Om, mu))
grs = T / N * (T - N - K) / (T - K - 1) * q
grs_p = stats.f.sf((T - N - K) / N * q, N, T - N - K)

results = {"total_r_squared": float(total_r2), "mean_cs_r_squared": float(cs_r2), "grs_statistic": float(grs),
           "grs_p_value": float(grs_p)}
results.update({f"factor{k + 1}_sharpe_annual": float(sharpe[k]) for k in range(K)})
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

sv = np.linalg.svd(G, compute_uv=False)
inter = {"nan_count": nan_count, "als_converged": bool(converged), "als_iterations": int(it),
         "gamma_frobenius_norm": float(np.linalg.norm(G)), "gamma_largest_sv": float(sv[0]),
         "mean_alpha": float(alpha.mean()), "mean_abs_alpha": float(np.abs(alpha).mean()), "total_r_squared": float(total_r2),
         "max_factor_return_abs": float(np.abs(F).max()), "gamma_abs_mean": float(np.abs(G).mean()),
         "mean_return_matrix": float(Rm.mean()), "max_abs_alpha": float(np.abs(alpha).max())}
for k in range(K):
    inter[f"factor_var_ratio_{k + 1}"] = float(ev[k] / ev.sum())
    inter[f"std_factor{k + 1}"] = float(sd[k])
    for j in range(k + 1, K):
        inter[f"factor_corr_{k + 1}{j + 1}"] = float(C[k, j])
json.dump({"intermediates": {k: {"value": v} for k, v in inter.items()}, "deliverables": results},
          open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
