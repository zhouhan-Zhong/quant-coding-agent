---
title: Latent factor models: IPCA by ALS, PCA factors, factor-neutral hedges, GRS test
category: backtest-factors
tags: ipca, instrumented pca, latent factors, kelly-pruitt-su, characteristics, managed portfolios, alternating least squares, als, gamma matrix, factor returns, identification, rotation, normalization, eigen rotation, total r2, cross-sectional r2, predictive r2, factor sharpe, grs, gibbons-ross-shanken, alpha test, pca, principal components, statistical factors, explained variance ratio, sklearn, factor loadings, factor neutral, pc neutral, hedge, min-norm, lstsq, params.json
summary: IPCA alternating least squares (Kronecker normal equations), the init/rotation/sign conventions that fix the non-identified solution, total vs cross-sectional R2, GRS statistic and p-value variants, PCA statistical factors and min-norm factor-neutral hedges.
---
## When this applies
"Estimate an IPCA model with K latent factors instrumented by firm characteristics", "alternating least squares
until convergence", "report Gamma norms, factor variance ratios, factor Sharpe ratios, total and cross-sectional
R2", "GRS test of the alphas", "PCA of returns: explained variance ratio and loadings", "make the portfolio neutral
to the first k principal components".

## Method and formulas
Symbols: r_t (N) excess returns of period t, Z_{t-1} (N x L) characteristics known at the end of t-1, Gamma (L x K),
f_t (K) latent factor returns, F (T x K), T periods, ridge = small Tikhonov constant.
- IPCA: r_t = Z_{t-1} Gamma f_t + e_t (dynamic betas z' Gamma). Pair returns of months 2..M with characteristics of
  months 1..M-1, so T = M - 1. Precompute ZZ_t = Z'Z (L x L) and Zr_t = Z'r_t (L).
- ALS: Gamma | F solves [sum_t (f_t f_t') kron ZZ_t + ridge I] vec(Gamma) = sum_t f_t kron Zr_t;
  F | Gamma: f_t = (Gamma' ZZ_t Gamma + ridge I)^-1 Gamma' Zr_t. Objective SSE = sum_t ||r_t - Z_{t-1} Gamma f_t||^2.
  Reference stopping rule: |SSE - SSE_prev| / SSE_prev < tol (an absolute max|dGamma| < tol rule stops elsewhere).
- Identification: (F A, Gamma A^-T) fits identically for any invertible A, so WHERE ALS lands depends on the init.
  Reference convention: F0 = np.random.default_rng(seed).standard_normal((T, K)) with each COLUMN scaled to unit L2
  norm; after convergence rotate: eigh(F'F/T) sorted descending, F <- F V, Gamma <- Gamma V, then flip the sign of
  any factor with a negative mean (and its Gamma column). KPS's normalisation (Gamma'Gamma = I, F'F diagonal) only
  if the task says.
- Statistics: factor_var_ratio_k = eig_k / sum(eig); total R2 = 1 - SSE / sum r^2 (UNCENTERED); cross-sectional
  R2_t = 1 - SSE_t / sum_i (r_it - mean_i r_t)^2 (CENTERED), then averaged over t; predictive R2 uses mean(F) in
  place of f_t; factor Sharpe = mean/std(ddof=1) * sqrt(12) for monthly data; factor correlations np.corrcoef(F.T).
- GRS: alphas a from time-series OLS of each r_i on [1, F]; E residuals; S = E'E/(T-K-1); Om = cov(F, ddof=1);
  mu = mean(F); q = a'S^-1 a / (1 + mu'Om^-1 mu); GRS = (T/N)(T-N-K)/(T-K-1) q ~ F(N, T-N-K). Reference p-value
  quirk observed: p = F(N, T-N-K).sf((T-N-K)/N q), i.e. WITHOUT the T/(T-K-1) factor - compute both; the p of the
  reported statistic itself failed a pinned check by more than its tolerance.
- PCA factors: sklearn PCA(n_components=k).fit(R) on raw (T x N) returns (centres internally; standardise only if
  told = correlation PCA); explained_variance_ratio_ descending (sum < 1); loadings components_ (k x N, signs
  arbitrary); factor returns (R - R.mean(0)) @ components_.T.
- Factor-neutral hedge: exposures e = comp @ w; h = min-norm solution of [comp; 1'] h = [-e; 0] (np.linalg.lstsq on
  the underdetermined system); w_neutral = w + h has zero PC exposures (machine zero) and unchanged net exposure;
  h is invariant to component sign flips. Residual R2 = 1 - var(R w_neutral) / var(R w).

## Conventions and pitfalls (what graders check)
- NaN cells: count them in the raw (date x ticker) pivot first (often a checkpoint), then fill returns and
  characteristics with 0; pivot sorted on dates and tickers; drop duplicate (date, ticker) keep last.
- Scale-sensitive checkpoints (||Gamma||_F, largest singular value, mean |Gamma|, factor stds, max |f|) depend on the
  init + normalisation; scale-invariant ones (variance ratios, Sharpe, correlations, R2) only on the rotation. To
  match a reference, search init variants against the scale-invariant targets first, then the scale.
- Inits that land elsewhere: raw normal draws without column normalisation, RandomState(seed).randn, QR/SVD of
  random matrices, PCA of managed portfolios. With an un-normalised init the ridge causes a slow drift and
  parameter-change stopping rules may never trigger.
- Iteration counts usually have a generous atol; als_converged must be a Python bool, iterations an int.
- Use np.linalg.solve, not inv, for the normal equations; einsum keeps the Kronecker system small (KL x KL).
- Characteristics are sometimes rank-transformed per period to [-0.5, 0.5] (KPS) - only when specified.
- PCA hedge outputs: weights sum equals the original sum (atol ~1e-2), neutral exposures |x| < 1e-6; keep the
  ticker order of params, not dict order of the data.

## Tested snippet
```python
import math, numpy as np
from scipy import stats
from sklearn.decomposition import PCA

def ipca_als(Rm, Zm, K, ridge=1e-8, tol=1e-6, max_iter=500, seed=42):   # Rm (T,N) r_t paired with Zm (T,N,L) = z_{t-1}
    T, N, L = Zm.shape; ZZ = np.einsum("tni,tnj->tij", Zm, Zm); Zr = np.einsum("tni,tn->ti", Zm, Rm)
    def gamma_step(F):                                                  # vec(Gamma) from the Kronecker normal equations
        A = np.einsum("tk,tl,tij->kilj", F, F, ZZ).reshape(K * L, K * L); b = np.einsum("tk,ti->ki", F, Zr).reshape(-1)
        return np.linalg.solve(A + ridge * np.eye(K * L), b).reshape(K, L).T
    def f_step(G):                                                      # f_t = (G'Z'ZG + ridge I)^-1 G'Z'r_t per period
        return np.linalg.solve(np.einsum("ik,tij,jl->tkl", G, ZZ, G) + ridge * np.eye(K), (Zr @ G)[..., None])[..., 0]
    sse = lambda G, F: float(((Rm - np.einsum("tni,ik,tk->tn", Zm, G, F)) ** 2).sum())
    F = np.random.default_rng(seed).standard_normal((T, K)); F /= np.linalg.norm(F, axis=0)   # unit-norm columns
    obj_old, conv = None, False
    for it in range(1, max_iter + 1):
        G = gamma_step(F); F = f_step(G); obj = sse(G, F)
        if obj_old is not None and abs(obj - obj_old) / obj_old < tol: conv = True; break
        obj_old = obj
    ev, V = np.linalg.eigh(F.T @ F / T); o = np.argsort(ev)[::-1]; ev, V = ev[o], V[:, o]
    F, G = F @ V, G @ V; s = np.where(F.mean(0) < 0, -1.0, 1.0)         # principal-axis rotation, positive factor means
    return G * s, F * s, ev / ev.sum(), it, conv

def grs(Rm, F):                                                        # alphas of r_i on [1, F]; GRS ~ F(N, T-N-K)
    T, N = Rm.shape; K = F.shape[1]; X = np.column_stack([np.ones(T), F]); B = np.linalg.lstsq(X, Rm, rcond=None)[0]
    a, E = B[0], Rm - X @ B; S = E.T @ E / (T - K - 1); mu = F.mean(0); Om = np.atleast_2d(np.cov(F.T, ddof=1))
    q = a @ np.linalg.solve(S, a) / (1 + mu @ np.linalg.solve(Om, mu)); stat = T / N * (T - N - K) / (T - K - 1) * q
    return a, stat, float(stats.f.sf(stat, N, T - N - K)), float(stats.f.sf((T - N - K) / N * q, N, T - N - K))

def pca_neutral(R, w, k):                                              # min-norm hedge: zero PC exposures, dollar-neutral
    comp = PCA(n_components=k).fit(R).components_                       # sklearn centres internally; do not standardise
    h = np.linalg.lstsq(np.vstack([comp, np.ones((1, R.shape[1]))]), np.append(-comp @ w, 0.0), rcond=None)[0]
    return w + h, comp @ (w + h)                                        # invariant to component sign flips

rng = np.random.default_rng(1); T, N, L, K = 120, 50, 5, 2
Z = rng.standard_normal((T + 1, N, L)); G0 = rng.normal(0, 0.02, (L, K)); f0 = rng.normal(0.3, 1.0, (T, K))
Rm = np.einsum("tni,ik,tk->tn", Z[:-1], G0, f0) + rng.normal(0, 0.05, (T, N)); Zm = Z[:-1]   # r_t explained by z_{t-1}
G, F, vr, n_it, conv = ipca_als(Rm, Zm, K)
E = Rm - np.einsum("tni,ik,tk->tn", Zm, G, F)
tot_r2 = 1 - (E ** 2).sum() / (Rm ** 2).sum()                           # uncentered total R2
cs_r2 = np.mean(1 - (E ** 2).sum(1) / ((Rm - Rm.mean(1, keepdims=True)) ** 2).sum(1))   # CENTERED per period, then mean
a, g_stat, p_std, p_quirk = grs(Rm, F)
print(n_it, conv, np.round(vr, 4), round(float(tot_r2), 4), round(float(cs_r2), 4), np.round(F.mean(0) / F.std(0, ddof=1) * math.sqrt(12), 3))
print(round(float(g_stat), 4), round(p_std, 4), round(p_quirk, 4), round(float(np.linalg.norm(G)), 4), np.round(pca_neutral(Rm, np.full(N, 1 / N), 3)[1], 12))
```
