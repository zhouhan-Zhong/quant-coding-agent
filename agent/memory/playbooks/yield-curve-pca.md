---
title: Yield-curve PCA: level, slope, curvature, scenarios and hedging
category: rates-curves
tags: pca, principal components, yield curve, level, slope, curvature, eigenvalues, eigenvectors, loadings, explained variance, covariance, correlation, standardize, daily changes, treasury, tenors, reconstruction, scenario, shock, pca hedging, key rate, dv01, butterfly, 2s10s, par yields, factor exposures, krd exposures, liability
summary: How to run PCA on yield changes (not levels), fix loading signs, report explained variance and reconstructions, build PC scenarios and hedge a bond portfolio's PC exposures.
---
## When this applies
"Perform PCA on the daily changes of the Treasury curve", "report the first three components / explained
variance", "interpret level, slope, curvature", "reconstruct the curve with k components", "generate +/-2 sigma
PC scenarios", "hedge the portfolio against the first two factors", spread/butterfly statistics on tenors.

## Method and formulas
Symbols: Y (T x n) yields in percent by tenor (columns ordered short to long), dY = 100 * diff(Y) changes in bp,
X = matrix analysed (default: changes), V = loading matrix (columns = eigenvectors), lambda_k = eigenvalues.
- Centre: Z = X - mean(X). Covariance PCA: C = cov(Z, ddof=1) = Z'Z/(T-1). Standardised (correlation) PCA:
  divide each column by its std (ddof=1) first; use only when the task says "standardised"/"correlation matrix".
- Decompose: eigh(C) -> sort eigenvalues descending; V[:, k] unit-norm, mutually orthogonal; explained ratio
  = lambda_k / sum lambda; cumulative = cumsum. Equivalent to sklearn PCA (explained_variance_ uses T-1) and to
  SVD of Z (lambda_k = s_k^2/(T-1)).
- Scores (factor realisations): S = Z V; std(S[:, k], ddof=1) = sqrt(lambda_k). Reconstruction with k PCs:
  X_hat = mean + S[:, :k] V[:, :k]'; residual RMSE = sqrt(mean((X - X_hat)^2)); the fraction of variance kept
  equals the cumulative ratio.
- Interpretation: PC1 loadings all the same sign (level), PC2 monotone from negative at the short end to positive
  at the long end (slope), PC3 positive at both ends and negative in the belly (curvature) - or the mirror; signs
  are arbitrary and must be fixed by rule.
- Empirical proxies (bp): level = mean of tenors or the 10y, slope = 10y - 2y, curvature/butterfly =
  2*5y - 2y - 10y, term premium proxy = 30y - 3m. Spread stats: mean, std (ddof=1), max and its date (idxmax).
- Scenarios: shock_k = +/- m sqrt(lambda_k) V[:, k] (bp, m = 1, 2 or 3); shifted curve = last curve + shock/100
  (percent). Combined scenario = sum of chosen PC shocks. Historical scenario = worst score days.
- PCA hedging: portfolio exposure to PC k, e_k = sum_i KRD_i (or DV01_i) V_ik; hedge instrument j has
  h_jk = sum_i KRD_ij V_ik; solve H x = -e for the first k PCs (square -> np.linalg.solve, otherwise lstsq);
  residual exposure after hedging should be ~0 for the hedged PCs. PnL approx: dP = -sum_k e_k * score_k.
- KRD -> factor exposures in ALM/immunisation tasks: PCA of daily changes of the PAR yields on the same knots as the
  KRDs (eigh(np.cov(diff.T)), descending), e_k = KRD . v_k with each PC's sign flipped so the LIABILITY exposure is
  positive (apply the same flip to every instrument); top-3 explained share typically > 0.95. Exposures then enter a
  hedge objective sum_k (w . e_k,instr - e_k,liab)^2 (see bond-analytics-immunization).

## Conventions and pitfalls (what graders check)
- Use CHANGES for PCA unless the task explicitly says levels (levels are non-stationary: PC1 > 99%). Daily
  changes of a Treasury CSV in percent: multiply by 100 for bp. Ratios are scale-invariant; eigenvalues scale by
  1e4 between percent^2 and bp^2 - report in the units requested.
- Sort by date ascending BEFORE differencing (Treasury downloads are newest-first, dates MM/DD/YYYY); drop tenors
  with missing history (e.g. 2 Mo, 20 Yr) or rows with NaN consistently; state n_obs = rows after diff.
- Sign convention: flip V[:, 0] so its sum (or mean) is positive; flip V[:, 1] so long minus short > 0; flip
  V[:, 2] so the ends are positive; apply the SAME flips to scores. Checkers usually test "all PC1 loadings share
  a sign", explained ratios (1e-6), cumulative of top 3 in a range (0.9-0.999), and loadings up to sign (abs).
- ddof: np.cov default is ddof=1; np.var/np.std default ddof=0 - be explicit. Standardising changes loadings
  materially; do not mix covariance eigenvalues with correlation loadings.
- Reconstruction must include the mean; scores computed from the centred matrix.
- Scenario shocks in bp when applied to percent yields must be divided by 100; report shocked curves with the
  same tenor order as input.
- Hedge with k instruments for k PCs gives an exact solve; more PCs than instruments -> least squares; keep the
  instrument order and sign (negative = short the hedge).
- Reproducibility: eigh is deterministic; sklearn PCA(svd_solver='full') too; randomised solvers are not.

## Tested snippet
```python
import numpy as np, pandas as pd

def curve_pca(X, standardize=False, n_keep=3):
    """X: (T, n) matrix of daily changes (bp). Returns loadings (n, n_keep), eigenvalues, ratios, scores, mean, scale."""
    mu = X.mean(axis=0); sd = X.std(axis=0, ddof=1) if standardize else np.ones(X.shape[1])
    Z = (X - mu) / sd
    cov = np.cov(Z.T, ddof=1)                                  # covariance (or correlation if standardized)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]; evals, evecs = evals[order], evecs[:, order]
    V = evecs[:, :n_keep].copy()
    if V[:, 0].sum() < 0: V[:, 0] *= -1                        # PC1 (level): all loadings positive
    if n_keep > 1 and V[-1, 1] - V[0, 1] < 0: V[:, 1] *= -1    # PC2 (slope): long end minus short end positive
    if n_keep > 2 and V[0, 2] + V[-1, 2] < 0: V[:, 2] *= -1    # PC3 (curvature): ends positive, belly negative
    scores = Z @ V                                             # (T, n_keep); std(scores_k, ddof=1) = sqrt(evals_k)
    return V, evals, evals / evals.sum(), scores, mu, sd

def reconstruct(scores, V, mu, sd): return scores @ V.T * sd + mu

def scenarios(V, evals, k_sd=2.0):                             # +/- k_sd sigma shocks along each PC, in input units (bp)
    return {f"pc{i + 1}_{'up' if s > 0 else 'down'}": s * k_sd * np.sqrt(evals[i]) * V[:, i]
            for i in range(V.shape[1]) for s in (1, -1)}

def pca_hedge(krd_port, krd_hedges, V):                        # zero the portfolio's exposure to the first k PCs
    e_port = V.T @ krd_port                                    # (k,) exposures: sum_i KRD_i v_ik
    H = V.T @ krd_hedges                                       # (k, n_hedges)
    x, *_ = np.linalg.lstsq(H, -e_port, rcond=None)            # square -> exact solve; else least squares
    return x, e_port + H @ x

rng = np.random.default_rng(0); tenors = np.array([0.25, 1, 2, 3, 5, 7, 10, 30])
f = rng.standard_normal((1000, 3)) * [5, 2, 1]                 # synthetic level/slope/curvature factors in bp
L = np.c_[np.ones(8), (tenors - 5) / 10, ((tenors - 5) / 10) ** 2 - 0.3]
dY = f @ L.T + 0.3 * rng.standard_normal((1000, 8))            # daily yield changes (bp)
V, ev, ratio, sc, mu, sd = curve_pca(dY)
err = np.sqrt(np.mean((reconstruct(sc, V, mu, sd) - dY) ** 2))
x, resid = pca_hedge(np.array([0.1, 0.5, 1.0, 1.4, 2.0, 2.5, 3.0, 1.0]), np.c_[np.eye(8)[:, [1, 4, 6]]], V)
print(np.round(ratio[:3], 4).tolist(), round(float(ratio[:3].sum()), 6), np.round(V[:, 0], 3).tolist())
print(round(float(err), 4), np.round(sc.std(axis=0, ddof=1) ** 2 / ev[:3], 6).tolist(), np.round(x, 3).tolist(), np.abs(resid).max() < 1e-10)
```
