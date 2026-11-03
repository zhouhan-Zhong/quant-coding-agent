---
title: Copulas and dependence: fitting, model selection, sampling, tail dependence
category: risk
tags: copula, gaussian copula, t copula, student-t copula, clayton, gumbel, frank, archimedean, kendall tau, spearman rho, rank correlation, pseudo-observations, empirical cdf, mle, ifm, inference functions for margins, tau inversion, profile likelihood, degrees of freedom, sampling, simulation, cholesky, marshall-olkin, tail dependence, lower tail, upper tail, empirical tail dependence, aic, bic, model selection, best copula, negative dependence, equity pairs, dependence structure, joint distribution
summary: Densities, tau-to-parameter maps with floors for negative dependence, profiled t degrees of freedom, MLE/IFM fitting, AIC/BIC selection, sampling algorithms and model/empirical tail-dependence coefficients, with the pseudo-observation and seed conventions graders check.
---
## When this applies
"Fit Gaussian/t/Clayton/Gumbel copulas to every pair of stocks and pick the best by AIC", "pseudo-observations",
"Kendall's tau implied parameter", "simulate N joint scenarios from the copula with these margins", "tail
dependence coefficients (model and empirical)", "portfolio VaR with a copula".

## Method and formulas
Symbols: n obs, d dims, U = (u, v) pseudo-observations in (0,1), x = Phi^-1(u), rho = correlation parameter,
nu = t degrees of freedom, th = Archimedean parameter, tau = Kendall's tau, rho_S = Spearman's rho.
- Data: pivot closes (date x symbol), log returns diff(log P), dropna; pairs = itertools.combinations in the given
  symbol order, keys "S1-S2". tau = scipy.stats.kendalltau(r1, r2) (tau-b; rank-based, same on returns or ranks).
- Pseudo-observations: u_i = rank(x_i)/(n+1) (scipy.stats.rankdata, average ranks); alternatively IFM: fit the
  margins (e.g. t or normal) and set u = F_hat(x). Never use rank/n (u = 1 breaks ppf/log).
- Gaussian: c(u,v) = (1-rho^2)^(-1/2) exp(-(rho^2 (x^2+y^2) - 2 rho x y)/(2(1-rho^2))); tau = (2/pi) arcsin(rho),
  rho_S = (6/pi) arcsin(rho/2). Inversion: rho = sin(pi tau/2). No tail dependence (lambda = 0 for |rho| < 1).
- t copula: c = f_{t,nu,R}(t_nu^-1(u), t_nu^-1(v)) / prod f_{t,nu}(t_nu^-1(u_i)); bivariate joint density
  Gamma((nu+2)/2)/(Gamma(nu/2) nu pi sqrt(1-rho^2)) (1 + (x^2 - 2 rho x y + y^2)/(nu(1-rho^2)))^(-(nu+2)/2).
  Reference fit: rho = sin(pi tau/2) fixed, nu by bounded profile likelihood on (2.01, 49.9) (nu strictly in (2, 50)).
  Tail dependence (both tails): lambda = 2 t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho))).
- Clayton (th > 0): C = (u^-th + v^-th - 1)^(-1/th); c = (1+th)(uv)^(-1-th)(u^-th + v^-th - 1)^(-2-1/th);
  th = 2 tau/(1-tau); lambda_L = 2^(-1/th), lambda_U = 0.
- Gumbel (th >= 1): A = ((-ln u)^th + (-ln v)^th)^(1/th), C = exp(-A);
  c = C (uv)^-1 (ln u ln v)^(th-1) A^(1-2th) (A + th - 1); th = 1/(1-tau); lambda_U = 2 - 2^(1/th).
- Frank (th != 0): C = -(1/th) ln(1 + (e^(-th u) - 1)(e^(-th v) - 1)/(e^(-th) - 1)); tau = 1 - 4/th (1 - D1(th)),
  D1(th) = (1/th) int_0^th t/(e^t - 1) dt (scipy.integrate.quad); invert with brentq; no tail dependence.
- Negative tau: floor Clayton th at 1e-6 (LL ~ 0) and Gumbel th at 1 (independence, LL = 0) so LL/AIC/BIC stay finite
  on every row (checkers assert finiteness); or rotate (1-u) if the task asks for rotated copulas.
- Fitting by full MLE: maximise sum_i log c(u_i; theta) (minimize_scalar bounded / L-BFGS-B). AIC = 2k - 2LL,
  BIC = k ln n - 2LL (k = 1 for Gaussian/Clayton/Gumbel/Frank, 2 for t); best = lowest AIC on the SAME U.
- Sampling: Gaussian Z = N(0,I) @ chol(R)', U = Phi(Z). t: W = chi2_nu/nu, T = Z/sqrt(W), U = t_nu cdf(T).
  Clayton (Marshall-Olkin): V ~ Gamma(1/th, 1), E_j ~ Exp(1), U_j = (1 + E_j/V)^(-1/th). Gumbel: V ~ positive
  stable(1/th) (Chambers-Mallows-Stuck), U_j = exp(-(E_j/V)^(1/th)); Frank: V ~ logarithmic series(1 - e^-th),
  U_j = -(1/th) ln(1 + e^(-E_j/V)(e^-th - 1)). Map to margins with x_j = F_j^-1(U_j).
- Empirical tail dependence at q (0.95): lambda_U = #(u > q and v > q)/#(u > q), lambda_L = #(u < 1-q and v < 1-q)/
  #(u < 1-q) - conditional on the FIRST variable's exceedance count (reference); P(.)/q forms only if stated.

## Conventions and pitfalls (what graders check)
- Pseudo-observations with n+1; all u in (0,1); simulated pairs' Kendall tau within a few 1e-2 of the implied tau.
- Gaussian/t correlation from tau is rho = sin(pi tau/2), not the Pearson correlation of returns.
- Densities in log space (log1p, lgamma); copula density only (no marginal terms) unless a full likelihood is defined.
- Seeds: np.random.default_rng(seed); draw in the documented order (normals, then chi-square/gamma/exponential);
  Cholesky applied as Z @ L.T; make R positive definite (eigenvalue clipping) when empirical.
- Tail coefficients in [0,1]: Gaussian 0; t symmetric; Clayton lower only; Gumbel upper only.
- Summary fields: most common best copula (mode over pairs), mean tau over pairs, booleans as Python bool.
- Portfolio simulation: copula samples -> each asset's margin -> weighted returns; VaR/ES loss-positive.

## Tested snippet
```python
import math, numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

def pseudo_obs(X): return (stats.rankdata(X, axis=0)) / (X.shape[0] + 1)          # u in (0,1), shape (n, d)
def gauss_logpdf(u, rho):                                   # bivariate Gaussian copula density
    x, y = stats.norm.ppf(u[:, 0]), stats.norm.ppf(u[:, 1]); r2 = 1 - rho * rho
    return -0.5 * math.log(r2) - (rho * rho * (x * x + y * y) - 2 * rho * x * y) / (2 * r2)
def t_logpdf(u, rho, nu):                                   # bivariate t copula density
    x, y = stats.t.ppf(u[:, 0], nu), stats.t.ppf(u[:, 1], nu); r2 = 1 - rho * rho
    joint = (math.lgamma((nu + 2) / 2) - math.lgamma(nu / 2) - math.log(nu * math.pi) - 0.5 * math.log(r2)
             - (nu + 2) / 2 * np.log1p((x * x - 2 * rho * x * y + y * y) / (nu * r2)))
    return joint - stats.t.logpdf(x, nu) - stats.t.logpdf(y, nu)
def clayton_logpdf(u, th):                                  # th > 0; tau = th/(th+2); lambda_L = 2^(-1/th)
    a, b = u[:, 0], u[:, 1]
    return math.log1p(th) - (1 + th) * (np.log(a) + np.log(b)) - (2 + 1 / th) * np.log(a ** -th + b ** -th - 1)
def gumbel_logpdf(u, th):                                   # th >= 1; tau = 1 - 1/th; lambda_U = 2 - 2^(1/th); th = 1 -> 0
    x, y = -np.log(u[:, 0]), -np.log(u[:, 1]); A = (x ** th + y ** th) ** (1 / th)
    return -A - np.log(u[:, 0] * u[:, 1]) + (th - 1) * (np.log(x) + np.log(y)) + (1 - 2 * th) * np.log(A) + np.log(A + th - 1)
def fit_1param(logpdf, u, lo, hi):                          # MLE by bounded 1-d search; returns (param, loglik, aic)
    r = minimize_scalar(lambda p: -logpdf(u, p).sum(), bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    return float(r.x), -float(r.fun), 2 * 1 + 2 * float(r.fun)
def fit_by_tau(u, tau):                                     # tau inversion (+ profiled t nu); floors keep LL finite when tau <= 0
    rho = math.sin(math.pi * tau / 2); th_c, th_g = max(2 * tau / (1 - tau), 1e-6), max(1 / (1 - tau), 1.0)
    nu = minimize_scalar(lambda v: -t_logpdf(u, rho, v).sum(), bounds=(2.01, 49.9), method="bounded").x
    ll = dict(gaussian=gauss_logpdf(u, rho).sum(), student_t=t_logpdf(u, rho, nu).sum(), clayton=clayton_logpdf(u, th_c).sum(), gumbel=gumbel_logpdf(u, th_g).sum())
    k, n = dict(gaussian=1, student_t=2, clayton=1, gumbel=1), len(u)
    return {m: dict(ll=float(v), aic=float(2 * k[m] - 2 * v), bic=float(k[m] * math.log(n) - 2 * v)) for m, v in ll.items()}, dict(rho=rho, nu=nu, clayton=th_c, gumbel=th_g)
def emp_tail_dep(u, v, q=0.95):                             # (upper, lower), conditional on the FIRST variable's exceedances
    return float(((u > q) & (v > q)).sum() / max((u > q).sum(), 1)), float(((u < 1 - q) & (v < 1 - q)).sum() / max((u < 1 - q).sum(), 1))
def sample_t(n, rho, nu, rng):                              # Gaussian: drop the chi-square step and use norm.cdf
    L = np.linalg.cholesky([[1, rho], [rho, 1]]); z = rng.standard_normal((n, 2)) @ L.T
    return stats.t.cdf(z / np.sqrt(rng.chisquare(nu, n) / nu)[:, None], nu)
def sample_clayton(n, th, rng):                             # Marshall-Olkin: V ~ Gamma(1/th), E ~ Exp(1)
    V = rng.gamma(1 / th, 1.0, n); E = rng.exponential(1.0, (n, 2)); return (1 + E / V[:, None]) ** (-1 / th)

rng = np.random.default_rng(0); u = pseudo_obs(sample_clayton(3000, 2.0, rng)); tau = stats.kendalltau(u[:, 0], u[:, 1])[0]
fits = {"gauss": fit_1param(gauss_logpdf, u, -0.999, 0.999), "clayton": fit_1param(clayton_logpdf, u, 1e-3, 20), "gumbel": fit_1param(gumbel_logpdf, u, 1.0001, 20)}
print(round(tau, 4), {k: tuple(round(x, 3) for x in v) for k, v in fits.items()}, round(2 ** (-1 / fits["clayton"][0]), 4))
for U in (pseudo_obs(sample_t(1500, 0.7, 5, rng)), pseudo_obs(sample_t(1500, -0.05, 30, rng))):   # strong and slightly negative dependence
    tk = stats.kendalltau(U[:, 0], U[:, 1])[0]; res, par = fit_by_tau(U, tk)
    print(round(tk, 4), min(res, key=lambda m: res[m]["aic"]), {m: round(r["aic"], 2) for m, r in res.items()}, {k: round(float(v), 4) for k, v in par.items()}, emp_tail_dep(U[:, 0], U[:, 1]))
```
