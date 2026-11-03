---
title: Optimisation and root finding for calibration tasks
category: numerics
tags: scipy.optimize, minimize, l-bfgs-b, bounds, random restarts, parameter transform, logistic transform, logsumexp, log-likelihood, brentq, bracket expansion, newton, nelder-mead, noisy objective, slsqp, constraints, least squares, convergence check, reproducibility, calibration, penalty, breakeven, par spread, hedge weights
summary: Robust patterns for bounded MLE/least-squares calibration (restarts, transforms, finite objectives), scalar root finding with bracket expansion, derivative-free and constrained optimisation, and the convergence/reporting conventions graders check.
---
## When this applies
Any "calibrate parameters by maximum likelihood / least squares", "find the root / critical price / implied
parameter", "minimum-variance or target-return portfolio with constraints", "fit a model to an option chain",
objective functions that are noisy (Monte Carlo inside) or have bounds.

## Method and formulas
- Bounded smooth problems: scipy.optimize.minimize(fun, x0, method="L-BFGS-B", bounds=[(lo, hi), ...],
  options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8}). Objective must return a finite Python float; map invalid
  regions to a large penalty (1e10) only as a guard - prefer bounds. Minimise the NEGATIVE log-likelihood.
- Restarts: a moment-based x0 plus n >= 5 starts from rng = np.random.default_rng(seed) uniform within the bounds; clip
  every start into [lo, hi] (L-BFGS-B raises on x0 outside); keep the result with the smallest finite fun; record
  n_restarts, converged = bool(res.success), and whether any parameter sits at a bound (np.isclose(x, lo|hi)).
- Likelihoods: build log-densities with scipy.stats.<dist>.logpdf (never log(pdf)); mixtures via scipy.special.logsumexp
  over components; factorials via gammaln; sum, then negate. Scale parameters to O(1) (rates in decimals, vol-of-vol
  not 1e-6) so FD gradients (eps 1e-8) are accurate, or pass options={"eps": 1e-6}.
- Transforms for unconstrained solvers (BFGS, Nelder-Mead): positive x = exp(y); bounded x = lo + (hi - lo)/(1 + exp(-y))
  (inverse y = ln((x - lo)/(hi - x))); correlation rho = tanh(y). Optimise over y, report x. Jacobian adjustments are
  irrelevant for point estimates.
- Scalar roots: brentq(f, a, b, xtol=1e-12, maxiter=500) needs f(a) f(b) < 0; expand the bracket geometrically
  (hi *= 2, up to ~60 times) before giving up; root_scalar(method="newton", fprime=...) when a derivative is cheap, but
  guard with a bracket fallback; return NaN (not an exception) when no root exists in the admissible domain.
- Nelder-Mead for noisy/non-smooth objectives: options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000, "adaptive": True};
  fix the seed INSIDE the objective (common random numbers) so repeated evaluations are deterministic, otherwise the
  simplex chases noise. No native bounds - use transforms or clip + penalty.
- Constrained problems: SLSQP with constraints=[{"type": "eq", "fun": g}, {"type": "ineq", "fun": h}] (ineq means h >= 0),
  bounds for box limits, scale the objective to O(1) (variances of 1e-4 -> multiply by 1e4), options={"ftol": 1e-12,
  "maxiter": 500}; verify constraints after solving (|sum w - 1| < 1e-8) and that res.success is True. For analytic checks:
  unconstrained minimum variance w = Sigma^{-1} 1 / (1' Sigma^{-1} 1).
- Least squares calibration: scipy.optimize.least_squares(resid, x0, bounds=(lo, hi), method="trf", x_scale="jac",
  xtol=ftol=gtol=1e-12); residuals in price or vol units as the task says (vol residuals weight strikes evenly).
- Convergence checks to report: res.success, res.message, res.nit, |grad| (res.jac), spread of fun across restarts,
  parameter stability across restarts (max relative difference), and a re-run from the best point with tighter tolerance.
- Hessian/standard errors when asked: numerical Hessian of the NLL at the optimum (statsmodels.tools.numdiff.approx_hess)
  -> cov = inv(H); L-BFGS-B's res.hess_inv is an operator (res.hess_inv.todense()) and is only approximate.
- Linear-in-parameter roots need no solver: a par spread/par basis where PV is linear in the spread is
  s* = s0 - PV(s0)/((PV(s0 + h) - PV(s0))/h) exactly (one secant step, h = 1bp).
- Breakeven parameters (breakeven vol of a note, implied spread): brentq on price(param) - target over a bracket where the
  sign changes (check first; barrier prices are not monotone in vol, so scan a coarse grid for the bracket).
- Sequential fits with cross-slice constraints (SVI calendar no-arbitrage): add a smooth penalty, e.g.
  1e3 * mean(max(w_prev(k) - w(k), 0)^2) on a strike grid, rather than hard constraints; 3 deterministic starts clipped
  into the bounds; keep the best fun.
- Multi-term objectives (hedge = KRD mismatch + factor mismatch + stress errors): scale terms to O(1) (bps/100, x10 on the
  term whose tolerance is tightest); the weights decide which grader tolerance binds - check each target after solving.

## Conventions and pitfalls (what graders check)
- Determinism: one seeded generator, starts in a fixed order, no dict-order dependence; identical results on re-run.
- Feasibility: never return parameters outside the bounds (np.clip the final x); check the Feller or other model
  conditions as bool flags, do not enforce unless told.
- Objective pathologies: NaN from log(0) or sqrt(negative) stalls L-BFGS-B silently (fun stays at x0) -> guard with
  np.errstate and finite checks; flat penalties create false convergence - prefer transforms.
- Numeric hygiene: cast with float()/int()/bool() before JSON; a negative LL (-res.fun) is the likelihood value to report;
  write log_likelihood, not the negative.
- Root finding on prices: use the no-arbitrage bounds to decide "no solution" (NaN + converged False) instead of hitting
  the bracket edge.
- Tolerances: brentq default xtol 2e-12 is fine; checkers compare roots to 1e-8..1e-6 and reprices to 1e-6 bp.
- Portfolio weights: sum to 1 within 1e-8, non-negative if long-only (clip -1e-12 noise to 0), report in the asset order
  of the input file.

## Tested snippet
```python
import math
import numpy as np
from scipy.optimize import minimize, brentq

def minimize_restarts(fun, bounds, x0=None, n_restarts=5, seed=42, method="L-BFGS-B", **opts):
    """Bounded minimisation with seeded random restarts; keeps the best finite result. Objective must return a finite
    float (return a large penalty like 1e10 for invalid parameters). x0 and random starts are clipped into the bounds."""
    rng = np.random.default_rng(seed); lo, hi = (np.array([b[i] for b in bounds], float) for i in (0, 1))
    starts = ([np.clip(np.asarray(x0, float), lo, hi)] if x0 is not None else []) + [rng.uniform(lo, hi) for _ in range(n_restarts)]
    best = None
    for s in starts:
        res = minimize(fun, s, method=method, bounds=bounds, options=opts)
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun): best = res
    best.x = np.clip(best.x, lo, hi); best.at_bound = bool(np.any(np.isclose(best.x, lo)) or np.any(np.isclose(best.x, hi)))
    return best

def to_bounded(y, lo, hi):   return lo + (hi - lo) / (1.0 + np.exp(-y))    # logistic transform for unconstrained solvers
def from_bounded(x, lo, hi): return np.log((x - lo) / (hi - x))              # inverse (x strictly inside (lo, hi))

def brentq_expand(f, lo, hi, grow=2.0, max_expand=60, **kw):
    """Root of f with bracket expansion: moves hi up (or lo down toward 0 for positive domains) until the sign changes."""
    flo, fhi = f(lo), f(hi)
    for _ in range(max_expand):
        if flo * fhi < 0: return brentq(f, lo, hi, xtol=kw.get("xtol", 1e-12), maxiter=kw.get("maxiter", 500))
        hi *= grow; fhi = f(hi)
    raise ValueError("no sign change found")

def min_variance_weights(Sigma, mu=None, target=None, long_only=True):
    """SLSQP: min w'Sigma w s.t. sum w = 1 (and mu'w = target if given), 0 <= w <= 1 if long_only."""
    n = len(Sigma); cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target is not None: cons.append({"type": "eq", "fun": lambda w: float(mu @ w) - target})
    res = minimize(lambda w: float(w @ Sigma @ w) * 1e4, np.full(n, 1.0 / n), method="SLSQP",      # scale objective to O(1)
                   bounds=[(0.0, 1.0)] * n if long_only else None, constraints=cons, options={"ftol": 1e-12, "maxiter": 500})
    assert res.success, res.message
    return res.x
```
