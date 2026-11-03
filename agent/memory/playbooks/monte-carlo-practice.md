---
title: Monte Carlo practice: seeding, variance reduction, Greeks, path simulation
category: numerics
tags: monte carlo, simulation, seed, random number generator, antithetic variates, control variate, common random numbers, finite difference greeks, bump size, pathwise derivative, likelihood ratio, standard error, confidence interval, path dependent, daily steps, gbm paths, batching, memory, convergence, draw order, legacy seed, compound poisson, aggregate loss, relative bump, sub-daily steps, nested horizons, empirical quantile
summary: The MC conventions checkers rely on (seed API and draw order, se = std(ddof=1)/sqrt(n), discounting), variance reduction, CRN bump-and-revalue with relative bumps, compound-Poisson sums, empirical VaR/ES, and pathwise/LR Greeks for GBM.
---
## When this applies
Any task that says "simulate N paths with seed 42", "Monte Carlo price with standard error", "MC Greeks by
finite differences / pathwise / likelihood ratio", "antithetic or control variates", "convergence table",
path-dependent payoffs (Asian, barrier, lookback) with daily monitoring, hedging P&L simulations, "sensitivity to sigma /
kappa by bump-and-revalue", compound-Poisson (frequency x severity) loss simulation with VaR/ES.

## Method and formulas
- Terminal GBM: S_T = S0 exp((r - q - sigma^2/2) T + sigma sqrt(T) Z). Path: n_steps = round(252 T) for daily, dt = T/n_steps,
  logS = ln S0 + cumsum((r - q - sigma^2/2) dt + sigma sqrt(dt) Z, axis=1) - exact for GBM at any dt (no Euler bias).
  Prepend S0 only if the payoff includes t = 0 (ask the task). Discount with e^{-rT}; se = std(payoff, ddof=1)/sqrt(N);
  95% CI = +-1.96 se. Error shrinks as 1/sqrt(N) (convergence tables: nested prefixes of one draw).
- Seeding: `np.random.seed(42); np.random.standard_normal(...)` (legacy) and `np.random.default_rng(42).standard_normal`
  give DIFFERENT numbers. Use the API the task names; draw in a fixed order and shape (n_paths, n_steps) once; derive
  everything (all strikes, all maturities, all bumps) from the same matrix when the task implies one simulation.
  Fresh draws per scheme when the reference "simulates each strategy" (a shared draw changes pooled statistics).
  Draw ORDER is part of the answer: references that reproduce to 7 digits drew e.g. all Poisson counts for every path
  first, then one vector of sum(counts) severities (legacy np.random.seed once, severities looped in the task's order);
  per-time-step jump models drew diffusion Z, then jump counts/uniforms, then jump sizes. Nested horizons are read from
  ONE run (record at step round(tau/dt)), not from separate simulations.
- Antithetic: Z and -Z; fold estimators to pair averages before computing mean and se (n_pairs, ddof=1); never report
  the se of the 2N unfolded draws (understated by sqrt(2) and correlated). Do not add antithetic if the task defines
  se = std/sqrt(n) on plain paths - it changes the expected se semantics.
- Control variate with known mean m_X: Y_cv = Y - beta (X - m_X), beta = cov(Y, X)/var(X) (same sample is fine);
  variance reduction ratio var(Y)/var(Y_cv). Good controls: S_T (E = S0 e^{(r-q)T}), geometric Asian (closed form),
  European price for Asian/American payoffs.
- Finite-difference Greeks with common random numbers (same Z for all bumps): delta = (V(S+h) - V(S-h))/(2h),
  gamma = (V(S+h) - 2V(S) + V(S-h))/h^2, vega with h = 0.01, rho h = 0.001, theta = -(V(T-h) - V(T))/h with h = 1/252
  (= +dV/dT; flip sign for dV/dt). Typical h_S = 0.5 or 1% of S. Keep per-path FD vectors so the Greek has an se.
  Without CRN, gamma at h = 0.5 is pure noise even at 1e6 paths.
- Model-parameter Greeks (dP/dsigma, dP/dkappa, level shifts) by bump-and-revalue: draw the Z matrix ONCE and reuse it for
  base and every bump. A "1% bump" is RELATIVE (p (1 +- 0.01)) but the Greek is per unit: (P_up - P_dn)/(2 * 0.01 * p);
  dividing by 2 * 0.01 instead multiplies it by p (a classic oracle miss). Additive shifts (e.g. +-0.1 degree): /(2h).
- Compound sums per path without loops: S = np.bincount(np.repeat(np.arange(n), counts), weights=sev, minlength=n)
  (empty paths -> 0); np.add.reduceat gives the same sums but returns sev[start] for empty segments (zero them) and fails
  when trailing paths are empty.
- Empirical risk measures: VaR = sorted[ceil(alpha n) - 1] (lower order statistic) unless the task says np.quantile/
  percentile (linear interpolation); ES = mean of losses strictly above VaR (or >= - as defined).
- Sub-daily steps with daily monitoring: n_days = ceil(T * 365) (or 252), m steps per day, dt = T/(n_days m); test the
  path condition only when (step + 1) % m == 0. se of a discounted price = DF * std(payoff, ddof=1)/sqrt(n).
- Pathwise (Z = (ln(S_T/S0) - (r-q-sigma^2/2)T)/(sigma sqrt T), 1 = indicator ITM, df = e^{-rT}): call delta = df 1 S_T/S0;
  put delta = -df 1{S_T<K} S_T/S0; vega = df 1 S_T (sqrt(T) Z - sigma T) (sign flipped for puts); rho_call = df T K 1;
  theta = -r V + df 1 S_T ((r - q - sigma^2/2) + sigma Z/(2 sqrt T)) (dV/dT). Pathwise gamma does not exist for kinked
  payoffs (report null); pathwise fails for digitals.
- Likelihood ratio (weights multiply the discounted payoff): delta Z/(S0 sigma sqrt T);
  gamma (Z^2 - 1 - Z sigma sqrt T)/(S0 sigma sqrt T)^2; vega (Z^2 - 1)/sigma - Z sqrt T; rho Z sqrt T/sigma - T.
  Multi-step paths: delta/gamma use the first-step score Z_1/(S0 sigma sqrt(dt)); vega/rho sum the per-step scores.
  LR variance > pathwise variance (checkers assert pw_se < lr_se).
- Asian pathwise: delta = df 1{A>K} A/S0, vega = df 1{A>K} mean_i S_i (W_i - sigma t_i); Asian delta < European delta.
- Memory: an (N x n_steps) float64 matrix costs 8 N n_steps bytes (1e6 x 252 = 2 GB). Batch over paths (50k-100k) and
  accumulate sums/sums of squares, or step through time keeping only the running state (S, running max/min, running sum).

## Conventions and pitfalls (what graders check)
- MC vs closed form: |MC - BS| < 3 se (typically also < 0.5 absolute); Asian < European; put-call parity on MC prices
  holds to ~se; Greeks: delta_C - delta_P = e^{-qT}, rho_C - rho_P = K T e^{-rT}, same gamma/vega.
- Unavailable estimators -> JSON null (None), never NaN; convergence rows named exactly as the checker filters them.
- Do not round MC outputs; se >= 0; count paths as int; store the seed used.
- Barrier/lookback with daily monitoring are biased relative to continuous formulas (knock-outs too high, lookbacks
  too low); say so, and apply the 0.5826 sigma sqrt(dt) shift only when asked.
- Runtime: vectorise across paths; python loops only over time steps (<= a few thousand iterations).

## Tested snippet
```python
import math
import numpy as np

def mc_call_greeks(S0, K, T, r, q, sig, n_paths=200_000, seed=42, h=0.5, batch=50_000):
    """European call by MC: antithetic pairs (estimators folded to pair averages before mean/se), control variate
    (S_T with E[S_T] = F), common-random-number FD delta/gamma, pathwise delta/vega, likelihood-ratio delta/gamma.
    Batched to bound memory. se = std(ddof=1)/sqrt(n_pairs)."""
    rng = np.random.default_rng(seed); df, sqT, F = math.exp(-r * T), math.sqrt(T), S0 * math.exp((r - q) * T)
    rows = []
    for start in range(0, n_paths, batch):
        m = min(batch, n_paths - start) // 2; Z = rng.standard_normal(m); Z = np.concatenate([Z, -Z])      # antithetic
        ST = lambda S, v: S * np.exp((r - q - 0.5 * v * v) * T + v * sqT * Z)                           # same Z = CRN
        S_T = ST(S0, sig); pay = df * np.maximum(S_T - K, 0.0); itm = (S_T > K).astype(float)
        up, dn = df * np.maximum(ST(S0 + h, sig) - K, 0.0), df * np.maximum(ST(S0 - h, sig) - K, 0.0)
        M = np.column_stack([pay, S_T, (up - dn) / (2 * h), (up - 2 * pay + dn) / h ** 2,              # price, control, FD
                             df * itm * S_T / S0, df * itm * S_T * (sqT * Z - sig * T),                 # pathwise delta, vega
                             pay * Z / (S0 * sig * sqT), pay * (Z * Z - 1 - Z * sig * sqT) / (S0 * sig * sqT) ** 2])  # LR
        rows.append(0.5 * (M[:m] + M[m:]))                                                              # pair averages
    X = np.concatenate(rows)
    beta = np.cov(X[:, 0], X[:, 1])[0, 1] / X[:, 1].var(ddof=1)                                         # control-variate coefficient
    cv = X[:, 0] - beta * (X[:, 1] - F)
    est = lambda v: {"value": float(v.mean()), "se": float(v.std(ddof=1) / math.sqrt(len(v)))}
    return {"price": est(X[:, 0]), "price_cv": est(cv), "delta_fd": est(X[:, 2]), "gamma_fd": est(X[:, 3]),
            "delta_pw": est(X[:, 4]), "vega_pw": est(X[:, 5]), "delta_lr": est(X[:, 6]), "gamma_lr": est(X[:, 7])}

def compound_poisson_mc(lam, sample_sev, n_sim, seed=42):   # legacy-seed order: all counts, then all severities
    np.random.seed(seed); counts = np.random.poisson(lam, n_sim); sev = sample_sev(int(counts.sum()))
    return np.bincount(np.repeat(np.arange(n_sim), counts), weights=sev, minlength=n_sim)   # empty paths -> 0

def mc_var_es(losses, alpha):              # VaR = sorted[ceil(alpha n) - 1]; ES = mean of losses strictly above VaR
    s = np.sort(losses); v = s[math.ceil(alpha * len(s)) - 1]; return float(v), float(s[s > v].mean())

def crn_param_greek(price_fn, p, rel=0.01):  # price_fn(param) must reuse ONE fixed Z matrix; relative bump, per-unit Greek
    return (price_fn(p * (1 + rel)) - price_fn(p * (1 - rel))) / (2 * rel * p)
```
