---
title: Monte Carlo for barrier and path-dependent options: discretisation bias, Brownian bridge, continuity correction, variance reduction, path counts
category: derivatives-pricing
tags: barrier option, down-and-out, down-and-in, up-and-out, up-and-in, knock-in, knock-out, monte carlo, path dependent, discrete monitoring, continuous monitoring, monitoring dates, broadie glasserman kou, continuity correction, 0.5826, brownian bridge, hitting probability, crossing probability, antithetic variates, control variate, standard error, confidence interval, number of paths, time steps, euler scheme, exact gbm, log-euler, in-out parity, rebate, asian option, lookback, running maximum, seed, default_rng, vectorised, chunking, memory, tolerance
summary: Exact GBM path simulation, the difference between discretely and continuously monitored barrier prices, the Brownian-bridge survival weight exp(-2 ln(S_k/B) ln(S_{k+1}/B)/(sigma^2 dt)) and the BGK 0.5826 sigma sqrt(dt) barrier shift, antithetic/control-variate SEs, path-count sizing and the parity/bound invariants graders test.
---
## When this applies
"Price a down-and-out / up-and-in option by Monte Carlo with N paths and M steps", "antithetic variates", "compare the MC
price with the analytic barrier formula", "report the standard error / confidence interval", "discrete vs continuous
monitoring", Asian and lookback options by simulation, "knock-in + knock-out = vanilla" checks.

## Method and formulas
Symbols: S0 spot, K strike, B barrier, T years, M steps (dt = T/M), N paths, r, q, sigma annual, Z ~ N(0, 1).
- Exact GBM step (no discretisation error in the marginal law): S_{k+1} = S_k exp((r - q - sigma^2/2) dt + sigma sqrt(dt) Z).
  Draw with the API the task names: np.random.default_rng(seed).standard_normal((N, M)) vs np.random.seed(seed);
  np.random.normal - different streams; draw all steps per path in the order stated (paths x steps matrix).
- Discrete monitoring: knocked out when min_k S_k <= B at the M monitoring dates (strict vs non-strict inequality and
  inclusion of t = 0 / t = T as the task says). The discrete-monitoring price converges, as N -> inf, to the DISCRETE
  option value, which for knock-outs is ABOVE the continuous closed form (fewer chances to hit) by O(sqrt(dt)).
- Continuity correction (Broadie-Glasserman-Kou): price a discretely monitored barrier with the continuous formula using
  B_shifted = B exp(+/- 0.5826 sigma sqrt(dt)) (+ for up barriers, - for down); conversely compare a discrete MC with the
  continuous formula only after shifting, or expect a gap ~ a few percent for daily monitoring.
- Brownian bridge (continuous monitoring from discrete paths, unbiased): given S_k, S_{k+1} both above a down barrier,
  P(hit in between) = exp(-2 ln(S_k/B) ln(S_{k+1}/B)/(sigma^2 dt)); survival weight of a path = prod_k (1 - p_k) (zero
  if any monitored value is below B); price = e^{-rT} mean(payoff x weight). Sampling variant: knocked out when U_k < p_k.
  Up barrier: same with ln(B/S). Lookback/maximum corrections use the same shift logic (beta = 0.5826).
- In-out parity: knock-in = vanilla - knock-out (same barrier, no rebate) - a check and a control variate.
  Control variate: price_cv = price_MC - beta (vanilla_MC - vanilla_BS) with beta = Cov/Var from the same paths (or 1).
- Antithetic variates: pair Z and -Z; the estimator is the mean over pair averages; SE from the N/2 pair averages
  (not from the 2N correlated payoffs). Standard error SE = std(discounted payoffs, ddof=1)/sqrt(N); 95% CI = +/- 1.96 SE.
- Sizing: to reach a tolerance tol on the price, N >= (std_payoff/tol)^2; typical payoff std ~ 10-15 for S0 = 100 ->
  1e6 paths for tol = 0.01. Simulate in chunks (e.g. 50k paths x M steps per chunk, running sums of payoff and payoff^2)
  so memory stays below ~1 GB; vectorise over paths, loop over steps or chunks; never cut M below the monitoring count.
- Analytic references (continuous): down-and-out call with B <= K: lambda = (r - q + sigma^2/2)/sigma^2, y = ln(B^2/(S0 K))/
  (sigma sqrt(T)) + lambda sigma sqrt(T); C_di = S0 e^{-qT} (B/S0)^{2 lambda} N(y) - K e^{-rT} (B/S0)^{2 lambda - 2} N(y - sigma sqrt(T));
  C_do = C_BS - C_di. Other cases: Reiner-Rubinstein (see the barrier/reflection card). Geometric Asian closed form
  serves as the control variate for arithmetic Asians.
- Rebates: paid at hit (discount from the hit step time) or at expiry - as the task says; a barrier already breached at
  t = 0 -> out option worth the rebate/0, in option worth the vanilla.
- Greeks by MC: bump-and-revalue with the SAME seed (common random numbers), relative bumps (1% of S0), central differences.

## Conventions and pitfalls (what graders check)
- Invariants asserted: 0 <= knock-out <= vanilla, knock-in + knock-out ~ vanilla within 3 SE, discrete knock-out >=
  continuous knock-out (for the same inputs), MC within ~3 SE of the analytic reference, same seed -> identical outputs,
  SE > 0 and decreasing ~ 1/sqrt(N), antithetic SE <= plain SE, CI contains the analytic value, n_paths/n_steps written
  as ints equal to the task's values.
- Monitoring convention changes the price by percent-level amounts: read "daily monitoring (252 steps)", "monitored at
  the step dates", "continuous barrier" carefully and state which one the output represents.
- Time steps: dt = T/M with M = the number of monitoring dates (e.g. 252 x T for daily); do not add an extra step.
- Discounting once: payoff x e^{-rT}; the SE is on the DISCOUNTED payoffs.
- Memory: N x M float64 matrices (1e6 x 252 = 2 GB) crash the sandbox - chunk. Time: keep the whole run under the
  task's limit; 1e5 x 252 exact GBM vectorised takes ~1-2 s.
- Deterministic draws: no Python-level randomness (random module) mixed in; one rng instance per task, seeded once,
  draws in the documented order (e.g. all Z for the vanilla before the barrier if the task says so).
- Reporting: price, std_error, ci_low/ci_high, n_paths, n_steps, plus analytic price and abs/rel difference when asked;
  floats cast with float(), booleans like `within_ci` recomputed from the written numbers.

## Tested snippet
```python
import numpy as np
from scipy.stats import norm
def bs_call(S, K, T, r, q, sig):
    d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * T) / (sig * np.sqrt(T)); d2 = d1 - sig * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
def dao_call_analytic(S, K, B, T, r, q, sig):              # continuous down-and-out call, B <= K (Reiner-Rubinstein / Hull)
    lam = (r - q + 0.5 * sig**2) / sig**2; y = np.log(B**2 / (S * K)) / (sig * np.sqrt(T)) + lam * sig * np.sqrt(T)
    cdi = S * np.exp(-q * T) * (B / S) ** (2 * lam) * norm.cdf(y) - K * np.exp(-r * T) * (B / S) ** (2 * lam - 2) * norm.cdf(y - sig * np.sqrt(T))
    return bs_call(S, K, T, r, q, sig) - cdi
def dao_call_mc(S, K, B, T, r, q, sig, n_paths=60_000, n_steps=252, seed=42, bridge=True, antithetic=True):
    rng = np.random.default_rng(seed); dt = T / n_steps; half = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal((half, n_steps)); Z = np.vstack([Z, -Z]) if antithetic else Z
    path = np.exp(np.log(S) + np.cumsum((r - q - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * Z, axis=1))   # exact GBM
    prev = np.hstack([np.full((len(path), 1), float(S)), path[:, :-1]])
    surv = np.all(path > B, axis=1).astype(float)                                        # discrete monitoring at the step dates
    if bridge:                                                                           # P(no hit inside each step | endpoints above B)
        p_hit = np.exp(-2 * np.log(prev / B) * np.log(path / B) / (sig**2 * dt))
        surv *= np.prod(np.where((prev > B) & (path > B), 1 - p_hit, 0.0), axis=1)        # continuous-monitoring weight
    payoff = np.exp(-r * T) * np.maximum(path[:, -1] - K, 0) * surv
    if antithetic: payoff = 0.5 * (payoff[:half] + payoff[half:])                        # pair averages for the SE
    return float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(len(payoff)))
S, K, B, T, r, q, sig = 100, 100, 85, 1.0, 0.05, 0.0, 0.25
ana = dao_call_analytic(S, K, B, T, r, q, sig)
disc, se_d = dao_call_mc(S, K, B, T, r, q, sig, bridge=False); cont, se_c = dao_call_mc(S, K, B, T, r, q, sig)
B_shift = B * np.exp(-0.5826 * sig * np.sqrt(T / 252))                                   # BGK: continuous formula for a DISCRETE barrier
print(round(ana, 4), "analytic continuous |", round(disc, 4), "+-", round(se_d, 4), "discrete-monitoring MC |", round(cont, 4), "+-", round(se_c, 4), "bridge MC")
print(round(dao_call_analytic(S, K, B_shift, T, r, q, sig), 4), "= BGK-corrected analytic ~ discrete MC;", abs(cont - ana) < 3 * se_c, disc >= cont, cont <= bs_call(S, K, T, r, q, sig))
```
