---
title: Monte Carlo reproducibility and tolerance: seeds and generator calls, path counts, draw order, antithetic/control variates, standard errors, when a simulation "agrees" with a closed form, bootstrap conventions (stationary / block / iid)
category: numerics
tags: monte carlo, simulation, seed, default_rng, random seed, np.random.seed, generator, standard_normal, normal, draw order, paths, steps, antithetic, control variate, variance reduction, standard error, confidence interval, tolerance, agreement, closed form, analytical, convergence, bootstrap, stationary bootstrap, block bootstrap, iid bootstrap, block length, filtered historical simulation, fhs, resampling, reproducible
summary: Simulation results are pinned by graders only when the generator, seed, draw order and sizes match the task; the conventions: np.random.default_rng(seed) unless np.random.seed is named, draws made as one (paths x steps) standard-normal block in the stated order, antithetic pairs built as (Z, -Z) halving the independent draws, the standard error reported as sd(payoffs)/sqrt(N) of the DISCOUNTED payoffs; "agrees with the analytic value" means |MC - exact| <= 3 SE (or the task's tolerance), which requires enough paths - never shrink N to save time, vectorise; bootstrap variants (stationary with geometric block length mean 1/p, circular block, iid) give different numbers, so implement exactly the one named.
triggers: text:monte carlo, text:bootstrap
---
## When this applies
Any task with "simulate", "Monte Carlo", "paths", "seed", "bootstrap", "resample", "the simulated value should be within x
of the analytical value", "report the standard error / 95 % confidence interval".

## Generator and draw order
- Use np.random.default_rng(seed) and its methods (standard_normal, normal, uniform, choice, integers) unless the task
  names np.random.seed / np.random.normal or another library; the two streams differ, so the choice decides whether a
  pinned number matches.
- Draw the whole block at once in the order the task implies: Z = rng.standard_normal((N_paths, N_steps)) (paths as
  rows) unless the text says "for each time step draw N normals".  Drawing per path in a loop changes the stream.
- Antithetic variates: generate N/2 independent draws and use (Z, -Z); the estimate averages the pair means; the standard
  error uses the N/2 pair averages, not N payoffs.
- Control variates: estimate = mean(Y) - b (mean(X) - E[X]) with b = cov(X, Y) / var(X) estimated on the same sample
  (or the pilot sample if the task says).
- Multiple quantities from one simulation (price, delta, several strikes) must reuse the SAME paths - resimulating per
  quantity breaks the pinned values and inflates differences.
- Exact GBM step: S_{t+dt} = S_t exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z); Euler for other SDEs only when asked;
  for the Heston / CIR variance use the full-truncation scheme (max(v, 0)) unless the task names another.

## Sizes and tolerance
- Use exactly the paths/steps/repetitions the task states; if it gives a range, use the upper value.  Cutting N to fit
  the time limit is the most common cause of "MC not within tolerance" - vectorise or chunk the paths instead.
- Standard error SE = sd(discounted payoffs, ddof = 1) / sqrt(N); 95 % CI = estimate +/- 1.96 SE.
- "Agrees with the closed form" = |MC - exact| within the task's tolerance (e.g. 1 % relative or 3 SE).  Check the
  closed form independently (Black-Scholes, barrier formulas, first-passage probability P(hit) = exp(-2 a b / (sigma^2 T))
  style results) - if the gap exceeds 3 SE with N >= 1e5 the closed form or the payoff is wrong, not the noise.
- Report MC quantities with their SE; do not round before comparing.

## Bootstrap conventions (historical / filtered historical simulation)
- iid bootstrap: rng.integers(0, n, size=(B, h)) indices into the standardised residuals.
- Stationary bootstrap (Politis-Romano): block lengths geometric with mean 1/p (p = 1/21 means mean block 21), start
  indices uniform, wrap around; circular block bootstrap: fixed block length b, wrap around.  The task names the
  variant and p/b - the numbers differ by several percent between variants.
- Filtered historical simulation: standardise returns by the fitted (GARCH/DCC) volatility, resample the standardised
  residuals, rescale by the forecast volatility for each horizon step; VaR = -quantile(alpha) of the simulated P&L,
  ES = -mean of losses beyond VaR (sign convention as the task states).
- Seed the bootstrap with the task's seed and generator exactly as above; the number of bootstrap replications B is
  not negotiable.

## Invariants worth printing
- MC mean within 3 SE of any analytic benchmark the task provides; SE decreases ~ 1/sqrt(N) when N is doubled;
  antithetic estimate uses N/2 pairs; the same seed reproduces the same numbers on a rerun (compute twice if cheap).
