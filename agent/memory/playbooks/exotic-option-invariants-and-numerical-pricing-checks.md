---
title: Exotic option invariants and numerical-pricing checks: barrier in/out parity, lookback and Asian bounds, compound-option parity, American early-exercise boundary, finite-difference grid sanity, Greeks by finite difference vs pathwise, monotonicity in strike/maturity/volatility
category: derivatives-pricing
tags: barrier, knock in, knock out, in out parity, lookback, floating strike, fixed strike, asian, geometric, arithmetic, levy, curran, compound option, geske, call on call, put call parity, american option, early exercise boundary, critical price, barone adesi whaley, binomial, crr, finite difference, crank nicolson, grid, boundary conditions, interpolation, greeks, delta, gamma, vega, pathwise, likelihood ratio, bumping, monotonic, convex, positivity, sanity checks
summary: The relations graders test on exotic-option tasks and how to make a pricer satisfy them: knock-in + knock-out = vanilla; fixed-strike lookback calls decrease in strike and floating-strike lookbacks increase in maturity; geometric Asian <= arithmetic Asian; compound-option put-call parity on the underlying option; the American put early-exercise boundary S*(t) < K, increasing towards K at expiry, with S*(T) = K (no dividend) and the put price at S0 read from the grid by interpolation equal to the reported price; finite-difference Greeks must agree with pathwise/analytic Greeks within a few percent when paths and bumps are sized correctly.
triggers: text:barrier, text:lookback, text:asian option, text:compound option, text:early exercise, text:american put
---
## When this applies
"Price barrier / lookback / Asian / compound options", "early-exercise boundary", "compare the FD / binomial / BAW
prices", "report Greeks by finite difference and pathwise", "check put-call parity / in-out parity", "prices must be
monotonic in K and T".

## Relations a correct implementation satisfies (print them as INVARIANT lines)
- Barrier: down-and-in + down-and-out = vanilla (same K, T, rebate 0); knock-out <= vanilla; prices decrease as the
  barrier moves towards the spot for knock-outs.  Discretely monitored knock-out >= continuously monitored.
- Lookback: fixed-strike lookback call price is non-increasing in K; floating-strike lookback put/call non-decreasing in T
  and in sigma; a lookback is worth at least the corresponding vanilla.  Floating-strike lookback call payoff
  S_T - min S; fixed-strike lookback call payoff max(max S - K, 0) - do not swap them.
- Asian: geometric-average price <= arithmetic-average price (Jensen) for calls and the reverse ordering does not hold
  for puts in general - check the task's statement; Levy / Curran / Turnbull-Wakeman approximations must agree with MC
  within the MC standard error for the stated paths; all prices positive.
- Compound (Geske): call-on-call C_cc + K_1 e^{-r T_1} = put-on-call P_cc + C(S, T_2) for the SAME underlying option
  valued at T_1 compound maturity (parity of the compound options on the underlying call), prices positive and below the
  underlying option.  A parity error of several units means the critical price S* (where the underlying call equals K_1
  at T_1) was not solved by root finding.
- American options: P_am >= P_eu, P_am >= max(K - S, 0); early-exercise boundary S*(t) below K, increasing to K at
  expiry (no dividends); for calls on a non-dividend stock the boundary is infinite (never exercise).  Critical prices
  from Barone-Adesi-Whaley: S*_call increases with T, S*_put decreases with T; BAW vs CRR binomial (>= 500 steps) within
  about 1 %.
- Put-call parity (European): C - P = S e^{-qT} - K e^{-rT} within 1e-6 of a numerical pricer's accuracy.

## Finite-difference pricer sanity
- Grid: S from 0 to S_max ~ 4 K (or K e^{4 sigma sqrt(T)}), M spatial nodes (>= 200), N time steps (>= 200 or
  dt <= dS^2 / (sigma^2 S_max^2) for explicit schemes); Crank-Nicolson with Rannacher start avoids oscillations.
- Boundary conditions: put V(0, t) = K e^{-r (T - t)} (or K for American), V(S_max, t) = 0; call V(0, t) = 0,
  V(S_max, t) = S_max e^{-q (T-t)} - K e^{-r (T-t)}.
- The reported price at S0 is the grid value INTERPOLATED at S0 at t = 0 (np.interp over the S grid); the grid file
  written to disk must contain the same S axis the price was read from (first column S, one column per time), so that
  "grid at S0, t = 0 equals option_values" holds.
- Early-exercise boundary per time step: the largest S where V equals the intrinsic value (within a tolerance); must be
  < K and > 0 - if you get S* = S_max the comparison sign is reversed.
- Convergence table: price at (M, N), (2M, 2N), (4M, 4N); differences should shrink by ~ 4x for a second-order scheme.

## Greeks
- Finite difference: delta = (V(S + h) - V(S - h)) / 2h with h ~ 1 % of S, gamma = (V(S + h) - 2 V(S) + V(S - h)) / h^2,
  vega = (V(sigma + 0.01) - V(sigma - 0.01)) / 0.02 - reported PER UNIT of volatility unless the task says "per 1 %".
  Use common random numbers across the bumped runs or the FD Greek is pure noise.
- Pathwise: delta = e^{-rT} E[1{S_T > K} S_T / S0]; vega = e^{-rT} E[1{S_T > K} S_T (ln(S_T/S0) - (r + sigma^2/2) T) /
  sigma]; likelihood-ratio for digital payoffs.  FD and pathwise Greeks agree within 2-3 % only with >= 1e5 paths and the
  same seed; a 2x gap means one of them uses the wrong units (per 1 % vs per unit) or forgot the discount factor.

## Invariants worth printing
- In-out parity residual; monotonicity in K and T (np.all(np.diff(...) <= 0)); geometric <= arithmetic; positivity;
  S*(t) < K and increasing; grid-interpolated price equals the reported price; BAW vs binomial within 1 %;
  FD Greek vs pathwise Greek within 3 %.
