---
title: FFT / characteristic-function methods: grid spacing, truncation, damping, aliasing, compound-Poisson aggregate loss, Carr-Madan option pricing, how to validate against Monte Carlo or closed forms
category: numerics
tags: fft, fast fourier transform, characteristic function, compound poisson, aggregate loss, panjer, var, expected shortfall, es, cvar, tail, quantile, grid, step size, truncation, aliasing, damping, carr madan, lewis, cos method, lognormal severity, gamma severity, pareto, discretisation, interpolation, relative error, monte carlo check
summary: How to size an FFT grid (N a power of two, step h, span N h covering the mass with the tail below 1e-9), how to discretise a continuous severity on the grid without losing the mean (midpoint / rounding method), how to read VaR and ES off a discrete distribution (first grid point whose CDF >= alpha; ES as the conditional mean including the partial mass), and the checks that catch a 5-7 % bias: mean and variance of the FFT distribution must match the analytic moments, and the MC estimate with enough paths must land inside its own standard error.
triggers: text:fft, text:characteristic function, text:compound poisson
---
## When this applies
"Compute the aggregate loss distribution of a compound Poisson model with FFT", "VaR and ES at 95 / 99 / 99.5 % from the
FFT", "compare FFT with Monte Carlo", "price options with the characteristic function (Carr-Madan / COS)".

## Compound Poisson by FFT (frequency N ~ Poisson(lambda), severity X with density f)
1. Choose the grid: step h and size M = 2^k such that M h > the quantile you need with margin (for ES at 99.5 % the
   span must cover far beyond the 99.5 % point: aim at CDF(M h) > 1 - 1e-9, i.e. span ~ lambda E[X] + 10-15 standard
   deviations for light tails, much more for heavy tails).  Too small a span aliases tail mass back to the origin and
   biases VaR DOWN; too coarse a step biases ES and quantiles by up to h/2.
2. Discretise the severity so that MASS AND MEAN are preserved: midpoint method p_j = F((j + 1/2) h) - F((j - 1/2) h),
   p_0 = F(h/2); check sum_j p_j ~ 1 and sum_j j h p_j ~ E[X] (correct the last point for the lost tail if the task says).
   Using the density times h instead (f(jh) h) loses mass near zero for lognormal / gamma severities and is a common
   5 % error.
3. Aggregate: phi_X = FFT(p); phi_S = exp(lambda (phi_X - 1)) (Poisson), or (1 - q + q phi_X)^n for binomial,
   (1 - beta (phi_X - 1))^(-alpha) for negative binomial; g = Re(IFFT(phi_S)); clip tiny negatives; renormalise only if
   the task allows (and report how much mass was lost - it should be < 1e-8).
4. Read off risk measures on the grid: VaR_alpha = the smallest j h with CDF_j >= alpha (or the task's interpolation
   rule); ES_alpha = E[S | S > VaR] = (sum_{j: jh > VaR} j h g_j + partial mass at VaR) / (1 - alpha).  State whether
   ES includes the atom at VaR exactly as the task defines.
5. Validate with moments: E[S] = lambda E[X], Var[S] = lambda E[X^2] (compound Poisson).  If the FFT mean is off by
   > 0.5 % the discretisation or truncation is wrong; fix that before comparing with Monte Carlo.
6. Monte Carlo cross-check: simulate N_sim years (N ~ Poisson(lambda), sum of N severities) with the task's seed and
   path count; the MC quantile has sampling error ~ sqrt(alpha (1 - alpha) / N_sim) / f_S(VaR) - with 1e5 paths the
   99.5 % VaR is only known to a few percent, so a 5 % tolerance test needs the FFT to be unbiased, not the MC.

## Option pricing with the characteristic function (Carr-Madan)
- Damped call: psi(v) = e^{-rT} phi(v - (alpha + 1) i) / (alpha^2 + alpha - v^2 + i (2 alpha + 1) v); call price
  C(k) = e^{-alpha k} / pi * Re( integral_0^inf e^{-i v k} psi(v) dv ).  Grid: N = 4096, eta (v step) ~ 0.25, strike step
  lambda = 2 pi / (N eta); Simpson weights; alpha ~ 1.5 for calls (negative alpha for puts via put-call parity or the
  Lewis formula).  Check: at-the-money price equals the Black-Scholes price when phi is the Black-Scholes
  characteristic function (within 1e-6), and prices are decreasing and convex in strike.
- COS method: a = c1 - L sqrt(c2 + sqrt(c4)), b = c1 + L sqrt(c2 + sqrt(c4)) with L = 10; N = 256 terms is usually
  enough; check convergence by doubling N.

## Invariants worth printing
- Probability mass of the FFT distribution within 1e-8 of one; FFT mean within 0.5 % of lambda E[X].
- VaR non-decreasing in alpha; ES >= VaR for every alpha.
- Doubling the grid size changes VaR/ES by less than the task's tolerance (grid convergence).
- Monte Carlo estimate within 3 standard errors of the FFT value.
