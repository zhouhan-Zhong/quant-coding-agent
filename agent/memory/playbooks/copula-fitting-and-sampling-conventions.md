---
title: Copula fitting and sampling conventions: pseudo-observations, Kendall tau to parameter maps, Gaussian/Student-t/Clayton/Gumbel/Frank, tail-dependence coefficients, empirical tail estimates, sampling in [0,1]
category: risk
tags: copula, gaussian copula, student t copula, clayton, gumbel, frank, archimedean, kendall tau, spearman rho, pseudo observations, ranks, probability integral transform, tail dependence, upper tail, lower tail, lambda_u, lambda_l, maximum likelihood, log-likelihood, aic, bic, degrees of freedom, inverse cdf, sampling, conditional sampling, marshall olkin, rank correlation, empirical copula, pair, pairwise
summary: The standard conventions behind copula tasks: pseudo-observations u = rank/(n+1) (never rank/n), the exact tau-to-parameter maps (Gaussian rho = sin(pi tau/2); Clayton theta = 2 tau/(1-tau); Gumbel theta = 1/(1-tau); Frank by solving the Debye relation), the closed-form tail coefficients (Clayton lower 2^(-1/theta), Gumbel upper 2 - 2^(1/theta), Gaussian 0, t-copula lambda = 2 t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))), the empirical tail estimate P(U > q | V > q), and the sampling algorithms that keep U in (0,1).
triggers: text:copula
---
## When this applies
"Fit Gaussian / t / Clayton / Gumbel / Frank copulas to the return pairs", "report theta, rho, nu, log-likelihood, AIC",
"which copula fits best", "upper / lower tail dependence", "sample N pairs from the fitted copula", "Kendall's tau of
the sample must match the input within 0.02".

## Pseudo-observations (the step every grader checks)
- u_i = rank(x_i) / (n + 1) with average ranks for ties (scipy.stats.rankdata default); dividing by n puts a point
  exactly at 1 and breaks the inverse CDFs.  Fit the copula on (u, v), never on raw returns or z-scores unless the task
  says "parametric margins".
- Kendall's tau: scipy.stats.kendalltau (tau-b) on the raw pairs; Spearman: spearmanr.  Report sample tau from the
  SAME window the copula is fitted on.

## Parameter maps and likelihoods
- Gaussian: rho = sin(pi tau / 2) (moment estimate) or MLE on the Gaussian copula density with z = Phi^{-1}(u).
- Student-t: rho as above (same tau relation), nu by maximising the t-copula log-likelihood over nu in [2.1, 50]
  (profile likelihood); the t-copula log-likelihood must be >= the Gaussian one at the same rho as nu -> infinity -
  if yours is lower, the density formula is wrong (remember the marginal t densities in the denominator).
- Clayton: theta = 2 tau / (1 - tau) > 0 for tau > 0 (no fit for tau <= 0 unless rotated); density
  c = (1 + theta) (u v)^(-1-theta) (u^-theta + v^-theta - 1)^(-2-1/theta).
- Gumbel: theta = 1 / (1 - tau) >= 1; generator phi(t) = (-ln t)^theta.
- Frank: tau = 1 - 4/theta (1 - D_1(theta)) with Debye D_1(theta) = (1/theta) integral_0^theta t/(e^t - 1) dt; solve for
  theta numerically (brentq on [-50, 50] excluding 0).
- AIC = 2 k - 2 LL with k = 1 (Gaussian, Clayton, Gumbel, Frank) or 2 (t); "best" = lowest AIC unless the task names
  another criterion.

## Tail dependence
- Closed forms: Gaussian lambda_U = lambda_L = 0 (rho < 1); t: lambda = 2 t_{nu+1}( -sqrt((nu + 1)(1 - rho)/(1 + rho)) )
  (both tails); Clayton: lambda_L = 2^(-1/theta), lambda_U = 0; Gumbel: lambda_U = 2 - 2^(1/theta), lambda_L = 0;
  Frank: 0 both.  So "strongest upper tail" among {Clayton, Gumbel, Frank} is Gumbel and "strongest lower tail" is
  Clayton by construction - if the sample says otherwise, the parameters or the orientation (u vs 1-u) are wrong.
- Empirical estimate at level q: lambda_U(q) = P(U > q, V > q) / (1 - q) = #{u_i > q and v_i > q} / #{v_i > q};
  lambda_L(q) = #{u_i < 1-q... } analogously with q small.  Use the task's q (often 0.95 / 0.05) on the SAMPLED copula
  when the task asks for "tail dependence of the samples".

## Sampling (keep every u in (0, 1))
- Gaussian/t: draw correlated normals (Cholesky) -> u = Phi(z); for t divide by sqrt(W/nu), W ~ chi2(nu), then
  u = t_nu(x).
- Clayton (Marshall-Olkin): V ~ Gamma(1/theta, 1); E_1, E_2 ~ Exp(1); u_k = (1 + E_k / V)^(-1/theta).
- Gumbel: V ~ positive stable with alpha = 1/theta (Chambers-Mallows-Stuck), u_k = exp(-(E_k / V)^(1/theta)).
- Frank: conditional inversion, u_2 = -1/theta ln(1 + w (1 - e^-theta) / (w (e^{-theta u_1} - 1) - e^{-theta u_1})).
- Use the task's seed with np.random.default_rng(seed) and the stated N; clip to [1e-12, 1 - 1e-12] only if an inverse CDF
  needs it, and never produce values outside (0, 1).  Check: sampled Kendall tau within ~0.02 of the fitted tau for N >= 1e4.

## Invariants worth printing
- All u, v in (0, 1); sampled tau vs target tau gap; t-copula LL >= Gaussian LL; Clayton theta > 0 and Gumbel theta >= 1
  when tau > 0; lambda_U(Gumbel) > lambda_U(Frank) = 0 and lambda_L(Clayton) > 0.
