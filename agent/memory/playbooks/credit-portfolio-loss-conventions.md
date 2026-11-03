---
title: Credit portfolio loss conventions: PD/LGD/EAD, one-factor Gaussian copula (Vasicek), CreditMetrics migration, hazard-rate curves, marginal and component CVaR, sector decomposition
category: credit
tags: credit risk, probability of default, pd, lgd, ead, expected loss, unexpected loss, vasicek, one factor, asset correlation, gaussian copula, default correlation, loss distribution, var, cvar, expected shortfall, creditmetrics, transition matrix, migration, rating, cohort, generator matrix, hazard rate, survival probability, cds spread, recovery, marginal var, component var, euler allocation, sector, obligor, concentration, granularity, monte carlo, seed
summary: The default conventions behind credit-portfolio tasks: EL = PD x LGD x EAD per obligor, portfolio loss by simulating a one-factor model (Z_i = sqrt(rho) M + sqrt(1 - rho) e_i, default when Z_i < Phi^{-1}(PD_i)) with the task's seed and path count, VaR as the alpha-quantile of LOSS (positive numbers), CVaR as the mean loss beyond VaR, component/marginal CVaR by Euler allocation (average loss of obligor i in the tail scenarios) so that components sum to the portfolio CVaR, migration matrices from cohort counts with rows summing to one, and hazard rates from CDS spreads via lambda = s / (1 - R).
---
## When this applies
"Simulate the portfolio loss distribution and report VaR/CVaR at 99 / 99.9 %", "marginal / component CVaR per obligor
and per sector", "which obligor contributes most", "estimate a rating transition matrix from the cohort file",
"bootstrap hazard rates from CDS spreads", "expected and unexpected loss".

## Definitions to pin down
- EL_i = PD_i x LGD_i x EAD_i (LGD = 1 - recovery); portfolio EL = sum.  Unexpected loss (UL) = standard deviation of
  the loss distribution (analytic for independent obligors: sqrt(sum EAD_i^2 LGD_i^2 PD_i (1 - PD_i)) - the task may
  define UL as VaR - EL instead; read it.
- One-factor Gaussian (Vasicek): Z_i = sqrt(rho_i) M + sqrt(1 - rho_i) eps_i; default iff Z_i < Phi^{-1}(PD_i).  Draw M and
  eps with the task's generator and seed, M first then eps (or as stated); N scenarios exactly as stated.  Conditional
  PD: p_i(M) = Phi((Phi^{-1}(PD_i) - sqrt(rho_i) M) / sqrt(1 - rho_i)).  Large-homogeneous-portfolio loss quantile:
  L_alpha = LGD Phi((Phi^{-1}(PD) + sqrt(rho) Phi^{-1}(alpha)) / sqrt(1 - rho)).
- Losses are POSITIVE numbers; VaR_alpha = quantile(L, alpha) (np.quantile with the task's interpolation, or the
  empirical order statistic ceil(alpha N)); CVaR_alpha = mean of losses >= VaR (state whether >= or >).  Report both as
  amounts or as % of total EAD exactly as the task says.
- Euler / component allocation: CVaR contribution of obligor i = E[L_i | L >= VaR_alpha] (mean of obligor i's loss over
  the tail scenarios); the components sum to the portfolio CVaR (invariant).  Marginal CVaR = CVaR(portfolio) -
  CVaR(portfolio without i) when the task says "marginal"; the two differ - follow the wording.  "Top obligor" is the
  one with the largest component; ties broken as the task says (by EAD, by name).
- Sector decomposition: sum of components over obligors of each sector; every sector present in the obligor file must
  appear (zero contribution still listed).
- Expected shortfall from a parametric fit (if asked): Gaussian ES = mu + sigma phi(z_alpha) / (1 - alpha).

## Migration / transition matrices
- Cohort estimate: P_ij = (number of obligors in rating i at the start that are in j at the end) / (number in i at the
  start), withdrawals excluded or treated as the task says; rows sum to one (invariant); default row is absorbing
  (P_DD = 1).  Multi-year matrices: matrix powers of the one-year matrix (or the generator matrix exponential if the
  task uses continuous time: P(t) = expm(t Q), Q rows sum to zero, off-diagonal >= 0).
- Rating order and labels exactly as the input (AAA > ... > CCC > D); report counts as integers.

## Hazard rates and CDS
- Flat hazard from a CDS spread: lambda = s / (1 - R) (spread in decimal, R recovery); survival S(t) = exp(-lambda t);
  piecewise-constant bootstrap: match each quoted maturity's premium leg PV = protection leg PV (premium paid quarterly in
  arrears with accrual on default when the task says).  Risky annuity, CVA = (1 - R) sum EE(t_i) DF(t_i) (S(t_{i-1}) -
  S(t_i)).
- Report hazard rates and spreads in the task's unit (decimal vs bp vs %).

## Invariants worth printing
- sum of component CVaRs == portfolio CVaR (within 1e-6 relative); VaR_99.9 >= VaR_99 >= EL; losses >= 0; transition
  rows sum to one; survival probabilities decreasing in t; every sector of the input appears in the decomposition.
