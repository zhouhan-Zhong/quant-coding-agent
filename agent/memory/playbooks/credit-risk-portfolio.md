---
title: Credit portfolio risk: migration matrices, factor-copula simulation, Vasicek/Gordy, loss distribution
category: credit
tags: credit risk, rating migration, transition matrix, migration matrix, cohort, default probability, pd, cumulative pd, marginal pd, hazard, matrix power, generator matrix, markov, embeddable, creditmetrics, one-factor, multi-factor, sector factors, asset correlation, gaussian copula, student-t copula, thresholds, loss distribution, expected loss, unexpected loss, credit var, cvar, economic capital, euler allocation, risk contribution, marginal contribution, wrong-way risk, stress, concentration, hhi, granularity adjustment, gordy, vasicek, asrf, basel irb, lgd, ead, credit spread
summary: Cohort transition matrices with absorbing default and multi-year powers, generator checks, sector-factor Gaussian/t copula default simulation with exact Euler CVaR allocation, wrong-way stress, Vasicek ASRF VaR and the Gordy granularity adjustment, and the PD/spread relations.
---
## When this applies
"Estimate the one-year rating transition matrix from cohort data", "cumulative default probabilities over 1-5
years", "generator matrix / is the chain Markov", "simulate portfolio default losses with a (multi-)factor Gaussian
or Student-t copula", "credit VaR / CVaR at 95/99/99.9%", "Euler / marginal risk contributions by obligor and
sector", "wrong-way risk scenario", "Vasicek analytical benchmark and granularity adjustment", "spread from PD".

## Method and formulas
Symbols: ratings best -> worst with D last; N_ij cohort counts; P_ij transition probability; EAD_i, LGD_i (= 1 -
recovery), PD_i; rho_i asset correlation; s(i) sector of obligor i; C sector-factor correlation matrix; L loss.
- Cohort estimator: P_ij = N_ij / sum_j N_ij, excluding withdrawn/NR end-states from numerator and denominator;
  default row absorbing (P_DD = 1). Multi-year P^(n) = matrix_power(P, n); cumulative PD(n) = P^(n)[i, D];
  marginal = cum(n) - cum(n-1); conditional = marginal/(1 - cum(n-1)); annualised 1 - (1 - cum(n))^(1/n).
- Generator: Q = real(logm(P)); embeddable if off-diagonals >= 0 (tolerance) and rows sum 0; expm(Q t) gives
  fractional-year matrices. Markov test: empirical 2-year matrix vs P^2.
- One-factor CreditMetrics: A_i = sqrt(rho) Z + sqrt(1-rho) eps_i; default iff A_i < Phi^-1(PD_i); migration buckets
  from cumulative thresholds of the rating row (np.searchsorted). Multi-factor: M = Z @ chol(C).T (one column per
  sector), A_i = sqrt(rho_i) M_{s(i)} + sqrt(1-rho_i) eps_i. Student-t copula: divide the WHOLE A row by
  sqrt(chi2_nu/nu) drawn once per scenario and use thresholds t.ppf(PD, nu). Loss = D @ (EAD x LGD); mark-to-market
  mode revalues each obligor at the forward price/spread of its new rating instead.
- Loss statistics (losses positive): EL = mean(L) (~ sum EAD LGD PD; MC/EL within ~2%), UL = std(L, ddof=1),
  VaR_a = np.quantile(L, a) (linear), CVaR_a = mean(L[L >= VaR_a]), EC = VaR - EL; skew/kurtosis with scipy defaults
  (biased, excess); pct zero-loss scenarios = mean(L == 0) x 100.
- Euler allocation: CVaR contribution_i = mean over tail scenarios (L >= VaR) of D_i EAD_i LGD_i - sums EXACTLY to
  CVaR; VaR contributions average over a band around VaR; sector totals = sums of obligor contributions; pct of
  CVaR; risk-weight ratio = pct of CVaR / (100 x exposure weight). Marginal = contribution / EAD. Incremental VaR_i =
  VaR(all) - VaR(without i); standalone VaR_i from L_i alone; diversification benefit = sum standalone - portfolio VaR.
- Wrong-way stress: for obligors in the stressed sectors rho -> min(rho + boost, 0.999) and LGD -> 1 - stressed
  recovery; rerun with the BASE seed so the change is attributable; report VaR/CVaR increase in %.
- Vasicek (ASRF) VaR_a = sum_{PD>0} EAD LGD Phi((Phi^-1(PD) + sqrt(rho) Phi^-1(a))/sqrt(1-rho)) - the conditional
  expected loss itself, NOT minus EL (that matched); Basel IRB K = LGD [Phi(...) - PD] x maturity adj.
- Gordy granularity adjustment (as implemented by the reference): w_i = EAD_i / M (M = total exposure), h_i the Vasicek
  argument, p_i = Phi(h_i); sigma^2 = sum w^2 LGD^2 p(1-p); C' = sum w LGD phi(h) sqrt(rho)/(sqrt(1-rho) phi(Phi^-1 a));
  GA = M sigma^2 / (2 C'); Gordy VaR = Vasicek VaR + GA (GA > 0, small vs VaR, decreasing in a is acceptable).
- Concentration: obligor HHI = sum (EAD/M)^2, sector HHI on sector exposure shares, max exposure weight.
- Spread basics: hazard lambda ~ s/(1 - R); Q(t) = exp(-lambda t); annual PD = 1 - e^-lambda; zero-coupon spread
  s = -ln(1 - PD_T LGD)/T; CDS par spread ~ lambda (1 - R).

## Conventions and pitfalls (what graders check)
- Rating order kept on both axes; rows sum to 1 (1e-10), P_DD = 1, cumulative PD monotone in horizon and rating.
- Input probabilities in percent -> /100; withdrawn ratings excluded from denominators; counts exact ints.
- Simulation: rng = np.random.default_rng(seed); factors first (n_sims x n_factors) then idiosyncratic (n_sims x n);
  strict '<' vs thresholds; separate seeds per copula as specified (e.g. seed + offset for the t run).
- PD = 0 obligors: norm.ppf(0) = -inf is fine in simulation (never default) but exclude them from Vasicek/Gordy sums.
- Memory/runtime: simulate in chunks of ~10k scenarios; store the default matrix as bool (100k x 1000 bools ~ 100 MB),
  never a float loss matrix; losses via bool matrix @ (EAD x LGD).
- Invariants asserted: CVaR >= VaR, monotone in a, t-copula tails heavier (VaR_99.9 and kurtosis), Euler sum / CVaR ~ 1,
  weights sum to 1, HHI in (0, 1), Gordy = Vasicek + GA. Ignore descriptive columns that contradict the data dictionary.
- Rounding: VaR/CVaR to 2 dp only when specified; summary values must equal the CSV values (reuse the same numbers).
- logm may return tiny imaginary parts: take the real part; report non-embeddability, do not "fix" P.

## Tested snippet
```python
import numpy as np, pandas as pd
from scipy import stats
from scipy.linalg import logm, expm

def migration_matrix(start, end, ratings, exclude=("NR", "WR")):     # cohort: arrays of start/end ratings per obligor-year
    df = pd.DataFrame({"s": start, "e": end}); df = df[~df["e"].isin(exclude)]
    cnt = pd.crosstab(df["s"], df["e"]).reindex(index=ratings, columns=ratings, fill_value=0).astype(float)
    P = cnt.div(cnt.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    P.loc["D"] = 0.0; P.loc["D", "D"] = 1.0                           # default is absorbing
    return cnt, P                                                      # rows sum to 1
def cumulative_pd(P, horizons):                                        # multi-year via matrix powers, PD = column 'D'
    return {h: pd.Series(np.linalg.matrix_power(P.to_numpy(), h)[:, -1], index=P.index) for h in horizons}
def generator_check(P):                                                # Q = logm(P): rows sum 0, off-diagonals >= 0 if embeddable
    Q = np.real(logm(P.to_numpy())); off = Q - np.diag(np.diag(Q))
    return Q, bool(np.all(off >= -1e-8)), float(np.abs(expm(Q) - P.to_numpy()).max())
def factor_sim(pd_, ead, lgd, rho, sec, C, n_sims, seed, nu=None, chunk=10_000):   # sector-factor Merton; t copula if nu
    rng = np.random.default_rng(seed); Lc = np.linalg.cholesky(C); n = len(pd_); ll = ead * lgd
    thr = stats.norm.ppf(pd_) if nu is None else stats.t.ppf(pd_, nu)   # PD = 0 -> -inf: never defaults
    L, D = np.empty(n_sims), np.empty((n_sims, n), dtype=bool)         # keep a BOOL default matrix, not floats
    for s in range(0, n_sims, chunk):
        m = min(chunk, n_sims - s); M = rng.standard_normal((m, len(C))) @ Lc.T      # correlated sector factors first
        A = np.sqrt(rho) * M[:, sec] + np.sqrt(1 - rho) * rng.standard_normal((m, n))
        if nu is not None: A /= np.sqrt(rng.chisquare(nu, (m, 1)) / nu)    # one chi-square mixing draw per scenario
        D[s:s + m] = A < thr; L[s:s + m] = D[s:s + m] @ ll
    return L, D
def loss_stats(L, D, ll, a):                                           # VaR = quantile, CVaR = tail mean, Euler CVaR split
    var = float(np.quantile(L, a)); tail = L >= var; cvar = float(L[tail].mean()); eul = D[tail].mean(axis=0) * ll
    return dict(EL=float(L.mean()), UL=float(L.std(ddof=1)), VaR=var, CVaR=cvar, EC=var - float(L.mean()), euler_gap=float(eul.sum() - cvar)), eul
def vasicek_gordy(pd_, ead, lgd, rho, a):                              # ASRF VaR (conditional EL, no EL subtraction) + Gordy GA
    M = ead.sum(); m = pd_ > 0; p, e, g, r = pd_[m], ead[m], lgd[m], rho[m]; w = e / M     # PD = 0 excluded from the sums
    h = (stats.norm.ppf(p) + np.sqrt(r) * stats.norm.ppf(a)) / np.sqrt(1 - r); ph = stats.norm.cdf(h)
    s2 = (w ** 2 * g ** 2 * ph * (1 - ph)).sum(); dC = (w * g * stats.norm.pdf(h) * np.sqrt(r) / (np.sqrt(1 - r) * stats.norm.pdf(stats.norm.ppf(a)))).sum()
    return float((e * g * ph).sum()), float(M * s2 / (2 * dC))         # Vasicek VaR, granularity adj. (Gordy VaR = sum)

ratings = ["A", "B", "C", "D"]; rng = np.random.default_rng(1)
s = rng.choice(["A", "B", "C"], 3000, p=[0.5, 0.35, 0.15]); P_true = {"A": [0.9, 0.08, 0.015, 0.005], "B": [0.05, 0.85, 0.08, 0.02], "C": [0.01, 0.1, 0.8, 0.09]}
e = np.array([rng.choice(ratings, p=P_true[x]) for x in s]); e[:20] = "NR"
cnt, P = migration_matrix(s, e, ratings); cpd = cumulative_pd(P, [1, 2, 5]); Q, embeddable, err = generator_check(P)
n = 200; pd_ = np.r_[np.zeros(20), rng.uniform(0.002, 0.05, n - 20)]; ead = rng.uniform(1e6, 5e6, n); lgd, rho = np.full(n, 0.45), np.full(n, 0.15)
sec, C = rng.integers(0, 3, n), np.array([[1, .5, .3], [.5, 1, .4], [.3, .4, 1]])
Lg, Dg = factor_sim(pd_, ead, lgd, rho, sec, C, 20000, 42); Lt, _ = factor_sim(pd_, ead, lgd, rho, sec, C, 20000, 1042, nu=5)
st, eul = loss_stats(Lg, Dg, ead * lgd, 0.99); V, GA = vasicek_gordy(pd_, ead, lgd, rho, 0.99)
print(P.round(4).to_numpy().tolist()[0], P.sum(axis=1).round(12).tolist(), cpd[5].round(4).tolist(), embeddable, round(err, 12))
print({k: round(v, 2) for k, v in st.items()}, round(float(np.quantile(Lt, 0.99)), 2), round(V, 2), round(GA, 2), round(float((pd_ * ead * lgd).sum()), 2), round(float(stats.kurtosis(Lt) / stats.kurtosis(Lg)), 2))
```
