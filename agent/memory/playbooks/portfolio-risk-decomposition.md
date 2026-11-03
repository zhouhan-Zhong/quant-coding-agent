---
title: Portfolio risk decomposition, risk parity, CVaR, vol targeting and Kelly sizing
category: portfolio
tags: portfolio, covariance, ewma covariance, ledoit-wolf, shrinkage, portfolio volatility, marginal var, component var, incremental var, risk contribution, risk budget, risk parity, inverse volatility, equal risk contribution, erc, regime risk budget, monthly rebalance, month end, minimum variance, maximum sharpe, tangency, mean-variance, efficient frontier, cvar optimization, rockafellar-uryasev, linprog, historical cvar, kelly criterion, fractional kelly, half kelly, leverage, volatility targeting, ex-ante volatility, leverage cap, position sizing, neutralization, beta neutral, dollar neutral, weights
summary: Exact decomposition formulas (marginal/component/incremental), risk-parity and mean-variance optimisers with SLSQP, the Rockafellar-Uryasev CVaR LP, month-end inverse-vol rebalancing with regime budgets, ex-ante vol targeting, neutralisation by projection and Kelly fractions, with the window/lag/ddof conventions graders test.
---
## When this applies
"Decompose portfolio volatility/VaR into contributions", "risk parity / equal risk contribution weights",
"inverse-volatility weights rebalanced monthly with a regime risk budget", "minimum-variance and maximum-Sharpe
portfolios with long-only constraints", "minimise CVaR with an LP", "Kelly (full/half) position sizes", "scale the
book to a 10% ex-ante volatility target with a leverage cap", "make the portfolio dollar/beta neutral".

## Method and formulas
Symbols: R (T x n) period returns, mu = mean vector, S = covariance (per period), w weights (sum 1 unless
leverage allowed), sigma_p = sqrt(w'Sw), z_a = Phi^-1(a), rf per period, A = annualisation factor (252 daily,
12 monthly, 52 weekly).
- Covariance: sample np.cov(R.T, ddof=1); EWMA S_t = lam S_{t-1} + (1-lam) r_t r_t' (lam 0.94, init sample cov);
  Ledoit-Wolf sklearn.covariance.LedoitWolf().fit(R).covariance_ (1/T, shrinks to scaled identity). Annualise S*A.
- Decomposition: marginal MRC_i = (S w)_i / sigma_p; component CRC_i = w_i MRC_i (sum = sigma_p); percent
  contribution = CRC_i/sigma_p. Parametric VaR = z_a sigma_p (- mu_p if the mean is included); component VaR_i =
  w_i z_a (S w)_i/sigma_p (sum = VaR); incremental VaR_i = VaR(w) - VaR(w with w_i = 0). Historical component ES_i =
  mean(w_i r_i | portfolio loss >= VaR) (sums exactly to ES).
- Risk parity: inverse-vol w_i = (1/sigma_i)/sum_j (1/sigma_j); ERC: SLSQP minimising sum_i (w_i (Sw)_i/(w'Sw) - 1/n)^2
  with sum w = 1, w >= 0 (or min 0.5 w'Sw - (1/n) sum ln w_i, then normalise). Check max |CRC_i/sigma_p - 1/n| < 1e-6.
- Month-end inverse-vol with regime budgets (reference conventions): rebalance on the LAST trading day of each month
  (month[t] != month[t+1]; the final data day has no successor, so it is not a rebalance); weights from the trailing
  window R[t-L:t] EXCLUDING day t (std ddof=1), normalised to 1, times the regime's budget; effective from the NEXT
  day; before the first rebalance hold equal weights 1/n (zero or inverse-vol initial weights failed); "mean weight
  of asset i" = mean over the rebalance vectors, not over days. Regimes: regime-models-and-black-litterman.
- Minimum variance / max Sharpe: SLSQP with sum w = 1 (+ w >= 0); closed forms S^-1 1/(1'S^-1 1) and tangency
  w proportional to S^-1 (mu - rf 1). Sharpe_annual = (w'mu - rf) A / (sigma_p sqrt A).
- CVaR LP (Rockafellar-Uryasev), S equally likely scenarios r_s, level beta: x = [w (n), alpha, u (S)]; minimise
  alpha + sum u_s/((1-beta) S) s.t. u_s >= -r_s'w - alpha, u_s >= 0, sum w = 1, w >= 0 (optional mu'w >= target);
  alpha* = VaR, objective = CVaR (loss-positive). linprog(method='highs').
- Strategy tail risk from realised returns: VaR = -np.percentile(r[-W:], 1), CVaR = -mean(r[-W:][r[-W:] <= -VaR]).
- Kelly: single bet f* = p - (1-p)/b; continuous f* = (mu - rf)/sigma^2; multivariate f* = S^-1 (mu - rf 1)
  (leverage = sum |f|); fractional k f*; growth g = rf + f'(mu - rf) - 0.5 f'Sf; Kelly VaR = z_a sqrt(f'Sf).
- Vol targeting: (i) realised: scale_t = target / (sigma_hat_{t-1} sqrt A), info up to t-1; (ii) ex-ante for a
  cross-sectional book whose weights are set at the close of t: scale = min(target / sqrt(w' Cov(R[t-L+1..t]) w A),
  cap) - the window INCLUDES day t; (iii) per-asset trend sizing w_i = signal_i (target/n)/sigma_i,ann from an EWMA
  variance, clipped to +-max leverage, applied with a one-day lag (backtest-conventions).
- Neutralisation: dollar + beta neutral w - X pinv(X'X) X'w with X = [1, beta], then rescale to gross 1;
  PC-neutral min-norm hedge via lstsq on [components; 1] (latent-factor-models).

## Conventions and pitfalls (what graders check)
- Sum checks: sum CRC == sigma_p (1e-10), sum component VaR == VaR, percent contributions sum to 1, ERC contributions
  equal (1e-6), weights sum to 1 (1e-8), long-only weights >= -1e-9 (clip tiny negatives).
- ddof=1 for sample covariance; do not mix estimators between the decomposition and the optimisation.
- Window alignment is the usual failure: windows that include the decision day, weights applied the same day as
  the rebalance, or a different pre-first-rebalance holding each move annualised return beyond 2% tolerances.
- Kelly on daily data: rf_daily = rf_annual/252 on the estimation window the task specifies, covariance ddof=1;
  Kelly VaR has no mean term; fractions may be negative.
- CVaR LP: with few scenarios alpha may be non-unique - report CVaR from the objective and VaR as the empirical
  quantile of the optimal portfolio if the task recomputes it.
- SLSQP: ftol 1e-12..1e-16, maxiter 1000, start from equal weights, check res.success.
- Monte Carlo sizing: rng = np.random.default_rng(seed), one FRESH draw per scheme in the specified order, wealth =
  W0 cumprod(1 + port), drawdown on the path prefixed with W0 (positive), "median" may be sorted[n//2] in references.
- Annualise AFTER per-period statistics: mean x A (compound only if stated), std x sqrt(A); Sharpe uses excess
  returns unless rf = 0 is stated. Output arrays in input asset order; floats via float(); no NaN in JSON.

## Tested snippet
```python
import math, numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize, linprog

def risk_contrib(w, S, a=0.99):                        # vol and parametric-VaR decomposition (sums are exact)
    sig = math.sqrt(w @ S @ w); mrc = S @ w / sig        # marginal: d sigma / d w_i
    crc = w * mrc; z = stats.norm.ppf(a)                 # component: sum(crc) == sig
    return sig, mrc, crc, z * sig, w * z * mrc           # VaR (zero mean), component VaR (sums to VaR)
def erc_weights(S):                                    # equal risk contribution, long-only, sum w = 1
    n = S.shape[0]; obj = lambda w: np.sum((w * (S @ w) / (w @ S @ w) - 1.0 / n) ** 2)
    r = minimize(obj, np.full(n, 1 / n), method="SLSQP", bounds=[(1e-9, 1)] * n,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"ftol": 1e-16, "maxiter": 1000})
    return r.x
def min_var_and_max_sharpe(mu, S, rf=0.0):             # long-only, fully invested; mu/S in the same period units
    n = len(mu); cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]; b = [(0, 1)] * n; w0 = np.full(n, 1 / n)
    mv = minimize(lambda w: w @ S @ w, w0, method="SLSQP", bounds=b, constraints=cons, options={"ftol": 1e-14}).x
    ms = minimize(lambda w: -(w @ mu - rf) / math.sqrt(w @ S @ w), w0, method="SLSQP", bounds=b, constraints=cons, options={"ftol": 1e-14}).x
    return mv, ms
def cvar_lp(R, beta=0.95):                             # Rockafellar-Uryasev LP: min alpha + mean(u)/(1-beta); x = [w, alpha, u]
    S_, n = R.shape; c = np.r_[np.zeros(n), 1.0, np.full(S_, 1.0 / ((1 - beta) * S_))]
    A_ub = np.c_[-R, -np.ones((S_, 1)), -np.eye(S_)]; b_ub = np.zeros(S_)       # loss_s - alpha - u_s <= 0, loss = -r_s'w
    A_eq, b_eq = [np.r_[np.ones(n), 0.0, np.zeros(S_)]], [1.0]                  # add a row -mu'w <= -target for a return floor
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n + [(None, None)] + [(0, None)] * S_, method="highs")
    return res.x[:n], float(res.x[n]), float(res.fun)   # weights, VaR (alpha), CVaR
def kelly(mu_excess, S, frac=1.0): f = frac * np.linalg.solve(S, mu_excess); return f, float(np.abs(f).sum())
def monthly_inverse_vol(R, dates, L, budget=lambda t: 1.0):   # month-END rebalance, window R[t-L:t] (excl. t), effective next day
    T, n = R.shape; m = pd.DatetimeIndex(dates).to_period("M"); is_me = np.r_[m[:-1] != m[1:], False]   # last data day: no rebalance
    w, W, reb = np.full(n, 1.0 / n), np.zeros((T, n)), []                # equal weight 1/n until the first rebalance
    for t in range(L, T):
        W[t] = w                                                         # weight in force today, set at an earlier close
        if is_me[t]: iv = 1 / R[t - L:t].std(axis=0, ddof=1); w = iv / iv.sum() * budget(t); reb.append(w)
    return W, reb
def vol_target_scale(w, R_win, target, cap, A=252):   # ex-ante scale from the trailing covariance, capped
    return min(target / math.sqrt(w @ np.cov(R_win, rowvar=False, ddof=1) @ w * A), cap)

D = np.diag([0.012, 0.009, 0.005]); R = np.random.default_rng(0).multivariate_normal([4e-4, 3e-4, 2e-4], D @ [[1, .3, .1], [.3, 1, .2], [.1, .2, 1]] @ D, 1000)
S = np.cov(R.T, ddof=1); mu = R.mean(axis=0); w = np.array([0.5, 0.3, 0.2]); sig, mrc, crc, var, cvar_c = risk_contrib(w, S)
w_erc = erc_weights(S); mv, ms = min_var_and_max_sharpe(mu, S, 0.0001); w_cv, var_lp, cvar = cvar_lp(R, 0.95)
print(round(sig * math.sqrt(252), 5), np.round(crc / sig, 4).tolist(), round(abs(crc.sum() - sig) + abs(cvar_c.sum() - var), 14), np.round(w_erc, 4).tolist(), np.round(risk_contrib(w_erc, S)[2] / risk_contrib(w_erc, S)[0], 5).tolist())
print(np.round(mv, 4).tolist(), np.round(ms, 4).tolist(), np.round(w_cv, 4).tolist(), round(var_lp, 5), round(cvar, 5), np.round(kelly(mu - 0.0001, S, 0.5)[0], 2).tolist())
W, reb = monthly_inverse_vol(R, pd.bdate_range("2021-01-04", periods=len(R)), 63); ret = (W[63:] * R[63:]).sum(axis=1)
tail = ret[-63:]; v99 = -np.percentile(tail, 1); print(len(reb), np.round(reb[0], 4).tolist(), round(float(ret.mean() * 252), 5), round(float(v99), 5), round(float(-tail[tail <= -v99].mean()), 5), round(vol_target_scale(w, R[-20:], 0.10, 3.0), 4))
```
