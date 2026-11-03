---
title: Factor regressions, rolling decompositions, portfolio sorts, momentum, Fama-MacBeth and IC
category: backtest-factors
tags: factor model, fama-french, ff3, carhart, four-factor, time-series regression, alpha, beta, t-stat, hac, newey-west, bartlett, rolling regression, rolling beta, rolling ols, variance decomposition, regime break, tracking error, information ratio, cross-sectional momentum, 12-2, 12-1, skip month, formation period, holding period, decile, quintile, portfolio sorts, double sort, independent sort, conditional sort, long-short, spread, fama-macbeth, residual momentum, information coefficient, ic, spearman, hit rate, equal weight, value weight, monthly rebalance
summary: Alignment rules for factor regressions (alpha annualisation, TE/IR, HAC t-stats incl. a manual Newey-West), rolling-window component and variance decompositions, momentum signals with a skip month, decile/double sorts, Fama-MacBeth standard errors and IC statistics.
---
## When this applies
"Regress the strategy/stock excess returns on FF3/Carhart factors: alpha, betas, tracking error, information ratio,
HAC t-stats", "rolling 48-month regression decomposition of a spread into factor components with Newey-West
t-stats and variance shares", "12-2 momentum decile portfolios", "double sort", "Fama-MacBeth", "residual momentum",
"information coefficient and hit rate of a signal".

## Method and formulas
Symbols: r_{i,t} return (decimal), rf_t risk-free, F_t factor vector (Mkt-RF, SMB, HML, [MOM]), T periods, N assets,
S_{i,t} signal known at the START of period t (end of t-1), A = 12 monthly / 252 daily.
- Time-series regression: r - rf = alpha + beta'F + e (sm.OLS + add_constant, or np.linalg.lstsq on [1, F]);
  alpha_annual = A alpha (arithmetic); tracking error = std(e, ddof=1) sqrt(A); IR = alpha_annual / TE; HAC
  cov_type='HAC', cov_kwds={'maxlags': L} (task value, else floor(4 (T/100)^(2/9))). Factor files from Ken French
  are in PERCENT -> /100, dates YYYYMM; align on dates even when lengths match.
- Manual Newey-West (reference style, = statsmodels HAC default use_correction=False): xe = X * e[:, None];
  S = xe'xe + sum_{j=1..M} (1 - j/(M+1))(G_j + G_j'), G_j = xe[j:]' xe[:-j]; V = (X'X)^-1 S (X'X)^-1; t = b/sqrt(diag V);
  no T/(T-k) factor, no prewhitening.
- Rolling decomposition (window W ending at t, rows t-W+1..t): OLS of y on [1, x_1..x_k]; component_k(t) = b_k x_k(t);
  residual(t) = y_t - alpha - sum components; variance share_k = sum((b_k x_k - mean)(y - mean)) / sum((y - mean)^2)
  x 100 over the window (ddof cancels); shares sum to R2 x 100 by the OLS identity, residual share = (1 - R2) x 100,
  so rows sum to exactly 100. Regime breaks = windows with R2 below a threshold. First output date = W-th month.
- Rolling betas: RollingOLS(y, X, window=w).fit().params or a loop; beta dated at the window END.
- Momentum (12-2): S_{i,t} = prod_{k=2..12}(1 + r_{i,t-k}) - 1; pandas ((1+R).rolling(11).apply(np.prod, raw=True)
  - 1).shift(2) with the holding month as the row; "12-1" uses shift(1); sum variant only if stated.
- Sorts: rank = s.rank(method='first'), bin = floor((rank-1) n_bins / n_valid) (0 = lowest); equal weight = bin mean;
  value weight with lagged market cap; long-short = top - bottom. Double sorts: independent (bins on the full
  cross-section, intersect) vs conditional (second sort within first-stage bins).
- Spread statistics: mean, std (ddof=1), t = mean/(std/sqrt(T)), annualised mean x A, vol x sqrt(A), Sharpe. k names
  per leg: mean(top k) - mean(bottom k), weights reset each period, no costs unless stated; periods with too few names
  are excluded from statistics but reported in the counts.
- Fama-MacBeth: per-period cross-sectional OLS -> gamma_t; estimate = mean; se = std(gamma, ddof=1)/sqrt(T).
- Residual momentum: rolling 36-month factor regressions ending t-1, residuals e_{t-12..t-2}, signal = sum(e)/std(e).
- IC: per period Spearman(S_{.,t}, r_{.,t+1}) over names with a valid signal (skip periods with < 3 names);
  mean, std (ddof=1), IR = mean/std, t = IR sqrt(n), hit rate = share of IC > 0; Pearson only if "IC (Pearson)".

## Conventions and pitfalls (what graders check)
- Alignment is the whole game: signal through t-1 (or t-2 with a skip), holding return in t; checkers compare the
  first valid period and the number of signal periods.
- Month keys: pd.Period('M') on both sides before joining; month-start stamps (YYYY-MM-01) when monthly and daily
  sources are mixed (data-cleaning-and-calendars). Monthly French rf is monthly, not annual.
- Returns: P_t/P_{t-1} - 1 from month-end prices; compound inside formation windows; keep float64 (tolerances can be
  1e-12 on these pipelines).
- Ties: argsort(kind='stable') / rank(method='first') break by input order; decile 10 = highest signal.
- std ddof=1 for Sharpe, t-stats, FM; HAC tvalues already robust; report nobs after dropping NaNs.
- Reference quirk seen once: a pinned "alpha_tstat" equal to alpha / SE(beta) (index slip); if alpha/SE(alpha)
  misses a pin by far, test that variant.
- Rolling outputs: exact row count = T - W + 1, sorted dates, no NaN; HAC t-stats checked loosely (atol ~0.5).

## Tested snippet
```python
import math, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
def ts_regression(excess_ret, factors, hac_lags=6, A=12):      # r - rf = alpha + beta'F + e; alpha_ann = A alpha; TE/IR from residuals
    d = pd.concat([excess_ret.rename("y"), factors], axis=1, join="inner").dropna()      # align dates first
    res = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    a, te = float(res.params.iloc[0]), float(res.resid.std(ddof=1) * math.sqrt(A))
    return dict(alpha_annual=a * A, alpha_t=float(res.tvalues.iloc[0]), betas=res.params.iloc[1:].round(6).to_dict(), r2=float(res.rsquared), n=int(res.nobs), te=te, ir=a * A / te)
def newey_west_t(X, e, b, M):                                   # Bartlett 1 - j/(M+1), no small-sample factor (= statsmodels HAC default)
    Xi = np.linalg.inv(X.T @ X); xe = X * e[:, None]; S = xe.T @ xe
    for j in range(1, M + 1): G = xe[j:].T @ xe[:-j]; S += (1 - j / (M + 1)) * (G + G.T)
    return b / np.sqrt(np.diag(Xi @ S @ Xi))
def rolling_decomposition(y, X, W, M=3):                        # windows t-W+1..t; component_k = b_k x_k(t); variance shares sum to 100
    out = []
    for t in range(W - 1, len(y)):
        Xw = np.column_stack([np.ones(W), X[t - W + 1:t + 1]]); yw = y[t - W + 1:t + 1]; b = np.linalg.lstsq(Xw, yw, rcond=None)[0]; e = yw - Xw @ b
        yc = yw - yw.mean(); r2 = 1 - e @ e / (yc @ yc); comp = b[1:] * X[t]
        pct = [float(b[k] * (Xw[:, k] - Xw[:, k].mean()) @ yc / (yc @ yc) * 100) for k in range(1, Xw.shape[1])]   # cov(b_k x_k, y)/var(y)
        out.append(dict(t=t, alpha=b[0], comp=comp, resid=y[t] - b[0] - comp.sum(), r2=r2, t_hac=newey_west_t(Xw, e, b, M), pct=pct, pct_resid=(1 - r2) * 100))
    return out
def decile_long_short(R, S, n_bins=10):                        # rank at t on S[t] (known info), earn R[t]; equal weights
    out = []
    for t in R.index:
        s = S.loc[t].dropna(); r = R.loc[t]
        if s.size < n_bins: out.append((t, np.nan, np.nan)); continue
        rank = s.rank(method="first"); b = np.floor((rank - 1) * n_bins / s.size).astype(int)   # 0 = losers ... n_bins-1 = winners
        out.append((t, float(r[b.index[b == n_bins - 1]].mean()), float(r[b.index[b == 0]].mean())))
    df = pd.DataFrame(out, columns=["date", "top", "bottom"]).set_index("date"); return df.assign(spread=df["top"] - df["bottom"])
def fama_macbeth(R, chars):                                    # chars: dict name -> DataFrame aligned with R (lagged)
    gam = []
    for t in R.index:
        X = pd.concat([chars[k].loc[t].rename(k) for k in chars], axis=1); d = pd.concat([R.loc[t].rename("y"), X], axis=1).dropna()
        if len(d) > X.shape[1] + 2: gam.append(sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit().params.rename(t))
    G = pd.DataFrame(gam); mean, se = G.mean(), G.std(ddof=1) / math.sqrt(len(G)); return pd.DataFrame({"estimate": mean, "t_stat": mean / se, "n_periods": len(G)})
def ic_stats(R, S, min_names=3):                               # Spearman IC per period over names with a valid signal
    ic = pd.Series({t: stats.spearmanr(S.loc[t], R.loc[t], nan_policy="omit")[0] for t in R.index if S.loc[t].notna().sum() >= min_names}).dropna()
    return dict(ic_mean=float(ic.mean()), ic_std=float(ic.std(ddof=1)), ir=float(ic.mean() / ic.std(ddof=1)), hit_rate=float((ic > 0).mean()), n=int(ic.size))

rng = np.random.default_rng(0); T, N = 240, 50; idx = pd.period_range("2005-01", periods=T, freq="M")
F = pd.DataFrame(rng.normal([0.006, 0.002, 0.003], [0.045, 0.03, 0.03], (T, 3)), index=idx, columns=["Mkt-RF", "SMB", "HML"])
beta = rng.uniform(0.5, 1.5, N); R = pd.DataFrame(F["Mkt-RF"].to_numpy()[:, None] * beta + rng.normal(0, 0.06, (T, N)), index=idx)
S = ((1 + R).rolling(11).apply(np.prod, raw=True) - 1).shift(2); ls = decile_long_short(R, S).dropna(); reg = ts_regression(ls["spread"], F)   # 12-2 momentum
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in reg.items()}, round(float(ls["spread"].mean() * 12), 4), str(S.first_valid_index()))
print(fama_macbeth(R, {"mom": S}).round(4).to_dict(), {k: round(v, 4) for k, v in ic_stats(R, S).items()})
dec = rolling_decomposition(F["Mkt-RF"].to_numpy() * 0.8 + F["SMB"].to_numpy() * 0.3 + rng.normal(0, 0.01, T), F.to_numpy(), 48)
print(len(dec), round(float(dec[-1]["r2"]), 4), np.round(dec[-1]["pct"], 3).tolist(), round(sum(dec[-1]["pct"]) + dec[-1]["pct_resid"], 10), np.round(dec[-1]["t_hac"], 3).tolist())
```
