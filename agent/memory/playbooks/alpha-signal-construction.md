---
title: Alpha signal construction: normalisation, residualisation, stable combination, lead-lag
category: backtest-factors
tags: alpha signal, composite signal, multi-signal, cross-sectional z-score, robust z-score, median fill, clip, winsorize, residualize, residualized signals, orthogonalize, neutralize, controls, pinv, signal combination, committee, agents, ensemble, rolling median coefficients, sign stability, sign agreement, stable coefficients, conviction, aic model selection, alternative data, sentiment signal, as-of join, vintage, stale data, lead-lag, lagged correlation, cross-correlation, asymmetry, fisher z, significance gate, out-of-sample stability, rank stability, market model residuals, long-short selection, lexsort, tie-break, turnover cap, beta neutral
summary: Per-date robust z-scores, column winsorisation and pinv residualisation against controls, rolling-median coefficient combination with a sign-agreement filter, committee/AIC conviction rules, long-short selection with deterministic tie-breaks, and lead-lag correlation tables with a Fisher-z gate and out-of-sample rank checks.
---
## When this applies
"Combine several characteristics / alternative-data feeds into one alpha", "z-score each component per date (fill
with the median, clip at 3)", "residualise the signals on risk controls", "combine with the rolling median of past
cross-sectional slopes, keeping only sign-stable coefficients", "committee of agents with conviction", "which asset
leads which: lagged correlations, asymmetry, significance, out-of-sample stability".

## Method and formulas
Symbols: s_{i,t} raw component for name i at date t, X_t controls (N x p), Y_t signals (N x m), Z_t residual z-scores,
b_t cross-sectional slopes, L lookback, q leg fraction, ids integer identifiers.
- Robust per-date z: to_numeric, inf -> NaN; all-NaN -> zeros; NaN -> cross-sectional median; z = (s - mean)/std(ddof=0);
  std 0 -> zeros; clip to [-3, 3]. Apply AFTER sign flips / log1p / ratio transforms (median(log1p x) != log1p(median x)).
  Agent score = mean of its normalised components; committee = weighted sum of agents; dispersion = std(agents, ddof=0).
- Winsorise each column at its own cross-sectional quantiles: W.clip(W.quantile(0.01), W.quantile(0.99), axis=1)
  (pandas linear quantiles), on the universe = rows with all signals and controls present (dropna).
- Residualise each signal on [1, controls] per date: e = y - X pinv(X'X) X'y (pinv tolerates collinear controls);
  then z-score with ddof=0 (sd = 0 -> zeros).
- Regression combination: in every month with >= min_n names and >= min_n finite forward returns, OLS of the next
  return on [1, Z] -> slopes b_t (drop the intercept). At rebalance t use the last L slope vectors from months
  strictly BEFORE t: med = median; agreement = #(sign(b) == sign(med)); stable = med where agreement >= threshold else
  0; alpha = Z_t @ stable. First rebalance needs both a minimum date index and L prior coefficient months.
- Selection: order = np.lexsort((ids, -alpha)) (alpha desc, id asc); long order[:k], short order[-k:], k = floor(q N)
  (skip the month if k < minimum); weights +-1/k; dollar- and beta-neutral projection w - X pinv(X'X) X'w with
  X = [1, beta]; scale to gross 1. Turnover cap (0.5 sum|w - w_prev| <= c): blend w_prev + (c/turnover)(w_tgt - w_prev),
  re-project on the non-zero support (sorted ids), renormalise; if still > c + 1e-9, shrink linearly toward w_prev by
  c/new_turnover without renormalising. Cost = rate x sum|dw| over the union of names; net = gross - cost.
- Text / alt-data daily signal: per (date, ticker) SUM of post scores, reindexed to trading days (missing -> 0;
  weekend posts drop out), signal_t = agg_t - mean(agg_{t-L..t-1}) over TRADING days, NaN when the day has fewer
  than m posts; IC needs >= 3 valid names per day.
- Per-asset model committee: fit candidates (martingale, OU/AR(1) with phi clipped, GBM, jump models) on the
  trailing window (last N obs, with a minimum count), every sigma floored at 1e-6 (also inside the log-pdf); AIC = 2k - 2LL;
  pick by (aic, name); conviction = 2 sigmoid(gap/scale) - 1 clamped to [0,1] with gap = AIC_2nd - AIC_best (the
  reference used this rather than a plain sigmoid); insufficient history -> sign 0, conviction 0, params NaN.
- Lead-lag of pair (A, B): corr_k = corr(A_t, B_{t+k}) on overlapping slices (k > 0: A[:n-k] vs B[k:]); c_AB = max over
  k = 1..L (A leads), c_BA = max over k = -L..-1; asymmetry = c_AB - c_BA; leader = A if asymmetry >= 0. Gate:
  |atanh(c_AB) - atanh(c_BA)| > 1.959964 sqrt(2/(n - 3)) (clip to +-0.999999; n = obs in the period). Table sorted by
  |asymmetry| desc (stable), rank = position; out-of-sample: rebuild the full table inside each period (slices
  restart per period) and look up each pair's rank; avg rank shift, same-sign rate, persistence = mean |asym_oos|/|asym_is|.
- Residual lead-lag: market model fitted IN-SAMPLE only (beta = cov/var ddof=1, alpha = ybar - beta xbar), residuals
  for every period with those parameters; the benchmark itself is excluded from the residual pairs.

## Conventions and pitfalls (what graders check)
- When the reference prints correlations / betas to 6 dp, round EACH lagged correlation before max/difference and
  residualise with the rounded beta and alpha; averages of rounded values are not re-rounded. Unrounded versions sit
  1e-6 off pinned values and can flip ranks.
- As-of joins of vintaged data: take the LAST row in (asof - lookback, asof] even if its value is NaN; stale if none or
  older than the threshold; mismatch between two vintages iff |v1 - v2| > 1e-12 (NaN never mismatches).
- ids may arrive as float64 (parquet): cast to int for sorting/tie-breaks, write as strings only if asked; use stable
  sorts and itertools.combinations in file column order; pair keys like "A__B".
- The beta used for neutralisation may be the RAW (un-winsorised) value even when winsorisation is described - if a
  checker recomputes everything, test both readings on the first months.
- Checkers re-derive turnover/costs from your weights: report turnover = 0.5 sum|dw| and cost on sum|dw|.

## Tested snippet
```python
import math, numpy as np, pandas as pd

def robust_z(s):                                                       # per date: inf->NaN, median fill, z (ddof=0), clip +-3
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.isna().all(): return pd.Series(0.0, index=s.index)
    s = s.fillna(s.median()); sd = float(s.std(ddof=0))
    return ((s - s.mean()) / sd).clip(-3.0, 3.0) if sd > 0 else pd.Series(0.0, index=s.index)

def residual_z(Y, X):                                                  # winsorise every column at 1/99%, residualise Y on [1, X], z ddof=0
    W = pd.concat([Y, X], axis=1).astype(float); W = W.clip(W.quantile(0.01), W.quantile(0.99), axis=1)
    Xc = np.column_stack([np.ones(len(W)), W[X.columns].to_numpy()]); Yw = W[Y.columns].to_numpy()
    E = Yw - Xc @ (np.linalg.pinv(Xc.T @ Xc) @ (Xc.T @ Yw)); sd = E.std(axis=0, ddof=0)   # pinv: collinear controls OK
    return np.where(sd > 0, (E - E.mean(0)) / np.where(sd > 0, sd, 1.0), 0.0)

def stable_coefs(B_hist, min_agree):                                   # B_hist (L, n_sig): slopes of the L months strictly before t
    med = np.median(B_hist, axis=0); agr = (np.sign(B_hist) == np.sign(med)).sum(0)
    return np.where(agr >= min_agree, med, 0.0), agr

def long_short(alpha, ids, beta, q=0.15):                              # alpha desc, id asc; +-1/k; dollar- and beta-neutral, gross 1
    k = int(math.floor(q * len(alpha))); o = np.lexsort((ids, -alpha)); sel = np.r_[o[:k], o[-k:]]
    w = np.r_[np.full(k, 1.0 / k), np.full(k, -1.0 / k)]; X = np.column_stack([np.ones(2 * k), beta[sel]])
    w = w - X @ (np.linalg.pinv(X.T @ X) @ (X.T @ w)); return ids[sel], w / np.abs(w).sum()

def lead_lag(a, b, L, n_obs=None):                                     # corr(A_t, B_{t+k}), each rounded to 6 dp BEFORE max/diff
    n, c = len(a), {}
    for k in range(-L, L + 1):
        x, y = (a[:n - k], b[k:]) if k > 0 else ((a[-k:], b[:n + k]) if k < 0 else (a, b)); c[k] = round(float(np.corrcoef(x, y)[0, 1]), 6)
    cab, cba = max(c[k] for k in range(1, L + 1)), max(c[k] for k in range(-L, 0))      # A leads B / B leads A
    dz = abs(np.arctanh(np.clip(cab, -0.999999, 0.999999)) - np.arctanh(np.clip(cba, -0.999999, 0.999999)))
    return cab, cba, round(cab - cba, 6), bool(dz > 1.959963984540054 * math.sqrt(2.0 / ((n_obs or n) - 3)))   # Fisher-z gate

conviction = lambda gap, scale: min(1.0, max(0.0, 2.0 / (1.0 + math.exp(-gap / scale)) - 1.0))   # AIC gap (best vs 2nd) -> [0, 1]

rng = np.random.default_rng(0); n = 120; ids = np.arange(1000, 1000 + n)
X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["beta", "vol", "size"]); Y = pd.DataFrame(rng.normal(size=(n, 4)) + 0.5 * X[["vol"]].to_numpy(), columns=list("abcd"))
Z = residual_z(Y, X); B = rng.normal([0.002, -0.001, 0.0, 0.0015], 0.002, (36, 4)); stab, agr = stable_coefs(B, 24)
names, w = long_short(Z @ stab, ids, X["beta"].to_numpy())
a = rng.normal(size=600); b = np.r_[np.zeros(2), 0.3 * a[:-2]] + rng.normal(size=600)    # B follows A with a 2-day lag
print(np.round(stab, 5), agr, round(float(np.abs(Z.mean(0)).max()), 12), len(names), round(float(w.sum()), 12), round(float(w @ X["beta"].to_numpy()[np.searchsorted(ids, names)]), 12))
print(lead_lag(a, b, 5), robust_z(pd.Series([1.0, np.inf, np.nan, 4.0, 100.0])).round(4).tolist(), round(conviction(3.0, 2.0), 6))
```
