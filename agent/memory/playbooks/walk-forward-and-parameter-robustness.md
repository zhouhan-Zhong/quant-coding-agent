---
title: Walk-forward optimisation, rolling in-sample/out-of-sample evaluation and parameter robustness
category: backtest-factors
tags: walk-forward, walk forward optimization, wfo, in-sample, out-of-sample, oos, is, training window, test window, rolling window, anchored window, expanding window, parameter grid, grid search, lookback, overfitting, parameter stability, robustness, stitched equity, oos sharpe, walk-forward efficiency, cross-validation, time series split, purged, embargo, combinatorial purged cv, cpcv, probability of backtest overfitting, pbo, deflated sharpe, mean reversion, z-score, bollinger, moving average, rebalance, warm-up
summary: Window arithmetic for rolling/anchored walk-forward (train W_tr, test W_te, step W_te), parameter selection on the train slice only, OOS stitching and Sharpe, walk-forward efficiency, parameter-stability tables, PBO/deflated Sharpe diagnostics, and the no-overlap/no-look-ahead invariants graders assert.
---
## When this applies
"Walk-forward optimisation of a lookback/threshold parameter", "fit on a 12-month window, trade the next 3 months, roll
forward", "report the out-of-sample Sharpe and the parameter chosen in each window", "in-sample vs out-of-sample
performance", "parameter sensitivity / robustness heatmap", "probability of backtest overfitting".

## Method and formulas
Symbols: N bars, W_tr train length, W_te test length (rows or calendar months), grid G of parameter values.
- Rolling windows: window k has train [s_k - W_tr, s_k) and test [s_k, s_k + W_te) with s_k = W_tr + k W_te, k = 0..K-1,
  K = floor((N - W_tr)/W_te) (a last partial test window is dropped unless the task keeps it). Anchored (expanding):
  train [0, s_k). Calendar version: month-end boundaries with pd.DateOffset(months=...); test months contiguous.
- For each window: compute the objective (Sharpe, total return, Calmar, profit factor) for every g in G on the TRAIN
  slice only; pick argmax (ties -> the smaller/first parameter in grid order, or the task's rule); apply g* to the test
  slice. Indicators on the test slice may warm up on prices BEFORE the test start (the lookback itself is not
  look-ahead), but the parameter choice must not touch test data. Signal at close t -> position from t+1 (shift(1)).
- Stitched OOS series: concatenate the test-slice returns in time order (each bar appears once) -> equity =
  cumprod(1 + r); OOS Sharpe = mean(r)/std(r, ddof=1) x sqrt(252) (daily) or sqrt(12) (monthly) - use the task's ddof and
  periods; annualised return = (prod(1 + r))^(252/n) - 1; max drawdown from the equity path.
- Walk-forward efficiency (WFE) = OOS annualised return / mean IS annualised return of the chosen parameters (or the
  Sharpe ratio version); WFE > 0.5 is the usual "robust" threshold. Report IS and OOS per window in a table.
- Parameter stability: the chosen g* per window, number of changes, fraction of windows at the modal value; robustness
  heatmap = objective over the whole grid on the full sample (or per window): a "plateau" means neighbours within x%.
- Combinatorial purged CV (Lopez de Prado): split into S blocks; for each of C(S, S/2) train combinations choose the
  best config IS, rank its OOS performance among all configs; logit lambda = ln(w/(1 - w)) of the relative rank;
  PBO = fraction of lambda < 0. Purging drops train bars whose labels overlap the test period; an embargo adds a gap.
- Deflated Sharpe: SR0 = sqrt(V[SR_n]) ((1 - gamma) Phi^{-1}(1 - 1/N) + gamma Phi^{-1}(1 - 1/(N e))), gamma = 0.5772;
  DSR = Phi[(SR - SR0) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)] with skew g3, kurtosis g4 of the returns.
- Mean-reversion example strategy: z = (P - MA_L)/SD_L; long when z < -k, short when z > k, flat otherwise (or exit at
  z crossing 0); Bollinger variants use k = 2; the walk-forward parameter is usually L (and/or k).

## Conventions and pitfalls (what graders check)
- Invariants asserted: test windows do not overlap and every OOS bar belongs to exactly one window; the number of
  windows K matches the formula; train/test boundaries as dates match the task's calendar; parameters in the output
  table belong to the grid (ints when the grid is ints); OOS Sharpe recomputed from the stitched returns equals the
  reported one; IS metrics computed only on train bars; no NaN rows in the stitched series (drop warm-up NaNs before
  metrics, keep the bar count the task expects).
- Look-ahead: using the whole sample to standardise (z-scores, scalers) before splitting leaks; fit scalers on the
  train slice. Indicator warm-up may use pre-test prices only when the task allows "price history before the window".
- Row-based vs calendar windows differ (126 rows != 6 months); follow the task's unit; months use calendar month ends.
- Rolling vs anchored: "expanding window" means anchored training; "rolling 12 months" means a fixed-length train.
- Ties in the objective: deterministic rule (first in grid order); report the objective value used for the choice.
- Annualisation: daily x sqrt(252), monthly x sqrt(12), weekly x sqrt(52); Sharpe with risk-free subtracted only when the
  task gives one; ddof=1 unless stated (pandas std default) - the pinned essentials apply.
- OOS period starts at s_0 = W_tr: the stitched equity has N - W_tr (rounded down to K W_te) bars, not N.
- Output hygiene: one row per window with window index (0- or 1-based as shown), train_start/train_end/test_start/test_end
  as ISO dates, chosen parameter, IS and OOS metric; summary JSON with oos_sharpe, n_windows, wfe as floats/ints.

## Tested snippet
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500))), index=pd.bdate_range("2018-01-01", periods=1500))
ret = px.pct_change().fillna(0.0)
def strat_returns(px, ret, L, k=1.0):                     # z-score mean reversion: signal at close t, position held over t+1
    ma = px.rolling(L).mean(); sd = px.rolling(L).std(ddof=1); z = (px - ma) / sd
    pos = pd.Series(np.where(z < -k, 1.0, np.where(z > k, -1.0, 0.0)), index=px.index)
    return (pos.shift(1) * ret).fillna(0.0)                 # one-bar lag: no look-ahead
def sharpe(r): return float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if r.std(ddof=1) > 0 else 0.0
grid, W_tr, W_te = [10, 20, 40, 60], 504, 126               # lookbacks; train 2 years, test 6 months (rows); rolling windows
oos, rows = [], []
for s in range(W_tr, len(px) - W_te + 1, W_te):             # test [s, s + W_te), train [s - W_tr, s)
    tr_idx, te_idx = px.index[s - W_tr:s], px.index[s:s + W_te]
    scores = {L: sharpe(strat_returns(px[:tr_idx[-1]], ret[:tr_idx[-1]], L)[tr_idx]) for L in grid}   # train slice only
    best = max(grid, key=lambda L: (scores[L], -L))                                                     # ties -> smaller L
    r_te = strat_returns(px[:te_idx[-1]], ret[:te_idx[-1]], best)[te_idx]   # warm-up uses pre-test prices; the choice does not
    oos.append(r_te); rows.append(dict(test_start=te_idx[0].date(), test_end=te_idx[-1].date(), lookback=best,
                                       is_sharpe=round(scores[best], 3), oos_sharpe=round(sharpe(r_te), 3)))
oos = pd.concat(oos); wf = pd.DataFrame(rows)
print(wf.to_string(index=False))
print("windows:", len(wf), "= floor((N - W_tr)/W_te) =", (len(px) - W_tr) // W_te, "| oos bars:", len(oos), "| unique:", bool(oos.index.is_unique),
      "| stitched OOS Sharpe:", round(sharpe(oos), 3), "| mean IS Sharpe:", round(float(wf["is_sharpe"].mean()), 3),
      "| parameter changes:", int((wf["lookback"].diff().fillna(0) != 0).sum()))
```
