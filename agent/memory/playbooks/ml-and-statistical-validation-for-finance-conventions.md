---
title: Machine learning and statistical validation conventions for finance: time-ordered splits, purging and embargo, leakage-free preprocessing, sklearn regularisation scales, default regularised logistic regression, seeds, classification and out-of-sample R2 metrics, probabilistic and deflated Sharpe ratios, bootstrap confidence intervals, multiple testing
category: backtest-factors
tags: machine learning, sklearn, scikit-learn, train test split, time series split, cross validation, purged k fold, purging, embargo, leakage, look-ahead, standardization, standard scaler, pca, ridge, lasso, elastic net, alpha, logistic regression, penalty, random forest, gradient boosting, random state, seed, accuracy, precision, recall, f1, roc auc, log loss, out of sample r2, hit rate, sharpe ratio, probabilistic sharpe ratio, psr, deflated sharpe ratio, dsr, bootstrap, block bootstrap, confidence interval, bonferroni, benjamini hochberg, p-value
summary: Split by time (no shuffling), purge training samples whose label window overlaps the test fold and embargo a gap after it; fit scalers / PCA / feature selection on the training fold only; sklearn Ridge minimises ||y - Xb||^2 + alpha ||b||^2 but Lasso minimises ||y - Xb||^2 / (2n) + alpha ||b||_1; LogisticRegression is L2-regularised with C = 1 by default (penalty=None for plain MLE); fix random_state; ROC AUC needs scores not labels; OOS R2 = 1 - SSE / SST against the stated benchmark mean; PSR uses the non-annualised Sharpe with skew and (non-excess) kurtosis; bootstrap with a seeded generator and the stated number of resamples.
triggers: text:cross-validation, text:cross validation, text:purged, text:embargo, text:lasso, text:ridge regression, text:random forest, text:logistic regression, text:deflated sharpe, text:probabilistic sharpe
---
## When this applies
"Train a classifier on features to predict next-day direction", "walk-forward / purged k-fold evaluation", "ridge / lasso
factor model", "report AUC, accuracy, out-of-sample R2", "is the Sharpe ratio significant (PSR / DSR)", "bootstrap a CI".

## Splits and leakage
- Features at t use information available at t; labels are forward returns (shift(-h)); drop the rows whose label is NaN.
- Never shuffle time series: `KFold(shuffle=False)` or `TimeSeriesSplit(n_splits)`; walk-forward = expanding or rolling
  training window followed by the next test block.
- Purging: remove training observations whose label interval [t, t + h] overlaps any test observation's interval.
  Embargo: also remove the first ceil(pct x N) observations after each test block (pct e.g. 1 %).
- Fit StandardScaler / PCA / imputers / feature selection on the training fold, then transform the test fold.

## Model defaults that change numbers
- Ridge(alpha): ||y - Xb||^2 + alpha ||b||^2; Lasso(alpha) and ElasticNet: (1 / (2 n)) ||y - Xb||^2 + alpha penalty -> the
  same alpha means different shrinkage; intercepts are not penalised (fit_intercept=True); standardise X first if stated.
- LogisticRegression default = L2 with C = 1.0 (C = 1 / lambda); plain maximum likelihood = penalty=None (sklearn >= 1.2)
  or statsmodels Logit.  Class weights only if stated.
- Tree ensembles and anything random: set random_state / seed exactly as stated; report n_estimators, max_depth used.

## Metrics
- Classification: accuracy, precision / recall / F1 for the positive class the task names; ROC AUC from predicted
  probabilities (`predict_proba[:, 1]`); log loss with probabilities clipped to [1e-15, 1 - 1e-15].
- Regression: OOS R2 = 1 - sum (y - yhat)^2 / sum (y - ybar)^2 where ybar is the benchmark the task names (historical
  training mean -> Campbell-Thompson; test-sample mean -> sklearn r2_score); hit rate = share of sign(yhat) == sign(y).
- Strategy from predictions: position at t from the prediction made with data up to t, applied to the return of t+1.

## Sharpe-ratio statistics
- Annualised Sharpe = mean(excess) / std(excess, ddof=1) x sqrt(periods per year).
- PSR(SR*) = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2) ) with SR per period (not annualised), g3 =
  skewness, g4 = kurtosis (normal = 3, i.e. NOT excess kurtosis).
- DSR = PSR with SR* = sqrt(V) ((1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e))), gamma = 0.5772..., V = variance of
  the N trial Sharpe ratios.
- t-stat of the mean return = mean / (std / sqrt(T)); Lo (2002) adjusts for autocorrelation if asked.

## Bootstrap and multiple testing
- iid bootstrap: resample T observations with replacement; block / stationary bootstrap keeps blocks of length L
  (state L).  rng = np.random.default_rng(seed); B resamples as stated; percentile CI = quantiles (alpha/2, 1 - alpha/2)
  of the B statistics (np.quantile default linear interpolation).
- Bonferroni: p_adj = min(1, m p); Benjamini-Hochberg: sort p ascending, largest k with p_(k) <= k q / m rejects 1..k.

## Invariants worth printing
- Max training date < min test date (minus the embargo) for every fold; number of test predictions = test rows; metrics
  in range (AUC in [0, 1]); the same seed reproduces the same numbers twice.
