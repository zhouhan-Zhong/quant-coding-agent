---
title: Realized volatility estimators (close-to-close, range-based, intraday)
category: volatility
tags: realized volatility, realized variance, historical volatility, close-to-close, parkinson, garman-klass, rogers-satchell, yang-zhang, gkyz, high low range, ohlc, intraday returns, bipower variation, jump detection, annualization, rolling window, volatility term structure, vol cone, ddof, volatility signature, microstructure noise, bandi-russell, sparse sampling, regular trading hours
summary: Exact formulas (including the Yang-Zhang k constant and ddof conventions) for every standard volatility estimator, intraday RV/BV, annualisation and rolling-window conventions.
---
## When this applies
"Estimate realized/historical volatility with Parkinson / Garman-Klass / Rogers-Satchell / Yang-Zhang", "compare
estimators", "rolling 21-day volatility", "volatility term structure or cone", "intraday realized variance and
bipower variation / jump component", "annualised vol from OHLC data", "RV at 1/2/5/15/30-minute sampling (volatility
signature)", "microstructure-noise variance and noise-corrected RV".

## Method and formulas
Daily bars i = 1..n with open O_i, high H_i, low L_i, close C_i and previous close C_{i-1}. Variances are DAILY; annualise
variance by x A (A = 252 trading days unless the task says 365/260) and vol = sqrt(A var). Logs everywhere.
- Close-to-close: r_i = ln(C_i/C_{i-1}); var_cc = (1/(n-1)) sum (r_i - rbar)^2 (pandas .std() default ddof=1);
  zero-mean variant (1/n) sum r_i^2 (use only if stated). Simple-return variant pct_change only if the task says "simple".
- Parkinson: var_P = (1/(4 n ln 2)) sum (ln(H_i/L_i))^2.
- Garman-Klass: var_GK = (1/n) sum [ 0.5 (ln(H_i/L_i))^2 - (2 ln 2 - 1) (ln(C_i/O_i))^2 ].
- Rogers-Satchell: var_RS = (1/n) sum [ ln(H_i/O_i) ln(H_i/C_i) + ln(L_i/O_i) ln(L_i/C_i) ] (drift-independent).
- Yang-Zhang: o_i = ln(O_i/C_{i-1}) (overnight), c_i = ln(C_i/O_i) (open-to-close);
  var_o = (1/(n-1)) sum (o_i - obar)^2, var_c = (1/(n-1)) sum (c_i - cbar)^2, k = 0.34 / (1.34 + (n+1)/(n-1));
  var_YZ = var_o + k var_c + (1 - k) var_RS. A window of n bars needs n+1 closes (for o_1).
- Garman-Klass-Yang-Zhang (GK with overnight): var = (1/n) sum [ o_i^2 + 0.5 (ln(H/L))^2 - (2 ln 2 - 1) c_i^2 ].
- Intraday (M returns r_j within a day): RV = sum r_j^2; bipower BV = (pi/2) sum_{j>=2} |r_j| |r_{j-1}| (Barndorff-Nielsen-
  Shephard definition; references omit the M/(M-1) small-sample factor - it moves BV by ~1.3% at M = 77, add it only if the
  task writes it); jump component J = max(RV - BV, 0); daily vol = sqrt(RV); annualise x 252. Realized quarticity
  RQ = (M/3) sum r_j^4 if a jump test statistic is requested.
- Intraday cleaning + sparse sampling (reference order): collapse duplicate timestamps (venue duplicates) with
  groupby(timestamp).median(); keep 09:30 <= time <= 16:00 INCLUSIVE (up to 391 one-minute prices); sort. The f-minute grid
  = rows with minute-of-session (60 h + m - 570) % f == 0 (pure subset, no forward fill: a missing minute drops a point);
  RV_f = sum (diff ln p)^2 within the day; n_f = the ACTUAL number of returns (76 instead of 77 when a bar is missing).
- Noise / volatility signature: RV_1min > RV_5min on most days signals microstructure noise. Bandi-Russell noise variance
  per day = RV_1min/(2 n_1min); bias-corrected RV_f = RV_f - 2 n_f * noise, where the reference used the GRAND MEAN of the
  daily noise estimates (not the day's own) and left it unclipped; ann vol = sqrt(max(RV_corr, 0) * 252); the grand vol is
  sqrt(max(mean_d RV_corr, 0) * 252) (not the mean of daily vols).
- Rolling: for window n, the label is the window's last date (right-aligned); min_periods = n; pandas
  rolling(n).std(ddof=1) for close-to-close; for range estimators apply the per-bar term then rolling(n).mean()
  (Parkinson/GK/RS are means of per-bar quantities, so rolling means are exact; YZ needs the window variances).
- Term structure / cone: evaluate each estimator over windows {5, 10, 21, 63, 126, 252} ending at the last date; the cone
  uses the distribution (min/quantiles/max) of each rolling series.
- Efficiency ratios: var_cc / var_est (Parkinson ~5.2, GK ~7.4 theoretical for continuous paths, lower in practice).

## Conventions and pitfalls (what graders check)
- ddof: close-to-close, var_o and var_c use ddof=1 (sample); Parkinson, GK, RS, GKYZ divide by n (not n-1); YZ k uses the
  window length n. Mixing these shifts results by ~n/(n-1).
- Clean OHLC first: H = max(H, O, C), L = min(L, O, C), drop rows with non-positive prices or H < L; zero-range bars give
  0 contributions (Parkinson) and can make RS slightly negative over tiny windows -> clip variance at 0 before sqrt.
- Discrete sampling bias: range estimators underestimate the true vol of a continuous process (max/min of 78 bars
  is below the true extreme: ~5-10% low); close-to-close is unbiased but noisy. Do not "correct" unless asked.
- Annualise once: sqrt(252 * daily_var); never sqrt(252) * sqrt(mean of annualised ...) twice. Report vol in decimals
  (0.2 = 20%) unless the task says percent.
- Overnight returns need the previous day's close aligned by shift(1) on the sorted date index; first row is NaN.
- Rolling outputs: drop the warm-up NaNs only if the task says so; otherwise keep rows with empty cells; sort by date.
- Intraday: group by trading date; returns within the day only (exclude the overnight jump unless told to include it);
  M = bars per day (78 for 5-minute US equity sessions of 6.5h).
- Typical asserted invariants: all vols > 0; YZ within ~30% of close-to-close over long windows; longer windows smoother
  (lower std of the rolling series); jump component >= 0; RV >= 0; column names exactly as specified.

## Tested snippet
```python
import math
import numpy as np, pandas as pd

def rv_estimators(o, h, l, c, c_prev=None, ann=252):
    """One OHLC window (n = len(c) bars) -> annualised vols sqrt(ann * var); c_prev = close before the window (YZ/GKYZ)."""
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c)); n = len(c)
    hl, co, ho, lo_, hc, lc = np.log(h / l), np.log(c / o), np.log(h / o), np.log(l / o), np.log(h / c), np.log(l / c)
    out = {"parkinson": np.sum(hl ** 2) / (4 * n * math.log(2)),
           "garman_klass": np.mean(0.5 * hl ** 2 - (2 * math.log(2) - 1) * co ** 2),
           "rogers_satchell": np.mean(ho * hc + lo_ * lc)}
    if c_prev is not None:
        cc = np.log(c / np.concatenate([[c_prev], c[:-1]]))
        out["close_to_close"] = np.var(cc, ddof=1)                               # sample variance (mean removed)
        oc = np.log(o / np.concatenate([[c_prev], c[:-1]]))                      # overnight log return
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        out["yang_zhang"] = np.var(oc, ddof=1) + k * np.var(co, ddof=1) + (1 - k) * out["rogers_satchell"]
        out["gkyz"] = np.mean(oc ** 2 + 0.5 * hl ** 2 - (2 * math.log(2) - 1) * co ** 2)
    return {k_: float(math.sqrt(max(v, 0.0) * ann)) for k_, v in out.items()}

def rolling_vol(df, window=21, ann=252):
    """df with columns open, high, low, close (DatetimeIndex sorted). Right-aligned rolling window labels."""
    rows = []
    for i in range(window, len(df)):                                             # window rows i-window .. i-1, prev close at i-window-1
        w = df.iloc[i - window:i]; cp = df["close"].iloc[i - window - 1] if i > window else None
        rows.append({"date": df.index[i - 1], **rv_estimators(w["open"], w["high"], w["low"], w["close"], cp, ann)})
    return pd.DataFrame(rows).set_index("date")

def realized_var_intraday(r_intraday, finite_sample=False):   # intraday log returns of ONE day
    r = np.asarray(r_intraday, float); m = len(r); f = m / (m - 1) if finite_sample else 1.0     # BN-S default: no M/(M-1)
    rv = float(np.sum(r ** 2)); bv = float(math.pi / 2 * f * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
    return rv, bv, max(rv - bv, 0.0)         # realized variance, bipower variation, jump component (daily, not annualised)

def intraday_rv_table(df, freqs=(1, 2, 5, 15, 30)):   # df: timestamp (datetime64), price at 1-minute resolution
    df = df.groupby("timestamp", as_index=False)["price"].median()                 # collapse venue duplicates
    tod = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df = df[(tod >= 570) & (tod <= 960)].assign(mos=tod - 570, date=df["timestamp"].dt.date).sort_values("timestamp")
    rows = []
    for d, g in df.groupby("date"):
        row = {"date": str(d)}
        for f in freqs:                                                          # sparse grid: minute-of-session % f == 0
            r = np.diff(np.log(g.loc[g["mos"] % f == 0, "price"].to_numpy())); row[f"RV_{f}"], row[f"n_{f}"] = float(r @ r), len(r)
        rows.append(row)
    out = pd.DataFrame(rows); noise = out["RV_1"] / (2 * out["n_1"])              # Bandi-Russell, per day
    out["RV_5_corr"] = out["RV_5"] - 2 * out["n_5"] * noise.mean()                 # grand-mean noise, unclipped
    return out
```
