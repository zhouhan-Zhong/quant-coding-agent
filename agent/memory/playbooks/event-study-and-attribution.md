---
title: Event studies, earnings surprises and Brinson attribution
category: backtest-factors
tags: event study, abnormal return, ar, car, caar, cumulative abnormal return, market model, estimation window, event window, out-of-sample residuals, t-test, bmp, boehmer, patell, standardized abnormal return, earnings surprise, sue, standardized unexpected earnings, beat, miss, consensus, guidance, guidance surprise, event-driven, event classification, alpha score, severity multiplier, 8-k events, brinson, brinson-fachler, brinson-hood-beebower, attribution, allocation effect, selection effect, interaction effect, benchmark, sector weights, cash drag, active return, month-end returns
summary: Market-model abnormal returns with CAR/BMP tests and day-0 alignment, SUE/surprise conventions, event-driven classification and scoring rules, and Brinson-Fachler allocation/selection/interaction with cash and drifting weights that reconcile exactly to the active return.
---
## When this applies
"Estimate abnormal returns around the announcement dates", "CAR over [-1,+1]/[-5,+5] and t-statistics",
"earnings surprise and SUE, classify beat/miss/meet", "classify 8-K events and score them into long/short/hold
signals", "sector attribution of the portfolio vs the benchmark
(allocation, selection, interaction)", "monthly attribution with quarterly rebalancing and a cash position".

## Method and formulas
Symbols: R_it stock return, R_mt market return (decimal, daily), day 0 = event day in trading-day index,
L = estimation-window length, w = event-window length; w_p, w_b = portfolio/benchmark sector weights at the
start of the period, R_p,i / R_b,i sector returns, R_b = sum_i w_b,i R_b,i benchmark return.
- Day-0 alignment: d0 = np.searchsorted(trading_dates, event_date, 'left') -> first trading day on/after the event
  (announcements after the close: use the next trading day if the task says so). Estimation window
  [d0-250, d0-31] (or as given), event window [d0-k, d0+k] inclusive (2k+1 days).
- Market model: regress R_i on R_m over the estimation window (OLS, intercept): alpha, beta;
  sigma_eps^2 = SSE/(L-2). Alternatives: constant-mean (AR = R_i - mean), market-adjusted (AR = R_i - R_m).
- AR_t = R_it - (alpha + beta R_mt); CAR = sum of AR over the window (no compounding; BHAR = prod(1+R_i) -
  prod(1+E[R]) only if asked); CAAR = mean of CAR across N events.
- Tests: t_AR = AR_t/sigma_eps; t_CAR = CAR/(sigma_eps sqrt(w)); SCAR_i = CAR_i/(sigma_eps,i sqrt(w)) (Patell
  adds the forecast-error correction (1 + 1/L + (R_m,t - mean R_m)^2/sum(R_m - mean)^2)); cross-sectional t for
  CAAR = mean(CAR)/(std(CAR, ddof=1)/sqrt(N)); BMP = mean(SCAR)/(std(SCAR, ddof=1)/sqrt(N)); two-sided p from the
  normal (or t with L-2 / N-1 df if specified).
- Earnings surprise: surprise = A - E; surprise_pct = (A - E)/|E| * 100; SUE = (A - E)/sd with sd = analyst
  dispersion or std of past surprises (sd == 0 -> 0); beat/miss if surprise_pct > +th / < -th (strict), else meet;
  YoY growth = (A - A_prior)/|A_prior| * 100; keep quarters in input order (positional matching).
- Event-driven scores (filings/news): classify each event with a fixed precedence cascade (e.g. officer-departure
  item + resignation sentence -> departure; "principal amount ... Notes" -> debt financing; else guidance vs
  consensus); guidance surprise % = (midpoint - consensus)/consensus x 100 (range -> (a+b)/2, "x plus or minus y%" ->
  x); score = base(type, direction) x severity multiplier (capped surprise size, role or deal-size bumps) x liquidity
  multiplier, exactly as the params define them; signal long if score >= +th, short if <= -th (inclusive), else hold;
  book net = sum of scores, gross = sum |scores|; rows in sorted event id order (parsing: text-and-filings-parsing).
- Market model fitted on an estimation (in-sample) period only and applied unchanged to later periods to build
  residual returns (lead-lag, residual signals); when the reference reports rounded alpha/beta, residualise with
  the rounded values.
- Brinson-Fachler (per period, beginning-of-period weights): allocation_i = (w_p,i - w_b,i)(R_b,i - R_b);
  selection_i = w_b,i (R_p,i - R_b,i); interaction_i = (w_p,i - w_b,i)(R_p,i - R_b,i); sum of the three over sectors
  = R_p - R_b exactly (requires sum w_p = sum w_b = 1). BHB variant: allocation_i = (w_p,i - w_b,i) R_b,i (same
  total, different per-sector split). Cash (w_b = 0): allocation = w_p,cash (R_cash - R_b), selection/interaction 0;
  a sector absent from the portfolio: allocation = -w_b,i (R_b,i - R_b), selection 0.
- Multi-period: effects computed each month with that month's starting weights and summed arithmetically
  (quarterly/annual totals = sums); portfolio and benchmark returns compounded: prod(1+R) - 1. Portfolio weights
  reset to targets on rebalance months and drift otherwise: w_i <- w_i (1 + R_p,i)/sum_j w_j (1 + R_p,j) (cash
  included with R_cash = annual rate/12). Benchmark weights from the task's schedule (per quarter).
- Monthly returns from daily closes: last close per calendar month (groupby(index.to_period('M')).last()),
  R_m = P_m/P_{m-1} - 1, anchored on the prior December for a January start.

## Conventions and pitfalls (what graders check)
- Window lengths are inclusive counts: [-5,+5] has 11 ARs; estimation window must not overlap the event window;
  report L and w as ints; drop events with insufficient history rather than shortening windows silently.
- Returns in decimal; CAR is a sum of decimals (reporting in percent only if asked); std ddof=1 for
  cross-sectional tests; use the SAME market series for estimation and event windows.
- Day-0 rule (on/after vs before) changes every AR - state it and follow the task text; weekends/holidays in the
  event list are the usual trap.
- Attribution reconciliation is asserted: allocation + selection + interaction == active return per period
  (1e-12) and summed over periods; cash contributes allocation only; weights are start-of-period, not average.
- Use the benchmark-sector return for the benchmark side and the portfolio's own instrument return for the
  portfolio side (sector ETF vs portfolio ETF mapping); a swapped mapping flips the selection sign.
- Earnings outputs keep input order; divide by |E| (negative consensus keeps the sign of the change); zero
  denominators -> 0.0 not inf; thresholds strict; rounding to 4 dp only when the spec says.
- Event classification is usually graded as exact lists (event types, item strings, signals) plus exact counts per
  type; numeric scores at rtol ~1e-3 - one misclassified event fails the whole unit, so test each rule on every event.
- JSON: lists for monthly effects in chronological order; floats via float(); no NaN.

## Tested snippet
```python
import math, numpy as np, pandas as pd
from scipy import stats

def event_study(r_i, r_m, dates, event_date, est=(-250, -31), win=(-5, 5)):
    """r_i, r_m: Series indexed by trading dates (sorted). Day 0 = first trading day >= event_date (searchsorted)."""
    d0 = int(np.searchsorted(dates, np.datetime64(event_date), side="left"))
    e = slice(d0 + est[0], d0 + est[1] + 1); X = np.c_[np.ones(e.stop - e.start), r_m.iloc[e]]
    beta, res, *_ = np.linalg.lstsq(X, r_i.iloc[e].to_numpy(), rcond=None)
    L = X.shape[0]; s2 = float(np.sum((r_i.iloc[e].to_numpy() - X @ beta) ** 2) / (L - 2))   # residual variance
    w = slice(d0 + win[0], d0 + win[1] + 1)
    ar = r_i.iloc[w].to_numpy() - (beta[0] + beta[1] * r_m.iloc[w].to_numpy())
    car = float(ar.sum()); n_w = len(ar)
    return dict(alpha=float(beta[0]), beta=float(beta[1]), ar=ar, car=car, t_car=car / math.sqrt(n_w * s2),
                scar=car / math.sqrt(n_w * s2), sigma_eps=math.sqrt(s2), day0=str(dates[d0])[:10])

def bmp_test(scars):                                            # Boehmer-Musumeci-Poulsen: cross-sectional t on SCARs
    s = np.asarray(scars); return float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))), float(2 * stats.norm.sf(abs(s.mean() / (s.std(ddof=1) / math.sqrt(len(s))))))

def sue(actual, expected, sd):                                  # standardized unexpected earnings; sd = analyst dispersion or past-surprise std
    return float((actual - expected) / sd) if sd != 0 else 0.0

def brinson_fachler(wp, wb, rp, rb):                            # dict sector -> beginning weights / returns, one period
    Rb = sum(wb[s] * rb[s] for s in wb); Rp = sum(wp[s] * rp[s] for s in wp); secs = sorted(set(wp) | set(wb))
    wp_, wb_ = ({s: d.get(s, 0.0) for s in secs} for d in (wp, wb))                     # missing sector -> weight 0 ...
    rb_, rp_ = {s: rb.get(s, rp.get(s)) for s in secs}, {s: rp.get(s, rb.get(s)) for s in secs}   # ... and inherits the other side's return
    alloc = {s: (wp_[s] - wb_[s]) * (rb_[s] - Rb) for s in secs}                         # CASH (wb = 0): wp_cash (r_cash - Rb)
    sel = {s: wb_[s] * (rp_[s] - rb_[s]) for s in secs}; inter = {s: (wp_[s] - wb_[s]) * (rp_[s] - rb_[s]) for s in secs}
    tot = sum(alloc.values()) + sum(sel.values()) + sum(inter.values())
    return dict(Rp=Rp, Rb=Rb, active=Rp - Rb, allocation=sum(alloc.values()), selection=sum(sel.values()),
                interaction=sum(inter.values()), reconcile=abs(tot - (Rp - Rb)), by_sector=dict(alloc=alloc, sel=sel, inter=inter))

rng = np.random.default_rng(0); dates = pd.bdate_range("2021-01-01", periods=600).to_numpy()
rm = pd.Series(rng.normal(0.0003, 0.01, 600)); ri = 0.0002 + 1.2 * rm + pd.Series(rng.normal(0, 0.015, 600)); ri.iloc[400] += 0.05
ev = event_study(ri, rm, dates, "2022-07-16")                   # a Saturday -> day 0 is the next trading day
print(ev["day0"], round(ev["beta"], 3), round(ev["car"], 4), round(ev["t_car"], 3), len(ev["ar"]))
print(bmp_test([0.8, 1.5, -0.2, 2.1, 0.9]), sue(1.10, 1.00, 0.05), sue(1.0, 1.0, 0.0))
bf = brinson_fachler({"Tech": 0.5, "Fin": 0.3, "CASH": 0.2}, {"Tech": 0.6, "Fin": 0.4}, {"Tech": 0.04, "Fin": 0.01, "CASH": 0.004}, {"Tech": 0.03, "Fin": 0.02})
print({k: round(v, 6) for k, v in bf.items() if k != "by_sector"})
```
