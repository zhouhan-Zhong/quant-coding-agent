---
title: Stress testing and scenario P&L: historical and hypothetical scenarios, sensitivity aggregation, correlation stress, reverse stress
category: risk
tags: stress test, stress testing, scenario analysis, scenario pnl, stressed pnl, historical scenario, hypothetical scenario, shock, factor shock, parallel shift, twist, steepener, flattener, worst case, worst scenario, risk contributors, top contributors, sensitivities, delta, dv01, cs01, vega, greeks aggregation, linear approximation, full revaluation, correlation stress, nearest psd, stressed var, stressed covariance, reverse stress test, liquidity horizon, square root of time, loss positive, 2008, covid, taper tantrum, scenario matrix, positions, market values
summary: Scenario P&L = positions x scenario returns (or sensitivities x shocks, with the second-order term), worst-window historical scenarios by compounding, per-position contributions that sum exactly to the scenario loss, correlation/vol stress with a PSD repair, stressed VaR and reverse stress, with the sign/unit conventions graders test.
---
## When this applies
"Compute the portfolio P&L under each stress scenario and rank them", "identify the largest risk contributors under the
worst scenario", "apply historical crisis windows to today's positions", "stress rates +100 bp / equities -20% / vol
+5 points", "stressed VaR / stressed correlations", "reverse stress test: what shock produces a loss of L".

## Method and formulas
Symbols: MV_i market value of position i (base currency, signed), r_i^s scenario return of asset/factor i in scenario s,
Delta_f sensitivity to factor f (DV01 per bp, equity delta in currency per 1% or per unit, vega per vol point).
- Linear scenario P&L: PnL_s = sum_i MV_i r_i^s (equities/funds/FX) or sum_f Delta_f x shock_f (rates in bp x DV01 per bp;
  vol points x vega; FX % x delta). Second order when Gammas are given: + 1/2 sum Gamma_f shock_f^2. Full revaluation
  when a pricer exists: PnL_s = V(theta + shock) - V(theta). Matrix form: PnL = Scenarios (S x F) @ Sensitivities (F).
- Contributions: contrib_{s,i} = MV_i r_i^s; they sum exactly to PnL_s (assert). Largest contributors = the most
  negative contributions (sort ascending; "top 3 risk contributors" = 3 largest losses, by absolute value only if the
  task says so). Percent contribution = contrib / PnL_s (only when PnL_s != 0).
- Historical scenarios: (a) a named window [d0, d1]: scenario return per asset = prod(1 + r_t) - 1 over the window
  (or sum of log returns, or sum of simple returns - follow the task), applied to today's MV; (b) worst h-day window:
  rolling h-day compounded returns, PnL per window end date, take argmin; report the window dates (end date, or start
  and end as the task formats them).
- Hypothetical shocks: parallel shift (all tenors + x bp), twist/steepener (short - a, long + b, linear in tenor or by
  pivot), equity crash (-20%), vol up (+5 points absolute, or x 1.5 relative - check wording), FX (+/-10% of the
  base-per-foreign rate; direction matters for foreign assets: domestic value = MV_foreign x S), credit spreads + x bp
  x CS01. Combined scenarios add the factor P&Ls (linear) or revalue jointly.
- Correlation stress: rho_s = rho + s (1 - rho) for off-diagonals (push toward 1), or set a block to a given value;
  repair PSD by eigenvalue clipping (eigh, clip at >= 1e-10, rescale to unit diagonal); covariance = D rho_s D with the
  (possibly stressed) vols D. Stressed VaR_99 = 2.3263 sqrt(w' Sigma_s w) for a normal model, or the VaR estimator of
  the task on the stress-period window (e.g. the 250 days around 2008).
- Reverse stress (linear book): along a shock direction d, the scale k with loss L*: k = -L*/(Delta . d); with several
  factors the minimal-norm shock is Delta L*/(Delta . Delta) (sign so the P&L equals -L*).
- Horizon scaling: h-day normal VaR = 1-day x sqrt(h) (no scaling for historical windows already h days long).
- Reporting: table scenario x (pnl, pnl_pct = pnl/total MV, rank), worst scenario name, contributions table for the
  worst scenario sorted by contribution; stressed vs base VaR ratio.

## Conventions and pitfalls (what graders check)
- Sign: scenario P&L is NEGATIVE for losses unless the task defines "loss" (positive). VaR/ES are loss-positive. Never
  mix: `stressed_loss = -pnl` when the template says loss.
- Units: shocks in the file may be in percent (-20) or decimals (-0.20) or bp (+100): read the header/task text and
  convert ONCE. Rates shocks apply to DV01 per bp (100 bp x DV01 = P&L), not to the price directly. Vol shocks are
  additive points of implied vol (0.20 -> 0.25) unless "relative".
- Invariants asserted: contributions sum to the scenario P&L (1e-8 relative), the worst scenario is the argmin of the
  written P&Ls, ranks consistent with the sorted P&Ls, stressed VaR >= base VaR when correlations and vols are pushed
  up, stressed covariance PSD, pnl_pct = pnl / total MV with the task's total (gross vs net - check), VaR99 >= VaR95,
  ES >= VaR, h-day scaling only where stated.
- Positions files: market value = quantity x price (x FX to base); short positions negative MV; cash earns nothing in
  stress unless a rate shock applies to it (follow the task).
- Historical windows: inclusive dates; use trading days present in the data; compounding vs summing changes results by
  ~1% for large moves - take the formula the task writes; missing assets in the window -> state the rule (0 or drop).
- Correlation stress must keep the diagonal at 1 and the matrix PSD; a non-PSD matrix gives sqrt of negatives (NaN) and
  fails the checker.
- Ranking ties: sort by pnl then by scenario name for determinism; "top 3" when fewer than 3 positions -> all of them.
- Output hygiene: scenario names exactly as in the input file, pnl as floats in base currency (not thousands unless
  told), ranks as ints starting at 1, dates ISO.

## Tested snippet
```python
import numpy as np, pandas as pd
from scipy.stats import norm
rng = np.random.default_rng(1)
assets = ["EQ_US", "EQ_EU", "UST10", "IG_CREDIT", "GOLD"]
pos = pd.Series([40e6, 25e6, 20e6, 10e6, 5e6], index=assets)                           # market values (base currency)
rets = pd.DataFrame(rng.normal(0, [0.012, 0.013, 0.004, 0.003, 0.010], (750, 5)), columns=assets, index=pd.bdate_range("2021-01-04", periods=750))
scen = pd.DataFrame({"EQ_US": [-0.20, -0.05, 0.0], "EQ_EU": [-0.22, -0.06, 0.0], "UST10": [0.03, 0.01, -0.02],   # scenario x asset returns
                     "IG_CREDIT": [-0.04, -0.01, -0.01], "GOLD": [0.05, 0.02, -0.03]}, index=["crash", "mild_selloff", "rates_up"])
pnl = scen @ pos                                                                           # linear P&L = sum position_i x return_i
contrib = scen.mul(pos, axis=1)                                                            # per-asset contributions (sum = scenario P&L)
worst = pnl.idxmin(); top3 = contrib.loc[worst].sort_values().head(3)                      # most negative contributors first
print(pnl.round(0).to_dict(), worst, top3.round(0).to_dict(), bool((contrib.sum(axis=1) - pnl).abs().max() < 1e-6))
h = 10; r10 = (1 + rets).rolling(h).apply(np.prod, raw=True) - 1                           # compounded h-day returns per asset
hist_pnl = r10.dropna() @ pos; w_end = hist_pnl.idxmin()
print(w_end.date(), round(float(hist_pnl.min()), 0), f"= worst {h}-day historical P&L (window ending that day)")
z = norm.ppf(0.99); cov = rets.cov(ddof=1); sig = float(np.sqrt(pos @ cov @ pos))
var99 = z * sig; es99 = norm.pdf(z) / 0.01 * sig
comp = pos * (cov @ pos) / sig * z                                                         # Euler components: sum exactly to VaR
print(round(var99, 0), round(es99, 0), bool(es99 >= var99), bool(abs(comp.sum() - var99) < 1e-6))
corr = rets.corr().to_numpy(); s = 0.5; stressed = corr + s * (1 - corr); np.fill_diagonal(stressed, 1.0)   # push correlations toward 1
w, v = np.linalg.eigh(stressed); stressed = v @ np.diag(np.clip(w, 1e-10, None)) @ v.T                        # nearest-PSD clip
d = np.sqrt(np.diag(stressed)); stressed = stressed / np.outer(d, d)
vol = rets.std(ddof=1).to_numpy(); cov_s = stressed * np.outer(vol, vol)
var_s = z * float(np.sqrt(pos.to_numpy() @ cov_s @ pos.to_numpy()))
print(round(var_s, 0), "= stressed-correlation VaR99;", bool(var_s >= var99), "(>= base when correlations rise)")
L_star = 5e6; delta = pos.to_numpy(); k = L_star / (delta @ delta)                   # minimal-norm shock vector giving P&L = -L_star
print(np.round(-k * delta * 100, 3).tolist(), "% asset shocks = reverse stress for a 5e6 loss; check:", round(float(delta @ (-k * delta)), 2))
```
