---
title: SABR and SVI smiles and volatility-surface conventions: Hagan lognormal and normal SABR formulas, calibration with fixed beta, raw SVI parametrisation and constraints, calendar and butterfly arbitrage checks, interpolation in total variance, moneyness and forward conventions
category: volatility
tags: sabr, hagan, alpha, beta, rho, nu, vol of vol, smile, skew, implied volatility, normal vol, lognormal vol, svi, raw svi, total variance, log moneyness, forward moneyness, volatility surface, vol surface, calendar arbitrage, butterfly arbitrage, durrleman, roger lee, wing, interpolation, extrapolation, calibration, least squares
summary: SABR vol from Hagan's formula with z/x(z) -> 1 at the money; calibrate alpha (often from the ATM vol), rho and nu with beta fixed; normal SABR / Bachelier vols for rates near zero; raw SVI w(k) = a + b(rho (k - m) + sqrt((k - m)^2 + sigma^2)) on k = ln(K/F) and total variance w = sigma_imp^2 T with b >= 0, |rho| < 1, sigma > 0, a + b sigma sqrt(1 - rho^2) >= 0, b(1 + |rho|) <= 2; no calendar arbitrage = w non-decreasing in T at fixed k; butterfly check via Durrleman's g(k) >= 0; interpolate in total variance against log-forward-moneyness.
triggers: text:sabr, text:svi, text:volatility surface, text:vol surface
---
## When this applies
"Calibrate SABR to the swaption / FX smile", "fit SVI slices", "check the surface for static arbitrage",
"interpolate the implied vol at a strike and maturity", "report RMSE of the fit in vol points".

## SABR (Hagan et al. 2002, lognormal implied vol)
- Inputs: forward F, strike K, expiry T, parameters alpha > 0, 0 <= beta <= 1, -1 < rho < 1, nu >= 0.
- fk = (F K)^((1 - beta) / 2); lfk = ln(F / K); z = (nu / alpha) fk lfk; x(z) = ln((sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho)).
- sigma_B = alpha / (fk (1 + (1-beta)^2/24 lfk^2 + (1-beta)^4/1920 lfk^4)) * (z / x(z))
  * (1 + ((1-beta)^2/24 alpha^2 / fk^2 + rho beta nu alpha / (4 fk) + (2 - 3 rho^2)/24 nu^2) T).
- At the money (K = F) use z / x(z) = 1: sigma_ATM = alpha / F^(1-beta) * (1 + (...) T) with fk = F^(1-beta); for |z| < 1e-7
  use the limit to avoid 0/0.
- Calibration: fix beta (task value; 0.5 rates, 1 equity/FX lognormal, 0 normal), fit (alpha, rho, nu) by least squares on
  vols (or alpha from the ATM vol by solving the cubic, then rho, nu); bounds rho in (-0.999, 0.999), nu > 0, alpha > 0.
- Prices: plug sigma_B into Black-76 with the same F, K, T and discount factor.  Normal (Bachelier) quotes are in absolute
  rate units (bps / 1e4) and need the normal SABR expansion or a Bachelier price.

## SVI (raw parametrisation, one slice per expiry)
- k = ln(K / F_T) with F_T = S exp((r - q) T) (or the given forward); w(k) = sigma_imp(k)^2 T.
- w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)); fit to market total variances (weights as stated, e.g. vega).
- Constraints: b >= 0, |rho| < 1, sigma > 0, a + b sigma sqrt(1 - rho^2) >= 0 (minimum variance non-negative), wing slopes
  b (1 + |rho|) <= 2 (Roger Lee bound in total-variance units).
- ATM total variance = a + b (-rho m + sqrt(m^2 + sigma^2)); implied vol = sqrt(w / T).

## Static-arbitrage checks
- Calendar: for every k, w(k, T2) >= w(k, T1) when T2 > T1 (compare on a common log-moneyness grid).
- Butterfly (per slice): g(k) = (1 - k w'/(2 w))^2 - (w'^2 / 4)(1/w + 1/4) + w''/2 >= 0 for all k (finite differences or
  the analytic SVI derivatives); equivalently call prices decrease in K and are convex in K.
- Report the number / location of violations rather than silently clipping.

## Interpolation and extrapolation
- Across maturities: linear in total variance w at fixed log-moneyness (or fixed delta), then sigma = sqrt(w / T).
- Across strikes: on the fitted smile (SABR / SVI) or a spline in (k, w) / (k, sigma); never interpolate in raw price
  space across maturities.  Beyond the quoted strikes extrapolate flat in implied vol unless the task says otherwise.

## Invariants worth printing
- Fitted vols positive and finite at every quoted strike; RMSE (vol points) and max abs error; ATM fitted vol within the
  tolerance of the quote; SVI constraint values (a + b sigma sqrt(1 - rho^2), b(1 + |rho|)); no calendar violations.
