---
title: Implied volatility solvers and closed-form approximations
category: volatility
tags: implied volatility, implied vol, iv, newton-raphson, vega, brentq, bisection, no-arbitrage bounds, brenner-subrahmanyam, corrado-miller, li approximation, hallerbach, deep itm, deep otm, inversion, root finding, option chain, black-76, forward, total variance, model prices
summary: Robust BS inversion (safeguarded Newton + brentq, bounds, ITM-to-OTM parity trick) and the ATM/non-ATM approximation formulas with their failure modes.
---
## When this applies
"Compute implied volatility from option prices", "invert Black-Scholes", IV smiles/surfaces from a chain,
"compare approximation formulas (Brenner-Subrahmanyam, Corrado-Miller, Li) with Newton", convergence flags,
RMSE of approximations against the true sigma on a synthetic grid.

## Method and formulas
Discounted spot Sd = S e^{-qT}, discounted strike Kd = K e^{-rT}, forward F = S e^{(r-q)T}.
- No-arbitrage bounds (solve only inside them, else NaN): call max(Sd - Kd, 0) < C < Sd; put max(Kd - Sd, 0) < P < Kd.
- Newton: sigma_{n+1} = sigma_n - (BS(sigma_n) - price) / vega(sigma_n), vega = Sd phi(d1) sqrt(T). Start at 0.3 or the
  Brenner-Subrahmanyam guess; stop when |BS - price| < tol (1e-10) or |step| < tol; cap at 100 iterations; if vega < 1e-12
  or the iterate leaves [1e-4, 5], switch to brentq on f(v) = BS(v) - price over [1e-4, 5] (xtol 1e-12).
- Deep ITM/OTM: vega -> 0, so Newton stalls and the price is insensitive to sigma; convert ITM calls to OTM puts via
  parity (P = C - Sd + Kd, identical IV) before solving; for |d| large the IV is poorly identified anyway -> report NaN
  when the price is within ~1e-10 of a bound.
- Brenner-Subrahmanyam (ATM, K = F): sigma ~ sqrt(2 pi / T) C / Sd. With spot-ATM (K = S) the forward is not the strike,
  so errors of ~0.02-0.04 are expected; checkers built on it expect those errors, do not "fix" them.
- Corrado-Miller (non-ATM): sigma ~ sqrt(2 pi / T) / (Sd + Kd) [ C - (Sd - Kd)/2 + sqrt( (C - (Sd - Kd)/2)^2 - (Sd - Kd)^2 / pi ) ].
  Negative discriminant (deep ITM/OTM, low vol, short T) -> NaN, not clipped (clipping shifts RMSE statistics).
- Li (2008) ATM exact cubic root: sigma = 4 sqrt(2/T) cos(2 pi/3 - arccos(-3 sqrt(pi) C / (4 Sd)) / 3), NaN if the arccos
  argument leaves [-1, 1]. Li non-ATM: t1 = C - (Sd - Kd)/2, disc = t1^2 - (Sd - Kd)^2 (1 + Sd/Kd)/(2 pi),
  sigma = sqrt(2 pi / T)(t1 + sqrt(disc))/(Sd + Kd); disc in (-1e-10, 0) -> 0, below -> NaN.
- Vectorised inversion: loop over rows (brentq is scalar) or vectorise Newton over arrays with masks; 1e4 rows take < 1 s.
- Forward-based (Black-76) inversion when F is known or parity-implied: undiscounted price c/DF with S = F, r = q = 0
  (DF = 1 for zero rates). Coin-margined (inverse) options quote premia in the coin: convert to USD with premium * F first
  (forward from parity: see option-quote-auditing); total variance w = sigma^2 T is what surface fits consume.
- Model-price IVs (Heston/FFT grids): brentq on [1e-4, 5] with xtol 1e-8 is enough for atol 5e-3 oracles; deep-OTM short-T
  model prices can be ~0 or slightly negative from quadrature error -> NaN, or reprice with the accurate integrator.

## Conventions and pitfalls (what graders check)
- If the task prints a formula, implement it literally even if it differs from the textbook (variants with or without a
  1/2 factor, different discriminant terms); oracle statistics are produced by that exact variant.
- Always divide by the dividend-discounted spot Sd, not S, when q > 0; use Kd = K e^{-rT} in CM/Li forms.
- error = sigma_est - sigma_true (signed); RMSE over non-NaN rows only; max/mean of |error| via pandas skipna; NaN rows
  stay in the CSV as empty cells (never inf, never 0 in place of NaN).
- Converged flags must be real Python bool; "all converged" = all(flags).
- ATM-only approximations must be NaN on non-ATM rows when the task says they apply to ATM only.
- Sort output rows as instructed (e.g. by sigma_true, T, moneyness); keep 10-decimal float_format if asked (error
  consistency tolerances are ~1e-9).
- IV from a chain: use mid = (bid + ask)/2, the OTM side per strike (put if K < F else call), filter zero bids and wide
  spreads as instructed, solve with brentq and T in years (days/365 unless told otherwise). Reference style for SVI/Dupire
  pipelines: OTM puts converted to calls by parity, Newton from the Brenner-Subrahmanyam start clipped to [0.05, 1.5],
  brentq fallback on [1e-3, 3], then keep only IVs inside the task's band (e.g. [0.05, 1.0]) - the survivor count is asserted.
- Smile sanity: IV must be the same for a call and a put at the same strike under parity; monotone in price.

## Tested snippet
```python
import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bs_price(S, K, T, r, q, sig, kind="call"):
    sq = sig * math.sqrt(T); d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / sq; d2 = d1 - sq
    if kind == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

def implied_vol(price, S, K, T, r, q, kind="call", lo=1e-4, hi=5.0, tol=1e-10):
    """Safeguarded Newton (vega), brentq fallback on [lo, hi]. ITM options are solved via the OTM twin (parity).
    Returns (sigma, converged); NaN when the price violates no-arbitrage bounds."""
    Sd, Kd = S * math.exp(-q * T), K * math.exp(-r * T)
    if kind == "call" and Kd < Sd:   price, kind = price - (Sd - Kd), "put"      # C - P = S e^-qT - K e^-rT
    elif kind == "put" and Kd > Sd:  price, kind = price + (Sd - Kd), "call"
    lower, upper = (0.0, Sd) if kind == "call" else (0.0, Kd)
    if not (lower < price < upper):
        return float("nan"), False
    sig = math.sqrt(2 * math.pi / T) * price / Sd                     # Brenner-Subrahmanyam start
    sig = 0.3 if not (lo < sig < hi) else sig
    for _ in range(100):
        f = bs_price(S, K, T, r, q, sig, kind) - price
        if abs(f) < tol: return float(sig), True
        d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
        vega = Sd * norm.pdf(d1) * math.sqrt(T)
        if vega < 1e-12: break
        new = sig - f / vega
        if not (lo < new < hi): break                                 # left the bracket -> brentq
        if abs(new - sig) < tol: return float(new), True
        sig = new
    g = lambda v: bs_price(S, K, T, r, q, v, kind) - price
    if g(lo) * g(hi) > 0: return float("nan"), False
    return float(brentq(g, lo, hi, xtol=1e-12, maxiter=500)), True

def corrado_miller(C, S, K, T, r, q=0.0):   # X = K e^{-rT}, Sd = S e^{-qT}; NaN when the discriminant is negative
    Sd, X = S * math.exp(-q * T), K * math.exp(-r * T)
    t1 = C - 0.5 * (Sd - X); disc = t1 * t1 - (Sd - X) ** 2 / math.pi
    return float("nan") if disc < 0 else math.sqrt(2 * math.pi / T) / (Sd + X) * (t1 + math.sqrt(disc))
```
