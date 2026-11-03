---
title: Black-Scholes prices, Greeks, parity and PDE residual
category: derivatives-pricing
tags: black-scholes, greeks, delta, gamma, vega, theta, rho, vanna, volga, put-call parity, dividend yield, pde residual, forward price, european option, option pricing, sensitivity surface, theta per day, vega per 1%, results.parquet, option table, no-arbitrage bounds, solve the pde
summary: Exact BS call/put formulas with continuous dividend yield, all first/second-order Greeks, parity and PDE identities a checker will assert, plus unit conventions.
---
## When this applies
Tasks asking for European option prices and Greeks "with continuous dividend yield", Greek surfaces over
moneyness x maturity grids, "verify the Black-Scholes PDE", put-call parity checks, forward prices, or any
exotic playbook that needs a vanilla BS leg (digitals, barriers, compound, cliquet, variance swaps).
"Solve the BS PDE (or any equivalent method)" for an option table: the closed form IS the exact PDE solution.

## Method and formulas
Symbols: S spot, K strike, T time to expiry in years, r continuously compounded risk-free rate, q continuous
dividend yield, sigma annualised volatility, N = standard normal cdf, phi = standard normal pdf.
- d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)
- Forward F = S e^{(r-q)T}; call C = e^{-rT}[F N(d1) - K N(d2)] = S e^{-qT} N(d1) - K e^{-rT} N(d2)
- Put P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1);  parity C - P = S e^{-qT} - K e^{-rT} = e^{-rT}(F - K)
- delta_C = e^{-qT} N(d1);  delta_P = e^{-qT}(N(d1) - 1) = -e^{-qT} N(-d1);  delta_C - delta_P = e^{-qT}
- gamma = e^{-qT} phi(d1) / (S sigma sqrt(T))   (same for call and put, > 0)
- vega = S e^{-qT} phi(d1) sqrt(T)   per unit of sigma (divide by 100 for "per 1 vol point"); same for call/put
- theta_C = -S e^{-qT} phi(d1) sigma / (2 sqrt(T)) - r K e^{-rT} N(d2) + q S e^{-qT} N(d1)   (dV/dt, per year)
- theta_P = -S e^{-qT} phi(d1) sigma / (2 sqrt(T)) + r K e^{-rT} N(-d2) - q S e^{-qT} N(-d1)
- rho_C = K T e^{-rT} N(d2);  rho_P = -K T e^{-rT} N(-d2);  rho_C - rho_P = K T e^{-rT}
- dividend sensitivity (epsilon): -T S e^{-qT} N(d1) for a call, +T S e^{-qT} N(-d1) for a put
- vanna = d delta / d sigma = -e^{-qT} phi(d1) d2 / sigma;  volga (vomma) = vega d1 d2 / sigma
- BS PDE: dV/dt + (r - q) S dV/dS + 0.5 sigma^2 S^2 d2V/dS2 - r V = 0, i.e. residual
  theta + (r - q) S delta + 0.5 sigma^2 S^2 gamma - r V == 0 for both call and put (|residual| ~ 1e-12).
- Limits: T -> 0 gives intrinsic max(S-K,0)/max(K-S,0); K -> 0 gives C -> S e^{-qT}; sigma -> 0 gives
  C -> max(S e^{-qT} - K e^{-rT}, 0).
- Homogeneity: C(S, K) = K C(S/K, 1) = S C(1, K/S) (used by forward-start and cliquet pricing).
- Calibration family (daily price file): log returns x = diff(ln close), sigma = std(x, ddof=1) sqrt(252),
  S0 = last close, n_returns = len(close) - 1; r and q as given in the task (q is often called D).

## Conventions and pitfalls (what graders check)
- Use the yield everywhere: delta_C in [0, e^{-qT}], not [0, 1]; delta_C - delta_P = e^{-qT}; PDE drift (r - q).
- theta is dV/dt (negative for most calls) per year; "per calendar day" = theta/365, "per trading day" = /252.
  Only convert when the task says so. Finite-difference theta sign: -(V(T - h) - V(T))/h = +dV/dT, the opposite sign.
- vega per unit sigma unless the task says "per 1%" (then /100). rho per unit r (per 1% -> /100).
- Grid outputs: sort by (T, moneyness); moneyness is usually S/K or K/S exactly as the task defines it; keep
  the column order from the task; write full precision; cast numpy scalars to float and flags to bool().
- Typical invariants asserted: gamma > 0, vega > 0 everywhere; call price non-decreasing in S, decreasing in K;
  put price <= K e^{-rT}, call <= S e^{-qT}; |parity error| < 1e-8; |PDE residual| < 1e-6; call delta monotone in S;
  gamma peaks near the money (slightly below K = S for calls with large sigma sqrt(T)); prices increase with T
  for calls when q = 0 (not guaranteed with q > 0 or for puts).
- Vectorise with numpy (works on arrays of S/K/T); guard T <= 0 or sigma <= 0 by returning intrinsic values.
- Use scipy.stats.norm.cdf / norm.pdf; never hand-rolled erf approximations (1e-7 errors fail tight oracles).
- Option-table checkers (invariants, no oracle numbers) typically assert: one row per input row in INPUT order
  (id column kept as str, compared with assert_series_equal), no NaN anywhere, parity at atol ~1e-3, bounds with
  ~1e-3 slack, call delta in [0, e^{-qT}] and put delta in [-e^{-qT}, 0] (1e-5 slack), gamma and vega >= -1e-6, and
  a finite-difference delta against bumped-spot rows if the input has them (ids like "<id>_up"/"<id>_dn", +-1% spot,
  atol ~5e-3) - so price every row, bumped ones included, with the same function.
- Clip prices at 0: deep-OTM, few-day options come out as ~-1e-90 by cancellation and fail "price >= 0".
- A numerical PDE (only if mandated): Crank-Nicolson in log-spot with Rannacher start-up; a 3-day option needs a grid
  spacing well below sigma sqrt(T) to meet 1e-3 parity - compare every row to the closed form.
- Never copy instruction header lines or identifier strings into output files (some checkers scan for them).

## Tested snippet
```python
import numpy as np
from scipy.stats import norm

def bs_greeks(S, K, T, r, q, sigma, kind="call"):
    """Black-Scholes with continuous dividend yield q. T in years, sigma annualised. Theta per year (dV/dt),
    vega/vanna/volga per unit of sigma, rho per unit of r. Works on numpy arrays."""
    S, K, T = np.asarray(S, float), np.asarray(K, float), np.asarray(T, float)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    eq, er, nd1 = np.exp(-q * T), np.exp(-r * T), norm.pdf(d1)
    F = S * np.exp((r - q) * T)                                        # forward price
    g = {"forward": F, "gamma": eq * nd1 / (S * sq), "vega": S * eq * nd1 * np.sqrt(T),
         "vanna": -eq * nd1 * d2 / sigma, "volga": S * eq * nd1 * np.sqrt(T) * d1 * d2 / sigma}
    if kind == "call":
        g["price"] = np.maximum(S * eq * norm.cdf(d1) - K * er * norm.cdf(d2), 0.0)   # clip -1e-90 cancellation noise
        g["delta"] = eq * norm.cdf(d1)
        g["theta"] = -S * eq * nd1 * sigma / (2 * np.sqrt(T)) - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1)
        g["rho"] = K * T * er * norm.cdf(d2)
    else:
        g["price"] = np.maximum(K * er * norm.cdf(-d2) - S * eq * norm.cdf(-d1), 0.0)
        g["delta"] = -eq * norm.cdf(-d1)
        g["theta"] = -S * eq * nd1 * sigma / (2 * np.sqrt(T)) + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)
        g["rho"] = -K * T * er * norm.cdf(-d2)
    return g

def pde_residual(g, S, r, q, sigma):      # theta + (r-q) S delta + 0.5 sigma^2 S^2 gamma - r V  (should be ~1e-12)
    return g["theta"] + (r - q) * S * g["delta"] + 0.5 * sigma ** 2 * S ** 2 * g["gamma"] - r * g["price"]

def parity_error(S, K, T, r, q, sigma):    # C - P - (S e^{-qT} - K e^{-rT})
    c, p = bs_greeks(S, K, T, r, q, sigma, "call"), bs_greeks(S, K, T, r, q, sigma, "put")
    return c["price"] - p["price"] - (S * np.exp(-q * T) - K * np.exp(-r * T))
```
