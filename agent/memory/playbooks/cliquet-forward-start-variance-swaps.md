---
title: Forward-start options, cliquets and variance-swap replication
category: volatility
tags: forward start option, rubinstein, cliquet, ratchet, reset option, local cap, global cap, floor, variance swap, fair variance strike, log contract, demeterfi, carr-madan replication, option chain, otm selection, trapezoid weights, vix formula, variance risk premium, realized variance, vega notional
summary: Rubinstein forward-start pricing (price vs return payoff discounting), cliquet sums with local caps as call spreads, and the discrete log-contract / VIX fair-variance formulas with chain-cleaning conventions.
---
## When this applies
"Forward-start / cliquet / ratchet option", "sum of N forward-start calls", "cliquet with local cap and floor",
"fair variance strike from an option chain", "replicate the log contract", "VIX-style calculation", "variance risk
premium", variance-swap P&L scenarios.

## Method and formulas
C_BS(S, K, tau) = BS call with yield q; homogeneity C(S, K) = S C(1, K/S).
- Forward-start call, strike alpha S(t_s) fixed at t_s, paying (S(t_e) - alpha S(t_s))^+ at t_e (price payoff):
  FS = S0 e^{-q t_s} C_BS(1, alpha, t_e - t_s). ATM (alpha = 1): FS = S0 e^{-q t_s} [e^{-q tau} N(d1) - e^{-r tau} N(d2)],
  d1 = (r - q + sigma^2/2) tau/(sigma sqrt tau), d2 = d1 - sigma sqrt tau, tau = t_e - t_s. Each leg uses its own tau, not T.
- Return payoff (S(t_e)/S(t_s) - alpha)^+ paid at t_e (unit notional): value = e^{-r t_s} C_BS(1, alpha, tau) -
  the reset date is discounted at r, NOT e^{-q t_s}. Price-denominated and return-denominated legs differ by S0 e^{(r-q) t_s}.
- Plain cliquet (no caps) = sum_{i=0}^{N-1} FS(i T/N, (i+1) T/N); grows with N (more ATM optionality); compare with the
  European C_BS(S0, S0, T) with the same q.
- Local floor f / cap c on each period return R_i = S_e/S_s - 1: min(max(R, f), c) = f + (R - f)^+ - (R - c)^+ =>
  leg = e^{-r t_e} f + e^{-r t_s}[C_BS(1, 1 + f, tau) - C_BS(1, 1 + c, tau)]. Uncapped, unfloored leg = e^{-r t_e}(e^{(r-q) tau} - 1).
  If all legs pay at T, replace e^{-r t_e} by e^{-r T} and multiply the option terms by e^{-r (T - t_e)}.
  Global cap/floor on the SUM has no closed form: MC with per-period lognormal returns, seed as instructed.
- Variance swap fair strike (Demeterfi et al., S* = F): K_var = (2 e^{rT}/T) sum_i (dK_i / K_i^2) Q(K_i), Q = OTM option mid
  (put if K < F, call if K >= F). dK_0 = K_1 - K_0, dK_last = K_last - K_{last-1}, dK_i = (K_{i+1} - K_{i-1})/2 inside - on the
  FILTERED (non-uniform) strike grid. General S*: K_var = (2/T)[rT - (S0 e^{rT}/S* - 1) - ln(S*/S0)] + the sums; the bracket
  vanishes when S* = F.
- VIX-style: sigma^2 = (2 e^{rT}/T) sum dK_i/K_i^2 Q(K_i) - (1/T)(F/K0 - 1)^2, K0 = first strike below F, Q(K0) = average of
  call and put mids; F from put-call parity at the strike with the smallest |C - P|: F = K + e^{rT}(C - P).
  VIX = 100 sqrt((T1 s1^2 w1 + T2 s2^2 w2) 365/30) with minute-based weights w1 = (N_T2 - N_30)/(N_T2 - N_T1), w2 = 1 - w1.
- Variance risk premium: VRP = K_var - sigma_ATM^2 (or realised - implied, as defined by the task); fair vol = sqrt(K_var).
  ATM IV at F by linear interpolation in strike between the kept strikes straddling F.
- Payoff: N_var (sigma_R^2 - K_var) with realised variance (252/n) sum (ln(S_i/S_{i-1}))^2 (no mean subtraction, unless told);
  vega notional N_vega = 2 K_vol N_var.
- Flat-vol sanity test: with BS prices at constant sigma on a dense wide grid (e.g. 0.2 F .. 4 F, step 0.5% of F) the
  replication returns sigma^2 to ~1e-5.

## Conventions and pitfalls (what graders check)
- Use the forward supplied in the data/params for the OTM split and for ATM; recompute S e^{(r-q)T} only if none is given.
- Chain cleaning order matters: filter to the target expiry first, build OTM side (bid/ask/mid of that side), then apply
  volume >= min, OTM bid > 0, (ask - bid)/mid <= max spread; count kept puts/calls; K == F counts as a call (strict K < F for puts).
- Rectangle rule with the raw spacing or trapezoid on the unfiltered chain misses 1e-3 tolerances when strikes are dropped.
- Forward-start legs: missing the e^{-q t_s} factor overstates prices by q t_s; summing legs must reproduce the cliquet
  total to 1e-6 -> write full precision.
- Sort configurations by (T, N); period_length = T/N; t_end - t_start = T/N to 1e-5; all leg prices > 0.
- "cliquet_increases_with_N" style flags must be real bools; ratios cliquet/european computed per T.

## Tested snippet
```python
import math
import numpy as np
from scipy.stats import norm

def bs_call(S, K, T, r, q, s):
    if T <= 0: return max(S - K, 0.0)
    sq = s * math.sqrt(T); d1 = (math.log(S / K) + (r - q + 0.5 * s * s) * T) / sq
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d1 - sq)

def forward_start_call(S0, t_s, t_e, r, q, s, alpha=1.0):
    """Rubinstein: call with strike alpha*S(t_s) set at t_s, paid at t_e. Homogeneity -> S0 e^{-q t_s} C_BS(1, alpha, t_e - t_s)."""
    return S0 * math.exp(-q * t_s) * bs_call(1.0, alpha, t_e - t_s, r, q, s)

def cliquet_local_caps(T, N, r, q, s, floor=None, cap=None):
    """Notional-1 cliquet: sum over periods of min(max(R_i, floor), cap), R_i = S_e/S_s - 1, each leg paid at t_e.
    min(max(R, f), c) = f + (R - f)^+ - (R - c)^+ ; E[e^{-r t_e} (S_e/S_s - a)^+] = e^{-r t_s} C_BS(1, a, t_e - t_s)
    (a RETURN payoff discounts the reset date with e^{-r t_s}; a PRICE payoff (S_e - a S_s)^+ uses S0 e^{-q t_s})."""
    dt = T / N; total = 0.0
    for i in range(N):
        t_s, t_e = i * dt, (i + 1) * dt; ratio_call = lambda a: math.exp(-r * t_s) * bs_call(1.0, a, dt, r, q, s)
        if floor is None: leg = math.exp(-r * t_e) * (math.exp((r - q) * dt) - 1.0)          # plain forward return
        else: leg = math.exp(-r * t_e) * floor + ratio_call(1 + floor)
        if cap is not None: leg -= ratio_call(1 + cap)
        total += leg
    return total

def var_swap_strike(Ks, otm_prices, F, T, r, vix_style=False, K0=None):
    """Demeterfi/CBOE discrete log-contract replication. Ks sorted, otm_prices = OTM option mid at each K.
    dK: one-sided at the ends, (K_{i+1} - K_{i-1})/2 inside. K_var = 2 e^{rT}/T sum dK/K^2 Q(K) [- (F/K0 - 1)^2 / T]."""
    Ks, Q = np.asarray(Ks, float), np.asarray(otm_prices, float); dK = np.empty(len(Ks))
    dK[0], dK[-1] = Ks[1] - Ks[0], Ks[-1] - Ks[-2]; dK[1:-1] = 0.5 * (Ks[2:] - Ks[:-2])
    kv = 2.0 * math.exp(r * T) / T * float(np.sum(dK / Ks ** 2 * Q))
    if vix_style: kv -= (F / K0 - 1.0) ** 2 / T
    return kv
```
