---
title: Short-rate models: Vasicek, CIR, Hull-White, caps and swaptions
category: rates-curves
tags: vasicek, cir, cox-ingersoll-ross, hull-white, short rate, mean reversion, zero-coupon bond, affine, a b functions, mle, maximum likelihood, non-central chi-square, ncx2, euler, feller condition, trinomial tree, calibration, caplet, floorlet, cap, floor, black-76, black 76, swaption, jamshidian, annuity, forward swap rate, convexity adjustment, bond option
summary: Closed-form bond prices and transition densities for Vasicek/CIR, MLE recipes, Hull-White tree calibration steps, Black-76 caplets/floorlets/swaptions, Jamshidian decomposition and the parity/invariant checks graders run.
---
## When this applies
"Calibrate Vasicek/CIR to this rate series", "price zero-coupon bonds / the model yield curve", "Feller condition",
"build a Hull-White trinomial tree matching the discount curve", "price caplets/caps/floors with Black-76",
"swaption price (Black or Jamshidian)", "forward rates implied by the model".

## Method and formulas
Symbols: k = mean-reversion speed, th = long-run mean, s = volatility (annualised, decimal rates), r = short
rate, tau = time to maturity in years, P(tau) = zero-coupon price, y = -ln P/tau = continuously compounded yield.
- Vasicek dr = k(th - r)dt + s dW: B = (1 - e^(-k tau))/k; A = exp((th - s^2/(2k^2))(B - tau) - s^2 B^2/(4k));
  P = A e^(-B r). Exact transition: r_{t+dt} | r_t ~ N(r_t e^(-k dt) + th(1 - e^(-k dt)), s^2 (1 - e^(-2k dt))/(2k)).
  MLE = OLS of r_{t+dt} on r_t: slope b, intercept a, residual sd sd_e -> k = -ln b/dt, th = a/(1-b),
  s = sd_e sqrt(-2 ln b/((1 - b^2) dt)).
- CIR dr = k(th - r)dt + s sqrt(r) dW: g = sqrt(k^2 + 2 s^2); den = (g + k)(e^(g tau) - 1) + 2g;
  B = 2(e^(g tau) - 1)/den; A = (2 g e^((k+g) tau/2)/den)^(2 k th/s^2); P = A e^(-B r). Feller: 2 k th > s^2
  (ratio 2 k th/s^2 > 1). Exact density: c = 2k/(s^2 (1 - e^(-k dt))), 2c r_{t+dt} ~ ncx2(df = 4 k th/s^2,
  nc = 2c r_t e^(-k dt)); log p = ncx2.logpdf(2c r_{t+dt}, df, nc) + ln(2c). Euler approximation:
  r_{t+dt} ~ N(r_t + k(th - r_t) dt, s^2 r_t dt), log p = norm.logpdf. Maximise the sum with L-BFGS-B, bounds
  k in [1e-3, 5], th in [1e-4, 0.5], s in [1e-4, 1], several starts (moment-based + seeded random), keep the best.
- Forward rate between model maturities: f(t1,t2) = -(ln P(t2) - ln P(t1))/(t2 - t1); instantaneous f = -d ln P/dtau.
- Hull-White dr = (theta(t) - a r)dt + s dW: P(t,T) = A(t,T) e^(-B r_t), B = (1 - e^(-a(T-t)))/a,
  ln A = ln(P(0,T)/P(0,t)) + B f(0,t) - s^2 (1 - e^(-2at)) B^2/(4a). Trinomial tree (two stages): dt, dR = s sqrt(3 dt),
  j_max = ceil(0.184/(a dt)); branching probabilities for node j: pu = 1/6 + (a^2 j^2 dt^2 - a j dt)/2,
  pm = 2/3 - a^2 j^2 dt^2, pd = 1/6 + (a^2 j^2 dt^2 + a j dt)/2 (modified branching at +/- j_max). Stage 2: with
  Arrow-Debreu prices Q_{m,j} (forward induction) set alpha_m = (ln sum_j Q_{m,j} e^(-j dR dt) - ln P(0,(m+1)dt))/dt,
  r_{m,j} = alpha_m + j dR. Calibrated tree reprices every input DF to ~1e-12.
- Caplet on forward F = F(t_{i-1}, t_i) (simple, accrual tau, payment DF(t_i), strike K, Black vol sig):
  d1 = (ln(F/K) + sig^2 t_{i-1}/2)/(sig sqrt(t_{i-1})), d2 = d1 - sig sqrt(t_{i-1});
  caplet = N tau DF(t_i) [F N(d1) - K N(d2)], floorlet = N tau DF(t_i) [K N(-d2) - F N(-d1)]. Vol time is the
  FIXING time t_{i-1}; the first caplet (fixing at 0) has only intrinsic value tau DF max(F-K,0).
  Cap = sum of caplets; cap - floor = payer swap = N sum tau_i DF(t_i)(F_i - K) (exact parity).
- Swaption (Black-76): annuity A = sum tau_i DF(t_i), forward swap rate S = (DF(T_0) - DF(T_n))/A,
  payer = N A [S N(d1) - K N(d2)], receiver = N A [K N(-d2) - S N(-d1)], d1 = (ln(S/K) + sig^2 T/2)/(sig sqrt T),
  T = option expiry = swap start. Bachelier (normal vol): payer = N A [(S-K) N(d) + sig_n sqrt(T) phi(d)],
  d = (S-K)/(sig_n sqrt T). Payer - receiver = N A (S - K).
- Jamshidian (one-factor affine models, Vasicek/HW/CIR): coupon vector c_i = K tau_i, c_n += 1; find r* with
  sum c_i P(T, t_i; r*) = 1 (brentq); payer swaption = sum c_i ZBP(T, t_i, strike K_i = P(T, t_i; r*)),
  receiver = sum c_i ZBC(...). Vasicek/HW bond option: sigma_p = s sqrt((1 - e^(-2kT))/(2k)) B(T, S),
  h = ln(P(0,S)/(P(0,T) K))/sigma_p + sigma_p/2, ZBC = P(0,S) N(h) - K P(0,T) N(h - sigma_p),
  ZBP = K P(0,T) N(-h + sigma_p) - P(0,S) N(-h).
- Convexity: futures - forward rate ~ s^2 t1 t2/2 (Ho-Lee) or the HW analogue; model yield curves are concave in
  tau because of the -s^2 terms in A.

## Conventions and pitfalls (what graders check)
- dt from the data frequency: calendar-daily series (weekends present) dt = 1/365, business-daily 1/252, monthly
  1/12; rates in percent -> divide by 100 before fitting; report parameters in decimal annual units.
- Use ncx2.logpdf (never log(pdf)); return a large penalty when any term is non-finite; clip starts into bounds
  (L-BFGS-B raises on out-of-bounds x0); Feller flag cast with bool().
- Bond-price invariants: 0 < P <= 1, P non-increasing in tau, y = -ln P/tau positive and finite, forward rates
  recomputed from the CSV must match the formula (rtol 1e-6), zero rates and yields consistent (y = z when
  both continuous).
- Black-76 uses the forward and DF of the payment date; do not discount with the fixing date; caps/floors on a
  schedule usually exclude the first (already-fixed) period only when the task says so.
- Parities checkers assert: cap - floor == payer swap value (1e-10), payer - receiver swaption == A (S - K),
  Jamshidian payer >= intrinsic A max(S-K, 0), prices increase with vol, caplets decrease in strike.
- Vol input "20" means 20% -> 0.20; strike "4.5" means 4.5% -> 0.045; keep notional separate from unit prices.
- Tree: probabilities must sum to 1 and be positive (choose j_max rule), the tree must reprice DFs; a one-stage
  tree with theta from the analytic formula does not match the curve to 1e-10.
- Vasicek allows negative rates; CIR data must be positive (clip 1e-8 before the log-likelihood if needed).

## Tested snippet
```python
import math, numpy as np
from scipy.stats import norm, ncx2
from scipy.optimize import brentq

def vasicek_P(tau, k, th, s, r):                         # P(tau) = A exp(-B r), dr = k(th - r)dt + s dW
    B = (1 - math.exp(-k * tau)) / k
    A = math.exp((th - s * s / (2 * k * k)) * (B - tau) - s * s * B * B / (4 * k))
    return A * math.exp(-B * r)

def cir_P(tau, k, th, s, r):                             # dr = k(th - r)dt + s sqrt(r) dW
    g = math.sqrt(k * k + 2 * s * s); eg = math.exp(g * tau); den = (g + k) * (eg - 1) + 2 * g
    B = 2 * (eg - 1) / den; A = (2 * g * math.exp((k + g) * tau / 2) / den) ** (2 * k * th / (s * s))
    return A * math.exp(-B * r)

def cir_loglik(p, r0, r1, dt):                           # exact transition density; r0 = r_t, r1 = r_{t+dt} arrays
    k, th, s = p; e = math.exp(-k * dt); c = 2 * k / (s * s * (1 - e))
    lp = ncx2.logpdf(2 * c * r1, 4 * k * th / (s * s), 2 * c * r0 * e) + math.log(2 * c)
    return float(lp.sum()) if np.all(np.isfinite(lp)) else -1e12

def black76(F, K, T, sig, df, tau, call=True):           # caplet (call) / floorlet (put) on a simple forward F
    if T <= 0: return df * tau * max((F - K) if call else (K - F), 0.0)
    sq = sig * math.sqrt(T); d1 = (math.log(F / K) + 0.5 * sq * sq) / sq; d2 = d1 - sq
    v = F * norm.cdf(d1) - K * norm.cdf(d2) if call else K * norm.cdf(-d2) - F * norm.cdf(-d1)
    return df * tau * v                                   # swaption: same with F = fwd swap rate, df*tau -> annuity

def vasicek_zbp(T, S, K, k, th, s, r0):                  # European put, expiry T, on P(T,S) with strike K
    PT, PS = vasicek_P(T, k, th, s, r0), vasicek_P(S, k, th, s, r0)
    sp = s * math.sqrt((1 - math.exp(-2 * k * T)) / (2 * k)) * (1 - math.exp(-k * (S - T))) / k
    h = math.log(PS / (PT * K)) / sp + sp / 2
    return K * PT * norm.cdf(-h + sp) - PS * norm.cdf(-h)

def jamshidian_payer(T, pay_t, taus, K, k, th, s, r0):   # payer swaption = sum c_i ZBP(T, t_i, P(T,t_i; r*))
    c = [K * tau for tau in taus]; c[-1] += 1.0
    rstar = brentq(lambda r: sum(ci * vasicek_P(t - T, k, th, s, r) for ci, t in zip(c, pay_t)) - 1.0, -1.0, 2.0)
    return sum(ci * vasicek_zbp(T, t, vasicek_P(t - T, k, th, s, rstar), k, th, s, r0) for ci, t in zip(c, pay_t))

k, th, s, r0 = 0.3, 0.05, 0.02, 0.04; rng = np.random.default_rng(0)
r = np.abs(0.04 + 0.0005 * np.cumsum(rng.standard_normal(2001)))          # stand-in daily short-rate path, dt = 1/252
print(round(vasicek_P(5, k, th, s, r0), 8), round(cir_P(5, k, th, s, r0), 8), round(cir_loglik((k, th, s), r[:-1], r[1:], 1 / 252), 2))
print(round(black76(0.04, 0.035, 1.0, 0.25, 0.95, 0.5), 8), round(jamshidian_payer(1.0, [2, 3, 4, 5], [1.0] * 4, 0.045, k, th, s, r0), 8), 2 * k * th > s * s)
```
