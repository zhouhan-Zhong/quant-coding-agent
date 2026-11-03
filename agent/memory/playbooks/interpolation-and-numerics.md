---
title: Interpolation, integration, finite differences and float/JSON hygiene
category: numerics
tags: interpolation, linear, log-linear, cubic spline, pchip, monotone, extrapolation, discount factor interpolation, vol interpolation, numerical integration, quad, gauss-legendre, trapezoid, simpson, non-uniform grid, finite differences, np.gradient, step size, numerical stability, log-space, clipping, errstate, float precision, json, nan, csv output, rounding, fixed decimals, key order, largest remainder, iso timestamps
summary: Which interpolant/integrator/difference scheme to use for curves, vols and payoffs, exact weight formulas, step-size rules, stability tricks, and the output-hygiene rules that decide whether JSON/CSV checkers pass.
---
## When this applies
Every task touches these: interpolating curves (discount factors, zero rates, vol smiles/surfaces), integrating
pricing formulas or option strips, bumping inputs for Greeks, and writing JSON/CSV deliverables compared with
tolerances.

## Method and formulas
- Interpolation choice: discount factors log-linear in t (equivalent to piecewise-constant forwards; linear in ln DF):
  DF(t) = exp((1 - w) ln DF_0 + w ln DF_1), w = (t - t0)/(t1 - t0); zero rates linear in t; outside the nodes use flat
  continuous-zero extrapolation DF(t) = exp(-z_edge t) (np.interp alone clips to the end DF, which is wrong for t > t_max).
  Vols: linear in total variance sigma^2 T across expiries at fixed moneyness; across strikes CubicSpline (smooth, can
  overshoot and create negative butterflies) or PchipInterpolator (shape-preserving, no overshoot, preferred for sparse
  or monotone data such as cdfs and curves); linear np.interp is the safe default when the task does not specify.
  Both scipy splines EXTRAPOLATE polynomially by default -> clip inputs to [x_min, x_max] for flat extrapolation or
  pass extrapolate=False (NaN outside). interp1d(..., bounds_error=False, fill_value=(y0, yN)) gives flat ends.
- Integration: scipy.integrate.quad(f, a, b, limit=200, epsabs=1e-12, epsrel=1e-10) for smooth 1-D integrands (returns
  (value, error)); infinite limits allowed but oscillatory integrands (Fourier) need a finite cutoff or weight="cos";
  Gauss-Legendre: x, w = np.polynomial.legendre.leggauss(n); int_a^b f = (b-a)/2 sum w_i f((b-a)/2 x_i + (a+b)/2), exact for
  polynomials of degree 2n-1 (n = 32-128 for pricing integrals); trapezoid on a non-uniform grid: weights
  w_0 = (x_1 - x_0)/2, w_i = (x_{i+1} - x_{i-1})/2, w_n = (x_n - x_{n-1})/2 (np.trapezoid(y, x); np.trapz is removed in numpy 2);
  the CBOE/replication convention instead uses FULL one-sided spacing at the ends (dK_0 = K_1 - K_0) - follow the task's
  words ("trapezoidal" vs "dK_i = (K_{i+1} - K_{i-1})/2 with one-sided ends"). Simpson: scipy.integrate.simpson(y, x=x).
- np.gradient(f, x, edge_order=2) on a non-uniform x is the 3-point second-order formula (exact for quadratics, one-sided
  at the ends); np.gradient twice for f'' is NOT the classic non-uniform 3-point f'' stencil - when a reference used
  np.gradient, only np.gradient (same edge_order, same axis) reproduces its numbers. leggauss(n) solves an eigenproblem
  (n = 2000 ~ 1 s): compute nodes once and cache; map to [0, U] by u = (x + 1) U/2, weights U/2 w.
- Finite differences: f'(x) ~ (f(x+h) - f(x-h))/(2h) with error O(h^2) + eps/h -> h ~ cbrt(eps) |x| ~ 6e-6 |x|;
  f'' ~ (f(x+h) - 2f(x) + f(x-h))/h^2, h ~ eps^{1/4} |x| ~ 1e-4 |x|; one-sided (f(x+h) - f(x))/h only at boundaries.
  Market-style bumps: 1% of spot (or 0.5-1.0 absolute), 1 vol point (0.01), 1 bp (1e-4) in rates, 1/252 in time; keep the
  same random numbers for MC bumps; report the bump sizes used.
- Stability: work in log space for products/likelihoods (logsumexp, gammaln), np.log1p/np.expm1 for small arguments,
  clip exponents to [-700, 700] before exp, write Black-Scholes terms as e^{-qT} N(d) (never N(d)/e^{qT}); guard 0/0 with
  np.errstate(divide="ignore", invalid="ignore") and np.where; sort before np.searchsorted; compare floats on grids
  with np.isclose (never ==) when selecting rows (moneyness == 1.0 after arithmetic fails).
- Day counts and dates: ACT/365: days/365.0, ACT/360: days/360.0, 30/360 US per schedule; business days with
  np.busday_count or pandas bdate_range; keep a sorted DatetimeIndex; year fraction convention per currency/instrument.

## Conventions and pitfalls (what graders check)
- JSON: cast np.float64 -> float, np.int64 -> int, np.bool_ -> bool (isinstance(x, bool) fails for numpy bools); NaN/inf are
  not valid JSON - write None (null) or omit; json.dump(obj, f, indent=2, allow_nan=False) raises early if something leaks.
- CSV: pandas writes NaN as empty cells (fine when the checker expects blanks); do not round unless the task specifies
  decimals; when another file must reproduce a value to 1e-12, write with full precision (repr) in both;
  float_format="%.10f" only when asked; keep exact column names/order; sort rows as specified; index=False.
- Rounding: Python round() is banker's rounding at exact halves; use it only where the spec asks for "rounded to n dp"
  and compare with tolerance in your own checks.
- Exact-string checkers (csv.DictReader rows compared to a re-derivation, regex like -?\d+\.\d{6}): write EVERY numeric
  cell as f"{x:.6f}" (strikes and quantities too: "100.000000"), ints as plain digits, missing as "", flags as the literal
  strings shown ("true"/"false"); build rows as dicts of str/int so pandas does not turn ints into 1.0. Keep the raw
  floats for sums; JSON summaries derived from the CSV (medians, max) must be computed from float(f"{x:.6f}") values to
  reconcile bit-for-bit; "values rounded to 6 dp" in JSON means round(x, 6) before dumping.
- Exact JSON key ORDER can be asserted (list(d) == expected): build dicts in the template order; never rewrite input files.
  A trailing metadata row that puts text into numeric columns (e.g. "chosen_method,<name>,,,") is written with csv.writer.
- Timestamps: ISO with trailing Z via ts.strftime("%Y-%m-%dT%H:%M:%SZ") on UTC; ACT/365 with intraday time =
  total_seconds/(365 * 86400); epoch-ms clocks stay int64 (np.searchsorted on the sorted clock for "first at/after").
- Integer allocations that must sum to a total: floor + largest remainder, ties to the earlier index (snippet).
- Equality of derived fields: compute a field once and reuse it (e.g. fair_vol = sqrt(K_var) stored, then
  fair_vol_minus_atm = fair_vol - iv_atm from the stored values) so cross-field identities hold to 1e-12.
- Interpolating vol in T directly or DF linearly produces 1e-3 level errors that fail 1e-6 reprice checks.
- Monotone data (cdf, cumulative weights, discount factors): PCHIP or linear, never not-a-knot cubic.
- Integration cutoffs: check convergence by doubling n / the cutoff; report the estimated error when quad gives one.
- Shapes: np.interp needs increasing x; CubicSpline needs strictly increasing x (drop duplicate strikes/dates first).

## Tested snippet
```python
import math
import numpy as np
from scipy.interpolate import PchipInterpolator

def df_loglinear(t_nodes, df_nodes, t):
    """Discount factors log-linear in t (piecewise-constant forwards); flat continuous-zero extrapolation beyond the ends."""
    t_nodes, df_nodes, t = np.asarray(t_nodes, float), np.asarray(df_nodes, float), np.asarray(t, float)
    z = -np.log(df_nodes) / t_nodes                                         # zero rates at the nodes
    out = np.exp(np.interp(t, t_nodes, np.log(df_nodes)))                   # np.interp clips -> flat log-DF outside; fix below
    out = np.where(t < t_nodes[0], np.exp(-z[0] * t), out); out = np.where(t > t_nodes[-1], np.exp(-z[-1] * t), out)
    return np.where(t <= 0, 1.0, out)

def gauss_legendre(f, a, b, n=64):          # smooth integrand on [a, b]; map nodes from [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n)
    return float(np.sum(w * f(0.5 * (b - a) * x + 0.5 * (a + b))) * 0.5 * (b - a))

def trapezoid_weights(x):                   # integral weights on a non-uniform grid (half spacing at the ends)
    x = np.asarray(x, float); w = np.empty_like(x)
    w[0], w[-1] = 0.5 * (x[1] - x[0]), 0.5 * (x[-1] - x[-2]); w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w

def central_diff(f, x, h=None, order=1):    # h ~ cbrt(eps)*|x| for f', eps^(1/4)*|x| for f''
    h = h or (6e-6 if order == 1 else 1.2e-4) * max(abs(x), 1.0)
    return (f(x + h) - f(x - h)) / (2 * h) if order == 1 else (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2

def json_safe(obj):                          # numpy scalars -> python; NaN/inf -> None (valid JSON for the grader)
    if isinstance(obj, dict): return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)): return [json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)): return bool(obj)
    if isinstance(obj, (int, np.integer)): return int(obj)
    if isinstance(obj, (float, np.floating)): return float(obj) if np.isfinite(obj) else None
    return obj

def flat_extrap_pchip(x, y):                 # shape-preserving interpolant with flat ends (clip inputs to the data range)
    p = PchipInterpolator(np.asarray(x, float), np.asarray(y, float))
    return lambda xq: p(np.clip(np.asarray(xq, float), x[0], x[-1]))

def largest_remainder(total, w):            # integers summing exactly to total; ties -> earlier index
    raw = total * np.asarray(w, float) / np.sum(w); q = np.floor(raw).astype(int)
    q[np.lexsort((np.arange(len(q)), -(raw - q)))[: int(total - q.sum())]] += 1
    return q
f6 = lambda x: f"{x:.6f}"                    # fixed-decimal CSV cell; summaries from float(f6(x)) reconcile exactly
```
