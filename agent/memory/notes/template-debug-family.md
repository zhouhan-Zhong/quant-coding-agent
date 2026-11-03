---
title: "Debug the template" tasks (template.py with planted bugs) and bootstrap-template tasks
category: meta
tags: template.py, debug, planted bugs, fix the code, canonical oracle, rtol 1e-6, bootstrap template, conventions.json, fixed leg, 30/360, act/360, ois discounting
summary: How to handle tasks that ship a buggy reference script and grade against a canonical recomputation with tight tolerance.
triggers: file:template.py, text:provided implementation contains bugs, text:fix the bugs
---
## When this applies
The data directory contains `template.py` (or the task says "the provided implementation contains bugs"), and the
checker recomputes a canonical oracle from the inputs with rtol ~1e-6. The deliverables are the template's
outputs (results.json, solution.json, CSV curves).

## Approach
1. Read the template fully; list every computation step against the task text and the standard formula.
2. Typical planted bugs (seen in public tasks): demeaning returns before an EWMA covariance when the spec says raw;
   `np.dot(r, r)` (scalar) where `np.outer(r, r)` is needed; hard-coded quantile (2.33) instead of
   `norm.ppf(0.99)`; ignoring a parameter from params.json (estimation window); multiplying instead of dividing by the
   portfolio vol in marginal risk; wrong day count on a swap leg (ACT/360 instead of 30/360); discounting IRS cash
   flows with the trial LIBOR curve instead of the OIS curve; bootstrapping curves in the wrong order (LIBOR before
   OIS); hard-wired pay/receive direction; off-by-one in caplet times (1-based vs 0-based); using the wrong expiry.
3. Keep the template's structure, variable names and output format; change only what is wrong. Do not "improve"
   conventions the task did not flag (that moves you away from the oracle).
4. Verify invariants the task implies (component risks sum to total, repricing residual ~1e-10 bp, parity).
5. Write outputs with full precision (`repr`-level floats); cross-file equalities are often tested at 1e-12.

## Curve-bootstrap templates in particular
- Pillar times from the valuation date with the stated day count; log-linear discount-factor interpolation; brentq with
  tight xtol (1e-14) per pillar; residual_bps = (model - input) * 1e4 must be ~0 for every quote.
- Fixed legs: 30/360 semi-annual (unless conventions.json says otherwise); float legs: ACT/360 quarterly with
  forwards from the projection curve, discounted on the OIS curve; PV01 = fixed-leg annuity * 1e-4.
- Keep OIS and LIBOR curves as separate objects even when their pillar dates coincide.
