---
title: What the hidden graders actually assert
category: meta
tags: grader, checker, pytest, tolerance, oracle, invariants, row count, columns, keys, bool, json, csv, deliverables, pass criteria
summary: The recurring assertion patterns of the hidden pytest checkers (86 public units studied) and how to satisfy each class of test.
---
## When this applies
Always. Every task is graded by a pytest file that reads the deliverables from the output directory and
asserts (a) structure, (b) financial invariants, (c) oracle values with tolerances. A single failing
assertion fails the unit, so structure and conventions matter as much as the model.

## The seven test classes (ordered by how often they fail)
1. STRUCTURE: file exists and is non-empty; JSON parses and is an object with EXACT key names; CSV has the
   EXACT column set (order usually free unless the task says "in this exact order"); exact row counts
   ("21 rows", "one row per maturity x strike"); parametrised tests over the maturity/strike grid; sort order
   ("sorted by T then K ascending"). Keys are compared literally: `n_returns` != `num_returns`, `S0` != `s0`.
2. TYPES: `isinstance(x, bool)` for flags (numpy.bool_ FAILS; cast with bool()), `isinstance(n, int)` for counts
   (write int(), not 692.0), floats finite (`math.isfinite`; NaN/inf in JSON fails or does not even parse),
   strings for dates ("YYYY-MM-DD"), `None`/null where the task says null (never NaN).
3. ORACLE VALUES: a reference implementation of the SAME method computed the number; typical tolerances are
   rtol 1e-6 for deterministic analytics (recomputed from the inputs by the checker), 1e-3 to 2% for calibrations,
   5% for Monte Carlo / optimisation outputs, often with an absolute floor. Hard-coded expected values appear for
   the fixed inputs (e.g. the number of returns in the file, the last close, an ATM price). To hit them: use
   EXACTLY the stated formula variant, ddof, window, seed/RNG API, compounding, day count and data cleaning rules.
4. INVARIANTS: no-arbitrage bounds (call >= max(S e^{-qT} - K e^{-rT}, 0), price <= S), monotonicity in strike /
   maturity / barrier, put-call parity (abs tol ~1e-6..1e-2), American >= European, in + out = vanilla for barriers,
   component risks sum to the total (1e-10), weights sum to 1, probabilities in [0,1], correlation in [-1,1],
   implied vols in (0.01, 1), positive prices, cumulative returns consistent with trade PnL, discount factors
   decreasing and in (0,1], VaR_99 >= VaR_95, ES >= VaR, 10-day VaR >= 1-day VaR.
5. CROSS-FILE CONSISTENCY: summary.json values must equal the corresponding CSV cells (often exact or 1e-6):
   compute summaries FROM the same DataFrame you wrote; write floats unrounded unless the task rounds.
6. SELF-CONSISTENCY FLAGS: `mc_validates`, `parity_holds`, `smile_present`, `feller_condition_satisfied`:
   the checker recomputes the condition from your own numbers and compares to your flag; define the flag with the
   task's threshold and compute it from the written values.
7. PLAUSIBILITY RANGES: sigma in (0.05, 0.6), Sharpe in (-3, 5), prices in a wide bracket, counts > 0, log-likelihood
   above a floor, kurtosis > 0 for daily returns. Fail only if the method is wrong or units are off (percent vs decimal,
   daily vs annual).

## Patterns worth remembering
- Templates in the task text with `<int>`, `<float>`, `<bool>` ARE the schema; literal numbers in them (e.g.
  "n_maturities": 3) are asserted exactly.
- "Debug the template" tasks (a template.py with planted bugs): the checker recomputes a canonical oracle with
  rtol 1e-6 — fix every bug (demeaning, outer vs dot, hard-coded z, ignored window parameter, inverted formula)
  and keep the template's structure and parameter names.
- Generic verifier tasks (params.json + glossary.json + solution.json "intermediates"): every checkpoint named in
  the instruction's solution.json template must be present and within tolerance; extra entries are harmless.
- Fixed constants that the checker reads back literally: seed, n_simulations, n_paths, window lengths — write them
  exactly as given (ints).
- Tests usually read CSV with pandas or csv.DictReader: an empty cell becomes NaN / "" — when the task says a cell is
  "literally empty", write NaN via pandas (not "nan", not 0).
- Rounding: only round where the task says so (then use round(x, n) on the written value and compute summaries
  from the rounded or unrounded values as the task implies); tolerance tests on rounded outputs can fail if you
  round too aggressively (6 dp is the usual minimum).
- Plots/HTML: checked only for existence and non-trivial size; use matplotlib Agg / plotly write_html(include_plotlyjs="cdn").
- Time limits: the checker is separate from the run; a slow script (minutes) is fine only within the card's timeout —
  vectorise MC and keep PDE grids moderate.
