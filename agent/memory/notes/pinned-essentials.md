---
title: Essentials that decide pass/fail (always shown)
category: meta
pinned: true
tags: essentials, checklist, grader, json, csv, bool, int, nan, exact names, seed, ddof, window
summary: Twelve rules distilled from the hidden graders of 86 public tasks; violating any one of them fails a unit.
---
1. Names are literal: file names, JSON keys, CSV columns, list order, sort order exactly as the task text/template.
   Literal numbers inside a template ("n_prices": 21) are asserted exactly.
2. Types are checked: flags -> Python bool (cast bool(); numpy.bool_ fails), counts -> int, numbers -> finite float.
   No NaN/Infinity anywhere in JSON; null only where the task says null. Dates -> "YYYY-MM-DD" strings.
3. Summary values must equal the cells they summarise: compute them from the values exactly as written (when the task
   rounds a table to N dp, aggregate the rounded values; otherwise never round).
4. Follow the stated formula variant literally (even when a textbook differs): ddof, annualisation factor, log vs
   simple returns, compounding, day count, 0- vs 1-based indices, which curve discounts which leg, sign conventions.
5. Windows and cleaning as stated and in order: sort -> dedupe -> ffill/dropna -> slice the last N rows; counts of
   survivors are often asserted.
6. Randomness: seed exactly as given (np.random.seed(seed) vs default_rng(seed) are different streams; use the API the
   task names, default_rng when it says "Generator"), draw in the task's order, one fresh draw per scenario.
7. Oracle tolerances are tight for analytics (1e-6 relative when the checker recomputes from inputs) and loose for
   MC (3-4 std errors, 5%): never trade exactness for speed in analytic steps; keep MC sizes as specified.
8. Invariants are asserted: parity, no-arbitrage bounds, monotonicity in strike/maturity, components sum to total,
   probabilities in [0,1], American >= European, in + out = vanilla, VaR99 >= VaR95, ES >= VaR, DF decreasing.
9. Self-consistency flags (mc_validates, parity_holds, smile_present ...) are recomputed by the grader from your own
   numbers with the task's threshold: compute them from the written values.
10. Read inputs from the real paths listed in the prompt (the task may call a file by another name); read run
    parameters from params.json when it exists; import shipped helper modules instead of re-implementing them.
11. Write only the deliverables into the output directory (create it first); re-open each file to confirm it parses.
12. Keep the script deterministic, single-file, under ~60 s, exit code 0; guard divisions, logs of non-positive
    values and empty slices so no NaN is produced silently.
