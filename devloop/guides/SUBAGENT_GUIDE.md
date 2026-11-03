# Guide: writing verified memory solutions for Agenthon T1 units

> Kept as used during the competition: the instructions for writing verified solutions and lesson cards; `devloop/prompts/solve.md` and `lesson.md` are derived from it. Paths are generalised: `<repo>` is this repository, `<kit>` a clone of the task kit, `<work>` a scratch directory.

You are producing *verified reference solutions* for public practice units of a quant-finance
coding benchmark. Each solution is a single, compact, self-contained Python script that passes the
unit's own offline checker. They will later be used as in-context examples for an LLM agent, so
clarity and compactness matter more than cleverness.

## Locations
- Units: `<kit>/units/<unit-id>/` with `instruction.md`, `card.toml`,
  `environment/data/*` (input data), `checks/test_outputs.py` (+ maybe `checks/reference_data/`).
- Project: `<repo>/`. Write solutions to
  `<repo>/agent/memory/solutions/<unit-id>.py`.
- Harness (run from the project dir):
  `cd <repo> && bash tests/solve_unit.sh <unit-id> agent/memory/solutions/<unit-id>.py <kit>`
  It runs the script with `DATA_DIR`/`OUTPUT_DIR` set, then runs the checker; the tail shows
  `N passed` or failures. `TAIL=60` env shows more lines. Output dir is printed as `OUT=/tmp/sol-out-XXXX`.
- Python: `python` (numpy, scipy, pandas, statsmodels, sklearn, arch, matplotlib, numba, ...).
  Do NOT pip install anything.

## Workflow per unit
1. Read `instruction.md` fully and `checks/test_outputs.py` fully (also any `checks/reference_data/*`).
   The checker is the ground truth: exact file names, exact column names, exact row counts,
   JSON key names, value ranges, oracle values with tolerances, boolean flags, sort order.
2. Look at the data files (`head`, shapes, column names, date ranges, NaNs).
3. Write the script, run the harness, iterate until **all tests pass**. If an oracle value cannot
   be matched after reasonable effort, note exactly which test fails and why, and move on.
4. Write a lesson card (see below).

## Script conventions (mandatory)
```python
# memory-solution
# unit: <unit-id> | category: <one of: derivatives-pricing, rates-curves, risk, volatility, backtest-factors, fx-commodities-crypto, microstructure, text-filings, credit, portfolio, data-engineering> | tags: comma, separated, lowercase
# deliverables: file1.json, file2.csv, ...
# <2-8 comment lines with the key formulas / method / conventions that made the checker pass>
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
```
- Read inputs via `os.path.join(DATA_DIR, "<file>")` (the instruction says `/app/<file>` or
  `/app/data/<file>`; the harness maps both to DATA_DIR). Write outputs via `os.path.join(OUT_DIR, ...)`.
- Keep it compact: target 60–150 lines; no argparse, no classes unless needed, no plotting unless a
  deliverable is an image. Vectorise; the whole script should run in < 60 s (hard cap 600 s).
- Deterministic: `np.random.seed(42)` / `default_rng(42)` when randomness is involved.
- Use plain `float()`/`int()`/`bool()` casts when dumping JSON (numpy types are not JSON serialisable;
  `bool` flags must be real Python bools). Never write NaN/inf into JSON — fill or cast.
- CSV: `index=False`, columns exactly as the instruction names them, in the stated order.
- **Canary hygiene: NEVER copy any line containing "canary", "GUID", or "BENCHMARK DATA" from the
  instruction into the script, comments, or outputs.** Do not paste instruction text verbatim.
- Do not hard-code oracle numbers from the checker as outputs. Compute them. (Hard-coded expected
  values are forbidden; the solution must be a genuine method.)
- Do not read anything under `checks/` at run time.

## Lesson card (mandatory, one per unit)
Write `<work>/lessons/<unit-id>.md`:
```markdown
# <unit-id>
category: <category>   tags: <tags>
## what the checker really tests
- (exact row counts, column sets, oracle values + tolerances, invariants, flags)
## method that passed
- (formulas / algorithm / parameter conventions, in 3-10 bullets)
## pitfalls & data quirks
- (anything that cost an iteration: date parsing, ddof, sort order, sign conventions, units, NaN handling)
## reusable snippets
```python
# <= 25 lines of the most reusable piece (loader, formula, output writer)
```
```

## Report back
When done with all assigned units, reply with a table: unit | tests passed/total | runtime | notes
(including any test you could not make pass). Do not modify any file outside
`agent/memory/solutions/<your-units>.py` and `<work>/lessons/`.
