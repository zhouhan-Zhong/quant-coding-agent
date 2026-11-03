# Guide: writing playbooks for the agent memory library

> Kept as used during the competition: the instructions for writing playbooks; `devloop/prompts/fold.md` carries its format and rules. Paths are generalised: `<repo>` is this repository, `<kit>` a clone of the task kit, `<work>` a scratch directory.

A playbook is a short, general method card that an LLM coding agent reads *before writing a Python
script* for a quant-finance task (pricing, risk, rates, backtests, data engineering...). The tasks are
graded by hidden pytest checkers that compare files (JSON/CSV) against a reference implementation with
tolerances and assert financial invariants. Playbooks must therefore be precise about formulas,
conventions and output hygiene — not tutorials.

## Location and format
Write each playbook to `<repo>/agent/memory/playbooks/<slug>.md` (slug = lowercase,
hyphens). Format (front matter is mandatory, keys exactly as shown):

```markdown
---
title: <Short title>
category: <derivatives-pricing | rates-curves | risk | volatility | backtest-factors | fx-commodities-crypto | microstructure | text-filings | credit | portfolio | data-engineering | numerics>
tags: <comma, separated, lowercase keywords that a task text would contain — include synonyms>
summary: <one line: what this card gives you>
---
## When this applies
<2-4 lines: task phrases / deliverables that signal this method>

## Method and formulas
<the exact formulas, in plain ASCII math (no LaTeX), with every symbol defined; the standard
textbook conventions (e.g. "q = continuous dividend yield", "T in years", "sigma annualised");
variants that tasks commonly ask for>

## Conventions and pitfalls (what graders check)
<bullet list: default parameters, ddof, annualisation, sign conventions, sort orders, edge cases,
numerical pitfalls, invariants a checker is likely to assert (parity, monotonicity, bounds),
typical tolerances>

## Tested snippet
```python
<<= 40 lines of numpy/scipy/pandas code implementing the core formula(s) as functions, self-contained,
verified to run>
```
```

## Rules
- Length: 50–120 lines per playbook. Dense and concrete; no filler, no history, no references list.
- Every snippet must be executed once with `python` before you save it (run it in a
  temporary file under /tmp and check it prints a sane number). Do not leave untested code.
- Use only numpy, scipy, pandas, statsmodels, arch (GARCH), scikit-learn — nothing else.
- Never include any GUID/UUID, the word "canary", or text copied from a task instruction.
- Playbooks are general-purpose: they describe methods, not a particular task's answer. Do not embed
  task-specific numeric answers or file names of a specific task; generic example file names like
  `results.json` are fine.
- Consult the lesson cards in `<work>/lessons/*.md` (written while solving public tasks):
  they contain the conventions that actually made checkers pass — fold the general lessons into the
  relevant playbooks (as conventions/pitfalls), without copying task-specific values.
- Verified solutions for reference style are in `<repo>/agent/memory/solutions/`.

## Report back
Reply with a list: slug | title | lines | snippet tested (y/n). Do not modify files elsewhere.
