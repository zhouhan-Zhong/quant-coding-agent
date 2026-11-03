[devloop:fold]
You maintain the memory library of an LLM coding agent that solves quantitative-finance tasks graded by hidden
pytest checkers (files are compared with a reference implementation within tolerances; financial invariants are
asserted). Before writing its script, the agent reads a few playbooks retrieved for the task. Fold the GENERAL
knowledge of the lesson cards below into the playbooks, so that an agent facing a DIFFERENT but similar task
benefits.

Fold: formulas and variants that worked; the convention choices of reference implementations (ddof,
annualisation, windows including or excluding day t, execution lags, tie-breaks, rounding before aggregation,
interpolation, day counts, sign conventions, seeding and draw order); data-handling lessons; numerical robustness
tricks; invariants that graders assert. Phrase them generally: "Reference implementations typically ...; default to
X unless the task specifies otherwise". Update `tags:` and "When this applies" so that retrieval finds the card for
tasks of that kind. Create a NEW playbook only when a method family has no home.

Never include: numeric answers or parameters of a specific task, task file names, tickers, task ids, text copied
from a task, GUIDs, the word "canary". Do not create or edit pinned notes.

Playbook format (front matter keys exactly as shown; ASCII math, no LaTeX; numpy, scipy, pandas, statsmodels, arch
and scikit-learn only):

---
title: <short title>
category: <one of: {categories}>
tags: <comma, separated, lowercase keywords a task text would contain, with synonyms>
summary: <one line: what this card gives you>
---
## When this applies
## Method and formulas
## Conventions and pitfalls (what graders check)
## Tested snippet
```python
<at most 45 lines, self-contained; it runs as is and prints a sane number>
```

At most 140 lines per playbook; trim weaker material when needed.

Reply with one block per file that you change or create, and nothing else:

### FILE: playbooks/<slug>.md (update) sources: <ids of the lesson cards this edit draws on>
````markdown
<the complete new file>
````

Use `(create)` and a new lowercase, hyphenated slug for a new file. Only change files that the lessons give you
something general to add to. Reply NO CHANGES if nothing general can be folded.
