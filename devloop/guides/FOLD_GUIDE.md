# Guide: folding lesson cards into the playbooks (memory library maintenance)

> Kept as used during the competition: the instructions for folding lesson cards into playbooks; `devloop/prompts/fold.md` and `devloop/guards.py` automate it. Paths are generalised: `<repo>` is this repository, `<kit>` a clone of the task kit, `<work>` a scratch directory.

Context: `<repo>/agent/memory/playbooks/*.md` are general method cards that an LLM
coding agent reads before writing a Python script for a quant-finance task. They were written from the first
~30 lesson cards. Since then ~43 more public tasks were solved and their lesson cards
(`<work>/lessons/<unit>.md`) record what the hidden graders actually checked, the method that
passed, pitfalls and reusable snippets. Your job: fold the GENERAL knowledge from the assigned lesson cards
into the playbooks, so that an agent facing a *different but similar* task benefits.

Read `devloop/guides/PLAYBOOK_GUIDE.md` first: the playbook format and rules there still apply
(front matter `title/category/tags/summary`; sections "When this applies", "Method and formulas",
"Conventions and pitfalls (what graders check)", "Tested snippet"; ASCII math; numpy/scipy/pandas/
statsmodels/arch/sklearn only; every snippet executed with `python` before saving).

## What to fold (and what not)
- DO fold: formulas and algorithm variants that worked; convention choices that reference implementations
  used (ddof, annualisation, windows including/excluding day t, execution lags, tie-breaks, rounding before
  aggregation, interpolation schemes, day counts, sign conventions, seeding/RNG API and draw order);
  data-handling lessons (header detection in Excel, parsing of "$1.2 M"/"0.02%" strings, XML footnotes,
  amendment replay order); numerical robustness tricks; invariants graders assert. Phrase them generally:
  "Benchmark reference implementations typically ...; default to X unless the task specifies otherwise".
- DO update `tags:` so retrieval finds the card for tasks of that kind (add the domain vocabulary a task
  text would contain), and update `When this applies`.
- DO create a NEW playbook when a method family has no home (e.g. execution/TCA and intraday volume
  scheduling; regime models (HMM/Baum-Welch) + Black-Litterman; cross-venue arbitrage / prediction markets;
  ETF holdings & flow pressure; cross-currency basis swaps; weather/temperature derivatives; IPCA/latent
  factors). Same format, 60-140 lines, tested snippet.
- DO NOT copy task-specific numeric answers, task file names, ticker-specific facts, or anything that only
  identifies one particular public task. No GUIDs, never the word "canary". No text copied from instructions.
- Keep each playbook <= 140 lines (trim weaker material if needed); keep snippets <= 45 lines.

## Process
1. Read the assigned lesson cards (and, when needed, the corresponding verified solution in
   `<repo>/agent/memory/solutions/<unit>.py`).
2. Edit the assigned playbooks in place (Edit tool, not full rewrites unless necessary) and/or create new ones.
3. Re-extract and run every snippet you changed or created (`python - <<EOF` from a temp file in your
   scratchpad); fix until it runs.
4. Run `cd <repo> && python -m pytest tests/test_memory.py -q -p no:cacheprovider`
   (must stay green) and `grep -rIlE "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|canary" agent/memory/playbooks` (must print nothing).

Only edit files under `<repo>/agent/memory/playbooks/` (the playbooks assigned to
you, plus new ones you create). Another agent edits the other playbooks concurrently — do not touch theirs.

## Report back
A table: playbook | new/updated | lines | what was folded (one line) | snippet tested (y/n).
