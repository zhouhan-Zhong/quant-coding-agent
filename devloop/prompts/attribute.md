[devloop:attribute]
You are analysing why an LLM coding agent failed a quantitative-finance programming task. A hidden pytest checker
graded the task. You see the task text, the checker's complaints, signals from the agent's log and its outputs.

Choose exactly one category:
{categories}

Rules:
- Pick the most specific cause that explains the checker failure. A rule-based label is given as a hint; override
  it only when the evidence clearly says otherwise.
- "conventions": the formats are right but numbers differ because of a convention (annualisation, ddof, calendars,
  units, windows, seeds, interpolation, day counts, signs). "invariants": a relation the task text states is
  violated (parity, bounds, MC vs analytic, totals). "checker-specific": the checker follows its own protocol.
- root_cause: one sentence about THIS task.
- general_lesson: what an agent facing a DIFFERENT but similar task should do, phrased generally ("When a task asks
  for X, ..."). No numbers, file names, tickers or ids from this task. Leave it empty if nothing general follows.
- tags: 3-8 lowercase keywords that a similar task's text would contain.

Reply with one JSON object and nothing else:
{"category": "...", "root_cause": "...", "general_lesson": "...", "tags": ["..."], "confidence": 0.0}
