[devloop:lesson]
You are writing a lesson card for the memory library of an LLM coding agent that solves quantitative-finance tasks
graded by hidden pytest checkers. You get one public practice task that the agent failed, together with what the
agent could not see: the task's checker source and a reference solution that passes it. Explain what the checker
really tests, the method that passes and why the agent's attempt failed.

The card is working material for curating the library: it may quote the checker's exact expectations. It must
never contain a GUID or a line marked as a canary.

Reply with the card in Markdown, exactly these sections:

# <unit-id>
category: <category>   tags: <comma, separated, lowercase>
## what the checker really tests
- exact row counts, column sets, oracle values with tolerances, invariants, flags
## method that passed
- formulas, algorithm and parameter conventions, 3-10 bullets
## why the agent failed
- the concrete difference between the agent's outputs and what the checker expects, and its likely cause
## pitfalls & data quirks
- anything that costs an iteration: date parsing, ddof, sort order, sign conventions, units, NaN handling
## general lesson
- 1-4 bullets that an agent facing a DIFFERENT but similar task should know (no numbers or file names of this task)
## reusable snippets
```python
# at most 25 lines of the most reusable piece (loader, formula, output writer)
```
