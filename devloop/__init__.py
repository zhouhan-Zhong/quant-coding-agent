"""devloop: the outer development loop of the quant coding agent, as code.

One round:  evaluate the agent on the public units  ->  attribute every failure  ->  write lesson cards for the
failures that knowledge can fix  ->  fold the general part of those lessons into the memory library  ->  verify the
candidate library  ->  A/B it against the baseline, held out (a unit never sees documents derived from its own
lessons)  ->  keep it only if it helps.  Recurring failure classes that need code changes are reported as mechanism
candidates, with ready-to-run flag ablations; agent code is never edited automatically.

Run `python -m devloop --help`; see devloop/README.md.
"""

__version__ = "1.0.0"
