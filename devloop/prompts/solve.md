[devloop:solve]
You are producing a verified reference solution for a public practice task of a quant-finance coding benchmark:
one compact, self-contained Python script that passes the task's own checker. You are shown the checker's source;
it is the ground truth for file names, columns, row counts, JSON keys, oracle values, tolerances, flags and sort
orders. The script will be shown to an LLM agent as an example when it solves OTHER tasks, so clarity and
compactness matter more than cleverness.

Script conventions (mandatory):

```python
# memory-solution
# unit: <unit-id> | category: <category> | tags: comma, separated, lowercase
# deliverables: file1.json, file2.csv
# <2-8 comment lines with the key formulas and conventions that made the checker pass>
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
```

- Read inputs with os.path.join(DATA_DIR, <file>) and write outputs with os.path.join(OUT_DIR, ...).
- 60-150 lines; no argparse, no plotting unless a deliverable is an image; it must run in under 60 seconds.
- Deterministic: np.random.default_rng(42), or the seed the task states.
- Cast with float()/int()/bool() when dumping JSON; never write NaN or inf into JSON.
- CSV: index=False, columns exactly as the task names them, in the stated order.
- Compute everything. Never paste expected numbers from the checker, and never read anything under checks/.
- Never copy a line containing a GUID, "canary" or "BENCHMARK DATA" into the script.

Reply with the complete script as ONE ```python block. After a failed run you get its output and the checker's
report; reply with the complete corrected script.
