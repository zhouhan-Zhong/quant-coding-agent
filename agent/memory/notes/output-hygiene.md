---
title: Output hygiene for JSON and CSV deliverables
category: meta
tags: json, csv, output, deliverables, nan, null, bool, int, float, rounding, sort, index, empty cell, encoding, parquet, html, png, os.makedirs, output_dir
summary: Exact rules for writing deliverable files so that structural and type assertions pass; includes a tested writer helper.
---
## When this applies
Every task. Deliverables are JSON objects/arrays, CSV tables, occasionally parquet, HTML or PNG.

## Rules
- Output directory: `OUT = os.environ.get("OUTPUT_DIR", "/app/output"); os.makedirs(OUT, exist_ok=True)`; write every
  file with `os.path.join(OUT, name)`. Never write anywhere else; never write extra files (no temporary CSVs, no logs).
- JSON scalars: cast before dumping — `float(x)`, `int(x)`, `bool(x)`, `str(x)`; numpy scalars raise TypeError or
  serialise wrongly; `numpy.bool_` is not `bool`. Dates as "YYYY-MM-DD" strings via `strftime("%Y-%m-%d")`.
- Never emit NaN or Infinity in JSON (`json.dump` writes the literal `NaN` which is invalid JSON and fails
  `json.load` in strict checkers). Replace with `None` only when the task allows null; otherwise fix the computation
  (guard division by zero, empty slices, log of non-positive numbers).
- Keep JSON key names, nesting and the order of list items exactly as the task template; do not add wrapper keys.
  Lists keep the task's ordering (e.g. tickers in input order, maturities ascending as listed).
- Counts are ints (`int(len(df))`), flags are bools, everything numeric else is float; write literal constants from
  the task (seed, n_paths, window) exactly.
- CSV: `df.to_csv(path, index=False)` unless the task wants the index; column names exactly as specified and in the
  specified order (`df = df[cols]`); dates formatted as strings before writing; sort as the task states
  (`sort_values(["T", "K"]).reset_index(drop=True)`).
- Empty cells: when the task wants blank cells (undefined IC, pnl on buy rows), write NaN (pandas writes "") or an
  empty string; never the strings "nan"/"None". Otherwise avoid NaN in CSV numeric columns (fill or drop per task).
- Precision: do not round unless told; if told "6 decimals" use `float_format="%.6f"` or `round(...)`; make summary
  values consistent with the CSV values they summarise (compute both from the same arrays).
- Bool columns in CSV are written as True/False strings — fine for pandas readers; if the task wants 0/1, cast to int.
- Parquet: `df.to_parquet(path, index=False)` (pyarrow installed). HTML plots: plotly `fig.write_html(path,
  include_plotlyjs="cdn")`; PNG: matplotlib with the Agg backend, `plt.savefig(path, dpi=100); plt.close()`.
- Encoding: default UTF-8; avoid non-ASCII in keys.
- Before finishing, re-open every written file (json.load / pd.read_csv) to make sure it parses.

## Tested snippet
```python
import json, math, os
import numpy as np, pandas as pd

OUT = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT, exist_ok=True)

def clean(obj):
    """Recursively convert numpy types; refuse NaN/inf (raise so the bug is visible instead of writing bad JSON)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not math.isfinite(f):
            raise ValueError("non-finite value in JSON output")
        return f
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).strftime("%Y-%m-%d")
    return obj

def write_json(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(clean(obj), f, indent=2)

def write_csv(name, df, cols=None, sort_by=None, float_format=None):
    if sort_by:
        df = df.sort_values(sort_by).reset_index(drop=True)
    if cols:
        df = df[cols]
    df.to_csv(os.path.join(OUT, name), index=False, float_format=float_format)

if __name__ == "__main__":
    write_json("summary.json", {"n": np.int64(21), "flag": np.bool_(True), "x": np.float64(1.5)})
    write_csv("t.csv", pd.DataFrame({"K": [2, 1], "T": [0.5, 0.5], "p": [1.0, np.nan]}), cols=["T", "K", "p"], sort_by=["T", "K"])
    print(json.load(open(os.path.join(OUT, "summary.json"))), open(os.path.join(OUT, "t.csv")).read())
```
