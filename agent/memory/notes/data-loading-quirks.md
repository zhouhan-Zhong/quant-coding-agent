---
title: Input data quirks and loading conventions
category: meta
tags: data, loading, csv, parquet, json, paths, /app/data, date parsing, duplicates, nan, ffill, window, estimation window, sort, column names, case, tickers, calendar, inner join, planted anomalies
summary: How task inputs are laid out and the cleaning conventions the reference implementations used; includes a robust loader snippet.
---
## When this applies
Every task with input files. Reference implementations are literal about cleaning order and windows, and the
checkers often hard-code counts derived from them (number of returns, valid rows, trades).

## Paths and layout
- The task text says `/app/<file>` or `/app/data/<file>`; the files are listed with their real absolute paths in
  the prompt — use those (or `DATA_DIR`/the listed path). Some task images RENAME a file on copy (e.g.
  `stock_chars.pqt` -> `/app/data/stock_data.parquet`); the prompt's file list tells you the real name.
- Helper modules shipped with the data (`fetch_data.py`, `html_to_text.py`, `template.py`) are importable from the
  data directory; a `template.py` with planted bugs is a "debug the template" task — fix the bugs, keep its structure.
- `params.json` holds run parameters (seed, windows, confidence levels, notional) — read every parameter from it
  instead of hard-coding; `glossary.json` is vocabulary only and is not needed.

## Cleaning conventions that matched the oracles
- Price files: `pd.read_csv(..., parse_dates=["date"])` (column names may be capitalised: lower-case them), sort by
  date, drop exact duplicate dates (`drop_duplicates("date", keep="first")`), then forward-fill prices if the task
  says so, then `dropna`. Order matters for the counts: sort -> dedupe -> ffill -> dropna.
- Log returns: `np.diff(np.log(close))` (n_prices - 1 returns); simple returns: `close.pct_change().dropna()`.
  std with ddof=1 unless told; annualise with sqrt(252) (daily), sqrt(12) (monthly), sqrt(52) (weekly).
- SPY-type calibration: S0 = last close, sigma = std(log returns, ddof=1) * sqrt(252), r and dividend yield as given
  in the task (do not infer them from data).
- Estimation windows: "use the most recent N observations" means slice `x[-N:]` AFTER cleaning, only if N < len(x).
- Multi-asset panels: align on common dates with an inner join; compute correlations on returns, not prices.
- Planted anomalies in "production feed" files: NaN cells, duplicated rows, outliers (e.g. +-0.42 daily returns) —
  apply exactly the rule the task states (threshold, not z-score, unless it says z-score) and count removals step by
  step on the surviving rows.
- Dates with mixed formats: parse each pattern explicitly with regex masks or `pd.to_datetime(format=...)`; write ISO.
- Monthly data: resample with `"ME"` (pandas >= 2.2; "M" is removed in pandas 3), month-end via `groupby(period).last()`;
  month labels like "YYYY-MM" are strings — match by `str(period)`.
- Treasury/FRED CSVs: missing values as "." or "ND" -> `na_values=[".", "ND"]`; column names with spaces ("1 Mo", "10 Yr").
- Intraday files: parse timestamps with timezone awareness only if the task says; sessions 09:30-16:00; count bars/day.
- Excel inputs: `pd.read_excel(path, sheet_name=...)` with openpyxl; header rows may be offset (`header=k`).
- Parquet: `pd.read_parquet(path)`; `.pqt` extension is still parquet.
- JSON inputs with nested dicts: keep the task's key order when iterating (`dict` preserves insertion order).
- Tickers: keep the file's spelling (e.g. "BRK-B"); sort with Python `sorted()` when the task says alphabetical.

## Tested snippet
```python
import os, numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")

def load_prices(name="prices.csv", date_col="date", price_col="close", ffill=False):
    df = pd.read_csv(os.path.join(DATA_DIR, name))
    df.columns = [c.strip().lower() for c in df.columns]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).drop_duplicates(date_col, keep="first").reset_index(drop=True)
    if ffill:
        df[price_col] = df[price_col].ffill()
    df = df.dropna(subset=[price_col])
    return df

def log_returns(close):
    close = np.asarray(close, dtype=float)
    return np.diff(np.log(close))

if __name__ == "__main__":
    import tempfile
    d = tempfile.mkdtemp(); DATA_DIR = d
    pd.DataFrame({"Date": ["2024-01-03", "2024-01-02", "2024-01-03", "2024-01-04"], "Close": [101.0, 100.0, 101.0, np.nan]}).to_csv(os.path.join(d, "prices.csv"), index=False)
    px = load_prices(ffill=True)
    print(len(px), px["close"].tolist(), log_returns(px["close"]).round(6).tolist())
```
