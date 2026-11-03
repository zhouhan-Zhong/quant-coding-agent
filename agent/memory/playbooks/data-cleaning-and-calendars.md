---
title: Data cleaning, calendars, returns, resampling and output hygiene
category: data-engineering
tags: csv, tsv, excel, read_csv, read_excel, parse dates, date format, timezone, utc, epoch milliseconds, sort, duplicates, drop_duplicates, source sequence, forward fill, ffill, dropna, missing data, nan, nan count, fred, trading calendar, business days, bdate_range, holidays, alignment, inner join, merge, as-of join, vintage, corporate actions, split adjustment, dividend adjustment, adjustment factor, outliers, winsorize, mad, returns, simple returns, log returns, pct_change, rolling window, min_periods, ewm, resample, monthly, month end, month start, ME, json output, csv output, nullable int, float precision, column order
summary: The cleaning order (sort -> dedupe -> ffill -> dropna), date/timezone/epoch parsing, messy-source recipes (FRED dots, SEC TSVs, offset Excel headers, "$1.2 M" strings), inner-join and month-start alignment, backward split/dividend adjustment, return conventions, pandas 3 rolling/ewm/resample defaults and NaN-free JSON/CSV outputs.
---
## When this applies
Any task that starts from CSV/TSV/Excel/parquet inputs ("clean the production feed", "align the series", "adjust for
splits and dividends", "daily/monthly returns", "rolling 20-day volatility", "resample to month-end", "monthly FRED
and daily series together") and any task whose checker counts rows (n_obs, n_valid, nan_count) or reads outputs.

## Method and formulas
Symbols: P_t price, r_t return, n_prices rows after cleaning, w window length, lam/alpha EWMA parameters.
- Loading: pd.read_csv(path); dates with pd.to_datetime(col, format=...); tz-aware strings -> utc=True then
  .dt.tz_localize(None) and .dt.normalize() for date keys. Numeric columns via pd.to_numeric(errors='coerce').
- Cleaning order that reproduces reference counts: sort_values(date, kind='stable') -> drop_duplicates(subset=date,
  keep='last' or as stated) -> ffill() (only if allowed) -> dropna() -> reset_index; count after each step. "Drop
  rows with any NaN" = dropna() without ffill; estimation windows ("last N days") are sliced AFTER cleaning.
- NaN accounting: count NaN cells of the RAW matrix first (often a checkpoint), then apply the stated fill (some
  references fill missing returns with 0, which differs from ffill).
- Keyed dedupe: keep the max source sequence per key with sort_values(key + [seq]).groupby(key).tail(1); count
  duplicate groups as (groupby(key).size() > 1).sum(); "keep last" means FILE order -> dedupe before any sort, then
  sort with kind='mergesort'; float64 ids from parquet -> int before tie-breaks.
- Alignment: inner join on the date key; returns computed AFTER alignment unless per-asset. Business-day grid
  pd.bdate_range (no holidays). Calendars implied by a data file (weekdays minus holidays): count windows in actual
  rows, never np.busday_count.
- FRED/Treasury CSVs: na_values=[".", "ND"] (blank fields are NaN anyway); daily -> resample("ME").last() (last VALID
  value of the month), then index.to_period("M").to_timestamp() so month-START stamps match monthly series; restrict
  the date range, ffill a series that ends early BEFORE dropna on the others.
- SEC flat files (TSV): sep="\t", dtype=str, keep_default_na=False; strip, collapse whitespace, upper-case keys; dates
  "%d-%b-%Y"; identifiers zfill (CIK 10 chars, CUSIP 9) - never let leading zeros become ints.
- Excel issuer files: read_excel(header=None, engine="openpyxl"), find the header row by content, drop footers and
  rows with non-numeric key fields; "$412,345.67 M" / "0.02%" / "13,579,246" parse via Decimal (etf-holdings-and-flows).
- Epoch milliseconds: pd.to_datetime(ms, unit="ms", utc=True).dt.floor("s") (drops ms jitter); filter [start, end + 1
  day); write "%Y-%m-%dT%H:%M:%SZ" or ts.isoformat().replace("+00:00", "Z").
- As-of joins: rows with date in (asof - window, asof]; the latest vintage row counts even if its value is NaN.
- Returns: simple pct_change(), log diff(log P); n_returns = n_prices - 1; compounding prod(1+r) - 1 / sum of logs;
  annualise mean x 252, std x sqrt(252) (ddof=1); calendar-time annualised log return = sum(log r) /
  ((last - first).days / 365.25).
- Corporate actions (backward, chronological): rows with date < action date are scaled. Split "a:b" (a new for b
  old): price x b/a, volume x a/b. Dividend D: price x (C_ex - D)/C_ex with the UNADJUSTED close on the ex-date (prior
  trading day if missing, searchsorted side='right' - 1); cumulative factors multiply; last row untouched.
- Outliers: winsorize at quantiles, |z| > k, or MAD rule |x - med| > k 1.4826 MAD; flag rather than drop unless told.
- Rolling: rolling(w) -> NaN for the first w-1 rows; rolling std ddof=1; shift(1) for past-only. EWMA:
  ewm(alpha=1-lam, adjust=False) = recursion x_t = lam x_{t-1} + (1-lam) s_t seeded with s_0 (adjust=True differs).
- Resampling: 'ME' month-end ('M' raises in pandas 3), 'QE', 'W-FRI'; groupby(index.to_period('M')).last() keeps
  the actual last trading date; .first() opens, .sum() volumes.
- Output hygiene: CSV index=False, exact column order, ISO date strings, NaN as empty cells, nullable ints via
  astype("Int64") (blank instead of 1.0); JSON with float()/int()/bool() and NaN/inf -> None.

## Conventions and pitfalls (what graders check)
- Row counts are asserted exactly -> follow the cleaning order literally; keep rule (first/last) changes values.
- Newest-first downloads: sort before diff/pct_change. Explicit date formats (dayfirst mismatches swap fields).
- Compare/merge only naive-UTC or same-zone timestamps; strip time before joining daily data.
- ffill BEFORE dropna and AFTER sorting; never bfill unless told.
- Adjusted vs raw prices: use what the spec says; back-adjust strictly before the action date.
- min_periods sets the first valid index; pandas rolling std ddof=1 vs numpy std ddof=0.
- Resample labels are period ends; month-start stamps need to_period("M").to_timestamp().
- np.bool_ fails isinstance(x, bool) and JSON; keys must be strings; keep float64 and round only at the final write.

## Tested snippet
```python
import io, json, math, numpy as np, pandas as pd

def load_prices(csv_text_or_path, date_col="date", fmt=None):          # sort -> dedupe -> (ffill) -> dropna
    df = pd.read_csv(csv_text_or_path)
    df[date_col] = pd.to_datetime(df[date_col], format=fmt, utc=True).dt.tz_localize(None).dt.normalize()   # tz-aware -> naive UTC dates
    df = df.sort_values(date_col, kind="stable").drop_duplicates(subset=date_col, keep="last")
    n_raw = len(df); df = df.ffill().dropna().reset_index(drop=True)      # ffill small gaps, then drop leading NaNs
    return df, {"n_rows_after_dedupe": int(n_raw), "n_valid": int(len(df))}
def returns(p, kind="simple"): return p.pct_change() if kind == "simple" else np.log(p).diff()   # first value NaN
def backward_adjust(dates, close, volume, actions):                      # actions: [{date, type: split 'a:b' | dividend amount}]
    close_a, vol_a, f_px, f_vol = close.astype(float).copy(), volume.astype(float).copy(), 1.0, 1.0
    for a in sorted(actions, key=lambda a: a["date"]):
        d = np.datetime64(pd.Timestamp(a["date"])); n_prior = int(np.searchsorted(dates, d, side="left"))
        if a["type"] == "split":
            new, old = (float(x) for x in a["ratio"].split(":")); fp, fv = old / new, new / old
        else:
            c_ex = close[int(np.searchsorted(dates, d, side="right")) - 1]; fp, fv = (c_ex - a["amount"]) / c_ex, 1.0
        close_a[:n_prior] *= fp; vol_a[:n_prior] *= fv; f_px *= fp; f_vol *= fv
    return close_a, vol_a, f_px, f_vol
def to_monthly(s, daily):                                                # FRED-style: '.'/blank -> NaN; daily -> last valid obs of the month
    s = pd.to_numeric(s, errors="coerce"); s = s.resample("ME").last() if daily else s
    s.index = s.index.to_period("M").to_timestamp(); return s            # month-START stamps line up with monthly series
def json_safe(o):                                                        # NaN/inf -> None, numpy scalars -> python
    if isinstance(o, dict): return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)): return [json_safe(v) for v in o]
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, (int, np.integer)): return int(o)
    if isinstance(o, (float, np.floating)): return None if not math.isfinite(float(o)) else float(o)
    return o

csv = "date,close,volume\n2024-01-03,101,1000\n2024-01-02,100,900\n2024-01-03,102,1100\n2024-01-04,,1200\n2024-01-05,104,1300\n2024-01-08,208,1400\n"
df, info = load_prices(io.StringIO(csv))
ca, va, fp, fv = backward_adjust(df["date"].to_numpy(), df["close"].to_numpy(), df["volume"].to_numpy(), [{"date": "2024-01-08", "type": "split", "ratio": "1:2"}])
r = returns(df["close"]); rsd = r.rolling(3, min_periods=3).std(ddof=1)   # NaN until 3 returns exist
print(info, df["date"].dt.strftime("%Y-%m-%d").tolist(), df["close"].tolist(), ca.tolist(), va.tolist(), fp, fv, r.round(4).tolist(), rsd.round(4).tolist())
fred = pd.read_csv(io.StringIO("observation_date,DGS10\n2024-01-30,4.0\n2024-01-31,.\n2024-02-01,4.1\n2024-02-29,\n"), na_values=["."], parse_dates=["observation_date"]).set_index("observation_date")["DGS10"]
ms = pd.Series([1704067200123, 1704096000456]); ts = pd.to_datetime(ms, unit="ms", utc=True).dt.floor("s")   # epoch ms with jitter -> whole seconds
px = pd.DataFrame({"k": ["a", "a", "b"], "seq": [2, 5, 1], "v": [1.0, 2.0, 3.0]}); dd = px.sort_values(["k", "seq"]).groupby("k", as_index=False).tail(1)
print(to_monthly(fred, True).to_dict(), ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(), dd["v"].tolist(), pd.Series([1.0, None]).astype("Int64").tolist())
print(json.dumps(json_safe({"n": np.int64(3), "x": np.float64(np.nan), "ok": np.bool_(True), "arr": np.array([1.5, np.inf])})))
```
