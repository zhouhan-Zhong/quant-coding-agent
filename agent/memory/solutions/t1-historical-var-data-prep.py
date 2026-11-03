# memory-solution
# unit: t1-historical-var-data-prep | category: risk | tags: data-cleaning, historical-var, mixed-date-formats, trading-calendar, drop-duplicates, outliers, listwise-nan, percentile-linear, equal-weight
# deliverables: results.json, solution.json
# header strip -> parse 3 date formats (YYYY-MM-DD, MM/DD/YYYY, DD-Mon-YY) per pattern -> drop dates not in calendar
# -> drop_duplicates(date, keep='last') in file order -> drop rows with any |r| > 0.25 (NaN-safe: NaN is not an outlier)
# -> dropna listwise -> sort by date; count rows removed at each step on the rows still present
# portfolio = row mean of 5 assets (equal weights); VaR = -np.percentile(p, q, method='linear'); worst day = min
import os, json
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "returns_raw.csv"))
df.columns = [c.strip() for c in df.columns]
assets = ["SPY", "TLT", "GLD", "IEF", "EFA"]
n_raw = len(df)

s = df["date"].astype(str).str.strip()
parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
for pat, fmt in [(r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"), (r"^\d{2}/\d{2}/\d{4}$", "%m/%d/%Y"), (r"^\d{2}-[A-Za-z]{3}-\d{2}$", "%d-%b-%y")]:
    m = s.str.match(pat)
    parsed[m] = pd.to_datetime(s[m], format=fmt)
df["date"] = parsed.dt.strftime("%Y-%m-%d")

cal = set(pd.read_csv(os.path.join(DATA_DIR, "trading_calendar.csv"))["date"].astype(str))
keep = df["date"].isin(cal); n_nontrading = int((~keep).sum()); df = df[keep]
n0 = len(df); df = df.drop_duplicates(subset="date", keep="last"); n_dup = n0 - len(df)
out = (df[assets].abs() > 0.25).any(axis=1)                        # NaN > 0.25 is False -> NaN-safe
n_out = int(out.sum()); df = df[~out]
n0 = len(df); df = df.dropna(subset=assets); n_nan = n0 - len(df)
df = df.sort_values("date").reset_index(drop=True)
n_clean = len(df)

port = df[assets].to_numpy(float).mean(axis=1)
i_min = int(np.argmin(port))
json.dump({"n_observations_clean": int(n_clean),
           "var_95_portfolio": float(-np.percentile(port, 5, method="linear")),
           "var_99_portfolio": float(-np.percentile(port, 1, method="linear")),
           "worst_day_return": float(port[i_min]), "worst_day_date": str(df["date"].iloc[i_min])},
          open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
json.dump({"cleaning_report": {"n_rows_raw": int(n_raw), "n_non_trading_days_removed": n_nontrading, "n_duplicates_removed": int(n_dup),
                               "n_outliers_removed": n_out, "n_missing_rows_removed": int(n_nan), "n_rows_after_cleaning": int(n_clean)}},
          open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
