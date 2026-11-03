# memory-solution
# unit: t1-double-sort | category: backtest-factors | tags: double-sort, dependent-sort, betting-against-beta, momentum, quintiles, equal-weight, turnover, transaction-costs, parquet
# deliverables: strategy_returns.csv
# input is /app/data/stock_data.parquet in the task image (a copy of the shipped stock_chars.pqt) -> accept either name
# per month: dropna(betabab_1260d, ret_12_7, ret_lead1m); skip if n < 5; sort (beta asc, identifier asc); q = min(5, 5*r//n + 1)
# keep q == 1, re-sort (ret_12_7 asc, identifier asc), same label formula on bucket size, keep q == 5 (empty -> skip month, keep prev weights)
# gross = mean ret_lead1m of corner; turnover = 1.0 for the first emitted portfolio else sum |w_t - w_prev| over the union of ids
# net = gross - 0.0015 * turnover; row labelled by the formation month-end
import os
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

path = next(p for p in (os.path.join(DATA_DIR, "stock_data.parquet"), os.path.join(DATA_DIR, "stock_chars.pqt")) if os.path.exists(p))
cols = ["betabab_1260d", "ret_12_7", "ret_lead1m"]
df = pd.read_parquet(path)[["identifier", "date"] + cols].dropna(subset=cols)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "identifier"]).reset_index(drop=True)

def labels(n):
    return np.minimum(5, (5 * np.arange(n)) // n + 1)

rows, prev_w = [], None
for date, grp in df.groupby("date", sort=True):
    n = len(grp)
    if n < 5:
        continue
    g = grp.sort_values(["betabab_1260d", "identifier"]).reset_index(drop=True)
    low = g[labels(n) == 1]
    low = low.sort_values(["ret_12_7", "identifier"]).reset_index(drop=True)
    target = low[labels(len(low)) == 5]
    k = len(target)
    if k == 0:
        continue                                                     # empty corner: skip, keep prev_w for turnover
    cur_w = {int(i): 1.0 / k for i in target["identifier"]}
    if prev_w is None:
        turnover = 1.0
    else:
        turnover = sum(abs(cur_w.get(i, 0.0) - prev_w.get(i, 0.0)) for i in set(prev_w) | set(cur_w))
    gross = float(target["ret_lead1m"].mean())
    rows.append({"date": date.strftime("%Y-%m-%d"), "net_return": gross - 0.0015 * turnover})
    prev_w = cur_w

pd.DataFrame(rows, columns=["date", "net_return"]).to_csv(os.path.join(OUT_DIR, "strategy_returns.csv"), index=False)
