# memory-solution
# unit: t1-cross-sectional-momentum | category: backtest-factors | tags: cross-sectional-momentum, long-short, 12-2-formation, data-cleaning, ffill, drop-duplicates, monthly-returns, sharpe, max-drawdown
# deliverables: results.json, portfolio.csv
# cleaning order: stable sort by date -> drop_duplicates(date, keep='last') -> ffill -> dropna   (118 clean rows here)
# monthly simple returns r_t = P_t/P_{t-1} - 1 (starts at 2nd clean row); signal_t = sum r_{t-12..t-2} (11 months, skip most recent)
# valid months t >= 12 of the return table; rank signal desc with ties broken by CSV column order (stable argsort on -signal)
# long = mean of top-3 r_t, short = mean of bottom-3 r_t, signal_return = long - short; cum = cumprod(1+r); total = cum[-1]-1
# ann_ret = (1+total)^(12/N) - 1; ann_vol = std(ddof=1)*sqrt(12); sharpe = ann_ret/ann_vol; mdd = max((peak-cum)/peak)
# checker tolerance is 1e-12 absolute -> keep plain float64 arithmetic, no rounding, write CSV with default float repr
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

raw = pd.read_csv(os.path.join(DATA_DIR, "prices.csv"))
tickers = [c for c in raw.columns if c != "date"]                      # original column order = tie-break order
df = raw.sort_values("date", kind="stable").drop_duplicates(subset="date", keep="last")
df = df.ffill().dropna().reset_index(drop=True)
num_clean_rows = int(len(df))

px = df[tickers].to_numpy(float)
rets = px[1:] / px[:-1] - 1.0                                          # (T-1, n_stocks); row t <-> date df.date[t+1]
ret_dates = df["date"].iloc[1:].tolist()
T = rets.shape[0]

rows = []
for t in range(12, T):
    sig = rets[t - 12:t - 1].sum(axis=0)                               # months t-12 .. t-2 inclusive (11 obs)
    order = np.argsort(-sig, kind="stable")                            # descending, ties -> lower column index first
    long_r = float(rets[t, order[:3]].mean()); short_r = float(rets[t, order[-3:]].mean())
    rows.append((ret_dates[t], long_r - short_r))

port = pd.DataFrame(rows, columns=["date", "signal_return"])
port["cumulative_return"] = np.cumprod(1.0 + port["signal_return"].to_numpy())
pr = port["signal_return"].to_numpy(); cum = port["cumulative_return"].to_numpy()
N = len(pr)
total_return = float(cum[-1] - 1.0)
ann_ret = float((1.0 + total_return) ** (12.0 / N) - 1.0)
ann_vol = float(pr.std(ddof=1) * math.sqrt(12.0))
peak = np.maximum.accumulate(cum)
res = {"num_clean_rows": num_clean_rows, "num_signal_months": int(N), "total_return": total_return,
       "annualized_return": ann_ret, "annualized_vol": ann_vol, "sharpe": ann_ret / ann_vol,
       "max_drawdown": float(((peak - cum) / peak).max())}
json.dump(res, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
port.to_csv(os.path.join(OUT_DIR, "portfolio.csv"), index=False)
