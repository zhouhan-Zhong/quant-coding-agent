# memory-solution
# unit: t1-intraday-volume-fitting-and-execution-scheduling | category: microstructure | tags: intraday-volume, volume-profile, vwap-schedule, walk-forward, ewma, winsorized, r-squared, largest-remainder, resample-5min, execution-scheduling
# deliverables: excluded_days.csv, model_performance.csv, best_model.json, final_schedule.csv
# 1-min bars -> keep 09:30..16:00 (bar start) -> groupby floor('5min') (only non-empty bins) -> keep 09:30..15:55 -> complete day = 78 bars.
# shares s[d,b] = V/sum_b V. Walk-forward: test day i >= 20 uses days i-20..i-1; score = mean daily R^2 = 1 - SSE/SST (SST==0 -> 1 if exact else 0).
# Models on the 20x78 share window: mean, median, EWMA with w_L = exp(-ln2 L/20) (L=0 most recent), winsorized mean (clip to 5th/95th pct per bar);
#   every profile renormalised to sum 1. Tie-break by fixed model order (stable argmax).
# Schedule: order_day may not be in the data -> build the 78-bar grid for that date; keep bars strictly after order datetime; window = last 20 complete
#   days before order_day; restrict + renormalise profile; integer allocation = floor + largest fractional remainder (ties -> earlier bar).
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["historical_mean_profile", "historical_median_profile", "ewma_profile", "winsorized_mean_profile"]
T, W = 78, 20
order = json.load(open(os.path.join(DATA_DIR, "order.json")))
order_dt, quantity = pd.Timestamp(order["datetime"]), int(order["quantity"])

df = pd.read_csv(os.path.join(DATA_DIR, "Stock_Intraday_Data.csv"), parse_dates=["datetime"]).sort_values("datetime")
tod = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
df = df[(tod >= 9 * 60 + 30) & (tod <= 16 * 60)]
df["bar"] = df["datetime"].dt.floor("5min")
bars = df.groupby("bar").agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"), Volume=("Volume", "sum")).reset_index()
btod = bars["bar"].dt.hour * 60 + bars["bar"].dt.minute
bars = bars[(btod >= 9 * 60 + 30) & (btod <= 15 * 60 + 55)].copy()
bars["date"] = bars["bar"].dt.normalize()
counts = bars.groupby("date").size()
complete_days = counts.index[counts == T]
excluded = counts.index[counts != T]
pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in sorted(excluded)]}).to_csv(os.path.join(OUT_DIR, "excluded_days.csv"), index=False)

# shares matrix: n_days x 78 (rows in chronological day order, columns in bar order)
vol = bars[bars["date"].isin(complete_days)].sort_values("bar")["Volume"].to_numpy(float).reshape(len(complete_days), T)
S = vol / vol.sum(axis=1, keepdims=True)

ew = np.exp(-math.log(2) * np.arange(W) / 20.0)[::-1]          # chronological window: oldest has lag 19, most recent lag 0
def profiles(win):                                             # win: W x T shares, chronological
    lo, hi = np.percentile(win, 5, axis=0), np.percentile(win, 95, axis=0)
    p = {"historical_mean_profile": win.mean(axis=0), "historical_median_profile": np.median(win, axis=0),
         "ewma_profile": (ew[:, None] * win).sum(axis=0) / ew.sum(),
         "winsorized_mean_profile": np.clip(win, lo, hi).mean(axis=0)}
    return {k: v / v.sum() for k, v in p.items()}

def r2(y, yhat):
    sst = float(((y - y.mean()) ** 2).sum())
    if sst == 0.0:
        return 1.0 if np.array_equal(y, yhat) else 0.0
    return 1.0 - float(((y - yhat) ** 2).sum()) / sst

scores = {m: [] for m in MODELS}
for i in range(W, len(S)):
    prof = profiles(S[i - W:i])
    for m in MODELS:
        scores[m].append(r2(S[i], prof[m]))
n_test = len(S) - W
avg = {m: float(np.mean(scores[m])) for m in MODELS}
perf = pd.DataFrame({"model_name": MODELS, "avg_r2": [avg[m] for m in MODELS], "num_test_days": n_test})
perf.to_csv(os.path.join(OUT_DIR, "model_performance.csv"), index=False)
best = MODELS[int(np.argmax([avg[m] for m in MODELS]))]       # argmax returns the first max -> fixed priority order
json.dump({"best_model": best, "best_avg_r2": avg[best], "num_test_days": int(n_test)},
          open(os.path.join(OUT_DIR, "best_model.json"), "w"), indent=2)

# final schedule on order_day (may lie after the sample): window = the 20 complete days strictly before order_day
order_day = order_dt.normalize()
prior = np.where(complete_days < order_day)[0]
win = S[prior[-W:]]
full = profiles(win)[best]
grid = pd.date_range(order_day + pd.Timedelta(hours=9, minutes=30), order_day + pd.Timedelta(hours=15, minutes=55), freq="5min")
keep = np.where(grid > order_dt)[0]
w = full[keep] / full[keep].sum()
raw = quantity * w
q = np.floor(raw).astype(int)
resid = quantity - int(q.sum())
frac = raw - q
order_idx = np.lexsort((np.arange(len(frac)), -frac))          # largest remainder first, earlier bar wins ties
q[order_idx[:resid]] += 1
sched = pd.DataFrame({"datetime": grid[keep].strftime("%Y-%m-%d %H:%M:%S"), "scheduled_quantity": q,
                      "predicted_bar_share": w, "cumulative_scheduled_quantity": np.cumsum(q)})
sched.to_csv(os.path.join(OUT_DIR, "final_schedule.csv"), index=False)
