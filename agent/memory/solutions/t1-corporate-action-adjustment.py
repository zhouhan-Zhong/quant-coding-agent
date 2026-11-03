# memory-solution
# unit: t1-corporate-action-adjustment | category: data-engineering | tags: corporate-actions, split-adjustment, dividend-adjustment, backward-adjustment, price-series, volume-adjustment
# deliverables: adjusted_prices.csv, results.json
# Backward adjustment, actions processed in chronological order; each action rescales only rows with date < action date.
#   split "a:b": price factor = b/a (4:1 -> 0.25), volume factor = a/b (x4) on prior rows.
#   dividend: factor = (C_ex - D) / C_ex with C_ex = UNADJUSTED close on the ex-date (nearest prior trading day if missing).
# cumulative_price_adjustment_factor = product of all factors; total dividends = plain sum of the "amount" fields.
# Last row stays untouched; volumes written as integers after rounding.
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

px = pd.read_csv(os.path.join(DATA_DIR, "prices.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
actions = json.load(open(os.path.join(DATA_DIR, "corporate_actions.json")))["corporate_actions"]
actions = sorted(actions, key=lambda a: a["date"])

dates = px["date"].to_numpy()
close_u = px["close"].to_numpy(float)
vol_u = px["volume"].to_numpy(float)
close_a, vol_a = close_u.copy(), vol_u.copy()
cum_px, cum_vol, total_div = 1.0, 1.0, 0.0

for a in actions:
    d = np.datetime64(pd.Timestamp(a["date"]))
    n_prior = int(np.searchsorted(dates, d, side="left"))          # rows strictly before the action date
    if a["type"] == "split":
        new, old = (float(x) for x in a["ratio"].split(":"))
        f_px, f_vol = old / new, new / old
    else:
        idx = int(np.searchsorted(dates, d, side="right")) - 1      # ex-date row, or nearest prior trading day
        c_ex = close_u[idx]
        f_px, f_vol = (c_ex - float(a["amount"])) / c_ex, 1.0
        total_div += float(a["amount"])
    close_a[:n_prior] *= f_px
    vol_a[:n_prior] *= f_vol
    cum_px *= f_px
    cum_vol *= f_vol

out = pd.DataFrame({"date": px["date"].dt.strftime("%Y-%m-%d"),
                    "close_unadjusted": close_u,
                    "close_adjusted": np.round(close_a, 4),
                    "volume_unadjusted": vol_u.astype(np.int64),
                    "volume_adjusted": np.round(vol_a).astype(np.int64)})
out.to_csv(os.path.join(OUT_DIR, "adjusted_prices.csv"), index=False)

results = {"n_actions_applied": int(len(actions)),
           "cumulative_price_adjustment_factor": round(float(cum_px), 8),
           "cumulative_volume_adjustment_factor": round(float(cum_vol), 4),
           "first_date_adjusted_close": round(float(close_a[0]), 4),
           "last_date_unadjusted_close": round(float(close_u[-1]), 4),
           "total_dividends_per_original_share": round(float(total_div), 4)}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
