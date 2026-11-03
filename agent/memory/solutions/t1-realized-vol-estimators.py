# memory-solution
# unit: t1-realized-vol-estimators | category: microstructure | tags: realized-variance, volatility-signature, microstructure-noise, bandi-russell, bipower-variation, intraday, sparse-sampling, dedup-median, rth-filter
# deliverables: results.json
# Clean: groupby timestamp -> median price (venue duplicates), keep time-of-day in [09:30, 16:00] inclusive, sort, group by date.
# Sampling at f minutes: keep rows with ((hour*60+minute) - 570) % f == 0 (no forward-fill; gaps just reduce n_f).
# RV_f = sum of squared log returns of the kept prices; n_f = number of returns.
# Bandi-Russell noise variance: eps2 = RV_1min / (2 n_1min) per day; grand mean over the 10 days.
# Corrected 5-min RV: RV_5 - 2 n_5 * grand_mean_noise (not clipped); ann vol = sqrt(max(corr, 0) * 252).
# Bipower (BN-S 2004 Def. 1): BV = (pi/2) sum |r_j||r_{j-1}| on the 5-min grid, daily scale, no M/(M-1) factor.
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "spy_intraday.csv"), parse_dates=["timestamp"])
df = df.groupby("timestamp", as_index=False)["mid_price"].median()
tod = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
df = df[(tod >= 9 * 60 + 30) & (tod <= 16 * 60)].sort_values("timestamp").reset_index(drop=True)
df["mod"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute - (9 * 60 + 30)
df["date"] = df["timestamp"].dt.date

FREQS = [1, 2, 5, 15, 30]
days = []
for d, g in df.groupby("date", sort=True):
    row = {"date": str(d)}
    r5 = None
    for f in FREQS:
        p = g.loc[g["mod"] % f == 0, "mid_price"].to_numpy(float)
        r = np.diff(np.log(p))
        row[f"RV_{f}min"] = float(np.sum(r ** 2))
        row[f"n_{f}min"] = int(len(r))
        if f == 5:
            r5 = r
    row["noise_var_est"] = row["RV_1min"] / (2.0 * row["n_1min"])            # Bandi-Russell (2006)
    row["BV_5min"] = float(math.pi / 2.0 * np.sum(np.abs(r5[1:]) * np.abs(r5[:-1])))
    days.append(row)

grand_noise = float(np.mean([x["noise_var_est"] for x in days]))
for x in days:
    x["RV_5min_corrected"] = float(x["RV_5min"] - 2.0 * x["n_5min"] * grand_noise)
    x["RV_5min_ann_vol"] = float(math.sqrt(max(x["RV_5min_corrected"], 0.0) * 252.0))

mean_corr = float(np.mean([x["RV_5min_corrected"] for x in days]))
summary = {"num_days": len(days), "grand_mean_noise_var": grand_noise, "mean_RV_5min_corrected": mean_corr,
           "grand_ann_vol": float(math.sqrt(max(mean_corr, 0.0) * 252.0))}
json.dump({"days": days, "summary": summary}, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
