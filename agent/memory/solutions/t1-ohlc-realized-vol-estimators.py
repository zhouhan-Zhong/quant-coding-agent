# memory-solution
# unit: t1-ohlc-realized-vol-estimators | category: volatility | tags: realized-vol, parkinson, garman-klass, rogers-satchell, yang-zhang, close-to-close, rolling-window, efficiency-ratio, term-structure, ohlc
# deliverables: calibration.json, full_sample_vol.json, rolling_vol_stats.csv, vol_term_structure.csv, summary.json
# work on the 692 rows that have a previous close (first row dropped); n_rolling = n_returns - W + 1 (= n_days - 21 for W=21)
# CC var = var(log C/C_prev, ddof=1); P = sum(ln(H/L)^2)/(4 n ln2); GK = mean(0.5 ln(H/L)^2 - (2ln2-1) ln(C/O)^2)
# RS = mean(ln(H/C)ln(H/O) + ln(L/C)ln(L/O)); YZ = var_o + k var_c + (1-k) RS, k = 0.34/(1.34 + (n+1)/(n-1)), var ddof=1
# annualized vol = sqrt(var * 252); efficiency = var(CC rolling variance series) / var(estimator rolling variance series)
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "spy_daily.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
n_days = len(df)
ohlc_valid = bool(((df.high >= df[["open", "close"]].max(axis=1)) & (df.low <= df[["open", "close"]].min(axis=1))).all())
d = df.iloc[1:].reset_index(drop=True).copy()
d["pc"] = df["close"].to_numpy()[:-1]
d["r"] = np.log(d.close / d.pc)                                            # close-to-close
d["hl2"] = np.log(d.high / d.low) ** 2
d["co"] = np.log(d.close / d.open)                                         # open-to-close
d["oc"] = np.log(d.open / d.pc)                                            # overnight
d["rs"] = np.log(d.high / d.close) * np.log(d.high / d.open) + np.log(d.low / d.close) * np.log(d.low / d.open)
d["gk"] = 0.5 * d.hl2 - (2 * math.log(2) - 1) * d.co ** 2
n_ret = len(d)

def variances(x):
    """Return the 5 per-period variance estimates for a block of rows (DataFrame)."""
    n = len(x); k = 0.34 / (1.34 + (n + 1) / (n - 1))
    cc = x.r.var(ddof=1); pk = x.hl2.sum() / (4 * n * math.log(2)); gk = x.gk.mean(); rs = x.rs.mean()
    yz = x.oc.var(ddof=1) + k * x.co.var(ddof=1) + (1 - k) * rs
    return np.array([cc, pk, gk, rs, yz])

NAMES = ["close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang"]
KEYS = ["cc_vol", "parkinson_vol", "garman_klass_vol", "rogers_satchell_vol", "yang_zhang_vol"]
ann = lambda v: np.sqrt(v * 252.0)

full = ann(variances(d))
json.dump({"n_days": int(n_days), "n_returns": int(n_ret), "S0": float(df.close.iloc[-1]), "ohlc_valid": ohlc_valid},
          open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)
json.dump({k: float(v) for k, v in zip(KEYS, full)}, open(os.path.join(OUT_DIR, "full_sample_vol.json"), "w"), indent=2)

def rolling_var(W):
    return np.array([variances(d.iloc[s:s + W]) for s in range(n_ret - W + 1)])     # (n_windows, 5)

rv = rolling_var(21); rvol = ann(rv)
eff = rv[:, 0].var(ddof=1) / rv.var(axis=0, ddof=1)
stats = pd.DataFrame({"estimator": NAMES, "mean_vol": rvol.mean(axis=0), "std_vol": rvol.std(axis=0, ddof=1),
                      "min_vol": rvol.min(axis=0), "max_vol": rvol.max(axis=0), "efficiency": eff})
stats.to_csv(os.path.join(OUT_DIR, "rolling_vol_stats.csv"), index=False)

ts = []
for W in [5, 10, 21, 63, 126, 252]:
    v = ann(rolling_var(W)); ts.append({"window": W, "cc_vol": float(v[:, 0].mean()), "yz_vol": float(v[:, 4].mean())})
pd.DataFrame(ts).to_csv(os.path.join(OUT_DIR, "vol_term_structure.csv"), index=False)

summary = {"n_rolling_windows": int(len(rv)), "cc_vol_annual": float(full[0]), "yz_vol_annual": float(full[4]),
           "parkinson_vol_annual": float(full[1]), "gk_vol_annual": float(full[2]), "rs_vol_annual": float(full[3]),
           "yz_efficiency": float(eff[4]), "parkinson_efficiency": float(eff[1]), "gk_efficiency": float(eff[2]), "rs_efficiency": float(eff[3]),
           "range_estimators_lower_variance": bool((eff[1:] > 1.0).all()), "yz_most_efficient": bool(np.argmax(eff) == 4),
           "vol_mean_range": [float(full.min()), float(full.max())]}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
