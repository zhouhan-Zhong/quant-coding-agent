# memory-solution
# unit: t1-yield-curve-pca-dynamics | category: rates-curves | tags: treasury, yield-curve, spreads, butterfly, pca, covariance, eigenvectors, forward-rates, nelson-siegel, least-squares
# deliverables: results.json
# Data is most-recent-first with MM/DD/YYYY dates -> parse and sort ascending before differencing.
# Spreads in bp = 100 * (yield differences in %); std with ddof=1 (sample); max date via idxmax.
# PCA: np.cov(dY.T) (ddof=1) of daily bp changes of 10 tenors, eigh -> sort descending, flip PC1 so sum > 0.
# Forwards: f(t1,t2) = (y2 t2 - y1 t1)/(t2 - t1) on last date, yields in percent treated as cc zero rates.
# Nelson-Siegel: fixed lambda = 1.5, least_squares on residuals from x0 = [4, -1, 1]; rmse = sqrt(mean(res^2)).
# Max daily move: argmax |diff(10 Yr)| in bp, report signed change and the date of the later observation.
import os, json, math
import numpy as np, pandas as pd
from scipy.optimize import least_squares

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "daily-treasury-rates.csv"))
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values("Date").reset_index(drop=True)
d = lambda ts: ts.strftime("%Y-%m-%d")

# 1. spreads (bp)
s2s10 = 100.0 * (df["10 Yr"] - df["2 Yr"])
bfly = 100.0 * (2.0 * df["5 Yr"] - df["2 Yr"] - df["10 Yr"])
tp = 100.0 * (df["30 Yr"] - df["3 Mo"])
imax = int(s2s10.idxmax())
spreads = {"spread_2s10s_mean_bp": float(s2s10.mean()), "spread_2s10s_std_bp": float(s2s10.std(ddof=1)),
           "spread_2s10s_max_date": d(df.loc[imax, "Date"]), "spread_2s10s_max_bp": float(s2s10.iloc[imax]),
           "butterfly_2s5s10s_mean_bp": float(bfly.mean()), "butterfly_2s5s10s_std_bp": float(bfly.std(ddof=1)),
           "term_premium_mean_bp": float(tp.mean()), "term_premium_std_bp": float(tp.std(ddof=1))}

# 2. PCA on covariance of daily bp changes (10 tenors)
tenors = ["1 Mo", "3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "30 Yr"]
taus = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 30], float)
dY = 100.0 * np.diff(df[tenors].to_numpy(float), axis=0)          # (N-1, 10)
cov = np.cov(dY.T, ddof=1)
evals, evecs = np.linalg.eigh(cov)
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]
pc1 = evecs[:, 0] * (-1.0 if evecs[:, 0].sum() < 0 else 1.0)
evr = evals / evals.sum()
pca = {"eigenvalues_top3": [float(v) for v in evals[:3]],
       "explained_variance_ratios_top3": [float(v) for v in evr[:3]],
       "cumulative_variance_top3": float(evr[:3].sum()),
       "pc1_loadings": [float(v) for v in pc1]}

# 3. forward rates on the last date (yields in %, treated as cc zero rates)
last = df.iloc[-1]
y = {t: float(last[c]) for t, c in zip(taus, tenors)}
fwd = lambda t1, t2: (y[t2] * t2 - y[t1] * t1) / (t2 - t1)
forwards = {"f_1y1y_pct": fwd(1, 2), "f_2y3y_pct": fwd(2, 5), "f_5y5y_pct": fwd(5, 10), "f_10y20y_pct": fwd(10, 30)}

# 4. Nelson-Siegel with fixed lambda
LAM = 1.5
def ns_fit(row):
    yy = row[tenors].to_numpy(float)
    x = taus / LAM
    f1 = (1 - np.exp(-x)) / x
    f2 = f1 - np.exp(-x)
    res = least_squares(lambda b: b[0] + b[1] * f1 + b[2] * f2 - yy, x0=[4.0, -1.0, 1.0])
    return {"date": d(row["Date"]), "beta0": float(res.x[0]), "beta1": float(res.x[1]), "beta2": float(res.x[2]),
            "rmse_pct": float(math.sqrt(np.mean(res.fun ** 2)))}
ns = {"first_date": ns_fit(df.iloc[0]), "last_date": ns_fit(df.iloc[-1])}

# 5. max absolute daily move in 10 Yr
d10 = 100.0 * df["10 Yr"].diff()
k = int(d10.abs().idxmax())
mdm = {"date": d(df.loc[k, "Date"]), "change_bp": float(d10.iloc[k])}

json.dump({"spreads": spreads, "pca": pca, "forward_rates": forwards, "nelson_siegel": ns, "max_daily_move": mdm},
          open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
