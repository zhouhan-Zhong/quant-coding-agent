# memory-solution
# unit: t1-credit-spread-decomposition | category: credit | tags: credit-spread, rolling-ols, newey-west, hac, bartlett, variance-decomposition, fred, resample, regime-break
# deliverables: decomposition_results.csv, variance_decomposition.csv, regime_breaks.csv
# FRED CSVs: na_values ["." and blank]; daily series -> month-end last valid obs via resample("ME").last(), re-indexed
#   to month-start to merge with monthly BAA/AAA. TEDRATE ffill after LIBOR cessation; drop months with BAA/AAA/DGS10/T10YIE NaN.
# Y = BAA - DGS10 on X = [1, BAA-AAA, TEDRATE, T10YIE]; 48-month rolling OLS (window ending at t), components = beta_k * x_k(t).
# Newey-West: S = G0 + sum_j w_j (G_j + G_j'), w_j = 1 - j/(m+1), m = 3, G_j = sum_t e_t e_{t-j} x_t x_{t-j}'; V = (X'X)^-1 S (X'X)^-1,
#   NO small-sample T/(T-k) correction; t = beta / sqrt(diag V).
# Variance share pct_k = Cov(beta_k X_k, Y)/Var(Y)*100 (same ddof for both), pct_resid = (1-R2)*100, sums to 100 exactly.
# Regime breaks: rows with R2 < 0.5.
import os
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

def load(name, monthly):
    s = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"), na_values=["."], parse_dates=["observation_date"])
    s = s.set_index("observation_date")[name].astype(float)
    if not monthly:
        s = s.resample("ME").last()                     # last valid observation of each calendar month
    s.index = s.index.to_period("M").to_timestamp()      # month-start stamps for alignment
    return s

df = pd.concat({"BAA": load("BAA", True), "AAA": load("AAA", True), "DGS10": load("DGS10", False),
                "TEDRATE": load("TEDRATE", False), "T10YIE": load("T10YIE", False)}, axis=1)
df = df.loc["2003-01-01":"2024-12-31"]
df["TEDRATE"] = df["TEDRATE"].ffill()
df = df.dropna(subset=["BAA", "AAA", "DGS10", "T10YIE"])

y_all = (df["BAA"] - df["DGS10"]).to_numpy()
X_all = np.column_stack([np.ones(len(df)), (df["BAA"] - df["AAA"]).to_numpy(), df["TEDRATE"].to_numpy(), df["T10YIE"].to_numpy()])
dates = df.index
W, M = 48, 3

def newey_west_t(X, e, beta):
    XtX_inv = np.linalg.inv(X.T @ X)
    xe = X * e[:, None]
    S = xe.T @ xe
    for j in range(1, M + 1):
        G = xe[j:].T @ xe[:-j]
        S += (1.0 - j / (M + 1.0)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return beta / np.sqrt(np.diag(V))

rows, vrows = [], []
for t in range(W - 1, len(df)):
    X, y = X_all[t - W + 1:t + 1], y_all[t - W + 1:t + 1]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    r2 = 1.0 - e @ e / np.sum((y - y.mean()) ** 2)
    tst = newey_west_t(X, e, beta)
    comp = beta[1:] * X[-1, 1:]
    rows.append({"date": dates[t].strftime("%Y-%m-%d"), "baa_treasury_spread": y[-1],
                 "default_component": comp[0], "liquidity_component": comp[1], "inflation_component": comp[2],
                 "residual": y[-1] - beta[0] - comp.sum(), "r_squared": r2,
                 "hac_tstat_default": tst[1], "hac_tstat_liquidity": tst[2], "hac_tstat_inflation": tst[3]})
    yc = y - y.mean()
    pct = [float(np.sum((beta[k] * X[:, k] - np.mean(beta[k] * X[:, k])) * yc) / np.sum(yc ** 2) * 100) for k in (1, 2, 3)]
    vrows.append({"date": rows[-1]["date"], "pct_default": pct[0], "pct_liquidity": pct[1], "pct_inflation": pct[2],
                  "pct_residual": (1.0 - r2) * 100})

dec = pd.DataFrame(rows).round(6)
var = pd.DataFrame(vrows).round(6)
dec.to_csv(os.path.join(OUT_DIR, "decomposition_results.csv"), index=False)
var.to_csv(os.path.join(OUT_DIR, "variance_decomposition.csv"), index=False)
dec.loc[dec["r_squared"] < 0.5, ["date", "r_squared"]].to_csv(os.path.join(OUT_DIR, "regime_breaks.csv"), index=False)
