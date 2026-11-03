# memory-solution
# unit: t1-fama-french-factor-model-new | category: backtest-factors | tags: fama-french, ff3, ols, newey-west, hac, grs-test, vif, rolling-beta, jensen-alpha, information-ratio, treynor, durbin-watson, matplotlib
# deliverables: results.json, regression_summary.csv, sector_summary.csv, rolling_betas.csv, factor_loadings.png, alpha_chart.png, r_squared_chart.png, rolling_betas.png
# pct_change returns, dropna, intersect with factor dates (2515 days); y = R - RF; X = [1, Mkt-RF, SMB, HML] via lstsq
# OLS SE = sqrt(diag(MSE (X'X)^-1)), MSE = SSR/(n-k-1); p-values two-tailed t(n-k-1)
# NW: L = floor(4 (n/100)^(2/9)) = 8; S = X'diag(e^2)X + sum_j w_j (G_j + G_j'), G_j = sum_t e_t e_{t-j} x_t x_{t-j}', w_j = 1 - j/(L+1); V = (X'X)^-1 S (X'X)^-1
# GRS = (T-N-K)/N * a' Sig^-1 a / (1 + mu' Om^-1 mu) with Sig = E'E/T, Om = cov(F) with 1/T; VIF_f = 1/(1-R2 of f on other two + const)
# rolling 252-day lstsq for all 10 stocks at once per window -> 2264 rows; std ddof=1
import os, json, math
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

px = pd.read_csv(os.path.join(DATA_DIR, "stock_prices.csv"), index_col=0, parse_dates=True).sort_index()
ff = pd.read_csv(os.path.join(DATA_DIR, "ff3_factors_daily.csv"), index_col=0, parse_dates=True).sort_index()
rets = px.pct_change().dropna()
idx = rets.index.intersection(ff.index)
rets, ff = rets.loc[idx], ff.loc[idx]
tickers = sorted(rets.columns); n, k = len(idx), 3
F = ff[["Mkt-RF", "SMB", "HML"]].to_numpy()
X = np.column_stack([np.ones(n), F])
XtX_inv = np.linalg.inv(X.T @ X)
L = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
SECTORS = {"IT": ["AAPL", "MSFT", "GOOGL"], "Services": ["AMZN", "MCD", "UNH"], "Finance": ["JPM", "GS", "V", "BRK-B"]}

def newey_west(e):
    S = (X * (e ** 2)[:, None]).T @ X
    for j in range(1, L + 1):
        G = (X[j:] * (e[j:] * e[:-j])[:, None]).T @ X[:-j]
        S += (1.0 - j / (L + 1.0)) * (G + G.T)
    return XtX_inv @ S @ XtX_inv

res, resid_mat, alphas = {}, [], []
for t in tickers:
    y = (rets[t] - ff["RF"]).to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b; ssr = float(e @ e)
    r2 = 1.0 - ssr / float(((y - y.mean()) ** 2).sum())
    mse = ssr / (n - k - 1)
    se_ols = math.sqrt(mse * XtX_inv[0, 0]); se_nw = math.sqrt(newey_west(e)[0, 0])
    t_ols, t_nw = b[0] / se_ols, b[0] / se_nw
    alpha_ann = b[0] * 252
    res[t] = {"alpha_daily": round(float(b[0]), 8), "alpha_annualized": round(float(alpha_ann), 6),
              "beta_mkt": round(float(b[1]), 6), "beta_smb": round(float(b[2]), 6), "beta_hml": round(float(b[3]), 6),
              "r_squared": round(r2, 6), "adj_r_squared": round(1 - (1 - r2) * (n - 1) / (n - k - 1), 6),
              "t_stat_alpha": round(float(t_ols), 4), "p_value_alpha": round(float(2 * stats.t.sf(abs(t_ols), n - k - 1)), 6),
              "t_stat_alpha_nw": round(float(t_nw), 4), "p_value_alpha_nw": round(float(2 * stats.t.sf(abs(t_nw), n - k - 1)), 6),
              "durbin_watson": round(float((np.diff(e) ** 2).sum() / ssr), 6),
              "information_ratio": round(float(alpha_ann / (e.std(ddof=1) * math.sqrt(252))), 6),
              "treynor_ratio": round(float(y.mean() * 252 / b[1]), 6), "ann_excess_return": round(float(y.mean() * 252), 6)}
    resid_mat.append(e); alphas.append(b[0])

# rolling 252-day betas (all stocks per window)
Y = (rets[tickers].to_numpy() - ff["RF"].to_numpy()[:, None]); W = 252
roll = np.array([np.linalg.lstsq(X[s:s + W], Y[s:s + W], rcond=None)[0][1] for s in range(n - W + 1)])
for j, t in enumerate(tickers):
    rb = roll[:, j]
    res[t].update({"rolling_beta_mkt_mean": round(float(rb.mean()), 6), "rolling_beta_mkt_std": round(float(rb.std(ddof=1)), 6),
                   "rolling_beta_mkt_min": round(float(rb.min()), 6), "rolling_beta_mkt_max": round(float(rb.max()), 6)})
roll_df = pd.DataFrame(roll, columns=tickers); roll_df.insert(0, "date", [d.strftime("%Y-%m-%d") for d in idx[W - 1:]])
roll_df.to_csv(os.path.join(OUT_DIR, "rolling_betas.csv"), index=False)

# GRS joint alpha test
E = np.column_stack(resid_mat); a = np.array(alphas); N = len(tickers)
Sig = E.T @ E / n; mu = F.mean(axis=0); Fd = F - mu; Om = Fd.T @ Fd / n
grs = (n - N - k) / N * float(a @ np.linalg.solve(Sig, a)) / (1.0 + float(mu @ np.linalg.solve(Om, mu)))
grs_p = float(stats.f.sf(grs, N, n - N - k))
# VIF
vif = []
for j in range(3):
    Xa = np.column_stack([np.ones(n), np.delete(F, j, axis=1)]); yf = F[:, j]
    ea = yf - Xa @ np.linalg.lstsq(Xa, yf, rcond=None)[0]
    vif.append(1.0 / (1.0 - (1.0 - (ea @ ea) / ((yf - yf.mean()) ** 2).sum())))

ranking = sorted(tickers, key=lambda t: -res[t]["alpha_annualized"])
out = {"meta": {"trading_days": int(n), "start_date": idx[0].strftime("%Y-%m-%d"), "end_date": idx[-1].strftime("%Y-%m-%d"),
                "num_stocks": N, "nw_lag": L},
       "stocks": res, "alpha_ranking": ranking,
       "grs_test": {"grs_statistic": round(grs, 4), "grs_p_value": round(grs_p, 6), "grs_rejected": bool(grs_p < 0.05), "df1": N, "df2": int(n - N - k)},
       "factor_diagnostics": {"vif_mkt_rf": round(vif[0], 4), "vif_smb": round(vif[1], 4), "vif_hml": round(vif[2], 4), "nw_lag": L}}
json.dump(out, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

cols = ["alpha_annualized", "beta_mkt", "beta_smb", "beta_hml", "r_squared", "adj_r_squared", "t_stat_alpha", "p_value_alpha",
        "t_stat_alpha_nw", "p_value_alpha_nw", "durbin_watson", "information_ratio", "treynor_ratio"]
summ = pd.DataFrame([{"ticker": t, **{c: res[t][c] for c in cols}} for t in tickers])
summ.to_csv(os.path.join(OUT_DIR, "regression_summary.csv"), index=False)
pd.DataFrame([{"sector": s, "avg_alpha_annualized": np.mean([res[t]["alpha_annualized"] for t in m]),
               "avg_beta_mkt": np.mean([res[t]["beta_mkt"] for t in m]), "avg_r_squared": np.mean([res[t]["r_squared"] for t in m])}
              for s, m in sorted(SECTORS.items())]).to_csv(os.path.join(OUT_DIR, "sector_summary.csv"), index=False)

# plots
xs = np.arange(N); fig, ax = plt.subplots(figsize=(11, 5))
for i, (c, lab) in enumerate([("beta_mkt", "Mkt-RF"), ("beta_smb", "SMB"), ("beta_hml", "HML")]):
    ax.bar(xs + (i - 1) * 0.27, [res[t][c] for t in tickers], 0.27, label=lab)
ax.set_xticks(xs); ax.set_xticklabels(tickers); ax.axhline(0, color="k", lw=0.8); ax.legend(); ax.set_title("Fama-French 3-Factor Loadings")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "factor_loadings.png"), dpi=100); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 5)); av = [res[t]["alpha_annualized"] for t in ranking]
ax.barh([t + (" *" if res[t]["p_value_alpha_nw"] < 0.05 else "") for t in ranking][::-1], av[::-1], color=["green" if v > 0 else "red" for v in av[::-1]])
ax.set_title("Annualized Jensen's Alpha (NW significance)"); fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "alpha_chart.png"), dpi=100); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(tickers, [res[t]["r_squared"] for t in tickers]); ax.set_title("R-squared — Fama-French 3-Factor Model")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "r_squared_chart.png"), dpi=100); plt.close(fig)
fig, ax = plt.subplots(figsize=(12, 6))
for t in tickers: ax.plot(idx[W - 1:], roll_df[t], lw=1, label=t)
ax.set_title("Rolling 252-Day Market Beta"); ax.set_xlabel("date"); ax.set_ylabel("rolling market beta"); ax.legend(ncol=5, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "rolling_betas.png"), dpi=100); plt.close(fig)
