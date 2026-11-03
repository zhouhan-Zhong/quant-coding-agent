# memory-solution
# unit: t1-alpha-hedge-strategy | category: backtest-factors | tags: cross-sectional-momentum, z-score, long-short, monthly-rebalance, transaction-costs, fama-french-regression, alpha, tracking-error, information-ratio, params-json-family
# deliverables: results.json, solution.json
# nan_count = NaN cells in the raw stock matrix (8); clean by filling NaN with 0 (tickers sorted alphabetically)
# signal at day t = mean of the trailing lookback_alpha returns R[t-60:t] (EXCLUDING day t), z-scored cross-sectionally
# rebalance on trade_start_day (61) and on the first trading day of every calendar month (22 events): long top-5 at +0.2,
# short bottom-5 at -0.2; cost = turnover sum|dw| * bps/1e4 charged on the rebalance day; new weights earn THAT day's return
# strategy series = days trade_start_day..end (443); ann_ret = mean*252, ann_vol = std(ddof=1)*sqrt252, Sharpe = ret/vol,
# mdd positive on cumprod(1+r); OLS of strategy returns on [1, Mkt_RF, SMB, HML]: alpha_ann = b0*252, TE = std(resid, ddof=1)*sqrt252
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

P = json.load(open(os.path.join(DATA_DIR, "params.json")))
AF = float(P["annualization_factor"])
df = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"))
fac = pd.read_csv(os.path.join(DATA_DIR, "factors.csv"))
df["date"] = pd.to_datetime(df["date"]); fac["date"] = pd.to_datetime(fac["date"])
stocks = sorted(c for c in df.columns if c != "date")
Rraw = df[stocks].apply(pd.to_numeric, errors="coerce").to_numpy(float)
nan_count = int(np.isnan(Rraw).sum())
R = np.nan_to_num(Rraw, nan=0.0)
T, n = R.shape
fac = fac.set_index("date").reindex(df["date"]).ffill().fillna(0.0)
Fm = fac[P["factor_names"]].to_numpy(float)

months = df["date"].dt.to_period("M").to_numpy()
month_start = np.r_[True, months[1:] != months[:-1]]
L, ts = int(P["lookback_alpha"]), int(P["trade_start_day"])
nL, nS = int(P["long_top_n"]), int(P["short_bottom_n"])
wL, wS = float(P["long_weight_per_stock"]), float(P["short_weight_per_stock"])
tc = P["transaction_cost_bps"] / 1e4

w = np.zeros(n); ret = np.zeros(T); n_reb = 0
for t in range(ts, T):
    cost = 0.0
    if t == ts or month_start[t]:                                    # rebalance: first trade day + each month start
        s = R[t - L:t].mean(axis=0)                                  # trailing-return signal
        z = (s - s.mean()) / s.std(ddof=1)
        order = np.argsort(z)
        new_w = np.zeros(n); new_w[order[-nL:]] = wL; new_w[order[:nS]] = -wS
        cost = np.abs(new_w - w).sum() * tc
        w = new_w; n_reb += 1
    ret[t] = float(w @ R[t]) - cost
x = ret[ts:]

ann_ret = x.mean() * AF; ann_vol = x.std(ddof=1) * math.sqrt(AF)
wealth = np.cumprod(1 + x); mdd = float((1 - wealth / np.maximum.accumulate(wealth)).max())
Xf = np.column_stack([np.ones(len(x)), Fm[ts:]])
b = np.linalg.lstsq(Xf, x, rcond=None)[0]
resid = x - Xf @ b
alpha_ann = b[0] * AF; te = resid.std(ddof=1) * math.sqrt(AF)

results = {"annualized_return": float(ann_ret), "annualized_volatility": float(ann_vol), "sharpe_ratio": float(ann_ret / ann_vol),
           "max_drawdown": mdd, "annualized_alpha": float(alpha_ann), "market_beta_residual": float(b[1]), "n_rebalances": n_reb}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

inter = {"nan_count": nan_count, "n_rebalances": n_reb, "annualized_alpha": float(alpha_ann), "market_beta_residual": float(b[1]),
         "smb_beta_residual": float(b[2]), "hml_beta_residual": float(b[3]), "tracking_error_annual": float(te),
         "information_ratio": float(alpha_ann / te), "hit_rate": float((x > 0).mean()), "sharpe_ratio": float(ann_ret / ann_vol)}
solution = {"intermediates": {k: {"value": v} for k, v in inter.items()}, "deliverables": results}
json.dump(solution, open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
