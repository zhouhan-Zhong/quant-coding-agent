# memory-solution
# unit: t1-regime-riskparity-cvar | category: portfolio | tags: absorption-ratio, eigenvalues, marchenko-pastur, regime-detection, inverse-volatility, risk-parity, monthly-rebalance, historical-cvar, params-json-family
# deliverables: results.json, solution.json
# clean: dropna rows (756 -> 726 = num_valid_observations), dedupe, sort; then the LAST estimation_window_days (504) rows
# regime obs t = 63..503 uses the trailing window R[t-63:t] (EXCLUDING day t): corr -> eigvalsh; AR = top-k sum / n_assets
# MP threshold = (1 + sqrt(n/window))^2; crisis if AR > 0.65, risk-off if AR > 0.45, else risk-on (strict >)
# rebalance on the last trading day of each calendar month (needs a following day -> 21, the final partial month is not one):
#   w = inv-vol (std ddof=1 of R[t-63:t]) normalised to 1, times risk_budget[regime_t]; new weights apply from the NEXT day
# before the first rebalance the book is EQUAL-WEIGHTED 1/n (full exposure); strategy returns over the 441 regime days
# ann_ret = mean*252; ann_vol = std(ddof=1)*sqrt252; Sharpe = ret/vol; mdd positive on cumprod(1+r); win_rate = frac > 0
# VaR99 = -percentile(last 63 returns, 1); CVaR = -mean(returns <= -VaR); mean_weight_stock_01 = mean over rebalance weights
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

P = json.load(open(os.path.join(DATA_DIR, "params.json")))
df = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"))
stocks = [c for c in df.columns if c != "date"]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
for c in stocks:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)
n_valid = int(len(df))
W = int(P.get("estimation_window_days") or n_valid)
df = df.iloc[-W:].reset_index(drop=True)
R = df[stocks].to_numpy(float); dates = df["date"]
T, n = R.shape
L, k = int(P["rolling_window"]), int(P["top_k_eigenvalues"])
mp_threshold = (1.0 + math.sqrt(n / L)) ** 2

# ---- rolling eigenvalue analysis / regimes (window excludes the observation day) ----
idx = np.arange(L, T)
ar, lam1, regime = [], [], []
for t in idx:
    ev = np.linalg.eigvalsh(np.corrcoef(R[t - L:t], rowvar=False))[::-1]
    a = ev[:k].sum() / ev.sum()
    ar.append(a); lam1.append(ev[0])
    regime.append("crisis" if a > P["regime_crisis_threshold"] else ("risk-off" if a > P["regime_risk_off_threshold"] else "risk-on"))
ar = np.array(ar); lam1 = np.array(lam1)
counts = {r: int(sum(1 for x in regime if x == r)) for r in ["crisis", "risk-off", "risk-on"]}

# ---- monthly rebalanced inverse-vol risk parity ----
months = dates.dt.to_period("M").to_numpy()
is_month_end = np.r_[months[:-1] != months[1:], False]              # last trading day of a calendar month
w_cur = np.ones(n) / n                                               # equal weight until the first rebalance
Wt = np.zeros((len(idx), n)); reb_w = []
for j, t in enumerate(idx):
    Wt[j] = w_cur
    if is_month_end[t]:
        iv = 1.0 / R[t - L:t].std(axis=0, ddof=1)
        w_cur = iv / iv.sum() * P["risk_budget"][regime[j]]
        reb_w.append(w_cur)                                          # effective from the next day
ret = (Wt * R[idx]).sum(axis=1)
reb_w = np.array(reb_w)

ann_ret = ret.mean() * 252; ann_vol = ret.std(ddof=1) * math.sqrt(252)
wealth = np.cumprod(1 + ret); mdd = float((1 - wealth / np.maximum.accumulate(wealth)).max())
tail = ret[-int(P["cvar_window"]):]
var99 = -float(np.percentile(tail, 100 * (1 - P["cvar_confidence"])))
cvar = -float(tail[tail <= -var99].mean())

results = {"num_valid_observations": n_valid, "annualized_return": float(ann_ret), "annualized_volatility": float(ann_vol),
           "sharpe_ratio": float(ann_ret / ann_vol), "max_drawdown": mdd, "cvar": cvar}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

inter = {"num_regime_observations": int(len(idx)), "mp_threshold": mp_threshold, "mean_absorption_ratio": float(ar.mean()),
         "std_absorption_ratio": float(ar.std(ddof=1)), "mean_largest_eigenvalue": float(lam1.mean()),
         "regime_count_crisis": counts["crisis"], "regime_count_risk_off": counts["risk-off"], "regime_count_risk_on": counts["risk-on"],
         "num_rebalances": int(len(reb_w)), "mean_weight_stock_01": float(reb_w[:, 0].mean()), "portfolio_var_99": var99,
         "cvar": cvar, "win_rate": float((ret > 0).mean())}
solution = {"intermediates": {k_: {"value": v} for k_, v in inter.items()}, "deliverables": results}
json.dump(solution, open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
