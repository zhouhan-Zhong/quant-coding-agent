# memory-solution
# unit: t1-kelly-var-sizing | category: portfolio | tags: kelly-criterion, parametric-var, monte-carlo, multivariate-normal, sharpe, max-drawdown, risk-budget, params-json-family
# deliverables: results.json, solution.json
# clean: drop rows with any NaN (1000 -> 970 valid), then stats on the LAST estimation_window_days (750) valid rows
# rf_daily = annual/252 (simple); mean_excess = mean - rf; cov ddof=1; f* = Sigma^-1 mu_excess; leverage = sum|f|
# parametric VaR (daily, 99%) = z * sqrt(f' Sigma f) (no mean term); scale = max_var_daily / VaR; f_var = scale * f
# MC: rng = default_rng(params.seed); ONE fresh rng.multivariate_normal(mean_raw, Sigma, size=(paths, days)) PER SCHEME,
#     drawn sequentially in the order full, half, var, equal-weight (the stream order matters for exact match)
# wealth = W0 * cumprod(1 + sims @ f) (no cash leg); Sharpe = pooled mean/std(ddof=1)*sqrt(252) over all path-days (no rf)
# max drawdown per path on [W0, wealth...]; "median" of terminal wealth / drawdown = sorted[n_paths // 2] (upper middle)
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

P = json.load(open(os.path.join(DATA_DIR, "params.json")))
df = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"))
assets = [c for c in df.columns if c != "date"]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
for c in assets:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().drop_duplicates("date").sort_values("date")        # anomaly handling: NaN rows / dup dates
n_valid = int(len(df))
W = int(P.get("estimation_window_days") or n_valid)
X = df[assets].to_numpy(float)[-W:]                                  # most recent W valid observations

rf = P["risk_free_annual"] / 252.0                                   # simple daily rf
mean_raw = X.mean(axis=0)
mean_exc = mean_raw - rf
Sigma = np.cov(X, rowvar=False, ddof=1)
f_full = np.linalg.solve(Sigma, mean_exc)                            # multi-asset Kelly f* = Sigma^-1 (mu - rf)
leverage = float(np.abs(f_full).sum())

z = norm.ppf(P["confidence_level"])
var_full = float(z * math.sqrt(f_full @ Sigma @ f_full))             # parametric daily VaR of the full-Kelly book
scale = float(P["max_var_daily"] / var_full)
f_var = f_full * scale
schemes = {"full": f_full, "half": 0.5 * f_full, "var": f_var, "equal": np.ones(len(assets)) / len(assets)}

W0, N, T = float(P["initial_capital"]), int(P["n_simulation_paths"]), int(P["n_days"])
rng = np.random.default_rng(int(P["seed"]))
res = {}
for name, f in schemes.items():                                      # sequential draws: order full, half, var, equal
    sims = rng.multivariate_normal(mean_raw, Sigma, size=(N, T))
    port = sims @ f
    wealth = W0 * np.cumprod(1.0 + port, axis=1)
    term = np.sort(wealth[:, -1])
    path = np.concatenate([np.full((N, 1), W0), wealth], axis=1)
    peak = np.maximum.accumulate(path, axis=1)
    mdd = np.sort(((peak - path) / peak).max(axis=1))
    res[name] = {"median_terminal_wealth": float(term[N // 2]), "mean_terminal_wealth": float(term.mean()),
                 "sharpe": float(port.mean() / port.std(ddof=1) * math.sqrt(252)),
                 "p50_drawdown": float(mdd[N // 2])}

results = {"num_valid_observations": n_valid,
           "median_terminal_wealth_full": res["full"]["median_terminal_wealth"],
           "median_terminal_wealth_var": res["var"]["median_terminal_wealth"],
           "sharpe_full": res["full"]["sharpe"], "sharpe_var": res["var"]["sharpe"],
           "p50_drawdown_full": res["full"]["p50_drawdown"], "p50_drawdown_var": res["var"]["p50_drawdown"]}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

inter = {f"mean_excess_{i}": float(mean_exc[i]) for i in range(len(assets))}
inter.update({"cov_00": float(Sigma[0, 0]), "cov_01": float(Sigma[0, 1]), "cov_12": float(Sigma[1, 2])})
inter.update({f"kelly_fraction_{i}": float(f_full[i]) for i in range(len(assets))})
inter.update({"total_kelly_leverage": leverage, "portfolio_var_full": var_full, "var_scale_factor": scale,
              "mean_terminal_wealth_full": res["full"]["mean_terminal_wealth"]})
solution = {"intermediates": {k: {"value": v} for k, v in inter.items()},
            "schemes": res, "deliverables": results}
json.dump(solution, open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
