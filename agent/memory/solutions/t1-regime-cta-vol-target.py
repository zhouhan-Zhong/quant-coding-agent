# memory-solution
# unit: t1-regime-cta-vol-target | category: backtest-factors | tags: cta, ema-crossover, ewma-vol-targeting, garch, arch, composite-volatility, regime-classification, percentile-thresholds, sharpe, params-json-family
# deliverables: results.json, solution.json
# log returns R = diff(log(px)); SMA-seeded EMA on prices; signal +1 if fast>slow else -1, 0 while slow EMA is NaN
# EWMA var seeded r[0]^2 (alpha=2/(lookback+1)); w = sig*(vol_target/n)/ann_ewma_vol clipped to +-max_lev; NaN -> 0
# per-asset GARCH(1,1): arch_model(100*r, mean='Zero', dist='normal', rescale=False); LRV = omega/1e4/(1-a-b) (daily dec.)
# composite vol = sqrt(mean_i cond_var_i) * sqrt(252) (RMS across assets); thresholds = np.percentile(comp, lo/hi pct)
# regime low if comp<=lo, high if comp>=hi; static port[t] = sum_i w[t-1,i] R[t,i], port[0]=0 (1259 rows)
# dynamic port[t] = sum_i w[t-1,i] * scale[t] * R[t,i]  (regime scale of day t applied to the LAGGED weight, i.e. NOT lagged itself)
# ann_ret = mean*252 (NOT exp-1); ann_vol = std(ddof=1)*sqrt252
# Sharpe = (ann_ret - rf)/ann_vol; mdd on exp(cumsum) path (positive magnitude); calmar = ann_ret/mdd
import os, json, math
import numpy as np, pandas as pd
from scipy.signal import lfilter
from arch import arch_model
import warnings; warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

P = json.load(open(os.path.join(DATA_DIR, "params.json")))
df = pd.read_csv(os.path.join(DATA_DIR, "prices.csv")).drop_duplicates("date").sort_values("date").dropna()
assets = [c for c in df.columns if c != "date"]
px = df[assets].to_numpy(float)
T, n = px.shape
R = np.diff(np.log(px), axis=0)                                      # 1259 x n log returns


def ema(x, span):                                                    # SMA-seeded EMA, NaN before the seed
    a = 2.0 / (span + 1); out = np.full_like(x, np.nan); out[span - 1] = x[:span].mean()
    for t in range(span, len(x)):
        out[t] = a * x[t] + (1 - a) * out[t - 1]
    return out


def ewma_var(r, span):                                               # sigma2[0] = r[0]^2
    a = 2.0 / (span + 1)
    return lfilter([a], [1.0, -(1 - a)], r ** 2, zi=[(1 - a) * r[0] ** 2])[0]


def perf(w, overlay=None):                                           # w aligned with R; one-day execution lag
    w_lag = np.vstack([np.zeros((1, n)), w[:-1]])
    if overlay is not None:                                          # regime scale applied AFTER the lag (same day as R[t])
        w_lag = w_lag * overlay[:, None]
    x = (w_lag * R).sum(axis=1)
    ann_ret = x.mean() * 252; ann_vol = x.std(ddof=1) * math.sqrt(252)
    sharpe = (ann_ret - P["risk_free_annual"]) / ann_vol
    wealth = np.exp(np.cumsum(x)); mdd = float((1 - wealth / np.maximum.accumulate(wealth)).max())
    return {"annualized_return": ann_ret, "annualized_volatility": ann_vol, "sharpe_ratio": sharpe,
            "max_drawdown": mdd, "calmar_ratio": ann_ret / mdd}


fast = np.column_stack([ema(px[:, i], P["fast_ema"]) for i in range(n)])
slow = np.column_stack([ema(px[:, i], P["slow_ema"]) for i in range(n)])
sig = np.where(fast > slow, 1.0, -1.0); sig[np.isnan(slow)] = 0.0
sig = sig[1:]                                                        # align with returns
ann_ewma = np.column_stack([np.sqrt(252.0 * ewma_var(R[:, i], P["vol_lookback"])) for i in range(n)])
tgt = P["vol_target"] / P["n_assets"]
lev = P["max_leverage_per_asset"]
w_static = np.nan_to_num(np.clip(sig * tgt / ann_ewma, -lev, lev))

# per-asset GARCH(1,1) on percent returns -> composite RMS vol -> regimes
pers, lrv, cvar = [], [], []
for i in range(n):
    res = arch_model(100.0 * R[:, i], mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
    a, b, om = float(res.params["alpha[1]"]), float(res.params["beta[1]"]), float(res.params["omega"]) / 1e4
    pers.append(a + b); lrv.append(om / (1 - a - b))
    cvar.append((np.asarray(res.conditional_volatility) / 100.0) ** 2)
comp = np.sqrt(np.mean(np.column_stack(cvar), axis=1)) * math.sqrt(252.0)
th_lo, th_hi = float(np.percentile(comp, P["regime_lo_pct"])), float(np.percentile(comp, P["regime_hi_pct"]))
low, high = comp <= th_lo, comp >= th_hi
scale = np.where(low, P["scale_low_vol"], np.where(high, P["scale_high_vol"], 1.0))

st, dy = perf(w_static), perf(w_static, scale)
results = {"dynamic_sharpe_ratio": dy["sharpe_ratio"], "sharpe_improvement": dy["sharpe_ratio"] - st["sharpe_ratio"],
           "regime_frac_high": high.mean(), "composite_vol_mean": comp.mean()}
results = {k: float(v) for k, v in results.items()}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

inter = {f"static_{k}": v for k, v in st.items()}
inter.update({f"dynamic_{k}": v for k, v in dy.items()})
inter.update({"avg_garch_persistence": np.mean(pers), "avg_long_run_variance": np.mean(lrv),
              "regime_frac_low": low.mean(), "regime_frac_mid": 1.0 - low.mean() - high.mean(),
              "regime_frac_high": high.mean(), "regime_thresh_lo": th_lo, "regime_thresh_hi": th_hi,
              "composite_vol_mean": comp.mean()})
solution = {"intermediates": {k: {"value": float(v)} for k, v in inter.items()}, "deliverables": results}
json.dump(solution, open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
