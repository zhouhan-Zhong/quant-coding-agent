# memory-solution
# unit: t1-crypto-funding-rate-basis-carry | category: fx-commodities-crypto | tags: perpetual-funding, basis-carry, ornstein-uhlenbeck, ols, adf, autocorrelation, regime-split, monte-carlo, exact-ou-transition, var-cvar
# deliverables: results.json, solution.json, funding_clean.csv, regime_stats.csv, ou_diagnostics.json, mc_stats.json
# Funding rates are decimals (1e-4 = 1bp); 3 settlements/day -> annualisation 1095; timestamps floor('s') to 'YYYY-MM-DDTHH:MM:SSZ'.
# Stats: std ddof=1, scipy skew/kurtosis bias=False, pct_positive in %, autocorr = pandas Series.autocorr(lag), adfuller(maxlag=24, 'c', 'AIC').
# OU by OLS on df = a + b f_{t-1}: kappa = -b/dt, mu = -a/b, sigma = resid_std(ddof=1)/sqrt(dt); half-life periods = ln2/(kappa dt),
#   days = /3; stationary std = sigma/sqrt(2 kappa). Per-regime fits use 1094 obs each (difference inside the regime).
# MC: exact transition f' = mu + (f - mu)e^{-k dt} + sigma sqrt((1 - e^{-2k dt})/(2k)) Z, default_rng(42), 50k paths x 1095 steps from the
#   last observed rate; annual return = sum of the 1095 simulated rates (x 1095/1095); VaR = sorted(-ret)[ceil(0.95 N) - 1] clamped >= 0,
#   CVaR = mean of losses strictly > VaR (0 if none); per-path mean rate distribution for simulated_funding_rate_distribution.
import os, json, math
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
P = json.load(open(os.path.join(DATA_DIR, "params.json")))
A, DT, N_P, H, SEED = P["annualization_periods"], P["ou_dt"], P["mc_num_paths"], P["mc_horizon_periods"], P["mc_random_seed"]

df = pd.read_csv(os.path.join(DATA_DIR, "funding_rates.csv"))
df = df[df["symbol"] == P["target_symbol"]].copy()
df["datetime"] = pd.to_datetime(df["funding_time"], unit="ms", utc=True).dt.floor("s")
df = df[(df["datetime"] >= pd.Timestamp(P["analysis_start"], tz="UTC")) & (df["datetime"] < pd.Timestamp(P["analysis_end"], tz="UTC") + pd.Timedelta(days=1))]
df = df.sort_values("datetime").reset_index(drop=True)
df["funding_rate_bps"] = df["funding_rate"] * 1e4
fmt = lambda s: s.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
df.assign(datetime=fmt(df["datetime"]))[["datetime", "symbol", "funding_rate", "funding_rate_bps"]].to_csv(os.path.join(OUT_DIR, "funding_clean.csv"), index=False)
f = df["funding_rate"].reset_index(drop=True)

def desc(x):
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "skewness": float(stats.skew(x, bias=False)),
            "kurtosis": float(stats.kurtosis(x, bias=False)), "pct_positive": float((x > 0).mean() * 100)}
def adf(x):
    r = adfuller(x.to_numpy(), maxlag=24, regression="c", autolag="AIC"); return float(r[0]), float(r[1])
def ou_fit(x):
    x = x.to_numpy(); y, lag = np.diff(x), x[:-1]
    X = np.column_stack([np.ones(len(lag)), lag]); beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta; rs = float(res.std(ddof=1)); a, b = float(beta[0]), float(beta[1])
    k, mu, sig = -b / DT, -a / b, rs / math.sqrt(DT)
    hl = math.log(2) / (k * DT) if k > 0 else float("nan"); sst = sig / math.sqrt(2 * k) if k > 0 else float("nan")
    return {"kappa": k, "mu": mu, "sigma": sig, "half_life_periods": hl, "half_life_days": hl / 3 if k > 0 else float("nan"),
            "stationary_std": sst, "ols_intercept": a, "ols_slope": b, "residual_std": rs,
            "ols_r_squared": float(1 - (res @ res) / np.sum((y - y.mean()) ** 2)), "num_observations_fit": int(len(y))}

d_full = desc(f); d_full.update({"min": float(f.min()), "max": float(f.max()), "median": float(f.median()),
                                 "annualized_mean": d_full["mean"] * A, "annualized_volatility": d_full["std"] * math.sqrt(A)})
ac = {f"lag_{k}": float(f.autocorr(k)) for k in (1, 3, 6, 12, 24)}
adf_s, adf_p = adf(f); ou = ou_fit(f)

split = pd.Timestamp(P["regime_split_date"], tz="UTC")
regs, csv_rows = {}, []
for i, (label, m) in enumerate([("bear_2022", df["datetime"] < split), ("recovery_2023", df["datetime"] >= split)], start=1):
    x = f[m.to_numpy()].reset_index(drop=True); d = desc(x); s_, p_ = adf(x); o = ou_fit(x)
    regs[f"period{i}"] = {"label": label, **d, "autocorr_lag1": float(x.autocorr(1)), "adf_statistic": s_, "adf_p_value": p_,
                          "ou_kappa": o["kappa"], "ou_sigma": o["sigma"], "_n": int(len(x)), "_nfit": o["num_observations_fit"]}
    csv_rows.append({"period": i, "label": label, "start_date": df.loc[m, "datetime"].min().strftime("%Y-%m-%d"),
                     "end_date": df.loc[m, "datetime"].max().strftime("%Y-%m-%d"), "num_obs": int(len(x)), **d,
                     "autocorr_lag1": float(x.autocorr(1)), "adf_statistic": s_, "adf_p_value": p_, "ou_kappa": o["kappa"], "ou_sigma": o["sigma"]})
pd.DataFrame(csv_rows).to_csv(os.path.join(OUT_DIR, "regime_stats.csv"), index=False)

# Monte Carlo basis carry with the exact OU transition
k, mu, sig = ou["kappa"], ou["mu"], ou["sigma"]
e, sd = math.exp(-k * DT), sig * math.sqrt((1 - math.exp(-2 * k * DT)) / (2 * k))
rng = np.random.default_rng(SEED); f0 = float(f.iloc[-1])
cur, cum = np.full(N_P, f0), np.zeros(N_P)
for _ in range(H):
    cur = mu + (cur - mu) * e + sd * rng.standard_normal(N_P); cum += cur
ann = cum * (A / H); avg_rate = cum / H
loss = np.sort(-ann); var = max(float(loss[math.ceil(P["var_confidence"] * N_P) - 1]), 0.0)
tail = loss[loss > var]; cvar = float(tail.mean()) if len(tail) else 0.0
pct = lambda x, q: float(np.percentile(x, q))
mc = {"mean_annual_return": float(ann.mean()), "std_annual_return": float(ann.std(ddof=1)), "var_95": var, "cvar_95": cvar,
      "pct_negative_return": float((ann < 0).mean() * 100),
      "percentiles": {f"p{q}": pct(ann, q) for q in (5, 25, 50, 75, 95)}, "mc_num_paths": N_P, "mc_seed": SEED, "initial_funding_rate": f0}

clean = lambda d: {kk: v for kk, v in d.items() if not kk.startswith("_")}
results = {"symbol": P["target_symbol"], "num_observations": int(len(f)),
           "date_range": {"start": df["datetime"].iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ"), "end": df["datetime"].iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ")},
           "descriptive_stats": d_full, "autocorrelation": ac, "adf_test": {"statistic": adf_s, "p_value": adf_p, "is_stationary": bool(adf_p < 0.05)},
           "regime_analysis": {"period1": clean(regs["period1"]), "period2": clean(regs["period2"]),
                               "mean_ratio_p2_over_p1": regs["period2"]["mean"] / regs["period1"]["mean"],
                               "autocorr_diff": regs["period2"]["autocorr_lag1"] - regs["period1"]["autocorr_lag1"]},
           "ou_parameters": {kk: ou[kk] for kk in ("kappa", "mu", "sigma", "half_life_periods", "half_life_days", "stationary_std",
                                                   "ols_intercept", "ols_slope", "residual_std", "num_observations_fit")},
           "mc_basis_carry": mc}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
inter = {"num_observations": int(len(f)), "mean_funding_rate": d_full["mean"], "std_funding_rate": d_full["std"], "annualized_mean": d_full["annualized_mean"],
         "adf_statistic": adf_s, "regime1_mean": regs["period1"]["mean"], "regime2_mean": regs["period2"]["mean"], "ou_kappa": k, "ou_mu": mu, "ou_sigma": sig,
         "half_life_periods": ou["half_life_periods"], "half_life_days": ou["half_life_days"], "stationary_std": ou["stationary_std"],
         "mc_mean_annual_return": mc["mean_annual_return"], "mc_var_95": var, "mc_cvar_95": cvar, "mc_pct_negative": mc["pct_negative_return"]}
json.dump({"intermediates": {kk: {"value": v} for kk, v in inter.items()}}, open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
json.dump({"full_sample": {"kappa": k, "mu": mu, "sigma": sig, "ols_intercept": ou["ols_intercept"], "ols_slope": ou["ols_slope"],
                           "ols_r_squared": ou["ols_r_squared"], "residual_std": ou["residual_std"], "num_observations": ou["num_observations_fit"]},
           "period1_bear": {"kappa": regs["period1"]["ou_kappa"], "sigma": regs["period1"]["ou_sigma"], "num_observations": regs["period1"]["_nfit"]},
           "period2_recovery": {"kappa": regs["period2"]["ou_kappa"], "sigma": regs["period2"]["ou_sigma"], "num_observations": regs["period2"]["_nfit"]}},
          open(os.path.join(OUT_DIR, "ou_diagnostics.json"), "w"), indent=2)
json.dump({"mc_num_paths": N_P, "mc_random_seed": SEED, "mc_horizon_periods": H, "initial_funding_rate": f0,
           "ou_params_used": {"kappa": k, "mu": mu, "sigma": sig},
           "simulated_funding_rate_distribution": {"mean": float(avg_rate.mean()), "std": float(avg_rate.std(ddof=1)),
                                                   "p5": pct(avg_rate, 5), "p50": pct(avg_rate, 50), "p95": pct(avg_rate, 95)},
           "annual_carry_return_distribution": {"mean": mc["mean_annual_return"], "std": mc["std_annual_return"], "var_95": var, "cvar_95": cvar,
                                                "pct_negative": mc["pct_negative_return"], **mc["percentiles"]}},
          open(os.path.join(OUT_DIR, "mc_stats.json"), "w"), indent=2)
