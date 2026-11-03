# memory-solution
# unit: t1-cme-hdd-option-pricing | category: derivatives-pricing | tags: weather-derivatives, hdd, burn-analysis, ornstein-uhlenbeck, monte-carlo, exact-transition, bump-and-revalue, common-random-numbers, ghcn-daily
# deliverables: results.json, solution.json, clean_daily.csv, yearly_hdd.csv, ou_diagnostics.json, mc_stats.json, greeks.json
# GHCN file has no header; TMAX/TMIN in tenths of degC; filter station, contract month, sample years; need both TMAX & TMIN on a day.
# HDD_d = max(base_c - (tmax+tmin)/2, 0); monthly index = sum over the month; payoff = max(HDD - K, 0) * tick; DF = exp(-r * days_to_expiry/365).
# OU on anomalies x = tavg - mean_by_day_of_month: OLS dx = a + b x on within-year adjacent days; dt = 1/365; kappa = -b/dt; sigma = sd(eps, ddof=1)/sqrt(dt); theta = -a/b.
# MC exact transition x' = x e^{-k dt} + theta(1 - e^{-k dt}) + sigma sqrt((1 - e^{-2k dt})/(2k)) Z, x0 = 0, HDD of day d from x before the step.
# Greeks with common random numbers (same Z): (P_up - P_down)/(2 h); temp h = 0.1 degC on seasonal means;
#   vega/kappa: relative 1% bump but divide by the ABSOLUTE bump h = 0.01*sigma (resp. 0.01*kappa), i.e. dP/dsigma, dP/dkappa.
import os, json
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

ct = json.load(open(os.path.join(DATA_DIR, "contract_terms.json")))
raw = pd.read_csv(os.path.join(DATA_DIR, "weather_observations.csv"), header=None,
                  names=["station_id", "date", "element", "value", "m_flag", "q_flag", "s_flag"], dtype={"date": str, "q_flag": str, "m_flag": str})
n_raw = int(len(raw))
base_c = (ct["base_temperature_f"] - 32.0) * 5.0 / 9.0
K, tick, r = float(ct["strike_hdd"]), float(ct["tick_value_usd"]), float(ct["risk_free_rate"])
df_disc = float(np.exp(-r * ct["days_to_expiry"] / 365.0))

w = raw[(raw["station_id"] == ct["station_id"]) & raw["element"].isin(["TMAX", "TMIN"])].copy()
w["date"] = pd.to_datetime(w["date"], format="%Y%m%d")
w = w[(w["date"].dt.month == ct["contract_month"]) & w["date"].dt.year.between(ct["sample_start_year"], ct["sample_end_year"])]
piv = w.pivot_table(index="date", columns="element", values="value", aggfunc="first").dropna() / 10.0
clean = pd.DataFrame({"date": piv.index.strftime("%Y-%m-%d"), "year": piv.index.year, "tmin_c": piv["TMIN"].values, "tmax_c": piv["TMAX"].values})
clean["tavg_c"] = (clean["tmax_c"] + clean["tmin_c"]) / 2.0
clean["hdd"] = np.maximum(base_c - clean["tavg_c"], 0.0)
clean = clean.sort_values("date").reset_index(drop=True)
clean.to_csv(os.path.join(OUT_DIR, "clean_daily.csv"), index=False)

yearly = clean.groupby("year", as_index=False)["hdd"].sum().rename(columns={"hdd": "monthly_hdd"})
yearly["payoff"] = np.maximum(yearly["monthly_hdd"] - K, 0.0) * tick
yearly.to_csv(os.path.join(OUT_DIR, "yearly_hdd.csv"), index=False)
burn_exp = float(yearly["payoff"].mean()); burn_price = burn_exp * df_disc

# OU fit on day-of-month anomalies
dom = pd.to_datetime(clean["date"]).dt.day.values
seas = clean.groupby(dom)["tavg_c"].mean()                     # index = day-of-month 1..31
seas_vec = seas.reindex(range(1, 32)).values
x = clean["tavg_c"].values - seas_vec[dom - 1]
yr = clean["year"].values
same = (yr[1:] == yr[:-1]) & (dom[1:] == dom[:-1] + 1)
dx, xl = (x[1:] - x[:-1])[same], x[:-1][same]
X = np.column_stack([np.ones(len(xl)), xl])
(a, b), res, *_ = np.linalg.lstsq(X, dx, rcond=None)
eps = dx - X @ np.array([a, b])
dt = 1.0 / 365.0
kappa, sigma, theta = -b / dt, float(np.std(eps, ddof=1)) / np.sqrt(dt), -a / b
json.dump({"alpha": float(a), "beta": float(b), "kappa": float(kappa), "theta": float(theta), "sigma": float(sigma),
           "n_pairs": int(len(dx)), "dt": dt, "resid_std": float(np.std(eps, ddof=1)),
           "r_squared": float(1.0 - (eps ** 2).sum() / ((dx - dx.mean()) ** 2).sum())},
          open(os.path.join(OUT_DIR, "ou_diagnostics.json"), "w"), indent=2)

# Monte Carlo with exact OU transition and common random numbers
n_days, n_paths, seed = len(seas_vec), int(ct["mc_num_paths"]), int(ct["mc_random_seed"])
Z = np.random.default_rng(seed).standard_normal((n_paths, n_days - 1))

def simulate(seas_m, kap, sig, th):
    e = np.exp(-kap * dt); sd = sig * np.sqrt((1.0 - np.exp(-2.0 * kap * dt)) / (2.0 * kap))
    xt = np.zeros(n_paths); hdd = np.zeros(n_paths)
    for d in range(n_days):
        hdd += np.maximum(base_c - (seas_m[d] + xt), 0.0)
        if d < n_days - 1:
            xt = xt * e + th * (1.0 - e) + sd * Z[:, d]
    pay = np.maximum(hdd - K, 0.0) * tick
    return df_disc * pay.mean(), hdd, pay

mc_price, hdd_sim, pay_sim = simulate(seas_vec, kappa, sigma, theta)
mc_se = float(df_disc * pay_sim.std(ddof=1) / np.sqrt(n_paths))
json.dump({"mc_price": float(mc_price), "mc_expected_payoff": float(pay_sim.mean()), "mc_std_error": mc_se,
           "mc_hdd_mean": float(hdd_sim.mean()), "mc_hdd_std": float(hdd_sim.std(ddof=1)),
           "mc_hdd_5th_pct": float(np.percentile(hdd_sim, 5)), "mc_hdd_95th_pct": float(np.percentile(hdd_sim, 95)),
           "mc_num_paths": n_paths, "mc_random_seed": seed}, open(os.path.join(OUT_DIR, "mc_stats.json"), "w"), indent=2)

h_t, h_rel = 0.1, 0.01
temp_delta = (simulate(seas_vec + h_t, kappa, sigma, theta)[0] - simulate(seas_vec - h_t, kappa, sigma, theta)[0]) / (2 * h_t)
vega = (simulate(seas_vec, kappa, sigma * (1 + h_rel), theta)[0] - simulate(seas_vec, kappa, sigma * (1 - h_rel), theta)[0]) / (2 * h_rel * sigma)
kappa_sens = (simulate(seas_vec, kappa * (1 + h_rel), sigma, theta)[0] - simulate(seas_vec, kappa * (1 - h_rel), sigma, theta)[0]) / (2 * h_rel * kappa)
greeks = {"temp_delta": float(temp_delta), "vega": float(vega), "kappa_sens": float(kappa_sens)}
json.dump({**greeks, "bump_temp_c": h_t, "bump_sigma_pct": h_rel, "bump_kappa_pct": h_rel}, open(os.path.join(OUT_DIR, "greeks.json"), "w"), indent=2)

diff_pct = float(abs(mc_price - burn_price) / burn_price * 100.0)
results = {"station_id": ct["station_id"], "contract_month": int(ct["contract_month"]), "base_temperature_c": float(base_c), "strike_hdd": K,
           "num_clean_days": int(len(clean)), "num_years": int(len(yearly)), "discount_factor": df_disc,
           "burn_expected_payoff": burn_exp, "burn_price": float(burn_price), "ou_kappa": float(kappa), "ou_theta": float(theta),
           "ou_sigma": float(sigma), "mc_price": float(mc_price), "mc_std_error": mc_se, "burn_mc_diff_pct": diff_pct, "greeks": greeks}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
inter = {"num_raw_rows": n_raw, "num_clean_days": int(len(clean)), "num_years": int(len(yearly)), "base_temperature_c": float(base_c),
         "mean_monthly_hdd": float(yearly["monthly_hdd"].mean()), "std_monthly_hdd": float(yearly["monthly_hdd"].std(ddof=1)),
         "burn_price": float(burn_price), "ou_kappa": float(kappa), "ou_sigma": float(sigma), "mc_price": float(mc_price),
         "mc_hdd_mean": float(hdd_sim.mean()), "burn_mc_diff_pct": diff_pct, **greeks}
json.dump({"intermediates": {k: {"value": v} for k, v in inter.items()}}, open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
