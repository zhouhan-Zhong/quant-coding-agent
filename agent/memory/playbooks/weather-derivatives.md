---
title: Weather derivatives: HDD/CDD indices, burn analysis, temperature OU Monte Carlo
category: derivatives-pricing
tags: weather derivatives, temperature, hdd, cdd, heating degree days, cooling degree days, cme weather, degree day option, weather futures, burn analysis, historical burn, ghcn, ghcn-daily, noaa, station, tmax, tmin, tenths of degree, seasonal mean, temperature anomaly, ornstein-uhlenbeck, mean reversion, monte carlo, tick value, temperature delta, vega, common random numbers
summary: Degree-day index and payoff definitions, GHCN-Daily cleaning, burn-analysis pricing, OU calibration on seasonal temperature anomalies, exact-transition Monte Carlo and CRN bump-and-revalue Greeks with the unit conventions the references used.
triggers: text:degree day, text:hdd, text:cdd, file:*ghcn*
---
## When this applies
"Price an HDD/CDD call or put on a monthly degree-day index (CME-style)", "burn analysis from historical station
temperatures", "clean GHCN-Daily TMAX/TMIN data", "fit a mean-reverting (OU) temperature model and price by Monte
Carlo", "temperature delta / vega / mean-reversion sensitivity", "compare burn and MC prices".

## Method and formulas
- Base temperature 65 F = (65 - 32) * 5/9 = 18.3333 C (do not round). Daily average tavg = (tmax + tmin)/2 (no rounding
  unless the task says). HDD_d = max(base - tavg, 0); CDD_d = max(tavg - base, 0); monthly index = sum over the contract
  month's days. Keep the index in the task's units (the reference worked in degC with the converted base; a degF index is
  9/5 x the degC one).
- Payoffs: call = tick * max(Index - K, 0); put = tick * max(K - Index, 0); optional cap: min(payoff, cap).
- Discounting: DF = exp(-r * days_to_expiry/365) (the reference used days/365, not /360 and not the month length).
- Burn analysis: for each historical year in the sample window compute the index and payoff; expected payoff = plain
  mean over years; price = DF * mean; std of the yearly index with ddof=1.
- GHCN-Daily csv has NO header: station_id, date (YYYYMMDD -> read as str), element, value, m_flag, q_flag, s_flag (flags
  mixed types -> dtype str); values in TENTHS of degC (/10). Filter station, element in {TMAX, TMIN}, contract month, year
  range; pivot date x element (aggfunc "first"); keep days with both (dropna). Report counts (clean days, years).
- Temperature OU on anomalies: seasonal mean mu(d) = mean tavg by day-of-month over the sample (or day-of-year / a Fourier
  fit if the task says); x = tavg - mu(d); regress dx_t = a + b x_t + eps on consecutive calendar days only (no pairs across
  years or gaps); dt = 1/365; kappa = -b/dt, theta = -a/b (~0), sigma = sd(eps, ddof=1)/sqrt(dt).
- MC: exact OU step x' = x e^{-kappa dt} + theta (1 - e^{-kappa dt}) + sigma sqrt((1 - e^{-2 kappa dt})/(2 kappa)) Z with
  x0 = 0 (or the last anomaly if told); day d's degree days use x BEFORE the step, so an n-day month needs n - 1 draws per
  path; temperature = mu(d) + x. Price = DF * mean(payoff); se = DF * std(payoff, ddof=1)/sqrt(n_paths).
- Greeks by bump-and-revalue with ONE Z matrix (common random numbers): temperature delta = (P(mu + h) - P(mu - h))/(2h)
  with h = 0.1 degree added to every seasonal mean (negative for HDD calls); vega = (P(sigma (1 + 1%)) - P(sigma (1 - 1%)))/
  (2 * 0.01 * sigma) (per unit sigma); kappa sensitivity likewise. Dividing by 2 * 0.01 instead multiplies the Greek by
  the parameter's value (a per-1%-relative sensitivity, not the per-unit one the references report).
- Diagnostics often requested: burn vs MC gap % = |MC - burn|/burn * 100; MC index mean/std/5th/95th percentiles; OU
  regression R^2; number of regression pairs.

## Conventions and pitfalls (what graders check)
- Exact counts: clean days = days with both TMAX and TMIN in the filtered window; years = distinct years; columns of the
  daily table exactly as specified (date ISO string, year, tmin, tmax, tavg, hdd) sorted by date.
- Yearly table: one row per year with index and payoff; payoff == tick * max(index - K, 0) recomputed by the checker.
- Seeds: default_rng(seed) (or the API named) and the stated n_paths; Z drawn once with shape (n_paths, n_days - 1).
- MC and burn prices can differ by 10-20% (the OU index variance differs from the sample's); checkers allow a band
  (e.g. gap < 20%) - do not tune the model to close it.
- Report bump sizes as given (e.g. bump_temp 0.1, bump_sigma_pct 0.01) next to the Greeks; ints for counts and seeds.
- A seasonal mean by day-of-month must cover every contract day (31 for January); missing days misalign the vector.

## Tested snippet
```python
import math
import numpy as np, pandas as pd

def load_ghcn(path, station, month, y0, y1):        # GHCN-Daily csv without header; values in tenths of degC
    raw = pd.read_csv(path, header=None, names=["station_id", "date", "element", "value", "m_flag", "q_flag", "s_flag"],
                      dtype={"date": str, "m_flag": str, "q_flag": str, "s_flag": str})
    w = raw[(raw["station_id"] == station) & raw["element"].isin(["TMAX", "TMIN"])].copy()
    w["date"] = pd.to_datetime(w["date"], format="%Y%m%d"); w = w[(w.date.dt.month == month) & w.date.dt.year.between(y0, y1)]
    p = w.pivot_table(index="date", columns="element", values="value", aggfunc="first").dropna() / 10.0
    return pd.DataFrame({"date": p.index, "tmin": p["TMIN"].values, "tmax": p["TMAX"].values, "tavg": (p["TMIN"] + p["TMAX"]).values / 2})

def degree_days(tavg, base=(65 - 32) * 5 / 9, kind="hdd"):
    return np.maximum(base - tavg, 0.0) if kind == "hdd" else np.maximum(tavg - base, 0.0)

def fit_anomaly_ou(dates, tavg, dt=1 / 365):        # seasonal mean by day of month; OLS dx = a + b x on consecutive days
    d = pd.DatetimeIndex(dates); seas = pd.Series(np.asarray(tavg)).groupby(d.day).mean(); x = np.asarray(tavg) - seas.loc[d.day].values
    ok = np.diff(d.values).astype("timedelta64[D]").astype(int) == 1              # no pairs across years / gaps
    xl, dx = x[:-1][ok], np.diff(x)[ok]; b, a = np.polyfit(xl, dx, 1); eps = dx - a - b * xl
    return {"kappa": -b / dt, "theta": -a / b, "sigma": eps.std(ddof=1) / math.sqrt(dt), "seasonal": seas.values, "n_pairs": int(ok.sum())}

def mc_degree_day_option(seas, kappa, sigma, theta, Z, K, tick, df, kind="hdd", call=True, dt=1 / 365):
    """Exact OU anomaly paths, x0 = 0; day d uses x BEFORE stepping; Z: (n_paths, n_days - 1), reused for every bump."""
    e = math.exp(-kappa * dt); sd = sigma * math.sqrt((1 - e * e) / (2 * kappa)); x = np.zeros(len(Z)); idx = np.zeros(len(Z))
    for d in range(len(seas)):
        idx += degree_days(seas[d] + x, kind=kind)
        if d < len(seas) - 1: x = x * e + theta * (1 - e) + sd * Z[:, d]
    pay = tick * np.maximum((idx - K) if call else (K - idx), 0.0)
    return df * pay.mean(), df * pay.std(ddof=1) / math.sqrt(len(pay)), idx

def crn_greeks(price, seas, kappa, sigma, h_temp=0.1, rel=0.01):   # price(seas, kappa, sigma) must reuse ONE Z matrix
    return {"temp_delta": (price(seas + h_temp, kappa, sigma) - price(seas - h_temp, kappa, sigma)) / (2 * h_temp),
            "vega": (price(seas, kappa, sigma * (1 + rel)) - price(seas, kappa, sigma * (1 - rel))) / (2 * rel * sigma),
            "kappa_sens": (price(seas, kappa * (1 + rel), sigma) - price(seas, kappa * (1 - rel), sigma)) / (2 * rel * kappa)}
```
