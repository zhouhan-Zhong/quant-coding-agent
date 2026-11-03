# memory-solution
# unit: t1-event-study-earnings | category: backtest-factors | tags: event-study, market-model, abnormal-returns, car, caar, corrado-rank-test, patell, bmp, kolari-pynnonen, log-returns, ols
# deliverables: market_model_params.csv, abnormal_returns.csv, car_per_event.csv, caar_summary.json, statistical_tests.json
# log returns, merged on common dates; event day 0 = event date position k; estimation = positions k-130..k-11 (120 obs), window k-5..k+5
# OLS via lstsq; AR = r - (a + b r_m); CAR sums; CAAR = cross-event mean per relative day
# Corrado: pool est residuals + 11 ARs, K=rankdata, U=K/(T+1)-0.5, sigma_i=sqrt(mean U_i^2) (ddof 0);
#   z_day0 = sum_i U_i0/sigma_i / sqrt(N); z_full = sum_i [sum_win U_it/(sigma_i sqrt(11))] / sqrt(N)
# Patell: s_i^2 = SSR/(L-2); C_it = 1 + 1/L + (Rm_t - Rm_bar)^2 / sum_est (Rm - Rm_bar)^2; SAR = AR/(s sqrt(C)); SCAR = sum SAR/sqrt(11)
# BMP t = mean(X)/(std(X, ddof=1)/sqrt(N)); KP adj: t *= sqrt((1-rho)/(1+(N-1) rho)), rho = mean pairwise corr of est residuals
#   over all event pairs with >= 10 common dates (same-ticker pairs included)
import os, json, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

st = pd.read_csv(os.path.join(DATA_DIR, "stock_prices.csv"), parse_dates=["date"]).set_index("date").sort_index()
mk = pd.read_csv(os.path.join(DATA_DIR, "market_prices.csv"), parse_dates=["date"]).set_index("date").sort_index()
ev = pd.read_csv(os.path.join(DATA_DIR, "events.csv"))
rs = np.log(st / st.shift(1)).dropna(how="all")
rm = np.log(mk["adj_close"] / mk["adj_close"].shift(1)).dropna()
idx = rs.index.intersection(rm.index)
rs, rm = rs.loc[idx], rm.loc[idx].to_numpy()
dates = [d.strftime("%Y-%m-%d") for d in idx]
pos = {d: i for i, d in enumerate(dates)}
REL = list(range(-5, 6))

params, ar_rows, car_rows, events = [], [], [], []
for _, e in ev.iterrows():
    k = pos.get(e.event_date)
    if k is None:
        continue
    est = list(range(max(0, k - 130), k - 10))                      # days -130..-11
    win = list(range(k - 5, k + 6))                                 # days -5..+5
    if len(est) < 60 or win[0] < 0 or win[-1] >= len(dates):
        continue
    y = rs[e.ticker].to_numpy()[est]; x = rm[est]
    if np.isnan(y).any():
        continue
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r2 = 1.0 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    yw = rs[e.ticker].to_numpy()[win]; xw = rm[win]
    ar = yw - (coef[0] + coef[1] * xw)
    params.append(dict(ticker=e.ticker, event_date=e.event_date, alpha=round(float(coef[0]), 8), beta=round(float(coef[1]), 8),
                       r_squared=round(float(r2), 8), n_obs=int(len(x))))
    for j, rd in enumerate(REL):
        ar_rows.append(dict(ticker=e.ticker, event_date=e.event_date, relative_day=rd, date=dates[win[j]], ar=round(float(ar[j]), 8)))
    car_rows.append(dict(ticker=e.ticker, event_date=e.event_date, car_full=round(float(ar.sum()), 8), car_pre=round(float(ar[:5].sum()), 8),
                         car_event=round(float(ar[5]), 8), car_post=round(float(ar[6:].sum()), 8)))
    events.append(dict(resid=resid, est_dates=[dates[i] for i in est], ar=ar, xm=x, xw=xw, L=len(x)))

pd.DataFrame(params).to_csv(os.path.join(OUT_DIR, "market_model_params.csv"), index=False)
pd.DataFrame(ar_rows).to_csv(os.path.join(OUT_DIR, "abnormal_returns.csv"), index=False)
pd.DataFrame(car_rows).to_csv(os.path.join(OUT_DIR, "car_per_event.csv"), index=False)

N = len(events)
AR = np.array([e["ar"] for e in events])                            # (N, 11)
caar = AR.mean(axis=0)
json.dump({"num_events": N, "caar_by_day": {str(d): round(float(caar[j]), 8) for j, d in enumerate(REL)},
           "caar_full": round(float(caar.sum()), 8), "caar_pre": round(float(caar[:5].sum()), 8),
           "caar_post": round(float(caar[6:].sum()), 8)}, open(os.path.join(OUT_DIR, "caar_summary.json"), "w"), indent=2)

# Corrado (1989) rank test with per-event rank scaling
z0, zf = 0.0, 0.0
for e in events:
    pooled = np.concatenate([e["resid"], e["ar"]]); T = len(pooled)
    U = rankdata(pooled) / (T + 1.0) - 0.5
    sig = np.sqrt((U ** 2).mean())
    z0 += U[-6] / sig; zf += U[-11:].sum() / (sig * np.sqrt(11.0))
z0 /= np.sqrt(N); zf /= np.sqrt(N)

# Kolari-Pynnonen adjusted BMP using Patell standardised residuals
sar0, scar = [], []
for e in events:
    xbar = e["xm"].mean()
    C = 1.0 + 1.0 / e["L"] + (e["xw"] - xbar) ** 2 / ((e["xm"] - xbar) ** 2).sum()
    s = np.sqrt((e["resid"] ** 2).sum() / (e["L"] - 2))
    sar = e["ar"] / (s * np.sqrt(C))
    sar0.append(sar[5]); scar.append(sar.sum() / np.sqrt(11.0))
cors = []
for a, b in itertools.combinations(range(N), 2):
    sa = pd.Series(events[a]["resid"], index=events[a]["est_dates"]); sb = pd.Series(events[b]["resid"], index=events[b]["est_dates"])
    com = sa.index.intersection(sb.index)
    if len(com) >= 10:
        cors.append(np.corrcoef(sa.loc[com], sb.loc[com])[0, 1])
rho = float(np.mean(cors))
adj = np.sqrt((1.0 - rho) / (1.0 + (N - 1) * rho))
bmp = lambda v: float(np.mean(v) / (np.std(v, ddof=1) / np.sqrt(N)))
json.dump({"corrado_z_full": round(float(zf), 8), "corrado_z_day0": round(float(z0), 8), "kp_rho_bar": round(rho, 8),
           "kp_t_full": round(bmp(scar) * adj, 8), "kp_t_day0": round(bmp(sar0) * adj, 8)},
          open(os.path.join(OUT_DIR, "statistical_tests.json"), "w"), indent=2)
