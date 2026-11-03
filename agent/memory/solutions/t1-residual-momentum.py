# memory-solution
# unit: t1-residual-momentum | category: backtest-factors | tags: fama-macbeth, residual-momentum, newey-west, expanding-window-oos, quintile-sort, spearman-ic, cross-sectional-zscore, parquet
# deliverables: fmb_results.json, monthly_oos_spread_returns.csv, monthly_fmb_coefficients.csv
# per month & model: dropna on target+regressors, need N >= max(80, 4(k+1)); sort by identifier; z-score ddof=0 (0 if sd==0); lstsq with const
# OOS from calendar-month index >= 120 (0-based, i.e. strictly after the 120th month): pred = X_t @ mean(all strictly prior monthly betas)
# quintiles: rank by (pred asc, identifier asc), q = min(5, 5*rank//N + 1); row only if Q1 and Q5 non-empty; IC = Spearman (avg ranks), NaN if constant
# NW t-stat (lag 6, Bartlett, 1/T autocov denominators): s2 = sum e^2/T + 2 sum_k w_k sum e_t e_{t-k}/T ; t = mean / sqrt(s2/T)
# empty CSV cells for slopes of signals not in the model and for undefined IC (NaN -> '' via to_csv)
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

CONTROLS = ["rvol_21d", "ivol_capm_21d", "ivol_capm_252d", "ivol_ff3_21d", "ivol_hxz4_21d", "beta_60m", "beta_dimson_21d",
            "betabab_1260d", "betadown_252d", "corr_1260d", "rskew_21d", "coskew_21d", "iskew_capm_21d", "iskew_ff3_21d", "iskew_hxz4_21d"]
RAW, RES = ["ret_12_7", "ret_6_1"], ["resff3_12_1", "resff3_6_1"]
MODELS = {"A": CONTROLS + RAW, "B": CONTROLS + RES, "C": CONTROLS + RAW + RES}
NW_LAG, BURN_IN = 6, 120

df = pd.read_parquet(os.path.join(DATA_DIR, "stock_chars.pqt"))
n_input = int(len(df))
df["date"] = pd.to_datetime(df["date"])
months = np.sort(df["date"].unique()); d2i = {d: i for i, d in enumerate(months)}

def nw_tstat(x, lag=NW_LAG):
    x = np.asarray(x, float); T = len(x); e = x - x.mean()
    s2 = (e * e).sum() / T
    for k in range(1, lag + 1):
        s2 += 2.0 * (1.0 - k / (lag + 1.0)) * (e[k:] * e[:-k]).sum() / T
    return float(x.mean() / math.sqrt(s2 / T))

def spearman(a, b):
    ra = pd.Series(a).rank(method="average").to_numpy(); rb = pd.Series(b).rank(method="average").to_numpy()
    return float("nan") if ra.std() == 0 or rb.std() == 0 else float(np.corrcoef(ra, rb)[0, 1])

coef_rows, spread_rows, summary = [], [], {}
for model, regs in MODELS.items():
    k = len(regs); min_n = max(80, 4 * (k + 1))
    betas, ics, spreads, n_months, n_oos = [], [], [], 0, 0
    for date, grp in df.groupby("date", sort=True):
        sub = grp[["identifier", "ret_lead1m"] + regs].dropna()
        if len(sub) < min_n:
            continue
        sub = sub.sort_values("identifier").reset_index(drop=True); N = len(sub)
        Z = sub[regs].to_numpy(float); mu = Z.mean(axis=0); sd = Z.std(axis=0, ddof=0)
        Z = np.where(sd > 0, (Z - mu) / np.where(sd > 0, sd, 1.0), 0.0)
        X = np.column_stack([np.ones(N), Z]); y = sub["ret_lead1m"].to_numpy(float)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        n_months += 1
        row = {"date": date.strftime("%Y-%m-%d"), "model": model, "n_assets": N, "alpha": beta[0]}
        row.update({s: (beta[1 + regs.index(s)] if s in regs else np.nan) for s in RAW + RES})
        coef_rows.append(row)
        if d2i[date] >= BURN_IN and betas:
            pred = X @ np.mean(np.vstack(betas), axis=0); ids = sub["identifier"].to_numpy()
            ic = spearman(pred, y)
            order = np.lexsort((ids, pred)); rank = np.empty(N, int); rank[order] = np.arange(N)
            q = np.minimum(5, (5 * rank) // N + 1); q1, q5 = q == 1, q == 5
            if q1.any() and q5.any():
                q1r, q5r = float(y[q1].mean()), float(y[q5].mean())
                spread_rows.append({"date": date.strftime("%Y-%m-%d"), "model": model, "n_assets": N, "q1_count": int(q1.sum()),
                                    "q5_count": int(q5.sum()), "q1_return": q1r, "q5_return": q5r, "q5_minus_q1": q5r - q1r, "ic": ic})
                spreads.append(q5r - q1r); n_oos += 1
                if np.isfinite(ic): ics.append(ic)
        betas.append(beta)
    B = np.vstack(betas)
    summary[model] = {"n_months": n_months, "n_oos": n_oos, "mean_ic": float(np.mean(ics)), "mean_spread": float(np.mean(spreads)),
                      "avg": {s: float(B[:, 1 + regs.index(s)].mean()) for s in regs}, "t": {s: nw_tstat(B[:, 1 + regs.index(s)]) for s in regs}}

C = summary["C"]
raw_max = max(abs(C["t"][s]) for s in RAW); res_max = max(abs(C["t"][s]) for s in RES)
beats = bool(summary["B"]["mean_ic"] > summary["A"]["mean_ic"] and summary["B"]["mean_spread"] > summary["A"]["mean_spread"] and res_max > raw_max)
res = {"task_id": "residual-momentum", "nw_lag": NW_LAG, "n_input_rows": n_input}
for m in "abc":
    res[f"n_months_model_{m}"] = int(summary[m.upper()]["n_months"]); res[f"n_oos_months_model_{m}"] = int(summary[m.upper()]["n_oos"])
for s in RAW + RES:
    res[f"avg_beta_model_c_{s}"] = C["avg"][s]; res[f"tstat_model_c_{s}"] = C["t"][s]
res["residual_beats_raw"] = beats
json.dump(res, open(os.path.join(OUT_DIR, "fmb_results.json"), "w"), indent=2)
pd.DataFrame(spread_rows).sort_values(["date", "model"]).to_csv(os.path.join(OUT_DIR, "monthly_oos_spread_returns.csv"), index=False)
pd.DataFrame(coef_rows, columns=["date", "model", "n_assets", "alpha"] + RAW + RES).sort_values(["date", "model"]).to_csv(
    os.path.join(OUT_DIR, "monthly_fmb_coefficients.csv"), index=False)
