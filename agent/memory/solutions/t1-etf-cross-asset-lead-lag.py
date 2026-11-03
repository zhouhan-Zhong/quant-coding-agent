# memory-solution
# unit: t1-etf-cross-asset-lead-lag | category: backtest-factors | tags: lead-lag, lagged-correlation, asymmetry, fisher-z, significance-gate, market-model, residual-returns, out-of-sample, subperiod-stability, etf
# deliverables: results.json, solution.json, lead_lag_results.csv, residual_lead_lag_results.csv
# r = ln(P_t/P_{t-1}); split RETURN dates: IS <= in_sample_end, OOS >= out_of_sample_start. Pairs = itertools.combinations(CSV column order).
# corr(k) = np.corrcoef(A[t], B[t+k]) (k>0: A[:n-k],B[k:]; k<0: A[|k|:],B[:n-|k|]); a_leads_b = max_{k=1..L}, b_leads_a = max_{k=-L..-1};
#   asym = a_leads_b - b_leads_a, leader = A if asym >= 0. Gate: |atanh(clip c_ab) - atanh(clip c_ba)| > 1.96*sqrt(2/(n-3)), n = period days.
# Round every lag corr to 6 dp BEFORE max/difference (bit-exact oracle match). Tables: all pairs sorted by |asym| desc (stable;
#   rank = 1-based position); top list = significant pairs only, first num_top_pairs; summary ratios left unrounded.
# Validation in any period = recompute the full table there and look up the pair's rank/asym; same_sign = sign(asym) equal.
# Market model fit on IS only (OLS with intercept, per non-SPY ETF); residuals e = r - alpha - beta*r_SPY with the REPORTED (6-dp)
#   alpha/beta (bit-exact oracle match), reused for OOS / 2022 / 2023; r_squared from the unrounded fit.
# Annualized return = sum(log r) / ((last_ret_date - first_ret_date).days / 365.25) on the full return series (not rounded).
import os, json, math, itertools
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
R6 = lambda x: round(float(x), 6)
d2s = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")

prm = json.load(open(os.path.join(DATA_DIR, "params.json")))
L, NTOP = int(prm["max_lag"]), int(prm["num_top_pairs"])
px = pd.read_csv(os.path.join(DATA_DIR, "etf_prices.csv"), parse_dates=["date"]).sort_values("date").set_index("date")
tick = list(px.columns)
R = np.log(px / px.shift(1)).iloc[1:]
IS, OOS = R[R.index <= pd.Timestamp(prm["in_sample_end"])], R[R.index >= pd.Timestamp(prm["out_of_sample_start"])]
LAGS = list(range(-L, L + 1))
ZCRIT = 1.959963984540054

def lag_corr(a, b, k):
    n = len(a)
    x, y = (a[:n - k], b[k:]) if k > 0 else ((a[-k:], b[:n + k]) if k < 0 else (a, b))
    return float(np.corrcoef(x, y)[0, 1])

def lead_lag_table(df, names):
    rows, n = [], len(df)
    for a, b in itertools.combinations(names, 2):
        x, y = df[a].to_numpy(), df[b].to_numpy()
        c = {k: R6(lag_corr(x, y, k)) for k in LAGS}              # oracle rounds correlations FIRST, then differences
        cab, cba = max(c[k] for k in range(1, L + 1)), max(c[k] for k in range(-L, 0))
        asym = R6(cab - cba)
        dz = abs(np.arctanh(np.clip(cab, -0.999999, 0.999999)) - np.arctanh(np.clip(cba, -0.999999, 0.999999)))
        rows.append(dict(etf_a=a, etf_b=b, corr_lag0=c[0], corr_a_leads_b=cab, corr_b_leads_a=cba, asymmetry=asym,
                         leader=a if asym >= 0 else b, lagger=b if asym >= 0 else a,
                         **{f"lag_{k}": c[k] for k in LAGS}, sig=bool(dz > ZCRIT * math.sqrt(2.0 / (n - 3)))))
    t = pd.DataFrame(rows)
    t = t.iloc[np.argsort(-t["asymmetry"].abs().to_numpy(), kind="stable")].reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    t.index = t["etf_a"] + "__" + t["etf_b"]
    return t

def write_table(t, name):
    cols = ["etf_a", "etf_b", "corr_lag0", "corr_a_leads_b", "corr_b_leads_a", "asymmetry", "leader", "lagger"] + [f"lag_{k}" for k in LAGS]
    out = t[cols].copy()
    num = [c for c in cols if c not in ("etf_a", "etf_b", "leader", "lagger")]
    out[num] = out[num].round(6)
    out.to_csv(os.path.join(OUT_DIR, name), index=False)

def top_pairs(t):
    top = t[t["sig"]].head(NTOP)
    return top, [dict(etf_a=r.etf_a, etf_b=r.etf_b, leader=r.leader, lagger=r.lagger, asymmetry=R6(r.asymmetry),
                      corr_a_leads_b=R6(r.corr_a_leads_b), corr_b_leads_a=R6(r.corr_b_leads_a)) for r in top.itertuples()]

same = lambda x, y: bool(np.sign(x) == np.sign(y))

def validate(top, t_oos):
    rows = []
    for i, (key, r) in enumerate(top.iterrows(), 1):
        o = t_oos.loc[key]
        rows.append(dict(etf_a=r.etf_a, etf_b=r.etf_b, in_sample_rank=i, in_sample_leader=r.leader, in_sample_lagger=r.lagger,
                         in_sample_asymmetry=R6(r.asymmetry), out_of_sample_rank=int(o["rank"]), out_of_sample_leader=o.leader,
                         out_of_sample_lagger=o.lagger, out_of_sample_asymmetry=R6(o.asymmetry), same_sign=same(o.asymmetry, r.asymmetry)))
    n = len(rows); ss = sum(r["same_sign"] for r in rows)
    summ = dict(same_sign_count=int(ss), sign_consistency_rate=R6(ss / n) if n else 0.0,
                avg_rank_shift=R6(np.mean([abs(r["out_of_sample_rank"] - r["in_sample_rank"]) for r in rows])) if n else 0.0,
                avg_abs_asymmetry_ratio=float(np.mean([abs(t_oos.loc[k, "asymmetry"]) / abs(top.loc[k, "asymmetry"]) for k in top.index])) if n else 0.0,
                num_pairs_still_top10=int(sum(r["out_of_sample_rank"] <= 10 for r in rows)))
    return rows, summ

# ---- raw lead-lag
t_is, t_oos = lead_lag_table(IS, tick), lead_lag_table(OOS, tick)
write_table(t_is, "lead_lag_results.csv")
top_raw, top_raw_json = top_pairs(t_is)
val_raw, pers_raw = validate(top_raw, t_oos)

# ---- in-sample SPY market model, residuals for every period with the IS alpha/beta
others = [c for c in tick if c != "SPY"]
x = IS["SPY"].to_numpy()
betas, E = {}, pd.DataFrame(index=R.index)
for c in others:
    y = IS[c].to_numpy()
    beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)); alpha = float(y.mean() - beta * x.mean())
    resid = y - alpha - beta * x
    betas[c] = dict(alpha=R6(alpha), beta=R6(beta), r_squared=R6(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()))
    E[c] = R[c] - betas[c]["alpha"] - betas[c]["beta"] * R["SPY"]
E_is, E_oos = E.loc[IS.index], E.loc[OOS.index]
t_ris, t_roos = lead_lag_table(E_is, others), lead_lag_table(E_oos, others)
write_table(t_ris, "residual_lead_lag_results.csv")
top_res, top_res_json = top_pairs(t_ris)
val_res, pers_res = validate(top_res, t_roos)

# ---- residual subperiod stability (2022, 2023) with the same IS fit
t_y = {y: lead_lag_table(E[E.index.year == y], others) for y in (2022, 2023)}
sub = []
for i, (key, r) in enumerate(top_res.iterrows(), 1):
    row = dict(etf_a=r.etf_a, etf_b=r.etf_b, in_sample_rank=i)
    for y in (2022, 2023):
        o = t_y[y].loc[key]
        row.update({f"rank_{y}": int(o["rank"]), f"leader_{y}": o.leader, f"asymmetry_{y}": R6(o.asymmetry), f"same_sign_{y}": same(o.asymmetry, r.asymmetry)})
    row["same_sign_both_years"] = bool(row["same_sign_2022"] and row["same_sign_2023"])
    sub.append(row)
sub_sum = {"num_pairs_same_sign_2022": int(sum(r["same_sign_2022"] for r in sub)), "num_pairs_same_sign_2023": int(sum(r["same_sign_2023"] for r in sub)),
           "num_pairs_same_sign_both_years": int(sum(r["same_sign_both_years"] for r in sub)),
           "avg_rank_shift_2022": R6(np.mean([abs(r["rank_2022"] - r["in_sample_rank"]) for r in sub])) if sub else 0.0,
           "avg_rank_shift_2023": R6(np.mean([abs(r["rank_2023"] - r["in_sample_rank"]) for r in sub])) if sub else 0.0}

results = {
    "num_trading_days": int(len(px)), "num_return_days": int(len(R)), "num_etfs": int(len(tick)),
    "date_range": {"start": d2s(px.index[0]), "end": d2s(px.index[-1])},
    "sample_split": {"in_sample_start": d2s(IS.index[0]), "in_sample_end": d2s(IS.index[-1]), "in_sample_return_days": int(len(IS)),
                     "out_of_sample_start": d2s(OOS.index[0]), "out_of_sample_end": d2s(OOS.index[-1]), "out_of_sample_return_days": int(len(OOS))},
    "num_significant_pairs": int(t_is["sig"].sum()), "top_lead_lag_pairs": top_raw_json,
    "out_of_sample_validation": val_raw, "persistence_summary": pers_raw, "market_model_betas": betas,
    "num_significant_residual_pairs": int(t_ris["sig"].sum()), "residual_top_lead_lag_pairs": top_res_json,
    "residual_out_of_sample_validation": val_res, "residual_persistence_summary": pers_res,
    "residual_subperiod_validation": sub, "residual_subperiod_summary": sub_sum,
    "signal_assessment": {"raw_persists_oos": bool(pers_raw["sign_consistency_rate"] > 0.5),
                          "residual_improves_persistence": bool(pers_res["same_sign_count"] >= pers_raw["same_sign_count"])},
}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

yrs = (R.index[-1] - R.index[0]).days / 365.25
V = lambda v: {"value": v}
inter = {
    "num_trading_days": V(results["num_trading_days"]), "num_return_days": V(results["num_return_days"]), "num_etfs": V(results["num_etfs"]),
    "spy_annualized_return": V(float(R["SPY"].sum() / yrs)), "tlt_annualized_return": V(float(R["TLT"].sum() / yrs)),
    "corr_spy_tlt": V(R6(np.corrcoef(R["SPY"], R["TLT"])[0, 1])),
    "raw_top1_leader": V(top_raw_json[0]["leader"]), "raw_top1_lagger": V(top_raw_json[0]["lagger"]),
    "raw_top1_asymmetry": V(top_raw_json[0]["asymmetry"]), "raw_same_sign_count": V(pers_raw["same_sign_count"]),
    "raw_sign_consistency_rate": V(pers_raw["sign_consistency_rate"]),
    "tlt_beta_to_spy": V(betas["TLT"]["beta"]), "uso_beta_to_spy": V(betas["USO"]["beta"]),
    "residual_top1_leader": V(top_res_json[0]["leader"]), "residual_top1_lagger": V(top_res_json[0]["lagger"]),
    "residual_top1_asymmetry": V(top_res_json[0]["asymmetry"]), "residual_same_sign_count": V(pers_res["same_sign_count"]),
    "residual_sign_consistency_rate": V(pers_res["sign_consistency_rate"]),
    "residual_top1_rank_2022": V(sub[0]["rank_2022"]), "residual_top1_rank_2023": V(sub[0]["rank_2023"]),
    "residual_num_pairs_same_sign_both_years": V(sub_sum["num_pairs_same_sign_both_years"]),
}
json.dump({"intermediates": inter}, open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
