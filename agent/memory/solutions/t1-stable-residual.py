# memory-solution
# unit: t1-stable-residual | category: backtest-factors | tags: residualized-signals, winsorize, cross-sectional-ols, pinv, zscore-ddof0, rolling-median-coefficients, sign-stability-filter, long-short, beta-neutral-projection, turnover-cap, transaction-costs, parquet
# deliverables: summary.json, monthly_results.csv, coefficients_last_rebalance.csv, final_month_weights.csv
# Checker recomputes everything from the panel, so mirror it: drop_duplicates(identifier, date, keep=last) on FILE order, then mergesort by (date, id).
# Month t: universe = dropna(14 signals + 15 controls); winsorize each column at pandas quantile(.01/.99); residual of each signal on [1, controls]
#   via pinv(X'X) X'y; z-score ddof=0 (sd==0 -> zeros). beta_60m used for projection/exposure = RAW (un-winsorized) universe value (reference does this).
# Coef month: universe >= 80 and >= 80 non-NaN ret_lead1m -> OLS slopes. Rebalance idx >= 96 with >= 36 prior coef months: np.median of last 36,
#   keep if #(sign == sign(median)) >= 24 else 0; alpha = Z @ stable; order = lexsort((ids, -alpha)); long top floor(.15n), short order[-n_short:].
# w = w_raw - X pinv(X'X) X'w_raw (X = [1, beta]); /gross. Turnover cap 0.30: blend, re-project on support (sorted ids, beta fallback raw->0), /gross,
#   if still > 0.300000001 shrink toward w_prev to exactly 0.30 (no renorm). cost = 0.001 * sum|dw|, turnover = 0.5 sum|dw|; missing ret -> 0.
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
SIG = ["ret_12_7", "ret_6_1", "resff3_12_1", "resff3_6_1", "seas_1_1an", "seas_1_1na", "seas_2_5an", "seas_2_5na",
       "seas_6_10an", "seas_6_10na", "seas_11_15an", "seas_11_15na", "seas_16_20an", "seas_16_20na"]
CTL = ["rvol_21d", "ivol_capm_21d", "ivol_capm_252d", "ivol_ff3_21d", "ivol_hxz4_21d", "beta_60m", "beta_dimson_21d", "betabab_1260d",
       "betadown_252d", "rskew_21d", "coskew_21d", "iskew_capm_21d", "iskew_ff3_21d", "iskew_hxz4_21d", "corr_1260d"]
CAP, TOL, COST, START, LB, AGREE, MINN = 0.30, 0.300000001, 0.0010, 96, 36, 24, 80

raw = pd.read_parquet(os.path.join(DATA_DIR, "stock_chars.pqt"))
df = raw.copy(); df["date"] = pd.to_datetime(df["date"])
df = df.drop_duplicates(["identifier", "date"], keep="last").sort_values(["date", "identifier"], kind="mergesort").reset_index(drop=True)
df["identifier"] = df["identifier"].astype(int)
dates = sorted(pd.to_datetime(df["date"].unique()))

ols = lambda X, y: np.linalg.pinv(X.T @ X) @ (X.T @ y)
def proj(w, b):
    X = np.column_stack([np.ones(len(w)), b]); return w - X @ (np.linalg.pinv(X.T @ X) @ (X.T @ w))
def zs(v):
    sd = float(v.std(ddof=0)); return np.zeros_like(v) if sd == 0.0 or not np.isfinite(sd) else (v - float(v.mean())) / sd
turn = lambda a, b: 0.5 * sum(abs(a.get(i, 0.0) - b.get(i, 0.0)) for i in set(a) | set(b))

panels, coefs, rawm = {}, {}, {}
for dt, g in df.groupby("date", sort=True):
    rawm[dt] = (dict(zip(g.identifier.tolist(), g.beta_60m.astype(float))), dict(zip(g.identifier.tolist(), g.ret_lead1m.astype(float))))
    u = g.dropna(subset=SIG + CTL)
    if len(u) == 0: continue
    wz = u[SIG + CTL].astype(float)
    wz = wz.clip(wz.quantile(0.01), wz.quantile(0.99), axis=1)
    Xc = np.column_stack([np.ones(len(u))] + [wz[c].values for c in CTL])
    Z = np.column_stack([zs(wz[s].values - Xc @ ols(Xc, wz[s].values)) for s in SIG])
    p = {"ids": u.identifier.to_numpy(int), "beta": u.beta_60m.to_numpy(float), "ret": u.ret_lead1m.to_numpy(float), "Z": Z, "n": len(u)}
    panels[dt] = p
    m = ~np.isnan(p["ret"])
    if p["n"] >= MINN and m.sum() >= MINN:
        coefs[dt] = ols(np.column_stack([np.ones(m.sum()), Z[m]]), p["ret"][m])[1:]

cdates, w_prev, recs, last = sorted(coefs), {}, [], None
for idx, dt in enumerate(dates):
    if idx < START: continue
    prior = [d for d in cdates if d < dt]
    if len(prior) < LB: continue
    H = np.array([coefs[d] for d in prior[-LB:]])
    med = np.median(H, axis=0); agr = (np.sign(H) == np.sign(med)[None, :]).sum(axis=0); stab = np.where(agr >= AGREE, med, 0.0)
    if dt not in panels or panels[dt]["n"] < MINN: continue
    p = panels[dt]; n, ids = p["n"], p["ids"]; alpha = p["Z"] @ stab
    nl = ns = int(math.floor(0.15 * n))
    if nl < 10 or ns < 10: continue
    order = np.lexsort((ids, -alpha)); sel = np.concatenate([order[:nl], order[-ns:]])
    wp = proj(np.concatenate([np.full(nl, 1.0 / nl), np.full(ns, -1.0 / ns)]), p["beta"][sel]); gross = float(np.abs(wp).sum())
    if gross == 0.0: continue
    w_tgt = dict(zip(ids[sel].tolist(), (wp / gross).tolist()))
    rbeta, rret = rawm[dt]; fbeta = dict(zip(ids.tolist(), p["beta"].tolist()))
    bt = lambda i: fbeta[i] if i in fbeta else (float(rbeta[i]) if np.isfinite(rbeta.get(i, np.nan)) else 0.0)
    trn = turn(w_prev, w_tgt)
    if trn <= CAP: w_fin = dict(w_tgt)
    else:
        lam = CAP / trn
        w_mid = {i: w_prev.get(i, 0.0) + lam * (w_tgt.get(i, 0.0) - w_prev.get(i, 0.0)) for i in set(w_prev) | set(w_tgt)}
        sup = sorted(i for i, v in w_mid.items() if v != 0.0)
        if not sup: continue
        wr = proj(np.array([w_mid[i] for i in sup]), np.array([bt(i) for i in sup])); gm = float(np.abs(wr).sum())
        if gm == 0.0: continue
        w_re = dict(zip(sup, (wr / gm).tolist())); nt = turn(w_prev, w_re)
        if nt > TOL:
            mu = CAP / nt
            w_fin = {i: w_prev.get(i, 0.0) + mu * (w_re.get(i, 0.0) - w_prev.get(i, 0.0)) for i in set(w_prev) | set(w_re)}
        else: w_fin = w_re
    gr = sum(w * rret[i] for i, w in w_fin.items() if w != 0.0 and np.isfinite(rret.get(i, np.nan)))
    dw = sum(abs(w_fin.get(i, 0.0) - w_prev.get(i, 0.0)) for i in set(w_fin) | set(w_prev))
    recs.append({"date": dt.strftime("%Y-%m-%d"), "n_eligible": n, "n_long": nl, "n_short": ns, "n_coeff_months_used": LB,
                 "gross_return": float(gr), "transaction_cost": COST * dw, "net_return": float(gr - COST * dw), "turnover": 0.5 * dw,
                 "beta_exposure": float(sum(w * bt(i) for i, w in w_fin.items())), "dollar_exposure": float(sum(w_fin.values())),
                 "gross_leverage": float(sum(abs(w) for w in w_fin.values()))})
    last = dict(date=recs[-1]["date"], med=med, agr=agr, stab=stab, alpha=dict(zip(ids.tolist(), alpha.tolist())), prev=dict(w_prev),
                tgt=w_tgt, fin=w_fin, bt=bt, ret=rret)
    w_prev = dict(w_fin)

M = pd.DataFrame(recs); M.to_csv(os.path.join(OUT_DIR, "monthly_results.csv"), index=False)
pd.DataFrame({"signal": SIG, "median_coef": last["med"], "sign_agreement_count": last["agr"].astype(int),
              "stable_coef": last["stab"]}).to_csv(os.path.join(OUT_DIR, "coefficients_last_rebalance.csv"), index=False)
keep = sorted({i for d in (last["prev"], last["tgt"], last["fin"]) for i, v in d.items() if v != 0.0})
W = pd.DataFrame({"date": last["date"], "identifier": keep, "alpha": [last["alpha"].get(i, 0.0) for i in keep],
                  "prev_weight": [last["prev"].get(i, 0.0) for i in keep], "target_weight": [last["tgt"].get(i, 0.0) for i in keep],
                  "final_weight": [last["fin"].get(i, 0.0) for i in keep], "beta_60m": [last["bt"](i) for i in keep],
                  "ret_lead1m": [last["ret"].get(i, np.nan) for i in keep]})
W = W.sort_values(["final_weight", "identifier"], ascending=[False, True], kind="mergesort")
W.assign(identifier=W.identifier.astype(str)).to_csv(os.path.join(OUT_DIR, "final_month_weights.csv"), index=False)

net, grs = M.net_return.to_numpy(), M.gross_return.to_numpy()
vol = float(net.std(ddof=0)); ann_r, ann_v = 12 * float(net.mean()), math.sqrt(12) * vol
wealth = np.concatenate([[1.0], np.cumprod(1.0 + net)]); peaks = np.maximum.accumulate(wealth)
summary = {"task_id": "stable-residual", "n_input_rows": int(len(raw)), "n_unique_dates": len(dates),
           "first_rebalance_date": recs[0]["date"], "last_rebalance_date": recs[-1]["date"], "n_tradable_months": len(recs),
           "n_skipped_months": int(max(0, len(dates) - START) - len(recs)), "mean_monthly_gross_return": float(grs.mean()),
           "mean_monthly_net_return": float(net.mean()), "monthly_volatility_net": vol, "annualized_return_net": ann_r,
           "annualized_volatility_net": ann_v, "annualized_sharpe_net": ann_r / ann_v if ann_v != 0 else 0.0,
           "cumulative_net_return": float(np.prod(1.0 + net) - 1.0), "max_drawdown_net": float(((peaks - wealth) / peaks).max()),
           "avg_turnover": float(M.turnover.mean()), "max_turnover": float(M.turnover.max()),
           "avg_abs_beta_exposure": float(M.beta_exposure.abs().mean()), "max_abs_beta_exposure": float(M.beta_exposure.abs().max()),
           "avg_names_long": float(M.n_long.mean()), "avg_names_short": float(M.n_short.mean())}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
