# memory-solution
# unit: t1-multimodal-alpha-fusion-edgar-cot-gdelt | category: backtest-factors | tags: multimodal, edgar, gdelt, cot, committee, cross-sectional-zscore, audit-features, aic-model-selection, ou, merton, gbm, long-short, params-json-family
# deliverables: committee_panel.csv, process_diagnostics.csv, alpha_panel.csv, portfolio_returns.csv, modality_diagnostics.csv, results.json, solution.json
# audit: bulletins in (asof-365d, asof] (none -> severity 0, count 0, days_since 365); vendor v1/v2 latest row per field in (asof-lookback, asof];
#   stale if no row or newest older than threshold; mismatch if |v1-v2| > 1e-12 (NaN never mismatches, non-numeric unequal -> 1.0)
# normalise per date AFTER the sign/log1p transform: inf->NaN, median fill (all-NaN -> 0), z ddof=0 (sd 0 -> 0), clip +-3; agent = mean of components
# process (trailing <= 12 prior forward returns, min 6): all sigmas ddof=0 floored at 1e-6; martingale uses mean 0 with std about the mean;
#   OU: phi = OLS slope clipped [1e-4, .999], intercept with clipped phi, + stationary first-obs term; Merton: 1.4-sigma jump split, lambda clipped (1e-6, 1-1e-6)
#   k = 1/2/3/5; AIC ties alphabetical; conviction = 2*sigmoid(gap/scale) - 1 (checker convention, NOT the plain sigmoid in the text)
# portfolio: rank by (final desc, ticker); |signal| >= min_trade; top-3 positive long, bottom-3 negative short, +-0.5 gross; turnover vs previous (start 0)
import os, json, math
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
rd = lambda f: pd.read_csv(os.path.join(DATA_DIR, f))
P = json.load(open(os.path.join(DATA_DIR, "params.json")))
FL, K = 1e-6, {"martingale": 1, "gbm": 2, "ou": 3, "merton_jump_diffusion": 5}
K.update(P.get("model_parameter_counts", {}))
KEY = ["asof_date", "ticker"]

# ---- 1. base panel (24 dates x 16 issuers)
uni = rd("issuer_universe.csv")
dates = pd.to_datetime(rd("rebalance_dates.csv")["rebalance_date"]).dt.strftime("%Y-%m-%d").tolist()
X = pd.DataFrame([(d, t) for d in dates for t in uni["ticker"]], columns=KEY).merge(uni[["ticker", "sector", "cot_bucket"]], on="ticker", how="left")
fund, act, news, fwd = rd("fundamental_features.csv"), rd("issuer_activity_features.csv"), rd("gdelt_news_features.csv"), rd("forward_returns.csv")
X = (X.merge(fund, on=KEY, how="left").merge(act, on=KEY, how="left").merge(news, on=KEY, how="left")
     .merge(rd("cot_macro_features.csv"), on=["asof_date", "cot_bucket"], how="left").merge(fwd, on=KEY, how="left", validate="one_to_one"))
assert len(X) == len(dates) * len(uni)

# ---- 2. trailing audit features
bul = rd("vendor_bulletins.csv"); bul["d"] = pd.to_datetime(bul["bulletin_date"])
refs = [rd("vendor_data_v1.csv"), rd("vendor_data_v2.csv")]
fields = sorted(set(refs[0]["field_name"]) | set(refs[1]["field_name"]))
by = []
for v in refs:
    v["d"] = pd.to_datetime(v["asof_date"])
    by.append({k: g.sort_values("d", kind="stable") for k, g in v.groupby(["ticker", "field_name"])})
aud = []
for d, tk in zip(X["asof_date"], X["ticker"]):
    a = pd.Timestamp(d)
    b = bul[(bul["ticker"] == tk) & (bul["d"] <= a) & (bul["d"] > a - pd.Timedelta(days=365))]
    mc, ms, st = 0, 0.0, 0
    for fld in fields:
        last = []
        for g in by:
            h = g.get((tk, fld))
            h = None if h is None else h[(h["d"] <= a) & (h["d"] > a - pd.Timedelta(days=P["reference_lookback_days"]))]
            last.append(h.iloc[-1] if h is not None and len(h) else None)
        newest = [r["d"] for r in last if r is not None]
        st += int(not newest or (a - max(newest)).days > P["reference_stale_threshold_days"])
        if last[0] is not None and last[1] is not None:
            try:
                diff = abs(float(last[0]["field_value"]) - float(last[1]["field_value"]))
            except (TypeError, ValueError):
                diff = 0.0 if last[0]["field_value"] == last[1]["field_value"] else 1.0
            if diff > 1e-12:
                mc, ms = mc + 1, ms + diff
    aud.append(dict(bulletin_severity_sum_365d=float(b["severity"].sum()), severe_bulletin_count_365d=int((b["severity"] >= P["severe_bulletin_cutoff"]).sum()),
                    days_since_last_bulletin=float((a - b["d"].max()).days) if len(b) else 365.0, reference_mismatch_count=mc,
                    reference_mismatch_abs_sum=ms, reference_stale_field_count=st, bulletin_rows=len(b)))
X = pd.concat([X, pd.DataFrame(aud, index=X.index)], axis=1)


# ---- 3-4. committee agents and fusion
def norm(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    s = s.fillna(s.median()); sd = float(s.std(ddof=0))
    return ((s - s.mean()) / sd).clip(-3.0, 3.0) if sd > 0 else pd.Series(0.0, index=s.index)


c = lambda col: pd.to_numeric(X[col], errors="coerce")
AGENTS = {
    "filing_agent_score": [c("revenue_growth_ttm"), c("gross_margin_ttm"), -c("debt_to_assets"), c("cash_to_assets"), -c("share_count_growth_yoy"),
                           -c("days_since_last_10q"), -c("days_since_last_10k"), -c("count_8k_trailing_90d")],
    "issuer_activity_agent_score": [c("insider_net_buy_value_30d"), c("insider_net_buy_value_90d"), c("insider_buy_count_90d"), -c("insider_sell_count_90d"),
                                    c("officer_net_buy_value_90d"), c("director_net_buy_value_90d")],
    "news_agent_score": [np.log1p(c("article_count_7d")), np.log1p(c("article_count_30d")), np.log1p(c("source_count_30d")), c("avg_tone_7d"), c("avg_tone_30d"),
                         c("article_count_7d") / np.maximum(c("article_count_30d"), 1.0), -np.log1p(c("theme_risk_count_30d")),
                         -np.log1p(c("theme_supply_count_30d")), -np.log1p(c("theme_macro_count_30d"))],
    "audit_agent_score": [-c("bulletin_severity_sum_365d"), -c("severe_bulletin_count_365d"), c("days_since_last_bulletin"), -c("reference_mismatch_count"),
                          -c("reference_mismatch_abs_sum"), -c("reference_stale_field_count")],
}
for name, comps in AGENTS.items():
    X[name] = pd.concat([s.groupby(X["asof_date"]).transform(norm) for s in comps], axis=1).mean(axis=1)
A = X[list(AGENTS)].to_numpy(float)
X["committee_score"] = sum(P[w] * X[a] for w, a in zip(["filing_weight", "issuer_activity_weight", "news_weight", "audit_weight"], AGENTS))
X["committee_dispersion"] = A.std(axis=1, ddof=0)
X["agreement_multiplier"] = np.maximum(0.0, 1.0 - P["disagreement_penalty"] * X["committee_dispersion"])
X["committee_alpha"] = X["committee_score"] * X["agreement_multiplier"]


# ---- 5. stochastic-process layer (AIC model selection on trailing realised returns)
def nll(x, m, s):
    s = max(float(s), FL)
    return float(np.sum(0.5 * np.log(2 * np.pi * s * s) + (x - m) ** 2 / (2 * s * s)))


def fit(r):
    mu, sd = float(r.mean()), max(float(r.std()), FL)
    out = dict(gbm_mu=mu, gbm_sigma=sd)
    aic = {"martingale": 2 * K["martingale"] + 2 * nll(r, 0.0, sd), "gbm": 2 * K["gbm"] + 2 * nll(r, mu, sd)}
    x, y = r[:-1], r[1:]
    vx = float(((x - x.mean()) ** 2).mean())
    phi = 0.0 if vx <= 1e-12 else float(((x - x.mean()) * (y - y.mean())).mean() / vx)
    phi = min(max(phi, 1e-4), 0.999)
    icpt = float(y.mean() - phi * x.mean())
    ou_mu, ou_sig = icpt / max(1 - phi, 1e-6), max(float(np.sqrt(((y - icpt - phi * x) ** 2).mean())), FL)
    theta = -math.log(phi)
    out.update(ou_theta=theta, ou_mu=ou_mu, ou_sigma=ou_sig, ou_half_life_days=math.log(2) / max(theta, 1e-6) * P["forward_return_days"])
    aic["ou"] = 2 * K["ou"] + 2 * (nll(y, icpt + phi * x, ou_sig) + nll(r[:1], ou_mu, ou_sig / max(math.sqrt(max(1 - phi * phi, 1e-12)), FL)))
    jm = np.abs(r - mu) > P["jump_threshold_sigma"] * sd
    jmean, jstd = (float(r[jm].mean() - mu), max(float(r[jm].std()), FL)) if jm.any() else (0.0, sd)
    mm, msd = (float(r[~jm].mean()), max(float(r[~jm].std()), FL)) if (~jm).any() else (mu, sd)
    lam = min(max(float(jm.mean()), 1e-6), 1 - 1e-6)
    pdf = lambda m, s: np.exp(-0.5 * np.log(2 * np.pi * s * s) - (r - m) ** 2 / (2 * s * s))
    mix = (1 - lam) * pdf(mm, msd) + lam * pdf(mm + jmean, math.sqrt(msd * msd + jstd * jstd))
    aic["merton_jump_diffusion"] = 2 * K["merton_jump_diffusion"] + 2 * float(-np.log(np.maximum(mix, 1e-15)).sum())
    out.update(merton_mu=mm, merton_sigma=msd, merton_jump_intensity=lam, merton_jump_mean=jmean, merton_jump_std=jstd)
    exp_ret = {"martingale": 0.0, "gbm": mu, "ou": icpt + phi * float(r[-1]), "merton_jump_diffusion": mm + lam * jmean}
    ranked = sorted((a, m) for m, a in aic.items() if np.isfinite(a))  # ties -> alphabetical model name
    best = ranked[0][1]
    conv = 1.0 if len(ranked) == 1 else min(1.0, max(0.0, 2.0 / (1.0 + math.exp(-max(ranked[1][0] - ranked[0][0], 0.0) / P["process_conviction_scale"])) - 1.0))
    er = exp_ret[best]
    out.update({f"{m.replace('_jump_diffusion', '')}_aic": a for m, a in aic.items()})
    out.update(selected_process=best, process_expected_return=er, process_sign=int(np.sign(er)), process_conviction=conv)
    return out


PCOLS = ["asof_date", "ticker", "history_count", "selected_process", "martingale_aic", "gbm_mu", "gbm_sigma", "gbm_aic", "ou_theta", "ou_mu", "ou_sigma",
         "ou_half_life_days", "ou_aic", "merton_mu", "merton_sigma", "merton_jump_intensity", "merton_jump_mean", "merton_jump_std", "merton_aic",
         "process_expected_return", "process_sign", "process_conviction"]
hist = {t: g.sort_values("asof_date") for t, g in fwd.groupby("ticker")}
proc = []
for d, tk in zip(X["asof_date"], X["ticker"]):
    h = hist[tk]
    r = h.loc[h["asof_date"] < d, "forward_return_21d"].tail(P["process_trailing_window"]).to_numpy(float)
    row = dict(asof_date=d, ticker=tk, history_count=len(r))
    row.update(fit(r) if len(r) >= P["process_min_history"] else dict(selected_process="insufficient_history", process_sign=0, process_conviction=0.0))
    proc.append(row)
proc = pd.DataFrame(proc).reindex(columns=PCOLS)
X = X.merge(proc, on=KEY, how="left")
X["final_signal"] = np.where(X["process_sign"] == 0, 0.0, X["committee_alpha"] * X["process_sign"] * X["process_conviction"])

# ---- 6-7. ranking and long/short selection
parts = []
for d, g in X.groupby("asof_date", sort=True):
    g = g.sort_values(["final_signal", "ticker"], ascending=[False, True]).copy()
    g["alpha_rank"] = np.arange(1, len(g) + 1)
    elig = g["final_signal"].abs() >= P["min_trade_signal"]
    lg, sh = g[elig & (g["final_signal"] > 0)].head(P["long_top_n"]).index, g[elig & (g["final_signal"] < 0)].tail(P["short_bottom_n"]).index
    g["selection"], g["portfolio_weight"] = "neutral", 0.0
    g.loc[lg, "selection"], g.loc[sh, "selection"] = "long", "short"
    g.loc[lg, "portfolio_weight"], g.loc[sh, "portfolio_weight"] = (0.5 / len(lg) if len(lg) else 0.0), (-0.5 / len(sh) if len(sh) else 0.0)
    parts.append(g)
X = pd.concat(parts)

# ---- 8. realised portfolio returns
prev, rows, eq = pd.Series(0.0, index=uni["ticker"]), [], float(P["initial_capital"])
for d, g in X.groupby("asof_date", sort=True):
    w = g.set_index("ticker")["portfolio_weight"].reindex(uni["ticker"]).fillna(0.0)
    to = 0.5 * float((w - prev).abs().sum()); prev = w
    gross = float((g["portfolio_weight"] * g["forward_return_21d"]).sum()); tc = to * P["transaction_cost_bps"] / 1e4
    eq *= 1.0 + gross - tc
    rows.append(dict(asof_date=d, long_count=int((g["selection"] == "long").sum()), short_count=int((g["selection"] == "short").sum()),
                     gross_return=gross, turnover=to, transaction_cost=tc, net_return=gross - tc, equity_curve=eq))
port = pd.DataFrame(rows)

# ---- outputs
X = X.sort_values(KEY).reset_index(drop=True)
X[["asof_date", "ticker", "sector", "cot_bucket"] + list(AGENTS) + ["committee_score", "committee_dispersion", "agreement_multiplier", "committee_alpha"]].to_csv(
    os.path.join(OUT_DIR, "committee_panel.csv"), index=False)
X[PCOLS].to_csv(os.path.join(OUT_DIR, "process_diagnostics.csv"), index=False)
X.sort_values(["asof_date", "alpha_rank", "ticker"])[["asof_date", "ticker", "sector", "cot_bucket", "committee_alpha", "process_sign", "process_conviction",
                                                      "final_signal", "alpha_rank", "selection", "portfolio_weight", "entry_date", "exit_date",
                                                      "forward_return_21d"]].to_csv(os.path.join(OUT_DIR, "alpha_panel.csv"), index=False)
port.to_csv(os.path.join(OUT_DIR, "portfolio_returns.csv"), index=False)
miss = lambda df: X[[col for col in df.columns if col not in KEY]].replace([np.inf, -np.inf], np.nan).isna().sum(axis=1)
X["_fm"], X["_am"], X["_nm"] = miss(fund), miss(act), miss(news)
diag = X.groupby("asof_date", sort=True).agg(num_issuers=("ticker", "size"), fundamental_missing_cells=("_fm", "sum"), issuer_activity_missing_cells=("_am", "sum"),
                                             news_missing_cells=("_nm", "sum"), audit_bulletin_rows_365d=("bulletin_rows", "sum"),
                                             audit_mismatch_issuers=("reference_mismatch_count", lambda s: int((s > 0).sum())),
                                             news_nonzero_issuers=("article_count_30d", lambda s: int((s > 0).sum())),
                                             process_history_ready_issuers=("history_count", lambda s: int((s >= P["process_min_history"]).sum()))).reset_index()
diag.to_csv(os.path.join(OUT_DIR, "modality_diagnostics.csv"), index=False)

net, n, af = port["net_return"], len(port), float(P["annualization_factor"])
cum = float(port["equity_curve"].iloc[-1] / P["initial_capital"] - 1.0)
ann_ret, ann_vol = (1.0 + cum) ** (af / n) - 1.0, float(net.std(ddof=0) * math.sqrt(af))  # population std; sharpe = mean*af/ann_vol
eqc = np.concatenate([[P["initial_capital"]], port["equity_curve"].to_numpy()])
results = {"num_issuers": int(len(uni)), "num_rebalance_dates": len(dates), "num_backtest_periods": int(n),
           "num_process_ready_dates": int((X.groupby("asof_date")["selected_process"].apply(lambda s: (s != "insufficient_history").any())).sum()),
           "num_news_active_dates": int((diag["news_nonzero_issuers"] > 0).sum()), "cumulative_return": cum, "annualized_return": float(ann_ret),
           "annualized_volatility": ann_vol, "sharpe_ratio": float(net.mean() * af / ann_vol) if ann_vol > 0 else 0.0,
           "max_drawdown": float((eqc / np.maximum.accumulate(eqc) - 1.0).min()), "avg_turnover": float(port["turnover"].mean()),
           "best_rebalance_date": port.loc[net.idxmax(), "asof_date"], "worst_rebalance_date": port.loc[net.idxmin(), "asof_date"]}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
pick = lambda d, side: X[(X["asof_date"] == d) & (X["selection"] == side)].sort_values("alpha_rank")["ticker"].tolist()
solution = {"first_rebalance_date": dates[0], "first_rebalance_longs": pick(dates[0], "long"), "first_rebalance_shorts": pick(dates[0], "short"),
            "last_rebalance_date": dates[-1], "last_rebalance_longs": pick(dates[-1], "long"), "last_rebalance_shorts": pick(dates[-1], "short"),
            "selected_process_counts": {k: int(v) for k, v in sorted(X["selected_process"].value_counts().items())}}
json.dump(solution, open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
