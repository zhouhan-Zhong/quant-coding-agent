# memory-solution
# unit: t1-sentiment-factor-alpha | category: backtest-factors | tags: sentiment-scoring, bag-of-words, engagement-weighting, momentum-signal, long-short, vol-targeting, ex-ante-covariance, transaction-costs, spearman-ic, ols-alpha, params-json-family
# deliverables: results.json, solution.json
# valid post = likes >= 0; tokens = re.sub(r"[^a-z\s]", "", text.lower()).split() (delete non-letters, split on space);
# sentiment = #bullish - #bearish tokens; final = sentiment * (1 + ln(1 + likes + 2 rt + 3 replies))
# daily aggregate = SUM of final scores per (calendar date, ticker), reindexed to trading days (missing -> 0, weekend posts dropped);
# signal[t] = agg[t] - mean(agg[t-5..t-1]) (trading days), NaN where the ticker has < 3 valid posts that day
# weights at t: +-1/(nL+nS) on top/bottom signals; for t >= 5+20: scale = min(target / sqrt(w' Cov(R[t-19..t]) w * 252), cap)
# P&L: x[t+1] = w[t] . R[t+1] - sum|w[t]-w[t-1]| * bps/1e4 (cost of the trade made at t) -> 498 days; IC = spearman(sig[t], R[t+1])
# mean_weighted_score / num_scored_nonzero use valid posts dated on TRADING days only (reference); alpha_tstat (reference quirk) =
# alpha_daily / SE(beta) -- the textbook alpha/SE(alpha) = -1.55 fails the atol 0.1 check
import os, json, math, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

P = json.load(open(os.path.join(DATA_DIR, "params.json")))
words = json.load(open(os.path.join(DATA_DIR, "sentiment_words.json")))
BULL, BEAR = set(words["bullish"]), set(words["bearish"])
posts = pd.read_json(os.path.join(DATA_DIR, "posts.jsonl"), lines=True)
v = posts[posts["likes"] >= 0].copy()                      # the only invalid records: likes sentinel (-999)


def sentiment(text):
    toks = re.sub(r"[^a-z\s]", "", str(text).lower()).split()
    return sum(t in BULL for t in toks) - sum(t in BEAR for t in toks)


v["sent"] = v["text"].map(sentiment)
eng = 1 + np.log1p(v["likes"] + 2 * v["retweets"] + 3 * v["replies"]) if P.get("engagement_weight", True) else 1.0
v["fs"] = v["sent"] * eng
v["date"] = pd.to_datetime(v["timestamp"]).dt.normalize()

pr = pd.read_csv(os.path.join(DATA_DIR, "prices.csv"), parse_dates=["date"])
px = pr.pivot_table(index="date", columns="ticker", values="close").sort_index()
tick = sorted(px.columns); px = px[tick]; dates = px.index
R = px.pct_change().to_numpy(); T, n = R.shape
mkt = pd.read_csv(os.path.join(DATA_DIR, "market_factor.csv"), parse_dates=["date"]).set_index("date")["mkt_rf"].reindex(dates).to_numpy()

on_td = v["date"].isin(dates)
nz = v.loc[on_td, "fs"]; nz = nz[nz != 0]
g = v.groupby(["date", "ticker"])
A = g["fs"].sum().unstack().reindex(index=dates, columns=tick).fillna(0.0)
N = g.size().unstack().reindex(index=dates, columns=tick).fillna(0)
L, VL = int(P["momentum_lookback_days"]), int(P["vol_lookback_days"])
sig = (A - A.shift(1).rolling(L).mean()).where(N >= P["volume_filter_min_posts"]).to_numpy()
has_sig = ~np.isnan(sig).all(axis=1)

ics = []
for t in range(T - 1):                                         # IC: signal[t] vs next-day return
    m = ~np.isnan(sig[t]) & ~np.isnan(R[t + 1])
    if m.sum() >= 3:
        c = spearmanr(sig[t][m], R[t + 1][m])[0]
        if np.isfinite(c):
            ics.append(c)
ics = np.array(ics)

nL, nS = int(P["long_top_n"]), int(P["short_bottom_n"])
W = np.zeros((T, n))
for t in range(T):
    m = ~np.isnan(sig[t])
    if m.sum() < nL + nS:
        continue
    idx = np.where(m)[0]; o = idx[np.argsort(sig[t][idx], kind="stable")]
    w = np.zeros(n); w[o[-nL:]] = 1.0 / (nL + nS); w[o[:nS]] = -1.0 / (nL + nS)
    if P.get("vol_target_annual") and t >= L + VL:             # ex-ante vol target with full trailing covariance
        C = np.cov(R[t - VL + 1:t + 1], rowvar=False, ddof=1)
        vol = math.sqrt(max(w @ C @ w, 0.0) * 252)
        if vol > 0:
            w = w * min(P["vol_target_annual"] / vol, P["max_leverage_scale"])
    W[t] = w
turn = np.abs(np.diff(np.vstack([np.zeros(n), W]), axis=0)).sum(axis=1)
tc = P["transaction_cost_bps"] / 1e4
first = int(np.argmax(has_sig))
days = np.arange(first, T - 1)
x = np.array([W[t] @ np.nan_to_num(R[t + 1]) - turn[t] * tc for t in days])   # one-day execution lag
mk = mkt[days + 1]

total = float(np.prod(1 + x) - 1); ann = float(x.mean() * 252); vol = float(x.std(ddof=1) * math.sqrt(252))
wealth = np.cumprod(1 + x); mdd = float((1 - wealth / np.maximum.accumulate(wealth)).max())
X = np.column_stack([np.ones(len(x)), mk]); b = np.linalg.lstsq(X, x, rcond=None)[0]
res = x - X @ b; s2 = res @ res / (len(x) - 2); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
r2 = 1 - (res @ res) / ((x - x.mean()) @ (x - x.mean()))
mean_ic, std_ic = float(ics.mean()), float(ics.std(ddof=1))

results = {"num_valid_posts": int(len(v)), "mean_weighted_score": float(nz.mean()), "total_return": total,
           "annualized_return": ann, "sharpe_ratio": ann / vol, "max_drawdown": mdd, "mean_ic": mean_ic,
           "ic_ir": mean_ic / std_ic, "alpha_annualized": float(b[0] * 252), "beta": float(b[1])}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
inter = {"num_posts": int(len(posts)), "num_valid_posts": int(len(v)), "num_tickers": n, "num_scored_nonzero": int(len(nz)),
         "mean_weighted_score": results["mean_weighted_score"], "num_signal_days": int(has_sig.sum()),
         "num_ic_observations": int(len(ics)), "mean_ic": mean_ic, "std_ic": std_ic, "ic_ir": mean_ic / std_ic,
         "annualized_volatility": vol, "win_rate": float((x > 0).mean()), "alpha_daily": float(b[0]), "beta": float(b[1]),
         "r_squared": float(r2), "alpha_tstat": float(b[0] / se[1]), "total_return": total, "annualized_return": ann,
         "sharpe_ratio": ann / vol, "max_drawdown": mdd, "alpha_annualized": float(b[0] * 252)}
json.dump({"intermediates": {k: {"value": val} for k, val in inter.items()}, "deliverables": results},
          open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
