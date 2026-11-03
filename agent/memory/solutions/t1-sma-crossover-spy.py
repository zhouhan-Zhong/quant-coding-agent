# memory-solution
# unit: t1-sma-crossover-spy | category: backtest-factors | tags: sma-crossover, golden-cross, long-only, next-open-execution, whole-shares, sharpe, max-drawdown, plotly
# deliverables: results.json, trades.csv, daily_portfolio.csv, cumulative_returns.html
# signals from adj_close SMA(50)/SMA(200) rolling means (NaN warm-up -> no signal); cross detected on day i
#   (prev <= / >= and now > / <), executed at day i+1 adj_open; shares = floor(cash / px); one position at a time
# daily value = cash + shares*adj_close; ret = pct_change (first = 0); cum = cumprod(1+ret); mdd = max((peak-cum)/peak)
# ann_ret = mean*252, ann_vol = std(ddof=1)*sqrt(252), sharpe = ann_ret/ann_vol; num_trades = #BUY rows; win_rate over SELL pnl>0
# forced close on last day at adj_close (also a SELL row); pnl empty string on BUY rows; monetary values rounded to 2 dp
import os, json, math
import numpy as np, pandas as pd
import plotly.graph_objects as go

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "spy_prices.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df["fast"] = df["adj_close"].rolling(50).mean()
df["slow"] = df["adj_close"].rolling(200).mean()
fast, slow = df["fast"].to_numpy(), df["slow"].to_numpy()
aopen, aclose, dates = df["adj_open"].to_numpy(float), df["adj_close"].to_numpy(float), df["date"].dt.strftime("%Y-%m-%d").tolist()
n = len(df)

cash, shares, in_pos, entry_cost, pending = 100000.0, 0, False, 0.0, None
trades, values = [], []
for i in range(n):
    if pending is not None:                                   # execute yesterday's signal at today's adj_open
        typ, sdate = pending; px = aopen[i]
        if typ == "BUY":
            shares = math.floor(cash / px); cash -= shares * px; entry_cost = shares * px; in_pos = True
            trades.append(dict(ticker="SPY", type="BUY", signal_date=sdate, exec_date=dates[i], price=round(px, 2), shares=shares, pnl=""))
        else:
            proceeds = shares * px; cash += proceeds
            trades.append(dict(ticker="SPY", type="SELL", signal_date=sdate, exec_date=dates[i], price=round(px, 2), shares=shares, pnl=round(proceeds - entry_cost, 2)))
            shares, in_pos = 0, False
        pending = None
    if i >= 1 and i < n - 1 and not (np.isnan(fast[i - 1]) or np.isnan(slow[i - 1]) or np.isnan(fast[i]) or np.isnan(slow[i])):
        if not in_pos and fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
            pending = ("BUY", dates[i])                       # golden cross
        elif in_pos and fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
            pending = ("SELL", dates[i])                      # death cross
    if i == n - 1 and in_pos:                                 # forced close at last adj_close
        px = aclose[i]; proceeds = shares * px; cash += proceeds
        trades.append(dict(ticker="SPY", type="SELL", signal_date=dates[i], exec_date=dates[i], price=round(px, 2), shares=shares, pnl=round(proceeds - entry_cost, 2)))
        shares, in_pos = 0, False
    values.append(cash + shares * aclose[i])

vals = pd.Series(values)
rets = vals.pct_change().fillna(0.0)
cum = (1.0 + rets).cumprod()
peak = cum.cummax(); mdd = float(((peak - cum) / peak).max())
ann_ret = float(rets.mean() * 252); ann_vol = float(rets.std(ddof=1) * math.sqrt(252))
final_cap = float(vals.iloc[-1])
sells = [t for t in trades if t["type"] == "SELL"]; buys = [t for t in trades if t["type"] == "BUY"]
wins = sum(1 for t in sells if t["pnl"] > 0)
res = {"SPY": {"total_return": (final_cap - 100000.0) / 100000.0, "final_capital": round(final_cap, 2),
               "annualized_return": ann_ret, "annualized_volatility": ann_vol,
               "sharpe_ratio": ann_ret / ann_vol if ann_vol > 0 else 0.0, "max_drawdown": mdd,
               "calmar_ratio": ann_ret / mdd if mdd > 0 else 0.0, "num_trades": len(buys),
               "win_rate": wins / len(sells) if sells else 0.0, "total_pnl": round(sum(t["pnl"] for t in sells), 2),
               "num_trading_days": int(n)}}
json.dump(res, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
pd.DataFrame(trades, columns=["ticker", "type", "signal_date", "exec_date", "price", "shares", "pnl"]).to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)
pd.DataFrame({"date": dates, "cumulative_return": cum.values}).to_csv(os.path.join(OUT_DIR, "daily_portfolio.csv"), index=False)

fig = go.Figure(go.Scatter(x=df["date"], y=cum.values, mode="lines", name="SPY"))
fig.add_hline(y=1.0, line_dash="dash")
fig.update_layout(title="SMA Crossover Momentum: SPY Cumulative Returns", xaxis_title="Date", yaxis_title="Cumulative Return")
fig.write_html(os.path.join(OUT_DIR, "cumulative_returns.html"), include_plotlyjs="cdn")
