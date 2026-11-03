# DELIVERABLES: results.json, trades.csv, daily_portfolio.csv, cumulative_returns.html
import json, math, os
import numpy as np, pandas as pd
import plotly.graph_objects as go
df = pd.read_csv("/app/spy_prices.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df["ema50"] = df["adj_close"].ewm(span=50, adjust=False).mean()
df["ema200"] = df["adj_close"].ewm(span=200, adjust=False).mean()
cash = 100000.0; shares = 0; in_pos = False; trades = []; values = []
pending = None  # ("BUY"/"SELL", signal_date)
entry_cost = 0.0
n = len(df)
for i in range(n):
    row = df.iloc[i]
    if pending is not None:
        typ, sdate = pending
        px = float(row["adj_open"])
        if typ == "BUY":
            shares = math.floor(cash / px); cash -= shares * px; in_pos = True; entry_cost = shares * px
            trades.append({"ticker": "SPY", "type": "BUY", "signal_date": sdate.strftime("%Y-%m-%d"), "exec_date": row["date"].strftime("%Y-%m-%d"), "price": round(px, 2), "shares": shares, "pnl": ""})
        else:
            proceeds = shares * px; pnl = proceeds - entry_cost; cash += proceeds
            trades.append({"ticker": "SPY", "type": "SELL", "signal_date": sdate.strftime("%Y-%m-%d"), "exec_date": row["date"].strftime("%Y-%m-%d"), "price": round(px, 2), "shares": shares, "pnl": round(pnl, 2)})
            shares = 0; in_pos = False
        pending = None
    if i >= 1:
        f0, s0, f1, s1 = df["ema50"].iloc[i-1], df["ema200"].iloc[i-1], row["ema50"], row["ema200"]
        if (not in_pos) and pending is None and f0 <= s0 and f1 > s1 and i < n - 1:
            pending = ("BUY", row["date"])
        elif in_pos and pending is None and f0 >= s0 and f1 < s1 and i < n - 1:
            pending = ("SELL", row["date"])
    if i == n - 1 and in_pos:
        px = float(row["adj_close"]); proceeds = shares * px; pnl = proceeds - entry_cost; cash += proceeds
        trades.append({"ticker": "SPY", "type": "SELL", "signal_date": row["date"].strftime("%Y-%m-%d"), "exec_date": row["date"].strftime("%Y-%m-%d"), "price": round(px, 2), "shares": shares, "pnl": round(pnl, 2)})
        shares = 0; in_pos = False
    values.append(cash + shares * float(row["adj_close"]))
vals = pd.Series(values, index=df["date"])
rets = vals.pct_change().fillna(0.0)
cum = (1 + rets).cumprod()
run_max = cum.cummax(); mdd = float(((run_max - cum) / run_max).max())
ann_ret = float(rets.mean() * 252); ann_vol = float(rets.std(ddof=1) * math.sqrt(252))
final_cap = float(vals.iloc[-1]); total_ret = (final_cap - 100000) / 100000
sells = [t for t in trades if t["type"] == "SELL"]; buys = [t for t in trades if t["type"] == "BUY"]
wins = sum(1 for t in sells if t["pnl"] > 0)
res = {"SPY": {"total_return": total_ret, "final_capital": round(final_cap, 2), "annualized_return": ann_ret,
               "annualized_volatility": ann_vol, "sharpe_ratio": ann_ret / ann_vol if ann_vol else 0.0,
               "max_drawdown": mdd, "calmar_ratio": ann_ret / mdd if mdd else 0.0, "num_trades": len(buys),
               "win_rate": wins / len(sells) if sells else 0.0, "total_pnl": round(sum(t["pnl"] for t in sells), 2),
               "num_trading_days": int(n)}}
os.makedirs("/app/output", exist_ok=True)
json.dump(res, open("/app/output/results.json", "w"), indent=2)
pd.DataFrame(trades, columns=["ticker", "type", "signal_date", "exec_date", "price", "shares", "pnl"]).to_csv("/app/output/trades.csv", index=False)
pd.DataFrame({"date": df["date"].dt.strftime("%Y-%m-%d"), "cumulative_return": cum.values}).to_csv("/app/output/daily_portfolio.csv", index=False)
fig = go.Figure(); fig.add_trace(go.Scatter(x=df["date"], y=cum.values, mode="lines", name="SPY"))
fig.add_hline(y=1.0, line_dash="dash"); fig.update_layout(title="EMA Crossover Momentum: SPY Cumulative Returns", xaxis_title="Date", yaxis_title="Cumulative Return")
fig.write_html("/app/output/cumulative_returns.html", include_plotlyjs="cdn")
