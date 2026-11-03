# memory-solution
# unit: t1-bollinger-backtest-aapl | category: backtest-factors | tags: bollinger-bands, mean-reversion, trailing-stop, risk-sizing, transaction-costs, next-open-execution, plotly, trade-analysis
# deliverables: results.json, trades.csv, daily_portfolio.csv, cumulative_returns.html, trade_analysis.json
# bands on adj_close: mid = SMA(20), band = mid +/- 2*rolling std(20, ddof=1); BUY signal close < lower, SELL signal close > upper
# trailing stop: peak starts at the signal-day close, updated with each close in position; SELL when close < 0.95*peak (bollinger exit wins)
# signal day t -> fill at t+1 adj_open; costs: buy cost = sh*px*1.001 + 9.99, sell net = sh*px*0.999 - 9.99; pnl = net - cost
# sizing: risk = 0.02*cash, stop_dist = fill_open - lower_band(t); sh = min(floor(risk/stop_dist), floor((cash-9.99)/(px*1.001)))
#   KEY: if stop_dist <= 0 (open gaps below the band) -> 0 shares -> the signal is consumed with NO trade (gives the 6-trade oracle)
# metrics as in other backtests (mean*252, std ddof=1*sqrt252, mdd on cum curve); intra-trade dd over adj_close from entry to exit fill (inclusive)
import os, json, math
import numpy as np, pandas as pd
import plotly.graph_objects as go

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
FEE, COMM, RISK, STOP, CAP0 = 9.99, 0.001, 0.02, 0.05, 100000.0

df = pd.read_csv(os.path.join(DATA_DIR, "aapl_prices.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
mid = df["adj_close"].rolling(20).mean(); sd = df["adj_close"].rolling(20).std(ddof=1)
upper, lower = (mid + 2 * sd).to_numpy(), (mid - 2 * sd).to_numpy()
aopen, aclose = df["adj_open"].to_numpy(float), df["adj_close"].to_numpy(float)
dates = df["date"].dt.strftime("%Y-%m-%d").tolist(); n = len(df)

cash, shares, in_pos, pending, peak = CAP0, 0, False, None, None
buy_cost, entry_i, entry_px = 0.0, None, None
trades, analysis, values = [], [], []
def close_trade(px, i, sdate, reason):
    global cash, shares, in_pos
    net = shares * px * (1 - COMM) - FEE; cash += net; pnl = round(net - buy_cost, 2)
    trades.append(dict(ticker="AAPL", type="SELL", signal_date=sdate, exec_date=dates[i], price=round(px, 2), shares=shares, pnl=pnl))
    seg = aclose[entry_i:i + 1]; run = np.maximum.accumulate(seg)
    analysis.append(dict(trade_num=len(analysis) + 1, entry_date=dates[entry_i], exit_date=dates[i], entry_price=round(entry_px, 2),
                         exit_price=round(px, 2), pnl=pnl, exit_reason=reason, intra_trade_max_drawdown=round(float(((run - seg) / run).max()), 6)))
    shares, in_pos = 0, False

for i in range(n):
    if pending is not None:                                       # fill yesterday's signal at today's adj_open
        typ, sdate, sig_i, reason = pending; px = aopen[i]
        if typ == "BUY":
            by_cash = math.floor((cash - FEE) / (px * (1 + COMM)))
            stop_dist = px - lower[sig_i]
            sh = min(math.floor(RISK * cash / stop_dist), by_cash) if stop_dist > 0 else 0
            if sh > 0:
                shares = sh; buy_cost = sh * px * (1 + COMM) + FEE; cash -= buy_cost; in_pos = True
                entry_i, entry_px, peak = i, px, aclose[sig_i]
                trades.append(dict(ticker="AAPL", type="BUY", signal_date=sdate, exec_date=dates[i], price=round(px, 2), shares=sh, pnl=""))
        else:
            close_trade(px, i, sdate, reason)
        pending = None
    if in_pos:
        peak = max(peak, aclose[i])
    if i < n - 1 and not np.isnan(lower[i]):
        if not in_pos and aclose[i] < lower[i]:
            pending = ("BUY", dates[i], i, None)
        elif in_pos and aclose[i] > upper[i]:
            pending = ("SELL", dates[i], i, "bollinger")
        elif in_pos and aclose[i] < peak * (1 - STOP):
            pending = ("SELL", dates[i], i, "stop_loss")
    if i == n - 1 and in_pos:
        close_trade(aclose[i], i, dates[i], "forced_close")
    values.append(cash + shares * aclose[i])

vals = pd.Series(values); rets = vals.pct_change().fillna(0.0); cum = (1.0 + rets).cumprod()
pk = cum.cummax(); mdd = float(((pk - cum) / pk).max())
ann_ret = float(rets.mean() * 252); ann_vol = float(rets.std(ddof=1) * math.sqrt(252)); final_cap = float(vals.iloc[-1])
sells = [t for t in trades if t["type"] == "SELL"]; buys = [t for t in trades if t["type"] == "BUY"]
res = {"AAPL": {"total_return": (final_cap - CAP0) / CAP0, "final_capital": round(final_cap, 2), "annualized_return": ann_ret,
                "annualized_volatility": ann_vol, "sharpe_ratio": ann_ret / ann_vol if ann_vol > 0 else 0.0, "max_drawdown": mdd,
                "calmar_ratio": ann_ret / mdd if mdd > 0 else 0.0, "num_trades": len(buys),
                "win_rate": sum(t["pnl"] > 0 for t in sells) / len(sells) if sells else 0.0,
                "total_pnl": round(sum(t["pnl"] for t in sells), 2), "num_trading_days": int(n)}}
json.dump(res, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
json.dump({"trades": analysis}, open(os.path.join(OUT_DIR, "trade_analysis.json"), "w"), indent=2)
pd.DataFrame(trades, columns=["ticker", "type", "signal_date", "exec_date", "price", "shares", "pnl"]).to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)
pd.DataFrame({"date": dates, "cumulative_return": cum.values}).to_csv(os.path.join(OUT_DIR, "daily_portfolio.csv"), index=False)
fig = go.Figure(go.Scatter(x=df["date"], y=cum.values, mode="lines", name="AAPL"))
fig.add_hline(y=1.0, line_dash="dash")
fig.update_layout(title="Bollinger Band Mean Reversion: AAPL Cumulative Returns", xaxis_title="Date", yaxis_title="Cumulative Return")
fig.write_html(os.path.join(OUT_DIR, "cumulative_returns.html"), include_plotlyjs="cdn")
