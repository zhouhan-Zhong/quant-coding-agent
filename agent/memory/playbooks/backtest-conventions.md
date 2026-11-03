---
title: Backtest conventions: execution timing, sizing, costs, rebalancing, trade logs and metrics
category: backtest-factors
tags: backtest, backtesting, trading strategy, signal, execution, next open, next close, look-ahead, lookahead bias, position sizing, whole shares, cash account, transaction costs, commission, bps, slippage, turnover, trade log, trades.csv, entry exit, equity curve, cumulative return, daily portfolio, annualized return, annualized volatility, sharpe, sortino, max drawdown, calmar, win rate, hit rate, number of trades, buy and hold, sma crossover, ema crossover, golden cross, death cross, bollinger, stop loss, trend following, cta, volatility targeting, ewma volatility, execution lag, lagged weights, overlay, monthly rebalance, month start, month end, long-short, params.json, off-by-one
summary: Event-loop and vectorised conventions that reproduce reference backtests: signal at close -> fill next open, whole-share sizing, cost formulas, trade rows, forced close, EMA variants, one-day weight lag with overlays, rebalance calendars, cost timing, and the metric definitions (mean x 252, ddof, drawdown path).
---
## When this applies
"Backtest this rule on daily OHLC data", "trades.csv / daily_portfolio.csv", "EMA(50/200) golden-cross strategy",
"trend-following with EWMA volatility targeting and a leverage cap", "monthly-rebalanced long-short of the top/bottom
k names with transaction costs", "regime overlay on the strategy", "total/annualised return, Sharpe, max drawdown,
Calmar, win rate, number of trades".

## Method and formulas
Symbols: C_t close (adjusted), O_t open, n trading days, cash0 initial capital, c = proportional cost (bps/1e4),
fee = fixed fee per order, s = slippage fraction, A = 252.
- Timing (no look-ahead): indicators from data up to the close of day t generate the signal on day t; the order
  fills on day t+1 at O_{t+1} ("next open") or C_{t+1} ("next close"); positions are marked at every close:
  value_t = cash + shares * C_t. No new signal on the last day; open positions are force-closed at C_{n-1}.
- Loop order each day: (1) fill the pending order from yesterday, (2) evaluate today's signal (may create a new
  pending order even if a fill happened at the open), (3) forced close if last day, (4) record value.
- Sizing: whole shares q = floor((cash - fee)/(px (1 + c))); q = 0 means the signal is consumed and NO trade is
  logged; fractional shares only if the task says. Cash: buy_cost = q px (1 + c) + fee; sell_net = q px (1 - c) - fee;
  trade pnl = sell_net - buy_cost; slippage buys at px (1 + s), sells at px (1 - s). Risk-based size q =
  min(floor(risk_frac cash / stop_distance), cash-limited q); stop_distance <= 0 -> skip.
- Indicators: SMA(w) = rolling(w).mean() (NaN for the first w-1 rows); Bollinger mid +/- k rolling std(ddof=1).
  EMA has two reference variants: (a) pandas ewm(span=k, adjust=False).mean() starts at the first price with no
  NaN warm-up (early whipsaw crosses are part of the reference - do not add a warm-up filter); (b) SMA-seeded EMA:
  NaN for the first k-1 rows, seed = mean of the first k values, then e_t = a x_t + (1-a) e_{t-1}, a = 2/(k+1).
  Crossover on day t needs t-1 and t: golden fast_{t-1} <= slow_{t-1} and fast_t > slow_t (death: >= and <).
  Trailing stop: peak starts at the signal-day close (or fill price per spec), updates each close; exit if C_t < (1-x) peak.
- Vectorised weights with a one-day execution lag: port_t = sum_i w_{t-1,i} r_{t,i}, port_0 = 0 KEPT (rows = number
  of returns; averages such as gross leverage include that row). Overlays dated t (regime scale, vol scale) multiply
  the LAGGED weight on the return day: port_t = sum_i w_{t-1,i} scale_t r_{t,i} (scaling w_t before the lag failed).
- Trend sizing (CTA): signal_t = +1 if fast > slow else -1, 0 while the slow EMA is NaN, aligned to returns with
  sig[1:] (price index t+1 = close of return day t); sigma_ann = sqrt(252 EWMA var) with s2_0 = r_0^2 (= pandas
  ewm(span, adjust=False) on r^2); w = signal (target/n_assets)/sigma_ann clipped to +-max leverage, NaN -> 0.
- Rebalance calendars: "monthly" = first trading day of each month (month_t != month_{t-1}) PLUS the initial trade
  day, or last trading day (month_t != month_{t+1}; the last data day has no successor and is not a rebalance).
  Cross-sectional rebalance on day t with a signal from R[t-L:t] (EXCLUDING t): the new weights already earn r_t
  and the cost sum|w_new - w_old| x bps/1e4 is charged on day t. Weights formed from data through the close of t
  earn r_{t+1}, and the cost of that trade is charged in the t+1 return.
- Equity metrics: ret_t = value_t/value_{t-1} - 1 (ret_0 = 0); cum = prod(1 + ret) starts at 1.0; total_return =
  value_{n-1}/cash0 - 1; annualized_return = mean(ret) A (params-family references use mean x 252 even where the
  text writes exp(.) - 1 or compounding) or CAGR (1 + total)^(A/n) - 1 only when named; vol = std(ddof=1) sqrt(A);
  Sharpe = (ann_ret - rf)/vol; mdd = max(1 - W/cummax W) positive, W = cumprod(1+r) (simple) or exp(cumsum r)
  (log); Calmar = ann_ret/mdd; hit rate = mean(ret > 0) over strategy days; Sortino uses downside std.
- Trade metrics: num_trades = BUY rows; win_rate = share of SELL rows with pnl > 0; total_pnl = sum of SELL pnl;
  profit factor = gross profit/|gross loss|; holding days = exit exec date - entry exec date; turnover = traded value
  / average equity. Close-to-close vectorised variant: pos_t = signal_{t-1}, ret_t = pos_t r_t - c |pos_t - pos_{t-1}|;
  buy-and-hold benchmark = floor(cash0/O_1) shares bought at the first fill.

## Conventions and pitfalls (what graders check)
- Final capital/num_trades oracles need: whole shares, next-open fills, pending fills BEFORE new signals, no signal
  on the last day, forced close at the last close, exact cost formula (fractional shares drift ~0.1%).
- trades.csv: ticker, type (BUY/SELL), signal_date, exec_date, price, shares, pnl; pnl EMPTY on BUY rows; price/pnl
  rounded to 2 dp in the CSV only (cash uses unrounded prices); forced-close rows have signal_date == exec_date.
- daily_portfolio.csv: one row per trading day, cumulative_return starts at 1.0 and ends at 1 + total_return.
- Use ADJUSTED open/close consistently; std ddof=1; mean x 252 vs CAGR differ materially on long samples.
- Look-ahead: a signal window that includes day t, or weights applied the day they are computed, inflates Sharpe
  far outside tolerance; check the first trade date against the first full window.
- Before the first rebalance hold exactly what the spec says (flat, or equal weight 1/n in some references).
- A cap in params (e.g. max position) may be unused by the reference when the weighting rule is explicit
  (+-w per name); implement the literal rule first and test the cap only if numbers disagree.
- Monthly summaries vary: some references use std ddof=0, CAGR (1 + cum)^(12/n) - 1 and a drawdown path prefixed with
  the initial capital - follow the formula in the task text; params-family default is ddof=1 and mean x A.
- Risk-sized entries: stop distance <= 0 means q = 0 (skip). One position at a time unless told.
- JSON: cast float()/int(); results keyed by ticker when several symbols; HTML charts only need to exist.

## Tested snippet
```python
import math, numpy as np, pandas as pd
def metrics(ret, A=252, log=False, rf=0.0):            # ann = mean*A (reference default), vol ddof=1, mdd on the wealth path
    ret = np.asarray(ret, float); W = np.exp(np.cumsum(ret)) if log else np.cumprod(1 + ret)
    ann, vol = float(ret.mean() * A), float(ret.std(ddof=1) * math.sqrt(A)); mdd = float((1 - W / np.maximum.accumulate(W)).max())
    return dict(annualized_return=ann, annualized_volatility=vol, sharpe_ratio=(ann - rf) / vol if vol > 0 else 0.0, max_drawdown=mdd, calmar_ratio=ann / mdd if mdd > 0 else 0.0)
def run_backtest(df, signal, cash0=100000.0, comm=0.001, fee=0.0, slip=0.0):   # df: date/open/close (adjusted); signal at close i -> fill at open i+1
    dates, aopen, aclose = df["date"].tolist(), df["open"].to_numpy(float), df["close"].to_numpy(float)
    n = len(df); cash, sh, pending, buy_cost, trades, values = cash0, 0, None, 0.0, [], []
    for i in range(n):
        if pending is not None:                                         # 1) fill yesterday's signal at today's open
            typ, sdate = pending; pending = None
            if typ == "BUY" and sh == 0:
                px = aopen[i] * (1 + slip); q = math.floor((cash - fee) / (px * (1 + comm)))
                if q > 0:                                               # q == 0 -> signal consumed, no trade
                    sh, buy_cost = q, q * px * (1 + comm) + fee; cash -= buy_cost
                    trades.append(dict(type="BUY", signal_date=sdate, exec_date=dates[i], price=round(px, 2), shares=q, pnl=""))
            elif typ == "SELL" and sh > 0:
                px = aopen[i] * (1 - slip); net = sh * px * (1 - comm) - fee; cash += net
                trades.append(dict(type="SELL", signal_date=sdate, exec_date=dates[i], price=round(px, 2), shares=sh, pnl=round(net - buy_cost, 2))); sh = 0
        if i < n - 1 and signal[i] != 0: pending = ("BUY" if signal[i] > 0 else "SELL", dates[i])   # 2) never on the last day
        if i == n - 1 and sh > 0:                                       # 3) forced close at the last close
            net = sh * aclose[i] * (1 - comm) - fee; cash += net; trades.append(dict(type="SELL", signal_date=dates[i], exec_date=dates[i], price=round(aclose[i], 2), shares=sh, pnl=round(net - buy_cost, 2))); sh = 0
        values.append(cash + sh * aclose[i])                            # mark-to-market at the close
    eq = pd.Series(values, index=dates); ret = eq.pct_change().fillna(0.0); sells = [t for t in trades if t["type"] == "SELL"]
    m = dict(metrics(ret), total_return=float(eq.iloc[-1] / cash0 - 1), final_capital=float(eq.iloc[-1]), num_trades=sum(t["type"] == "BUY" for t in trades),
             win_rate=float(np.mean([t["pnl"] > 0 for t in sells])) if sells else 0.0, total_pnl=round(float(sum(t["pnl"] for t in sells)), 2), num_trading_days=n)
    return m, pd.DataFrame(trades), pd.DataFrame({"date": dates, "cumulative_return": (1 + ret).cumprod().to_numpy()})
def ema_sma_seeded(x, span):                           # NaN before index span-1, seed = SMA of the first span values
    a = 2.0 / (span + 1); out = np.full(len(x), np.nan); out[span - 1] = x[:span].mean()
    for t in range(span, len(x)): out[t] = a * x[t] + (1 - a) * out[t - 1]
    return out                                         # pandas ewm(span=, adjust=False) instead starts at x[0] with no warm-up
def lagged_perf(w, R, overlay=None, log=True):         # port_t = sum_i w_{t-1,i} (x overlay_t) R_{t,i}; row 0 (= 0) is KEPT
    wl = np.vstack([np.zeros((1, R.shape[1])), w[:-1]]) * (1.0 if overlay is None else np.asarray(overlay)[:, None])
    x = (wl * R).sum(axis=1); return x, metrics(x, log=log), float(np.abs(wl).sum(axis=1).mean())   # avg gross leverage incl. row 0
month_flags = lambda d: (lambda m: (np.r_[True, m[1:] != m[:-1]], np.r_[m[:-1] != m[1:], False]))(pd.DatetimeIndex(d).to_period("M"))   # first / last day of month

rng = np.random.default_rng(0); n = 800; dates = pd.bdate_range("2020-01-01", periods=n); px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (n, 2)), axis=0))
R = np.diff(np.log(px), axis=0); fast, slow = (np.column_stack([ema_sma_seeded(px[:, i], s) for i in range(2)]) for s in (20, 60))
sig = np.where(fast > slow, 1.0, -1.0); sig[np.isnan(slow)] = 0.0; sig = sig[1:]          # price index t+1 = close of return day t
vol = np.sqrt(252 * pd.DataFrame(R ** 2).ewm(span=30, adjust=False).mean().to_numpy())    # EWMA variance seeded with r_0^2
x, m, lev = lagged_perf(np.nan_to_num(np.clip(sig * (0.10 / 2) / vol, -2, 2)), R); first, last = month_flags(dates)
up, dn = (fast[:-1, 0] <= slow[:-1, 0]) & (fast[1:, 0] > slow[1:, 0]), (fast[:-1, 0] >= slow[:-1, 0]) & (fast[1:, 0] < slow[1:, 0])   # NaN compares False
mb, trades, daily = run_backtest(pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": px[:, 0] * (1 + rng.normal(0, 0.002, n)), "close": px[:, 0]}), np.r_[0, np.where(up, 1, np.where(dn, -1, 0))], fee=1.0)
print(len(x), {k: round(v, 4) for k, v in m.items()}, round(lev, 4), int(first.sum()), int(last.sum()))
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in mb.items()}, len(trades), daily["cumulative_return"].iloc[0])
```
