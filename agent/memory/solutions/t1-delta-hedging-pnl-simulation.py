# memory-solution
# unit: t1-delta-hedging-pnl-simulation | category: risk | tags: delta-hedging, discrete-rebalancing, transaction-costs, escrowed-dividend, black-scholes, hedging-pnl, cash-account, time-varying-iv
# deliverables: results.json
# Escrowed-dividend BS: S_eff(t) = S[t] - sum_{ex > t} D e^{-r (ex - t)/252}; T_rem = (252 - t)/252; sigma = iv[t];
#   d1 = (ln(S_eff/K) + (r + s^2/2) T_rem)/(s sqrt T_rem); price = S_eff N(d1) - K e^{-rT} N(d2); hedge delta = N(d1) (no q-discount).
# Book: day 0 sell call at model price, buy delta0 shares (+ buy-side TC). Each day t = 1..252: cash *= e^{r dt};
#   on ex-div day credit position_prev * D; if t % freq == 0 and t < 252 rebalance to delta(t) (TC = |dN| S * bps/1e4,
#   buy rate if dN > 0 else sell rate). Day 252: sell all shares (sell TC), pay max(S - K, 0); pnl = final cash.
# n_rebalances counts only interior rebalances (251 / 50 / 25 / 11); total TC includes the initial buy and final liquidation.
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

S = pd.read_csv(os.path.join(DATA_DIR, "stock_path.csv")).sort_values("day")["close"].to_numpy(float)
IV = pd.read_csv(os.path.join(DATA_DIR, "iv_path.csv")).sort_values("day")["implied_vol"].to_numpy(float)
divs = pd.read_csv(os.path.join(DATA_DIR, "dividends.csv"))
div_map = {int(d): float(a) for d, a in zip(divs["ex_div_day"], divs["amount"])}
fees = pd.read_csv(os.path.join(DATA_DIR, "fee_schedule.csv"))
rate = {str(t).strip().lower(): float(b) / 1e4 for t, b in zip(fees["trade_type"], fees["rate_bps"])}
K, r, NDAYS = 100.0, 0.03, 252
dt = 1.0 / NDAYS

def pv_divs(t):
    return sum(a * math.exp(-r * (ex - t) * dt) for ex, a in div_map.items() if ex > t)

def price_delta(t):
    tau = (NDAYS - t) * dt; s = IV[t]; S_eff = S[t] - pv_divs(t)
    d1 = (math.log(S_eff / K) + (r + 0.5 * s * s) * tau) / (s * math.sqrt(tau)); d2 = d1 - s * math.sqrt(tau)
    return S_eff * norm.cdf(d1) - K * math.exp(-r * tau) * norm.cdf(d2), norm.cdf(d1), S_eff

def tc(d_shares, px):
    return abs(d_shares) * px * (rate["buy"] if d_shares > 0 else rate["sell"])

def simulate(freq):
    price0, pos, _ = price_delta(0)
    cost0 = tc(pos, S[0])
    cash = price0 - pos * S[0] - cost0
    total_tc, total_div, n_reb = cost0, 0.0, 0
    for t in range(1, NDAYS + 1):
        cash *= math.exp(r * dt)                                     # 1. interest on the bank
        if t in div_map:                                             # 2. dividend on the position carried overnight
            cash += pos * div_map[t]; total_div += pos * div_map[t]
        if t < NDAYS and t % freq == 0:                              # 3. rebalance strictly inside (0, 252)
            _, new_pos, _ = price_delta(t)
            trade = new_pos - pos; c = tc(trade, S[t])
            cash -= trade * S[t] + c; total_tc += c; pos = new_pos; n_reb += 1
    c = tc(-pos, S[NDAYS])                                           # terminal liquidation at the sell rate
    cash += pos * S[NDAYS] - c; total_tc += c
    payoff = max(S[NDAYS] - K, 0.0)
    cash -= payoff
    return {"freq_days": int(freq), "n_rebalances": int(n_reb), "total_transaction_cost": float(total_tc),
            "total_dividends_received": float(total_div), "hedging_pnl": float(cash)}

p0, d0, se0 = price_delta(0)
out = {"t0": {"price": float(p0), "delta": float(d0), "S_eff": float(se0), "pv_divs": float(pv_divs(0))},
       "terminal": {"final_stock": float(S[NDAYS]), "call_payoff": float(max(S[NDAYS] - K, 0.0))},
       "simulations": [simulate(f) for f in (1, 5, 10, 22)]}
json.dump(out, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
