# memory-solution
# unit: t1-digital-barrier-options | category: derivatives-pricing | tags: digital-options, cash-or-nothing, asset-or-nothing, gap-option, barrier, reflection-principle, spy-calibration
# deliverables: calibration.json, digital_prices.csv, gap_prices.csv, barrier_digital_prices.csv, summary.json
# key formulas (continuous dividend yield q, mu = r - q - sigma^2/2 for log-price drift):
#   cash call = e^{-rT} N(d2), asset call = S e^{-qT} N(d1), BS call = asset_call - K * cash_call
#   gap call(K1 payoff strike, K2 trigger) = S e^{-qT} N(d1(K2)) - K1 e^{-rT} N(d2(K2))
#   down-and-out cash call (H < S, H < K): e^{-rT}[ N((ln(S/K)+mu T)/(s√T)) - (H/S)^{2mu/s^2} N((ln(H^2/(S K))+mu T)/(s√T)) ]
#   up-and-out cash put  (H > S, H > K): e^{-rT}[ N((ln(K/S)-mu T)/(s√T)) - (H/S)^{2mu/s^2} N((ln(K S/H^2)-mu T)/(s√T)) ]
#   knock-in = vanilla - knock-out
import os, json
import numpy as np, pandas as pd
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

px = pd.read_csv(os.path.join(DATA_DIR, "spy_daily.csv"), parse_dates=["date"]).sort_values("date")
close = px["close"].to_numpy(float)
logret = np.diff(np.log(close))
S0 = float(close[-1]); sigma = float(logret.std(ddof=1) * np.sqrt(252.0))
r, D = 0.05, 0.013
json.dump({"S0": S0, "sigma": sigma, "r": r, "D": D, "n_returns": int(len(logret))},
          open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)

def d1d2(S, K, T):
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - D + 0.5 * sigma ** 2) * T) / sq
    return d1, d1 - sq

maturities = [0.25, 0.50, 1.00]
moneyness = [0.90, 0.95, 1.00, 1.05, 1.10]
rows = []
for T in maturities:
    for m in moneyness:
        K = m * S0
        d1, d2 = d1d2(S0, K, T)
        er, eq = np.exp(-r * T), np.exp(-D * T)
        cash_call, cash_put = er * norm.cdf(d2), er * norm.cdf(-d2)
        asset_call, asset_put = S0 * eq * norm.cdf(d1), S0 * eq * norm.cdf(-d1)
        rows.append({"T": T, "moneyness": m, "K": K, "cash_call": cash_call, "cash_put": cash_put,
                     "asset_call": asset_call, "asset_put": asset_put,
                     "bs_call": asset_call - K * cash_call, "bs_put": K * cash_put - asset_put})
dig = pd.DataFrame(rows)
dig.to_csv(os.path.join(OUT_DIR, "digital_prices.csv"), index=False)

gap_rows = []
for T in maturities:
    for tm in [0.95, 1.00, 1.05]:
        K1, K2 = S0, tm * S0
        d1, d2 = d1d2(S0, K2, T)
        gap_rows.append({"T": T, "trigger_moneyness": tm, "K2": K2, "K1": K1,
                         "gap_call": S0 * np.exp(-D * T) * norm.cdf(d1) - K1 * np.exp(-r * T) * norm.cdf(d2)})
gap = pd.DataFrame(gap_rows)
gap.to_csv(os.path.join(OUT_DIR, "gap_prices.csv"), index=False)

mu = r - D - 0.5 * sigma ** 2
bar_rows = []
for T in maturities:
    sq = sigma * np.sqrt(T); er = np.exp(-r * T)
    for m in moneyness:
        K = m * S0
        # down-and-out cash call, H = 0.85 S0
        H = 0.85 * S0
        vanilla = er * norm.cdf((np.log(S0 / K) + mu * T) / sq)
        ko = er * (norm.cdf((np.log(S0 / K) + mu * T) / sq)
                   - (H / S0) ** (2 * mu / sigma ** 2) * norm.cdf((np.log(H * H / (S0 * K)) + mu * T) / sq))
        ko = max(ko, 0.0); ki = vanilla - ko
        bar_rows.append({"T": T, "moneyness": m, "K": K, "barrier_type": "down_out_cash_call", "H": H,
                         "knockout_price": ko, "knockin_price": ki, "vanilla_price": vanilla, "parity_error": abs(ko + ki - vanilla)})
        # up-and-out cash put, H = 1.15 S0
        H = 1.15 * S0
        vanilla = er * norm.cdf((np.log(K / S0) - mu * T) / sq)
        ko = er * (norm.cdf((np.log(K / S0) - mu * T) / sq)
                   - (H / S0) ** (2 * mu / sigma ** 2) * norm.cdf((np.log(K * S0 / (H * H)) - mu * T) / sq))
        ko = max(ko, 0.0); ki = vanilla - ko
        bar_rows.append({"T": T, "moneyness": m, "K": K, "barrier_type": "up_out_cash_put", "H": H,
                         "knockout_price": ko, "knockin_price": ki, "vanilla_price": vanilla, "parity_error": abs(ko + ki - vanilla)})
bar = pd.DataFrame(bar_rows)
bar.to_csv(os.path.join(OUT_DIR, "barrier_digital_prices.csv"), index=False)

def pick(df, T, col, **kw):
    sub = df[np.isclose(df["T"], T)]
    for k, v in kw.items():
        sub = sub[np.isclose(sub[k], v)] if isinstance(v, float) else sub[sub[k] == v]
    return float(sub[col].iloc[0])

mono = all(np.all(np.diff(dig[np.isclose(dig["T"], T)].sort_values("moneyness")["cash_call"].values) <= 1e-12) for T in maturities)
summary = {
    "n_digital_prices": int(len(dig)), "n_gap_prices": int(len(gap)), "n_barrier_prices": int(len(bar)),
    "max_cash_parity_error": float((dig["cash_call"] + dig["cash_put"] - np.exp(-r * dig["T"])).abs().max()),
    "max_asset_parity_error": float((dig["asset_call"] + dig["asset_put"] - S0 * np.exp(-D * dig["T"])).abs().max()),
    "max_bs_decomp_error": float((dig["bs_call"] - (dig["asset_call"] - dig["K"] * dig["cash_call"])).abs().max()),
    "max_barrier_parity_error": float(bar["parity_error"].max()),
    "atm_cash_call_T050": pick(dig, 0.5, "cash_call", moneyness=1.0),
    "atm_cash_put_T050": pick(dig, 0.5, "cash_put", moneyness=1.0),
    "atm_asset_call_T050": pick(dig, 0.5, "asset_call", moneyness=1.0),
    "atm_gap_call_T050": pick(gap, 0.5, "gap_call", trigger_moneyness=1.0),
    "atm_do_cash_call_T050": pick(bar[bar["barrier_type"] == "down_out_cash_call"], 0.5, "knockout_price", moneyness=1.0),
    "atm_uo_cash_put_T050": pick(bar[bar["barrier_type"] == "up_out_cash_put"], 0.5, "knockout_price", moneyness=1.0),
    "cash_call_increases_with_S": bool(mono),
    "do_leq_vanilla": bool((bar["knockout_price"] <= bar["vanilla_price"] + 1e-12).all()),
}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
