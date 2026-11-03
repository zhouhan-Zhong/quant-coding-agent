# memory-solution
# unit: t1-interest-rate-cap-floor | category: derivatives-pricing | tags: cap, floor, caplet, floorlet, black-76, put-call-parity, discount-factors, forward-rates
# deliverables: results.json
# Caplet i (0-based index i): fixing time T = i*tau (first caplet fixes today -> intrinsic value max(f-K,0)).
# Black: d1 = (ln(f/K) + 0.5 vol^2 T)/(vol sqrt T), d2 = d1 - vol sqrt T;
#   caplet = N*tau*df*(f N(d1) - K N(d2)), floorlet = N*tau*df*(K N(-d2) - f N(-d1)); df_i discounts the payment at (i+1)*tau.
# swap_value = N*tau*sum(df_i (f_i - K)); parity error = cap - floor - swap (0 up to rounding).
# Rounding as the instruction states (2 / 4 / 6 dp); checker tolerances are rtol 1% on cap/floor, atol 0.1 on sum(caplets).
import os, json, math
import numpy as np
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

p = json.load(open(os.path.join(DATA_DIR, "params.json")))
f = np.asarray(p["forward_rates"], float)
dfs = np.asarray(p["discount_factors"], float)
vol, K, N, tau = float(p["vol"]), float(p["strike"]), float(p["notional"]), float(p["period_length"])

caplets, floorlets = [], []
for i, (fi, dfi) in enumerate(zip(f, dfs)):
    T = i * tau                                   # fixing time of period i (0 for the first period)
    scale = N * tau * dfi
    if T <= 0:
        caplets.append(scale * max(fi - K, 0.0)); floorlets.append(scale * max(K - fi, 0.0))
        continue
    sq = vol * math.sqrt(T)
    d1 = (math.log(fi / K) + 0.5 * vol ** 2 * T) / sq
    d2 = d1 - sq
    caplets.append(scale * (fi * norm.cdf(d1) - K * norm.cdf(d2)))
    floorlets.append(scale * (K * norm.cdf(-d2) - fi * norm.cdf(-d1)))

cap, floor = float(sum(caplets)), float(sum(floorlets))
swap = float(N * tau * np.sum(dfs * (f - K)))
parity = cap - floor - swap

json.dump({"cap_price": round(cap, 2), "floor_price": round(floor, 2),
           "caplets": [round(float(c), 4) for c in caplets], "floorlets": [round(float(c), 4) for c in floorlets],
           "swap_value": round(swap, 2), "put_call_parity_error": round(parity, 6)},
          open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
