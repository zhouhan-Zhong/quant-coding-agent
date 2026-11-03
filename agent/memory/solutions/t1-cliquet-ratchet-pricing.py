# memory-solution
# unit: t1-cliquet-ratchet-pricing | category: derivatives-pricing | tags: cliquet, ratchet, forward-start, rubinstein, black-scholes, dividend-yield, spy-calibration
# deliverables: calibration.json, cliquet_prices.csv, forward_start_details.csv, summary.json
# SPY calibration family: close-to-close log returns, sigma = std(ddof=1)*sqrt(252), S0 = last close; r = 0.05, D = 0.013 (given).
# Forward-start ATM call (Rubinstein 1991), strike set at t_s, matures at t_e, tau = t_e - t_s:
#   FS = S0 e^{-D t_s} * [e^{-D tau} N(d1) - e^{-r tau} N(d2)],  d1 = (r - D + sigma^2/2) tau / (sigma sqrt(tau)), d2 = d1 - sigma sqrt(tau)
#   (= S0 e^{-D t_s} * C_BS(S=1, K=1, tau) -> each leg is a unit ATM call scaled by the dividend-discounted spot)
# Cliquet(T, N) = sum_{i=0}^{N-1} FS(i T/N, (i+1) T/N); European = C_BS(S0, K=S0, T) with the same dividend yield.
# Oracle matched to ~1e-12 with this convention; cliquet grows with N (more resets -> more ATM optionality).
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

px = pd.read_csv(os.path.join(DATA_DIR, "spy_daily.csv"), parse_dates=["date"]).sort_values("date")
close = px["close"].to_numpy(float)
x = np.diff(np.log(close))
S0 = float(close[-1]); r = 0.05; D = 0.013
sigma = float(x.std(ddof=1) * math.sqrt(252))

json.dump({"n_returns": int(len(x)), "return_mean": float(x.mean()), "return_std": float(x.std(ddof=1)),
           "return_skewness": float(skew(x)), "return_excess_kurtosis": float(kurtosis(x, fisher=True)),
           "annualized_vol": sigma, "S0": S0},
          open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)

def bs_call(S, K, tau):
    """Black-Scholes call with continuous dividend yield D."""
    if tau <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sq = sigma * math.sqrt(tau)
    d1 = (math.log(S / K) + (r - D + 0.5 * sigma * sigma) * tau) / sq
    return S * math.exp(-D * tau) * norm.cdf(d1) - K * math.exp(-r * tau) * norm.cdf(d1 - sq)

def forward_start_call(t_s, t_e):
    """ATM forward-start call: strike fixed at S(t_s), pays at t_e. Homogeneity -> S0 e^{-D t_s} * C_BS(1, 1, t_e - t_s)."""
    return S0 * math.exp(-D * t_s) * bs_call(1.0, 1.0, t_e - t_s)

maturities, resets = [1.0, 2.0, 3.0], [4, 12]
cfg_rows, fs_rows = [], []
for T in maturities:
    euro = bs_call(S0, S0, T)
    for N in resets:
        dt = T / N
        total = 0.0
        for i in range(N):
            t_s, t_e = i * dt, (i + 1) * dt
            p = forward_start_call(t_s, t_e)
            total += p
            fs_rows.append({"T": T, "N": N, "period_index": i, "t_start": t_s, "t_end": t_e, "forward_start_price": p})
        cfg_rows.append({"T": T, "N": N, "period_length": dt, "cliquet_price": total,
                         "european_price": euro, "ratio_cliquet_over_european": total / euro})

cfg = pd.DataFrame(cfg_rows).sort_values(["T", "N"]).reset_index(drop=True)
cfg.to_csv(os.path.join(OUT_DIR, "cliquet_prices.csv"), index=False)
pd.DataFrame(fs_rows).to_csv(os.path.join(OUT_DIR, "forward_start_details.csv"), index=False)

increases = all(float(cfg[(cfg["T"] == T) & (cfg["N"] == 12)]["cliquet_price"].iloc[0]) >
                float(cfg[(cfg["T"] == T) & (cfg["N"] == 4)]["cliquet_price"].iloc[0]) for T in maturities)
summary = {"n_configurations": int(len(cfg)), "total_forward_starts": int(len(fs_rows)),
           "max_cliquet_price": float(cfg["cliquet_price"].max()), "min_cliquet_price": float(cfg["cliquet_price"].min()),
           "mean_ratio_cliquet_over_european": float(cfg["ratio_cliquet_over_european"].mean()),
           "cliquet_increases_with_N": bool(increases)}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
