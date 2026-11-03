# memory-solution
# unit: t1-spread-option-kirk-margrabe | category: derivatives-pricing | tags: spread-option, kirk-approximation, margrabe, exchange-option, correlated-gbm, cholesky, monte-carlo, two-asset-calibration
# deliverables: calibration.json, margrabe_prices.csv, kirk_prices.csv, correlation_sensitivity.csv, summary.json
# Calibration: inner-join AAPL/MSFT on date, close-to-close log returns (692), sigma_i = std(ddof=1)*sqrt(252), rho = Pearson corr, S_i0 = last close.
# Margrabe (S1 = AAPL, S2 = MSFT, payoff max(S1-S2,0)): sig_x = sqrt(s1^2 + s2^2 - 2 rho s1 s2),
#   d1 = (ln(S1/S2) + (D2 - D1 + sig_x^2/2) T)/(sig_x sqrt T), C = S1 e^{-D1 T} N(d1) - S2 e^{-D2 T} N(d2)
# Kirk (payoff max(S1-S2-K,0)): F_i = S_i e^{(r-D_i)T}, w = F2/(F2+K), sig_K = sqrt(s1^2 - 2 rho s1 s2 w + s2^2 w^2),
#   d1 = (ln(F1/(F2+K)) + sig_K^2 T/2)/(sig_K sqrt T), C = e^{-rT}[F1 N(d1) - (F2+K) N(d2)]; K = 0 -> exactly Margrabe.
# MC: 500k paths per T, Cholesky of [[1,rho],[rho,1]], exact GBM terminal step, same normals reused for all K at that T.
# Note: AAPL << MSFT so the spread call is deep OTM (prices ~1e-5..1); checker is invariant-based (tol = max(3 se + 1e-3, 5%)).
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

a = pd.read_csv(os.path.join(DATA_DIR, "aapl_daily.csv"), parse_dates=["date"])[["date", "close"]]
b = pd.read_csv(os.path.join(DATA_DIR, "msft_daily.csv"), parse_dates=["date"])[["date", "close"]]
df = a.merge(b, on="date", suffixes=("_1", "_2")).sort_values("date").reset_index(drop=True)
x1 = np.diff(np.log(df["close_1"].to_numpy(float))); x2 = np.diff(np.log(df["close_2"].to_numpy(float)))
S1, S2 = float(df["close_1"].iloc[-1]), float(df["close_2"].iloc[-1])
s1, s2 = float(x1.std(ddof=1) * math.sqrt(252)), float(x2.std(ddof=1) * math.sqrt(252))
rho = float(np.corrcoef(x1, x2)[0, 1])
r, D1, D2 = 0.05, 0.005, 0.005

json.dump({"S1_0": S1, "S2_0": S2, "sigma1": s1, "sigma2": s2, "rho": rho, "n_returns": int(len(x1)),
           "D1": D1, "D2": D2, "r": r}, open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)

def margrabe(T, rh):
    sx = math.sqrt(s1 * s1 + s2 * s2 - 2 * rh * s1 * s2)
    sq = sx * math.sqrt(T)
    d1 = (math.log(S1 / S2) + (D2 - D1 + 0.5 * sx * sx) * T) / sq
    return S1 * math.exp(-D1 * T) * norm.cdf(d1) - S2 * math.exp(-D2 * T) * norm.cdf(d1 - sq), sx

def kirk(K, T, rh):
    F1, F2 = S1 * math.exp((r - D1) * T), S2 * math.exp((r - D2) * T)
    w = F2 / (F2 + K)
    sk = math.sqrt(s1 * s1 - 2 * rh * s1 * s2 * w + s2 * s2 * w * w)
    sq = sk * math.sqrt(T)
    d1 = (math.log(F1 / (F2 + K)) + 0.5 * sk * sk * T) / sq
    return math.exp(-r * T) * (F1 * norm.cdf(d1) - (F2 + K) * norm.cdf(d1 - sq))

maturities, mons = [0.25, 0.5, 1.0], [-0.10, -0.05, 0.0, 0.05, 0.10]
avg_spot = 0.5 * (S1 + S2)

mg = [{"T": T, "margrabe_price": margrabe(T, rho)[0], "sigma_exchange": margrabe(T, rho)[1]} for T in maturities]
pd.DataFrame(mg).to_csv(os.path.join(OUT_DIR, "margrabe_prices.csv"), index=False)
sigma_exchange = float(mg[0]["sigma_exchange"])

# Monte Carlo: Cholesky-correlated terminal GBM, one normal matrix per maturity shared across strikes
rng = np.random.default_rng(42); n_paths = 500_000
L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
rows = []
for T in maturities:
    Z = rng.standard_normal((n_paths, 2)) @ L.T
    ST1 = S1 * np.exp((r - D1 - 0.5 * s1 * s1) * T + s1 * math.sqrt(T) * Z[:, 0])
    ST2 = S2 * np.exp((r - D2 - 0.5 * s2 * s2) * T + s2 * math.sqrt(T) * Z[:, 1])
    disc = math.exp(-r * T)
    for m in mons:
        K = avg_spot * m
        pay = disc * np.maximum(ST1 - ST2 - K, 0.0)
        rows.append({"T": T, "K": K, "spread_moneyness": m, "kirk_price": kirk(K, T, rho),
                     "mc_price": float(pay.mean()), "mc_std_err": float(pay.std(ddof=1) / math.sqrt(n_paths))})
kp = pd.DataFrame(rows).sort_values(["T", "K"]).reset_index(drop=True)
kp.to_csv(os.path.join(OUT_DIR, "kirk_prices.csv"), index=False)

rho_grid = [-0.9, -0.5, 0.0, 0.5, 0.9]
cs = pd.DataFrame({"rho_grid": rho_grid, "margrabe_price": [margrabe(0.5, g)[0] for g in rho_grid],
                   "kirk_price_K0": [kirk(0.0, 0.5, g) for g in rho_grid]})
cs.to_csv(os.path.join(OUT_DIR, "correlation_sensitivity.csv"), index=False)

err = (kp["kirk_price"] - kp["mc_price"]).abs()
atm = np.isclose(kp["spread_moneyness"], 0.0)
summary = {"sigma_exchange": sigma_exchange,
           "max_mc_error_margrabe": float(err[atm].max()),
           "max_mc_error_kirk": float(err.max()),
           "mc_validates": bool(err.max() <= 3.0 * kp["mc_std_err"].max()),
           "rho_sensitivity_range": float(cs["margrabe_price"].max() - cs["margrabe_price"].min()),
           "mean_kirk_price": float(kp["kirk_price"].mean()),
           "margrabe_T050": float(margrabe(0.5, rho)[0])}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
