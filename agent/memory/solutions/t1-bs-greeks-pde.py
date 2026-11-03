# memory-solution
# unit: t1-bs-greeks-pde | category: derivatives-pricing | tags: black-scholes, greeks, pde-residual, put-call-parity, spy-calibration
# deliverables: calibration.json, greeks_surface.csv, pde_verification.csv, summary.json
# pattern: calibrate (S0 = last close, sigma = std(log returns, ddof=1)*sqrt(252)) -> analytic BS with dividend yield on a
#          (moneyness x maturity) grid -> PDE residual Theta + (r-D) S Delta + 0.5 sigma^2 S^2 Gamma - r V == 0 -> summary flags.
import os, json
import numpy as np, pandas as pd
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Step 1: calibration from SPY daily closes
px = pd.read_csv(os.path.join(DATA_DIR, "spy_daily.csv"), parse_dates=["date"]).sort_values("date")
close = px["close"].to_numpy(dtype=float)
logret = np.diff(np.log(close))                     # n_returns = len(close) - 1
S0 = float(close[-1])
ret_mean, ret_std = float(logret.mean()), float(logret.std(ddof=1))
sigma = ret_std * np.sqrt(252.0)
r, D = 0.05, 0.013
K = S0

# ---- Step 2/3: analytic Greeks with continuous dividend yield
def bs_all(S, K, T, r, q, sig):
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * sqT)
    d2 = d1 - sig * sqT
    eq, er = np.exp(-q * T), np.exp(-r * T)
    Nd1, Nd2, nd1 = norm.cdf(d1), norm.cdf(d2), norm.pdf(d1)
    call = S * eq * Nd1 - K * er * Nd2
    put = K * er * norm.cdf(-d2) - S * eq * norm.cdf(-d1)
    call_delta = eq * Nd1
    put_delta = eq * (Nd1 - 1.0)
    gamma = eq * nd1 / (S * sig * sqT)
    vega = S * eq * nd1 * sqT                       # per unit of sigma (not per 1%)
    call_theta = -S * eq * nd1 * sig / (2 * sqT) - r * K * er * Nd2 + q * S * eq * Nd1      # per year, dV/dt
    put_theta = -S * eq * nd1 * sig / (2 * sqT) + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)
    call_rho = K * T * er * Nd2
    put_rho = -K * T * er * norm.cdf(-d2)
    return dict(call_price=call, put_price=put, call_delta=call_delta, put_delta=put_delta, gamma=gamma,
                call_theta=call_theta, put_theta=put_theta, vega=vega, call_rho=call_rho, put_rho=put_rho)

moneyness = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
maturities = [0.25, 0.50, 1.00]
rows, pde_rows = [], []
for T in maturities:
    for m in moneyness:
        S = m * K
        g = bs_all(S, K, T, r, D, sigma)
        rows.append({"T": T, "moneyness": m, "S": S, **{k: float(v) for k, v in g.items()}})
        # ---- Step 4/5: PDE residual and parity
        res_c = g["call_theta"] + (r - D) * S * g["call_delta"] + 0.5 * sigma ** 2 * S ** 2 * g["gamma"] - r * g["call_price"]
        res_p = g["put_theta"] + (r - D) * S * g["put_delta"] + 0.5 * sigma ** 2 * S ** 2 * g["gamma"] - r * g["put_price"]
        parity = abs(g["call_price"] - g["put_price"] - (S * np.exp(-D * T) - K * np.exp(-r * T)))
        pde_rows.append({"T": T, "moneyness": m, "S": S, "call_pde_residual": float(res_c),
                         "put_pde_residual": float(res_p), "parity_error": float(parity)})

surf = pd.DataFrame(rows, columns=["T", "moneyness", "S", "call_price", "put_price", "call_delta", "put_delta", "gamma",
                                   "call_theta", "put_theta", "vega", "call_rho", "put_rho"]).sort_values(["T", "moneyness"])
pde = pd.DataFrame(pde_rows).sort_values(["T", "moneyness"])
surf.to_csv(os.path.join(OUT_DIR, "greeks_surface.csv"), index=False)
pde.to_csv(os.path.join(OUT_DIR, "pde_verification.csv"), index=False)

def atm(T, col):
    return float(surf[(np.isclose(surf["T"], T)) & (np.isclose(surf["moneyness"], 1.0))][col].iloc[0])

sub = surf[np.isclose(surf["T"], 0.5)]
upper_ok = all(-1e-12 <= row.call_delta <= np.exp(-D * row.T) + 1e-12 for row in surf.itertuples())
summary = {
    "n_grid_points": int(len(surf)), "n_maturities": len(maturities), "n_moneyness": len(moneyness),
    "max_call_pde_residual": float(pde["call_pde_residual"].abs().max()),
    "max_put_pde_residual": float(pde["put_pde_residual"].abs().max()),
    "max_parity_error": float(pde["parity_error"].max()),
    "atm_call_price_T025": atm(0.25, "call_price"), "atm_call_price_T050": atm(0.5, "call_price"), "atm_call_price_T100": atm(1.0, "call_price"),
    "atm_call_delta_T100": atm(1.0, "call_delta"), "atm_gamma_T100": atm(1.0, "gamma"), "atm_vega_T100": atm(1.0, "vega"),
    "max_gamma_moneyness_T050": float(sub.loc[sub["gamma"].idxmax(), "moneyness"]),
    "delta_call_range_valid": bool(upper_ok),
    "gamma_positive_everywhere": bool((surf["gamma"] > 0).all()),
    "vega_positive_everywhere": bool((surf["vega"] > 0).all()),
}
json.dump({"S0": S0, "sigma": float(sigma), "r": r, "D": D, "K": float(K), "n_returns": int(len(logret)),
           "return_mean": ret_mean, "return_std": ret_std}, open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
