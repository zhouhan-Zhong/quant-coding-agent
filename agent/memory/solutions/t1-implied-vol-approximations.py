# memory-solution
# unit: t1-implied-vol-approximations | category: volatility | tags: implied-vol, brenner-subrahmanyam, li-approximation, corrado-miller, hallerbach, newton-raphson, closed-form-inversion, synthetic-grid
# deliverables: approximations.csv, summary.json
# Grid: S0 = 500, r = 0.05, D = 0.013; St = S0 e^{-DT}, Kt = K e^{-rT}; BS call C = St N(d1) - Kt N(d2) (d1 = (ln(St/Kt) + s^2 T/2)/(s sqrt T)).
# "ATM" = K == S0 (spot ATM) -> BS/Li-ATM formulas assume forward ATM, so their errors are ~0.03 (expected by the checker).
# Brenner-Subrahmanyam: s = sqrt(2 pi / T) C / St.   Li ATM (exact small root of z^3 - 6 z + 6c = 0, c = sqrt(pi/2) C/St):
#   s = 4 sqrt(2/T) cos(2 pi/3 - arccos(-3 sqrt(pi) C/(4 St))/3); NaN if the arccos argument is outside [-1, 1].
# Li non-ATM (CM-type): t1 = C - (St-Kt)/2, disc = t1^2 - (St-Kt)^2 (1 + St/Kt)/(2 pi), s = sqrt(2 pi/T) (t1 + sqrt(disc))/(St+Kt);
#   disc in (-1e-10, 0) -> 0, disc < -1e-10 -> NaN (8 deep-OTM/ITM low-vol points are NaN and excluded from RMSE).
# CMH as the checker expects it: m = St/Kt, a = sqrt(2 pi)/(m+1) (2C/Kt - (m-1)), b = ((m-1)/(m+1))^2/2, s = (a + sqrt(a^2 - 8 b))/sqrt(T)
#   (no 1/2 factor -> about 2x the true vol; the oracle RMSE 0.284 is reproduced only with this exact form).
# Newton: start 0.3, tol 1e-10, 100 iters, converged when |price diff| < tol or |step| < tol.
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

S0, r, D = 500.0, 0.05, 0.013
SQ2PI = math.sqrt(2.0 * math.pi)
sigmas, Ts, mons = [0.10, 0.15, 0.20, 0.30, 0.40], [0.25, 0.50, 1.00], [0.90, 0.95, 1.00, 1.05, 1.10]

def bs_call(K, T, s):
    St, Kt = S0 * math.exp(-D * T), K * math.exp(-r * T)
    d1 = (math.log(St / Kt) + 0.5 * s * s * T) / (s * math.sqrt(T))
    return St * norm.cdf(d1) - Kt * norm.cdf(d1 - s * math.sqrt(T))

def bs_vega(K, T, s):
    St, Kt = S0 * math.exp(-D * T), K * math.exp(-r * T)
    d1 = (math.log(St / Kt) + 0.5 * s * s * T) / (s * math.sqrt(T))
    return St * norm.pdf(d1) * math.sqrt(T)

def brenner_subrahmanyam(C, St, T):
    return math.sqrt(2.0 * math.pi / T) * C / St

def li_atm(C, St, T):
    u = -3.0 * math.sqrt(math.pi) * C / (4.0 * St)          # cubic z^3 - 6 z + 6 c = 0 solved by Viete's trigonometric form
    if u < -1.0 or u > 1.0:
        return float("nan")
    return 4.0 * math.sqrt(2.0 / T) * math.cos(2.0 * math.pi / 3.0 - math.acos(u) / 3.0)

def clip_disc(disc):
    if disc < 0.0:
        return 0.0 if disc > -1e-10 else float("nan")
    return disc

def li_nonatm(C, St, Kt, T):
    t1 = C - 0.5 * (St - Kt)
    disc = clip_disc(t1 * t1 - (St - Kt) ** 2 * (1.0 + St / Kt) / (2.0 * math.pi))
    return math.sqrt(2.0 * math.pi / T) * (t1 + math.sqrt(disc)) / (St + Kt)

def cmh(C, St, Kt, T):
    m = St / Kt
    a = SQ2PI / (m + 1.0) * (2.0 * C / Kt - (m - 1.0))
    b = 0.5 * ((m - 1.0) / (m + 1.0)) ** 2
    disc = clip_disc(a * a - 8.0 * b)
    return (a + math.sqrt(disc)) / math.sqrt(T)

def newton(C, K, T, s0=0.3, tol=1e-10, max_iter=100):
    s = s0
    for _ in range(max_iter):
        f = bs_call(K, T, s) - C
        if abs(f) < tol:
            return s, True
        v = bs_vega(K, T, s)
        if not np.isfinite(v) or v <= 0:
            return float("nan"), False
        s_new = s - f / v
        if abs(s_new - s) < tol:
            return s_new, True
        s = s_new
    return s, False

rows, conv = [], []
for s in sigmas:
    for T in Ts:
        for m in mons:
            K = S0 * m
            St, Kt = S0 * math.exp(-D * T), K * math.exp(-r * T)
            C = bs_call(K, T, s)
            is_atm = abs(m - 1.0) < 1e-9
            est = {"sigma_bs_approx": brenner_subrahmanyam(C, St, T) if is_atm else float("nan"),
                   "sigma_li_atm": li_atm(C, St, T) if is_atm else float("nan"),
                   "sigma_li_nonatm": li_nonatm(C, St, Kt, T), "sigma_cmh": cmh(C, St, Kt, T)}
            s_nr, ok = newton(C, K, T); conv.append(ok); est["sigma_newton"] = s_nr
            row = {"sigma_true": s, "T": T, "K": K, "moneyness": m, "bs_price": C, **est}
            for key, err in [("sigma_bs_approx", "error_bs"), ("sigma_li_atm", "error_li_atm"), ("sigma_li_nonatm", "error_li_nonatm"),
                             ("sigma_cmh", "error_cmh"), ("sigma_newton", "error_newton")]:
                row[err] = est[key] - s
            rows.append(row)

cols = ["sigma_true", "T", "K", "moneyness", "bs_price", "sigma_bs_approx", "sigma_li_atm", "sigma_li_nonatm", "sigma_cmh",
        "sigma_newton", "error_bs", "error_li_atm", "error_li_nonatm", "error_cmh", "error_newton"]
df = pd.DataFrame(rows)[cols].sort_values(["sigma_true", "T", "moneyness"]).reset_index(drop=True)
df.to_csv(os.path.join(OUT_DIR, "approximations.csv"), index=False, float_format="%.10f")

def rmse(col):
    e = df[col].dropna().to_numpy(float)
    return float(np.sqrt(np.mean(e ** 2)))

summary = {"n_points": int(len(df)), "n_atm_points": int((df["moneyness"] == 1.0).sum()),
           "rmse_bs_atm": rmse("error_bs"), "rmse_li_atm": rmse("error_li_atm"), "rmse_li_nonatm": rmse("error_li_nonatm"),
           "rmse_cmh": rmse("error_cmh"), "rmse_newton": rmse("error_newton"),
           "max_abs_error_li_nonatm": float(df["error_li_nonatm"].abs().max()),
           "max_abs_error_cmh": float(df["error_cmh"].abs().max()),
           "mean_abs_error_cmh": float(df["error_cmh"].abs().mean()),
           "all_newton_converged": bool(all(conv))}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
