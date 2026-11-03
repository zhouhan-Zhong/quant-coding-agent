# memory-solution
# unit: t1-dupire-local-vol | category: volatility | tags: dupire, local-volatility, svi, implied-vol-surface, newton-raphson, put-call-parity, gatheral-total-variance, forward-variance, calendar-arbitrage, flat-vol-test, spy-options
# deliverables: calibration.json, implied_vol_surface.csv, svi_params.csv, local_vol_surface.csv, flat_vol_test.csv, forward_variance.csv, summary.json
# S0 = daily close ON the valuation date 2024-03-15 (509.82), not the last row of the history. r = 0.05, D = 0.013, T = days/365.
# Options file has many rows per contract -> aggregate per (expiry, type, strike): volume-weighted price, summed volume; keep volume > 10,
#   OTM only (C: K > S0, P: K < S0), puts -> calls by parity C = P + S0 e^{-DT} - K e^{-rT}; IV by Newton (vega) + brentq fallback, keep [0.05, 1].
# SVI per expiry on k = ln(K/F): w = a + b (rho (k-m) + sqrt((k-m)^2 + s^2)), L-BFGS-B with bounds a[1e-6,.5] b[1e-4,1] rho[-.999,.999] m[-1,1] s[1e-3,1],
#   least squares on total variance (|k| <= 0.35), rmse reported in vol terms. Calendar penalty keeps w_i(k) >= w_{i-1}(k) on the grid.
# Dupire in Gatheral form: v_loc = w_T / [1 - (k/w) w_k + 1/4(-1/4 - 1/w + k^2/w^2) w_k^2 + w_kk/2], analytic SVI w_k, w_kk, forward difference
#   in T between consecutive SVI slices (backward for the last). Flat-vol test: price-form Dupire with FD on analytic BS prices (sigma = 0.20).
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, brentq

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
r, D = 0.05, 0.013
VAL_DATE = pd.Timestamp("2024-03-15")

daily = pd.read_csv(os.path.join(DATA_DIR, "spy_daily.csv"), parse_dates=["date"])
S0 = float(daily.loc[daily["date"] == VAL_DATE, "close"].iloc[0])

opt = pd.read_csv(os.path.join(DATA_DIR, "spy_options.csv"), parse_dates=["date", "expiry"])
opt["pv"] = opt["price"] * opt["volume"]
agg = opt.groupby(["expiry", "type", "strike"], as_index=False).agg(pv=("pv", "sum"), volume=("volume", "sum"))
agg["mid_price"] = agg["pv"] / agg["volume"]
agg = agg[agg["volume"] > 10].copy()
agg["T"] = (agg["expiry"] - VAL_DATE).dt.days / 365.0
agg = agg[(agg["T"] > 0) & (((agg["type"] == "C") & (agg["strike"] > S0)) | ((agg["type"] == "P") & (agg["strike"] < S0)))].copy()
agg["call_price"] = np.where(agg["type"] == "C", agg["mid_price"],
                             agg["mid_price"] + S0 * np.exp(-D * agg["T"]) - agg["strike"] * np.exp(-r * agg["T"]))

def bs_call(K, T, s):
    sq = s * math.sqrt(T); d1 = (math.log(S0 / K) + (r - D + 0.5 * s * s) * T) / sq
    return S0 * math.exp(-D * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d1 - sq)

def implied_vol(price, K, T):
    lo, hi = max(S0 * math.exp(-D * T) - K * math.exp(-r * T), 0.0), S0 * math.exp(-D * T)
    if not (lo < price < hi):
        return float("nan")
    v = math.sqrt(2 * math.pi / T) * price / S0 if price > 0 else 0.2          # Brenner-Subrahmanyam start
    v = min(max(v, 0.05), 1.5)
    for _ in range(50):
        sq = v * math.sqrt(T); d1 = (math.log(S0 / K) + (r - D + 0.5 * v * v) * T) / sq
        diff = bs_call(K, T, v) - price; vega = S0 * math.exp(-D * T) * norm.pdf(d1) * math.sqrt(T)
        if abs(diff) < 1e-10: return v
        if vega < 1e-10: break
        vn = v - diff / vega
        if not (1e-3 < vn < 3.0): break
        if abs(vn - v) < 1e-12: return vn
        v = vn
    try: return brentq(lambda vv: bs_call(K, T, vv) - price, 1e-3, 3.0)
    except ValueError: return float("nan")

agg["implied_vol"] = [implied_vol(p, K, T) for p, K, T in zip(agg["call_price"], agg["strike"], agg["T"])]
agg = agg[(agg["implied_vol"] >= 0.05) & (agg["implied_vol"] <= 1.0)].copy()
agg["k"] = np.log(agg["strike"] / (S0 * np.exp((r - D) * agg["T"])))

# --- expiry selection: liquid slices (>= 15 OTM strikes with |k| <= 0.35) nearest to a 1M..9M ladder ---
counts = agg[agg["k"].abs() <= 0.35].groupby("expiry").size()
cand = agg.groupby("expiry")["T"].first().loc[counts[counts >= 15].index]
cand = cand[cand >= 0.08]
targets = [0.10, 0.17, 0.27, 0.42, 0.52, 0.77]
chosen = []
for tg in targets:
    e = (cand - tg).abs().idxmin()
    if e not in chosen: chosen.append(e)
chosen = sorted(chosen)
sel = agg[agg["expiry"].isin(chosen)].sort_values(["T", "strike"])
iv_df = pd.DataFrame({"K": sel["strike"], "T": sel["T"], "expiry": sel["expiry"].dt.strftime("%Y-%m-%d"),
                      "implied_vol": sel["implied_vol"], "option_type": sel["type"], "mid_price": sel["mid_price"]})
iv_df.to_csv(os.path.join(OUT_DIR, "implied_vol_surface.csv"), index=False)
json.dump({"S0": S0, "r": r, "D": D, "n_expiries": len(chosen), "n_options_filtered": int(len(agg)),
           "expiry_dates": [pd.Timestamp(e).strftime("%Y-%m-%d") for e in chosen]},
          open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)

# --- SVI fit per expiry (raw SVI in total variance) with a calendar penalty vs the previous slice ---
def svi_w(p, k):
    a, b, rho, m, s = p; km = k - m
    return a + b * (rho * km + np.sqrt(km * km + s * s))
def svi_derivs(p, k):
    a, b, rho, m, s = p; km = k - m; R = np.sqrt(km * km + s * s)
    return b * (rho + km / R), b * s * s / R ** 3                     # w_k, w_kk

K_grid = S0 * np.linspace(0.85, 1.15, 13)
bounds = [(1e-6, 0.5), (1e-4, 1.0), (-0.999, 0.999), (-1.0, 1.0), (1e-3, 1.0)]
svi_rows, params, Ts = [], [], []
prev = None
for e in chosen:
    g = sel[(sel["expiry"] == e) & (sel["k"].abs() <= 0.35)]
    T = float(g["T"].iloc[0]); k = g["k"].to_numpy(); w_obs = (g["implied_vol"] ** 2 * T).to_numpy()
    kg = np.log(K_grid / (S0 * math.exp((r - D) * T)))
    def obj(p):
        res = np.mean((svi_w(p, k) - w_obs) ** 2)
        if prev is not None:
            res += 1e3 * np.mean(np.clip(svi_w(prev, kg) - svi_w(p, kg), 0, None) ** 2)   # calendar: w non-decreasing in T
        return res
    best = None
    for x0 in ([w_obs.min() * 0.5, 0.1, -0.5, 0.0, 0.1], [w_obs.min() * 0.8, 0.05, -0.7, -0.05, 0.05], [w_obs.min() * 0.3, 0.2, -0.3, 0.05, 0.2]):
        x0 = [min(max(v, lo), hi) for v, (lo, hi) in zip(x0, bounds)]
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun: best = res
    p = best.x; prev = p
    rmse = float(np.sqrt(np.mean((np.sqrt(np.maximum(svi_w(p, k), 1e-12) / T) - g["implied_vol"].to_numpy()) ** 2)))
    svi_rows.append({"expiry": pd.Timestamp(e).strftime("%Y-%m-%d"), "T": T, "a": p[0], "b": p[1], "rho": p[2], "m": p[3], "sigma": p[4], "rmse": rmse})
    params.append(p); Ts.append(T)
svi = pd.DataFrame(svi_rows); svi.to_csv(os.path.join(OUT_DIR, "svi_params.csv"), index=False)

# --- Dupire local vol (Gatheral total-variance form), forward difference in T between slices ---
def local_vol(i, K):
    T = Ts[i]; F = S0 * math.exp((r - D) * T); k = np.log(K / F)
    w = svi_w(params[i], k); w_k, w_kk = svi_derivs(params[i], k)
    j, l = (i, i + 1) if i + 1 < len(Ts) else (i - 1, i)
    w_T = (svi_w(params[l], k) - svi_w(params[j], k)) / (Ts[l] - Ts[j])
    den = 1 - (k / w) * w_k + 0.25 * (-0.25 - 1 / w + k * k / w ** 2) * w_k ** 2 + 0.5 * w_kk
    v = w_T / den
    return np.sqrt(np.where((den > 0) & (v > 0), v, np.nan)), np.sqrt(w / T)

lv_rows, fv_rows = [], []
for i, T in enumerate(Ts):
    lv, iv = local_vol(i, K_grid)
    for K, a_, b_ in zip(K_grid, lv, iv):
        lv_rows.append({"K": K, "T": T, "moneyness": K / S0, "local_vol": a_, "implied_vol": b_})
    lv_atm, iv_atm = local_vol(i, np.array([S0]))
    w_atm = iv_atm[0] ** 2 * T
    fwd = 0.0
    if i + 1 < len(Ts):
        w_next = (local_vol(i + 1, np.array([S0]))[1][0]) ** 2 * Ts[i + 1]
        fwd = (w_next - w_atm) / (Ts[i + 1] - T)
    fv_rows.append({"T": T, "total_variance": w_atm, "forward_variance": fwd, "local_vol_atm": lv_atm[0], "implied_vol_atm": iv_atm[0]})
lv_df = pd.DataFrame(lv_rows); lv_df.to_csv(os.path.join(OUT_DIR, "local_vol_surface.csv"), index=False)
fv_df = pd.DataFrame(fv_rows); fv_df.to_csv(os.path.join(OUT_DIR, "forward_variance.csv"), index=False)

# --- flat-vol identity check: price-form Dupire on analytic BS prices with sigma = 0.20 ---
SIG = 0.20
flat_rows = []
for T in (0.25, 0.5, 1.0):
    for kr in np.linspace(0.8, 1.2, 9):
        K = kr * S0; hK = 1e-3 * K; hT = 1e-4
        C = lambda KK, TT: bs_call(KK, TT, SIG)
        C_T = (C(K, T + hT) - C(K, T)) / hT
        C_K = (C(K + hK, T) - C(K - hK, T)) / (2 * hK)
        C_KK = (C(K + hK, T) - 2 * C(K, T) + C(K - hK, T)) / hK ** 2
        lv = math.sqrt((C_T + (r - D) * K * C_K + D * C(K, T)) / (0.5 * K * K * C_KK))
        flat_rows.append({"K_ratio": kr, "T": T, "local_vol": lv, "implied_vol": SIG, "error": abs(lv - SIG)})
flat = pd.DataFrame(flat_rows); flat.to_csv(os.path.join(OUT_DIR, "flat_vol_test.csv"), index=False)

i50 = int(np.argmin(np.abs(np.array(Ts) - 0.5)))
interior = fv_df["forward_variance"].iloc[:-1]
json.dump({"n_grid_points": int(len(lv_df)), "local_vol_all_positive": bool(np.isfinite(lv_df["local_vol"]).all() and (lv_df["local_vol"] > 0).all()),
           "max_flat_vol_error": float(flat["error"].max()), "avg_svi_rmse": float(svi["rmse"].mean()),
           "local_vol_atm_T050": float(fv_df["local_vol_atm"].iloc[i50]), "implied_vol_atm_T050": float(fv_df["implied_vol_atm"].iloc[i50]),
           "forward_var_positive": bool((interior > 0).all())},
          open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
