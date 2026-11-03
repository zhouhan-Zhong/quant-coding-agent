# memory-solution
# unit: t1-localvol-barrier | category: volatility | tags: deribit, btc-options, quote-cleaning, put-call-parity, forward-inference, synthetic-calls, dupire, numpy-gradient, local-vol-monte-carlo, down-and-out, antithetic, daily-monitoring
# deliverables: cleaned_quotes.csv, surface_nodes.csv, call_surface.csv, local_vol_surface.csv, barrier_results.json, summary.json
# Cleaning order: dedupe (keep latest creation_timestamp per instrument) -> drop crossed (bid > ask) -> drop wide ((ask-bid)/mid > 0.95, mid > 0); expiry label = 2nd token of instrument_name.
# Pairing by (expiry, strike); forward per expiry from the pair with min |C_btc - P_btc| via BTC-unit parity: C_btc - P_btc = (F - K)/F  =>  F = K / (1 - (C_btc - P_btc)).
# premium_usd = premium_btc * F; synthetic call = P_usd + F - K for K < F ("put_parity"), direct call mid for K >= F ("call_direct"); r = q = 0, DF = 1.
# ttm = (expiration_ts - snapshot_ts) / (1000*86400*365); total_variance = BS implied var * T (brentq on synthetic call with S = F, r = 0).
# Dupire on the 5x7 target grid with np.gradient(edge_order=2): var = 2 C_T / (K^2 C_KK), C_KK = gradient applied twice; floor var 1e-8, cap vol 3.0.
# MC: S0 = spot_proxy, zero drift, log-Euler with bilinear local vol (clipped to the grid), ceil(T*365) days x 4 sub-steps, antithetic pairs, barrier checked only at daily closes.
import os, json
import numpy as np, pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

rules = json.load(open(os.path.join(DATA_DIR, "task_rules.json")))
meta = json.load(open(os.path.join(DATA_DIR, "snapshot_metadata.json")))
bc = json.load(open(os.path.join(DATA_DIR, "barrier_contract.json")))
raw = pd.read_csv(os.path.join(DATA_DIR, "deribit_btc_option_snapshot.csv"))
n_raw = len(raw); snap = int(meta["snapshot_ts"])

# ---- cleaning ----
dd = raw.sort_values("creation_timestamp").drop_duplicates("instrument_name", keep="last")
n_dup = n_raw - len(dd)
crossed = dd["bid_price_btc"] > dd["ask_price_btc"]
wide = ((dd["ask_price_btc"] - dd["bid_price_btc"]) / dd["mid_price_btc"] > rules["max_relative_spread"]) & (dd["mid_price_btc"] > 0)
valid = dd["bid_price_btc"] >= rules["min_bid_btc"]
cl = dd[~crossed & ~wide & valid].copy()
cl["expiry"] = cl["instrument_name"].str.split("-").str[1]
cl["ttm_years"] = (cl["expiration_timestamp"] - snap) / (1000.0 * 86400.0 * 365.0)
cl = cl.sort_values(["expiration_timestamp", "strike", "option_type"])
cl[["instrument_name", "expiry", "option_type", "strike", "bid_price_btc", "ask_price_btc", "mid_price_btc", "mark_iv_pct",
    "underlying_price_usd", "open_interest", "volume_usd", "creation_timestamp"]].to_csv(os.path.join(OUT_DIR, "cleaned_quotes.csv"), index=False)

# ---- pairing + forward inference + synthetic calls ----
calls = cl[cl["option_type"] == "call"].set_index(["expiry", "strike"]); puts = cl[cl["option_type"] == "put"].set_index(["expiry", "strike"])
idx = calls.index.intersection(puts.index)
pairs = pd.DataFrame({"C": calls.loc[idx, "mid_price_btc"], "P": puts.loc[idx, "mid_price_btc"], "ttm_years": calls.loc[idx, "ttm_years"]}).reset_index()
n_unmatched = len(cl) - 2 * len(pairs)

def bs_call(F, K, T, s):
    sq = s * np.sqrt(T); d1 = (np.log(F / K) + 0.5 * sq * sq) / sq
    return F * norm.cdf(d1) - K * norm.cdf(d1 - sq)

nodes = []
for e, g in pairs.groupby("expiry", sort=False):
    atm = g.loc[(g["C"] - g["P"]).abs().idxmin()]
    F = float(atm["strike"] / (1.0 - (atm["C"] - atm["P"])))
    for _, r in g.iterrows():
        K, T = float(r["strike"]), float(r["ttm_years"]); cu, pu = r["C"] * F, r["P"] * F
        side = "put_parity" if K < F else "call_direct"
        syn = pu + F - K if K < F else cu
        tv = syn - max(F - K, 0.0)
        iv = brentq(lambda s: bs_call(F, K, T, s) - syn, 1e-4, 10.0) if tv > 1e-8 else 0.0
        nodes.append({"expiry": e, "ttm_years": T, "strike": int(K), "forward_usd": F, "call_mid_usd": cu, "put_mid_usd": pu, "otm_side": side,
                      "synthetic_call_usd": syn, "log_moneyness": np.log(K / F), "total_variance": iv * iv * T})
nodes = pd.DataFrame(nodes).sort_values(["ttm_years", "strike"]).reset_index(drop=True)
nodes.to_csv(os.path.join(OUT_DIR, "surface_nodes.csv"), index=False)

# ---- target call surface with monotone/convex/intrinsic repair ----
TE, TK = rules["target_expiries"], np.array(rules["target_strikes"], float)
def repair(C, K, F):
    C = np.maximum(C.copy(), np.maximum(F - K, 0.0))
    for _ in range(200):
        prev = C.copy()
        for i in range(1, len(K) - 1):                     # lower any point above the chord of its neighbours
            w = (K[i] - K[i - 1]) / (K[i + 1] - K[i - 1]); C[i] = min(C[i], (1 - w) * C[i - 1] + w * C[i + 1])
        C = np.minimum.accumulate(C)                        # non-increasing in strike
        if np.allclose(C, prev): break
    return C
rows, Cmat, ttms, fwd = [], [], [], {}
for e in TE:
    g = nodes[nodes["expiry"] == e].set_index("strike"); F = float(g["forward_usd"].iloc[0]); T = float(g["ttm_years"].iloc[0])
    C = repair(g.loc[TK.astype(int), "synthetic_call_usd"].to_numpy(float), TK, F)
    Cmat.append(C); ttms.append(T); fwd[e] = F
    rows += [{"expiry": e, "ttm_years": T, "strike": int(k), "call_price_usd": float(c)} for k, c in zip(TK, C)]
surf = pd.DataFrame(rows).sort_values(["ttm_years", "strike"]).reset_index(drop=True)
surf.to_csv(os.path.join(OUT_DIR, "call_surface.csv"), index=False)
Cmat, ttms = np.array(Cmat), np.array(ttms)
order = np.argsort(ttms); Cmat, ttms, TE = Cmat[order], ttms[order], [TE[i] for i in order]

# ---- Dupire local vol on the grid ----
CT = np.gradient(Cmat, ttms, axis=0, edge_order=2)
CK = np.array([np.gradient(Cmat[i], TK, edge_order=2) for i in range(len(TE))])
CKK = np.array([np.gradient(np.gradient(Cmat[i], TK, edge_order=2), TK, edge_order=2) for i in range(len(TE))])
with np.errstate(divide="ignore", invalid="ignore"):
    lvar = 2.0 * CT / (TK[None, :] ** 2 * CKK)
lvar = np.where(np.isfinite(lvar), lvar, 0.0)
lvol = np.minimum(np.sqrt(np.maximum(lvar, rules["local_var_floor"])), rules["local_vol_cap"])
lv_rows = [{"expiry": TE[i], "ttm_years": ttms[i], "strike": int(TK[j]), "dC_dT": CT[i, j], "dC_dK": CK[i, j], "d2C_dK2": CKK[i, j], "local_vol": lvol[i, j]}
           for i in range(len(TE)) for j in range(len(TK))]
pd.DataFrame(lv_rows).to_csv(os.path.join(OUT_DIR, "local_vol_surface.csv"), index=False)

# ---- local-vol Monte Carlo: vanilla + down-and-out call, daily-close monitoring ----
S0, K, H = float(meta["spot_proxy_usd"]), float(bc["strike"]), float(bc["barrier"])
im = TE.index(bc["maturity_expiry"]); T = ttms[im]; n_paths, nspd = int(bc["n_paths"]), int(bc["n_steps_per_day"])
n_days = int(np.ceil(T * 365.0)); dt = T / (n_days * nspd); half = n_paths // 2
interp = RegularGridInterpolator((ttms, TK), lvol, method="linear", bounds_error=False, fill_value=None)
rng = np.random.default_rng(int(bc["random_seed"]))
lnS = np.full(2 * half, np.log(S0)); alive = np.ones(2 * half, bool)
for step in range(n_days * nspd):
    Z = rng.standard_normal(half); Z = np.concatenate([Z, -Z])
    t = np.clip(step * dt, ttms[0], ttms[-1]); S = np.clip(np.exp(lnS), TK[0], TK[-1])
    sig = interp(np.column_stack([np.full(2 * half, t), S]))
    lnS += -0.5 * sig * sig * dt + sig * np.sqrt(dt) * Z
    if (step + 1) % nspd == 0:
        alive &= np.exp(lnS) > H
ST = np.exp(lnS); van = np.maximum(ST - K, 0.0); dob = van * alive
se = lambda x: float((0.5 * (x[:half] + x[half:])).std(ddof=1) / np.sqrt(half))
surf_van = float(surf[(surf["expiry"] == bc["maturity_expiry"]) & (surf["strike"] == int(bc["vanilla_comparison_strike"]))]["call_price_usd"].iloc[0])
mc_van, mc_dob, p_hit = float(van.mean()), float(dob.mean()), float(1.0 - alive.mean())
bres = {"spot_proxy_usd": S0, "maturity_expiry": bc["maturity_expiry"], "strike": K, "barrier": H, "n_steps_per_day": nspd,
        "variance_reduction": "antithetic", "barrier_monitoring_mode": "daily_close_only", "forward_at_maturity": fwd[bc["maturity_expiry"]],
        "surface_vanilla_call_usd": surf_van, "localvol_mc_vanilla_call_usd": mc_van, "localvol_mc_vanilla_call_se": se(van),
        "down_and_out_call_usd": mc_dob, "down_and_out_call_se": se(dob), "barrier_hit_probability": p_hit, "survival_probability": 1.0 - p_hit,
        "knockout_discount_vs_vanilla_pct": (1.0 - mc_dob / mc_van) * 100.0}
json.dump(bres, open(os.path.join(OUT_DIR, "barrier_results.json"), "w"), indent=2)

slopes = [np.diff(Cmat[i]) / np.diff(TK) for i in range(len(TE))]
flags = {"call_surface_monotone_in_strike": bool(all(np.all(s <= 1e-8) for s in slopes)),
         "call_surface_convex_in_strike": bool(all(np.all(np.diff(s) >= -1e-8) for s in slopes)),
         "local_vol_positive": bool(np.all(lvol > 0)), "barrier_leq_vanilla": bool(mc_dob <= mc_van + 1e-8),
         "barrier_nonnegative": bool(mc_dob >= 0), "vanilla_mc_close_to_surface": bool(abs(mc_van - surf_van) < 450)}
summary = {"n_raw_rows": int(n_raw), "n_clean_rows": int(len(cl)), "n_paired_nodes": int(len(nodes)), "n_surface_nodes": int(len(surf)),
           "n_localvol_nodes": int(len(lv_rows)), "duplicates_removed": int(n_dup), "crossed_rows_removed": int(crossed.sum()),
           "wide_spread_rows_removed": int((wide & ~crossed).sum()), "unmatched_rows_removed": int(n_unmatched), **flags,
           "all_checks_passed": bool(all(flags.values()))}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
