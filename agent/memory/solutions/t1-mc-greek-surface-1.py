# memory-solution
# unit: t1-mc-greek-surface-1 | category: derivatives-pricing | tags: monte-carlo, greeks, pathwise, likelihood-ratio, finite-difference, common-random-numbers, asian-option, black-scholes, convergence, surface-plot
# deliverables: prices.json, greeks.json, consistency.json, delta_surface.csv, vega_surface.csv, convergence.csv, delta_surface.png, summary.json
# One RandomState(42) normal matrix Z (500k x 12): Asian paths S_i = S0 exp((r-q-s^2/2) t_i + s W_i), European uses S_12 (= S_T).
# FD Greeks with common random numbers (same Z for every bump): delta/gamma h=0.5, vega h=0.01, theta -(V(T-h)-V(T))/h with h=1/252, rho h=0.001.
# Pathwise (EU): delta = e^{-rT} 1{ITM} S_T/S0, vega = e^{-rT} 1{ITM} S_T (sqrt(T) Z - s T), rho = e^{-rT} T K 1{ITM}, theta = dV/dT (sign of FD theta).
# LR (EU): weights on discounted payoff: delta Z/(S0 s sqrtT); gamma (Z^2 - 1 - Z s sqrtT)/(S0 s sqrtT)^2; vega (Z^2-1)/s - Z sqrtT; rho Z sqrtT/s - T.
# Asian pathwise: delta = e^{-rT} 1{A>K} A/S0, vega = e^{-rT} 1{A>K} mean_i S_i (W_i - s t_i); Asian LR via first-step score (delta/gamma) and summed scores.
# Unavailable Greeks (PW gamma, LR theta, Asian PW gamma/theta/rho) are JSON null. Convergence rows use greek name "delta_call" (checker filters on it).
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

S0, K, r, q, sig, T = 100.0, 100.0, 0.05, 0.02, 0.30, 1.0
N, M = 500_000, 12
rng = np.random.RandomState(42)
Z = rng.standard_normal((N, M))                      # monthly increments
ZT = Z.sum(axis=1) / math.sqrt(M)                    # terminal standard normal (W_T / sqrt(T))
OPTS = ["european_call", "european_put", "asian_call", "asian_put"]

def bs(S, vol, tau, rr, kind):
    d1 = (math.log(S / K) + (rr - q + 0.5 * vol * vol) * tau) / (vol * math.sqrt(tau)); d2 = d1 - vol * math.sqrt(tau)
    e_q, e_r = math.exp(-q * tau), math.exp(-rr * tau)
    if kind == "call":
        return {"price": S * e_q * norm.cdf(d1) - K * e_r * norm.cdf(d2), "delta": e_q * norm.cdf(d1), "rho": K * tau * e_r * norm.cdf(d2),
                "gamma": e_q * norm.pdf(d1) / (S * vol * math.sqrt(tau)), "vega": S * e_q * norm.pdf(d1) * math.sqrt(tau),
                "theta": S * e_q * norm.pdf(d1) * vol / (2 * math.sqrt(tau)) - q * S * e_q * norm.cdf(d1) + rr * K * e_r * norm.cdf(d2)}
    return {"price": K * e_r * norm.cdf(-d2) - S * e_q * norm.cdf(-d1), "delta": e_q * (norm.cdf(d1) - 1.0), "rho": -K * tau * e_r * norm.cdf(-d2),
            "gamma": e_q * norm.pdf(d1) / (S * vol * math.sqrt(tau)), "vega": S * e_q * norm.pdf(d1) * math.sqrt(tau),
            "theta": S * e_q * norm.pdf(d1) * vol / (2 * math.sqrt(tau)) + q * S * e_q * norm.cdf(-d1) - rr * K * e_r * norm.cdf(-d2)}

def disc_payoffs(S=S0, vol=sig, tau=T, rr=r):
    """Discounted payoff vectors of the 4 options for the common normals Z (common random numbers)."""
    dt = tau / M; t = dt * np.arange(1, M + 1)
    W = np.cumsum(Z, axis=1) * math.sqrt(dt)
    Spath = S * np.exp((rr - q - 0.5 * vol * vol) * t + vol * W)
    ST, A = Spath[:, -1], Spath.mean(axis=1); df = math.exp(-rr * tau)
    return {"european_call": df * np.maximum(ST - K, 0.0), "european_put": df * np.maximum(K - ST, 0.0),
            "asian_call": df * np.maximum(A - K, 0.0), "asian_put": df * np.maximum(K - A, 0.0)}

def mean_se(x): return {"value": float(np.mean(x)), "se": float(np.std(x, ddof=1) / math.sqrt(len(x)))}
NULL = {"value": None, "se": None}

base = disc_payoffs()
prices = {o: {"mc_price": mean_se(base[o])["value"], "mc_se": mean_se(base[o])["se"]} for o in OPTS}
prices["european_call"]["bs_price"] = bs(S0, sig, T, r, "call")["price"]; prices["european_put"]["bs_price"] = bs(S0, sig, T, r, "put")["price"]
json.dump(prices, open(os.path.join(OUT_DIR, "prices.json"), "w"), indent=2)

# ---- finite differences (per-path estimators so that SE is available)
hS, hV, hT, hR = 0.5, 0.01, 1.0 / 252.0, 0.001
up, dn = disc_payoffs(S=S0 + hS), disc_payoffs(S=S0 - hS)
vup, vdn = disc_payoffs(vol=sig + hV), disc_payoffs(vol=sig - hV)
tdn = disc_payoffs(tau=T - hT)
rup, rdn = disc_payoffs(rr=r + hR), disc_payoffs(rr=r - hR)
fd = {o: {"delta": (up[o] - dn[o]) / (2 * hS), "gamma": (up[o] - 2 * base[o] + dn[o]) / hS ** 2, "vega": (vup[o] - vdn[o]) / (2 * hV),
          "theta": -(tdn[o] - base[o]) / hT, "rho": (rup[o] - rdn[o]) / (2 * hR)} for o in OPTS}

# ---- pathwise and likelihood-ratio estimators (per path)
dt = T / M; tg = dt * np.arange(1, M + 1); W = np.cumsum(Z, axis=1) * math.sqrt(dt)
Spath = S0 * np.exp((r - q - 0.5 * sig * sig) * tg + sig * W); ST, A = Spath[:, -1], Spath.mean(axis=1)
df, sqT = math.exp(-r * T), math.sqrt(T)
pw, lr = {o: {} for o in OPTS}, {o: {} for o in OPTS}
for o, sgn in [("european_call", 1.0), ("european_put", -1.0)]:
    itm = (sgn * (ST - K) > 0).astype(float)
    pw[o]["delta"] = df * itm * sgn * ST / S0
    pw[o]["vega"] = df * itm * sgn * ST * (sqT * ZT - sig * T)
    pw[o]["rho"] = -T * base[o] + df * itm * sgn * ST * T
    pw[o]["theta"] = -r * base[o] + df * itm * sgn * ST * ((r - q - 0.5 * sig * sig) + sig * ZT / (2 * sqT))
    lr[o]["delta"] = base[o] * ZT / (S0 * sig * sqT)
    lr[o]["gamma"] = base[o] * (ZT ** 2 - 1.0 - ZT * sig * sqT) / (S0 * sig * sqT) ** 2
    lr[o]["vega"] = base[o] * ((ZT ** 2 - 1.0) / sig - ZT * sqT)
    lr[o]["rho"] = base[o] * (ZT * sqT / sig - T)
for o, sgn in [("asian_call", 1.0), ("asian_put", -1.0)]:
    itm = (sgn * (A - K) > 0).astype(float); sdt = math.sqrt(dt)
    pw[o]["delta"] = df * itm * sgn * A / S0
    pw[o]["vega"] = df * itm * sgn * (Spath * (W - sig * tg)).mean(axis=1)
    lr[o]["delta"] = base[o] * Z[:, 0] / (S0 * sig * sdt)                      # density depends on S0 only through the first step
    lr[o]["gamma"] = base[o] * (Z[:, 0] ** 2 - 1.0 - Z[:, 0] * sig * sdt) / (S0 * sig * sdt) ** 2
    lr[o]["vega"] = base[o] * (((Z ** 2 - 1.0) / sig - Z * sdt).sum(axis=1))
    lr[o]["rho"] = base[o] * (Z.sum(axis=1) * sdt / sig - T)

greeks = {o: {g: {"fd": mean_se(fd[o][g]), "pathwise": mean_se(pw[o][g]) if g in pw[o] else NULL,
                  "lr": mean_se(lr[o][g]) if g in lr[o] else NULL} for g in ["delta", "gamma", "vega", "theta", "rho"]} for o in OPTS}
json.dump(greeks, open(os.path.join(OUT_DIR, "greeks.json"), "w"), indent=2)

# ---- consistency checks
bsc, bsp = bs(S0, sig, T, r, "call"), bs(S0, sig, T, r, "put")
g = lambda o, gr, m="fd": greeks[o][gr][m]["value"]
bs_err = {f"{gr}_{k}": float(abs(g(f"european_{k}", gr) - ref[gr]) / abs(ref[gr])) for k, ref in [("call", bsc), ("put", bsp)] for gr in ["delta", "gamma", "vega", "rho"]}
pcp = {"delta_diff": g("european_call", "delta") - g("european_put", "delta"), "gamma_diff": g("european_call", "gamma") - g("european_put", "gamma"),
       "rho_diff": g("european_call", "rho") - g("european_put", "rho")}
var = {"pw_variance": float(np.var(pw["european_call"]["delta"], ddof=1)), "lr_variance": float(np.var(lr["european_call"]["delta"], ddof=1))}
var["pw_se"], var["lr_se"] = math.sqrt(var["pw_variance"] / N), math.sqrt(var["lr_variance"] / N)
avse = {"price_call": bool(prices["asian_call"]["mc_price"] < prices["european_call"]["mc_price"]),
        "price_put": bool(prices["asian_put"]["mc_price"] < prices["european_put"]["mc_price"]),
        "delta_call": bool(g("asian_call", "delta") < g("european_call", "delta"))}
json.dump({"bs_errors": bs_err, "put_call_parity": pcp, "variance_ratio": var, "asian_vs_european": avse,
           "bs_reference": {"call": bsc, "put": bsp}}, open(os.path.join(OUT_DIR, "consistency.json"), "w"), indent=2)

# ---- closed-form delta / vega surfaces + 3D plot
Sg, Vg = np.linspace(70, 130, 13), np.linspace(0.10, 0.50, 9)
rows = [{"S0": s, "sigma": v, **{f"{nm}_eu": bs(s, v, T, r, kd)[gr] for gr in ["delta", "vega"] for nm, kd in [(f"{gr}_call", "call"), (f"{gr}_put", "put")]}} for s in Sg for v in Vg]
surf = pd.DataFrame(rows)
surf[["S0", "sigma", "delta_call_eu", "delta_put_eu"]].to_csv(os.path.join(OUT_DIR, "delta_surface.csv"), index=False)
surf[["S0", "sigma", "vega_call_eu", "vega_put_eu"]].to_csv(os.path.join(OUT_DIR, "vega_surface.csv"), index=False)
SS, VV = np.meshgrid(Sg, Vg, indexing="ij")
fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(SS, VV, surf["delta_call_eu"].to_numpy().reshape(13, 9), cmap="viridis")
ax.set_xlabel("S0"); ax.set_ylabel("sigma"); ax.set_zlabel("call delta"); ax.set_title("European call delta (BS)")
fig.savefig(os.path.join(OUT_DIR, "delta_surface.png"), dpi=100); plt.close(fig)

# ---- convergence of the European call delta (nested subsets of the same paths)
conv = []
for n in [10_000, 50_000, 100_000, 200_000, 500_000]:
    for meth, vec in [("fd", fd["european_call"]["delta"]), ("pathwise", pw["european_call"]["delta"]), ("lr", lr["european_call"]["delta"])]:
        ms = mean_se(vec[:n]); conv.append({"N": n, "method": meth, "greek": "delta_call", "value": ms["value"], "se": ms["se"]})
pd.DataFrame(conv).to_csv(os.path.join(OUT_DIR, "convergence.csv"), index=False)

# ---- summary flags
within = lambda a, b, tol: abs(a - b) / abs(b) < tol
dsurf = surf["delta_call_eu"].to_numpy().reshape(13, 9)
flags = {"prices_match_bs": bool(abs(prices["european_call"]["mc_price"] - bsc["price"]) < 0.5 and abs(prices["european_put"]["mc_price"] - bsp["price"]) < 0.5),
         "fd_greeks_within_5pct": bool(max(bs_err.values()) < 0.05),
         "pw_lr_delta_within_5pct": bool(within(g("european_call", "delta", "pathwise"), bsc["delta"], 0.05) and within(g("european_call", "delta", "lr"), bsc["delta"], 0.05)),
         "lr_gamma_within_10pct": bool(within(g("european_call", "gamma", "lr"), bsc["gamma"], 0.10)),
         "pcp_delta": bool(abs(pcp["delta_diff"] - math.exp(-q * T)) < 0.005), "pcp_gamma": bool(abs(pcp["gamma_diff"]) < 0.001),
         "pcp_rho": bool(abs(pcp["rho_diff"] - K * T * math.exp(-r * T)) < 1.0),
         "pw_variance_below_lr": bool(var["pw_variance"] < var["lr_variance"]),
         "asian_below_european": bool(all(avse.values())),
         "delta_surface_monotone": bool(np.all(np.diff(dsurf, axis=0) >= -1e-12)), "vega_surface_positive": bool((surf["vega_call_eu"] > 0).all()),
         "convergence_ok": bool(all(abs(c["value"] - bsc["delta"]) < 0.01 for c in conv if c["N"] == 500_000))}
flags["all_checks_passed"] = bool(all(flags.values()))
json.dump(flags, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
