# memory-solution
# unit: t1-stochvol-implied-surface-new | category: derivatives-pricing | tags: two-factor-heston, christoffersen-heston-jacobs, chiarella-ziveyi, characteristic-function, gauss-legendre, quad, implied-vol-surface, brentq, put-call-parity, dupire-local-vol, finite-differences, 3d-plot
# deliverables: option_prices.csv, iv_surface.csv, iv_surface.png, smile_slices.png, integration_comparison.csv, put_prices.csv, put_parity_check.csv, local_vol_surface.csv, local_vol_surface.png, summary.json
# Two independent CIR variance factors -> CF of ln S_T factorises: phi(u) = exp(iu(ln S0 + (r-q)T)) * prod_j exp(C_j + D_j v_j0)
#   with the Heston "little trap" C_j, D_j (xi = kappa - rho sigma iu, d = sqrt(xi^2 + sigma^2(u^2 + iu)), g = (xi-d)/(xi+d)).
# Price: C = S e^{-qT} P1 - K e^{-rT} P2, P_j = 1/2 + (1/pi) int Re[e^{-iu lnK} phi_j(u)/(iu)] du, phi_1 = phi(u-i)/phi(-i).
# GL(2000 nodes, [0,100]) vs scipy quad([1e-8,50], limit=200): quad truncates the tail -> diffs > 1e-5 at T=1/12, so GL is chosen
#   (decision made against a high-accuracy reference integral). Direct put: P = K e^{-rT}(1-P2) - S e^{-qT}(1-P1).
# IV by brentq on [1e-4, 5] (xtol=1e-8). Dupire on the discrete grid: C_T, C_K via np.gradient (one-sided at edges), C_KK central,
#   sigma_loc^2 = (C_T + r K C_K)/(0.5 K^2 C_KK) at interior strikes; den < 1e-15 -> NaN, floor variance at 1e-10.
# Speed: leggauss(2000) is an O(n^3) eigenproblem (~1 s) -> compute the nodes once and cache; use np.exp (not math.exp) on arrays.
import os, json, math, csv
import numpy as np, pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

S0, r, q = 110.0, 0.03, 0.0
k1, k2 = 2.0 ** (-1.5), 2.0 ** 1.5
FACTORS = [(k1, 0.04, math.sqrt(k1 * 0.04), -0.5, 0.04), (k2, 0.04, math.sqrt(k2 * 0.04), -0.99, 0.04)]  # kappa, theta, sigma, rho, v0
K_grid = np.linspace(79, 165, 12)
T_grid = np.linspace(1 / 12, 1.0, 12)

def cf(u, T):
    """Characteristic function of ln S_T for the two-factor model (u may be complex / array)."""
    u = np.asarray(u, dtype=complex)
    out = 1j * u * (math.log(S0) + (r - q) * T)
    for kap, th, sig, rho, v0 in FACTORS:
        xi = kap - rho * sig * 1j * u
        d = np.sqrt(xi ** 2 + sig ** 2 * (u ** 2 + 1j * u))
        g = (xi - d) / (xi + d)
        e = np.exp(-d * T)
        out = out + kap * th / sig ** 2 * ((xi - d) * T - 2 * np.log((1 - g * e) / (1 - g))) + (xi - d) / sig ** 2 * (1 - e) / (1 - g * e) * v0
    return np.exp(out)

def integrands(u, T, lnK):
    """Re parts of the P1 and P2 integrands at frequencies u (array) for strikes lnK (array) -> (len(u), len(lnK))."""
    F = S0 * math.exp((r - q) * T)
    ph2 = cf(u, T); ph1 = cf(u - 1j, T) / F
    ex = np.exp(-1j * np.outer(u, lnK))
    return (ex * (ph1 / (1j * u))[:, None]).real, (ex * (ph2 / (1j * u))[:, None]).real

_GL_CACHE = {}
def probs_gl(T, Ks, n=2000, upper=100.0):
    if n not in _GL_CACHE:                                   # leggauss(2000) costs ~1 s (eigenvalue problem): compute once
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    nodes, weights = _GL_CACHE[n]
    u = 0.5 * (nodes + 1) * upper; w = 0.5 * upper * weights
    f1, f2 = integrands(u, T, np.log(Ks))
    return 0.5 + w @ f1 / math.pi, 0.5 + w @ f2 / math.pi

def probs_quad(T, K, lo=1e-8, hi=50.0, limit=200):
    lnK = np.array([math.log(K)])
    I1 = quad(lambda x: integrands(np.array([x]), T, lnK)[0][0, 0], lo, hi, limit=limit)[0]
    I2 = quad(lambda x: integrands(np.array([x]), T, lnK)[1][0, 0], lo, hi, limit=limit)[0]
    return 0.5 + I1 / math.pi, 0.5 + I2 / math.pi

def call_put(P1, P2, K, T):
    disc, fwd = K * math.exp(-r * T), S0 * math.exp(-q * T)
    return fwd * P1 - disc * P2, disc * (1 - P2) - fwd * (1 - P1)

# --- Step 2: integration method comparison at the shortest maturity -------------------------------------------
T0 = T_grid[0]
P1g, P2g = probs_gl(T0, K_grid)
P1ref, P2ref = probs_gl(T0, K_grid, n=2000, upper=200.0)          # high-accuracy reference (wider range) for the choice
rows_cmp, err_quad, err_gl = [], [], []
for i, K in enumerate(K_grid):
    p_gl = call_put(P1g[i], P2g[i], K, T0)[0]
    p_q = call_put(*probs_quad(T0, K), K, T0)[0]
    p_ref = call_put(P1ref[i], P2ref[i], K, T0)[0]
    err_quad.append(abs(p_q - p_ref)); err_gl.append(abs(p_gl - p_ref))
    rows_cmp.append([K, T0, p_q, p_gl, abs(p_q - p_gl)])
if max(err_gl) < 1e-5 and max(err_quad) >= 1e-5:
    chosen = "gauss_legendre"
elif max(err_quad) < 1e-5 and max(err_gl) >= 1e-5:
    chosen = "scipy_quad"
else:
    chosen = "gauss_legendre" if max(err_gl) <= max(err_quad) else "scipy_quad"
with open(os.path.join(OUT_DIR, "integration_comparison.csv"), "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["K", "T", "price_quad", "price_gl", "abs_diff"])
    for row in rows_cmp:
        wr.writerow([f"{row[0]:.6f}", f"{row[1]:.6f}", f"{row[2]:.10f}", f"{row[3]:.10f}", f"{row[4]:.3e}"])
    wr.writerow(["chosen_method", chosen, "", "", ""])

# --- Steps 1, 3, 4: price surface, implied vols, puts ----------------------------------------------------------
calls = np.zeros((12, 12)); puts = np.zeros((12, 12))
for j, T in enumerate(T_grid):
    if chosen == "gauss_legendre":
        P1, P2 = probs_gl(T, K_grid)
    else:
        P1, P2 = map(np.array, zip(*[probs_quad(T, K) for K in K_grid]))
    for i, K in enumerate(K_grid):
        calls[i, j], puts[i, j] = call_put(P1[i], P2[i], K, T)

def bs_call(K, T, s):
    sq = s * math.sqrt(T); d1 = (math.log(S0 / K) + (r - q + 0.5 * s * s) * T) / sq
    return S0 * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d1 - sq)

def implied_vol(price, K, T):
    f = lambda s: bs_call(K, T, s) - price
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-8)
    except ValueError:
        return float("nan")

iv = np.array([[implied_vol(calls[i, j], K_grid[i], T_grid[j]) for j in range(12)] for i in range(12)])
put_parity = calls - S0 * np.exp(-q * T_grid)[None, :] + K_grid[:, None] * np.exp(-r * T_grid)[None, :]
parity_diff = np.abs(puts - put_parity)

def save_matrix(mat, Ks, name):
    df = pd.DataFrame(mat, index=Ks, columns=T_grid); df.index.name = "K"
    df.to_csv(os.path.join(OUT_DIR, name), float_format="%.10g")

save_matrix(calls, K_grid, "option_prices.csv"); save_matrix(iv, K_grid, "iv_surface.csv"); save_matrix(puts, K_grid, "put_prices.csv")
pd.DataFrame({"K": np.repeat(K_grid, 12), "T": np.tile(T_grid, 12), "put_direct": puts.ravel(), "put_parity": put_parity.ravel(),
              "abs_diff": parity_diff.ravel()}).to_csv(os.path.join(OUT_DIR, "put_parity_check.csv"), index=False)

# --- Step 5: Dupire local vol on the discrete grid (interior strikes) ------------------------------------------
dK, dT = K_grid[1] - K_grid[0], T_grid[1] - T_grid[0]
C_T = np.gradient(calls, dT, axis=1)                       # forward/backward at T edges, central inside
C_K = np.gradient(calls, dK, axis=0)[1:-1]
C_KK = (calls[2:] - 2 * calls[1:-1] + calls[:-2]) / dK ** 2
K_int = K_grid[1:-1]
num = C_T[1:-1] + (r - q) * K_int[:, None] * C_K + q * calls[1:-1]
den = 0.5 * K_int[:, None] ** 2 * C_KK
local_vol = np.where(den < 1e-15, np.nan, np.sqrt(np.maximum(num / np.where(den < 1e-15, 1.0, den), 1e-10)))
save_matrix(local_vol, K_int, "local_vol_surface.csv")

# --- Plots ----------------------------------------------------------------------------------------------------
def surface_plot(Ks, Z, title, zlabel, name):
    KK, TT = np.meshgrid(Ks, T_grid, indexing="ij")
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(KK, TT, Z, cmap="viridis")
    ax.set_xlabel("Strike K"); ax.set_ylabel("Maturity T"); ax.set_zlabel(zlabel); ax.set_title(title)
    fig.savefig(os.path.join(OUT_DIR, name), dpi=100); plt.close(fig)

surface_plot(K_grid, iv, "IV Surface (K, T)", "Implied Volatility", "iv_surface.png")
surface_plot(K_int, local_vol, "Local Vol Surface (K, T)", "Local Volatility", "local_vol_surface.png")
fig, ax = plt.subplots(figsize=(8, 5))
for j in (0, 3, 6, 11):
    ax.plot(K_grid, iv[:, j], marker="o", label=f"T = {T_grid[j]:.4f}")
ax.set_xlabel("Strike K"); ax.set_ylabel("Implied Volatility"); ax.set_title("Volatility smiles"); ax.legend()
fig.savefig(os.path.join(OUT_DIR, "smile_slices.png"), dpi=100); plt.close(fig)

# --- Summary --------------------------------------------------------------------------------------------------
atm = int(np.argmin(np.abs(K_grid - S0))); atm_int = int(np.argmin(np.abs(K_int - S0))); j05 = int(np.argmin(np.abs(T_grid - 0.5)))
summary = {"chosen_method": chosen, "iv_min": float(np.nanmin(iv)), "iv_max": float(np.nanmax(iv)), "iv_mean": float(np.nanmean(iv)),
           "atm_iv_T0p5": float(iv[atm, j05]), "skew_T0p083": float(iv[0, 0] - iv[-1, 0]), "skew_T1p0": float(iv[0, -1] - iv[-1, -1]),
           "num_nan": int(np.isnan(iv).sum()), "put_parity_max_diff": float(parity_diff.max()),
           "local_vol_atm_T0p5": float(local_vol[atm_int, j05]),
           "local_vol_range": [float(np.nanmin(local_vol)), float(np.nanmax(local_vol))]}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
