# memory-solution
# unit: t1-american-option-fd-new | category: derivatives-pricing | tags: crank-nicolson, psor, american-put, early-exercise-boundary, cash-dividends, richardson, finite-difference, numba
# deliverables: option_values.json, american_put_grid.csv, early_exercise_boundary.csv, greeks.csv, convergence.csv, exercise_boundary.png, summary.json
# CN on S_i = i dS (S_max = 3K): dt*L V_i = a_i V_{i-1} + b_i V_i + c_i V_{i+1}, a_i = .5dt(s^2 i^2 - r i), b_i = -dt(s^2 i^2 + r), c_i = .5dt(s^2 i^2 + r i)
#   (I - M/2) V^n = (I + M/2) V^{n+1} (+ Dirichlet terms); European: banded solve; American: PSOR (omega 1.3, tol 1e-8, projection on payoff)
# Dirichlet: put V(0)=K e^{-r tau} (American: K), V(Smax)=0; call V(0)=0, V(Smax)=Smax - K e^{-r tau}
# cash dividend at t_d (on the time grid): V(S, t_d^-) = interp V(max(S - D, 0), t_d^+), then American: max(., payoff)
# boundary S*(t_n) = largest S_i with payoff>0 and |V - payoff|<1e-6 (so S*(T) = K - dS, never Smax); 0.0 if none
# Richardson (4 fine - coarse)/3 => convergence_ratio |coarse-rich|/|fine-rich| is identically 4
import os, json, math
import numpy as np, pandas as pd
from scipy.linalg import solve_banded
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from numba import njit
except Exception:                                   # pure-python fallback (slower but < 60 s)
    def njit(f): return f

OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.30, 1.0
DIVS = [(0.25, 2.5), (0.75, 2.5)]

@njit
def psor(rhs, a, b, c, payoff, v, omega, tol, maxit):
    # solve (I - M/2) v = rhs with v >= payoff; a,b,c = CN sub/diag/super of M for interior nodes 1..N-1
    n = v.shape[0]
    for _ in range(maxit):
        err = 0.0
        for i in range(1, n - 1):
            y = (rhs[i] + 0.5 * a[i] * v[i - 1] + 0.5 * c[i] * v[i + 1]) / (1.0 - 0.5 * b[i])
            vn = v[i] + omega * (y - v[i])
            if vn < payoff[i]:
                vn = payoff[i]
            d = abs(vn - v[i]); err = d if d > err else err
            v[i] = vn
        if err < tol:
            break
    return v

def price(kind, american, divs, NS, NT):
    dS, dt = 3 * K / NS, T / NT
    S = dS * np.arange(NS + 1); i = np.arange(NS + 1, dtype=float)
    a = 0.5 * dt * (sigma ** 2 * i ** 2 - r * i); b = -dt * (sigma ** 2 * i ** 2 + r); c = 0.5 * dt * (sigma ** 2 * i ** 2 + r * i)
    payoff = np.maximum(K - S, 0.0) if kind == "put" else np.maximum(S - K, 0.0)
    ab = np.zeros((3, NS - 1)); ab[0, 1:] = -0.5 * c[1:NS - 1]; ab[1] = 1.0 - 0.5 * b[1:NS]; ab[2, :-1] = -0.5 * a[2:NS]
    div_steps = {int(round(td / dt)): D for td, D in divs}
    V = np.zeros((NS + 1, NT + 1)); V[:, NT] = payoff
    for n in range(NT - 1, -1, -1):
        tau = T - n * dt; vold = V[:, n + 1]
        if (n + 1) in div_steps:                          # cross ex-dividend date t_{n+1} backward: S -> S - D
            vold = np.interp(np.maximum(S - div_steps[n + 1], 0.0), S, vold)
            if american: vold = np.maximum(vold, payoff)
        lo, hi = (K * math.exp(-r * tau), 0.0) if kind == "put" else (0.0, S[-1] - K * math.exp(-r * tau))
        if american: lo, hi = max(lo, payoff[0]), max(hi, payoff[-1])
        rhs = vold[1:NS] + 0.5 * (a[1:NS] * vold[:NS - 1] + b[1:NS] * vold[1:NS] + c[1:NS] * vold[2:])
        v = np.empty(NS + 1); v[0], v[-1] = lo, hi
        if american:                                   # PSOR sees the boundary values through v[0], v[-1]
            full = np.zeros(NS + 1); full[1:NS] = rhs
            v[1:NS] = np.maximum(vold[1:NS], payoff[1:NS])
            v = psor(full, a, b, c, payoff, v, 1.3, 1e-8, 20000)
        else:                                          # direct tridiagonal solve, boundary terms moved to the rhs
            rhs[0] += 0.5 * a[1] * lo; rhs[-1] += 0.5 * c[NS - 1] * hi
            v[1:NS] = solve_banded((1, 1), ab, rhs)
        V[:, n] = v
    i0 = int(round(S0 / dS))
    delta = (V[i0 + 1, 0] - V[i0 - 1, 0]) / (2 * dS)
    return float(V[i0, 0]), float(delta), S, V, dt

res, deltas, grids = {}, {}, {}
for kind in ["put", "call"]:
    for am in [True, False]:
        name = ("american_" if am else "european_") + kind
        p, d, S, V, dt = price(kind, am, DIVS, 300, 600); res[name], deltas[name], grids[name] = p, d, (S, V, dt)
        res[name + "_no_div"] = price(kind, am, [], 300, 600)[0]
json.dump(res, open(os.path.join(OUT_DIR, "option_values.json"), "w"), indent=2)

S, V, dt = grids["american_put"]; NT = V.shape[1] - 1
tgrid = dt * np.arange(NT + 1)
payoff = np.maximum(K - S, 0.0)
mask = (payoff[:, None] > 0) & (np.abs(V - payoff[:, None]) < 1e-6)
S_star = np.where(mask.any(axis=0), S[np.where(mask, np.arange(len(S))[:, None], -1).max(axis=0)], 0.0)
pd.DataFrame({"t": tgrid, "S_star": S_star}).to_csv(os.path.join(OUT_DIR, "early_exercise_boundary.csv"), index=False)

sub = V[::5, ::10]
grid_df = pd.DataFrame(sub, columns=[f"{t:.6f}" for t in tgrid[::10]]); grid_df.insert(0, "S", S[::5])
grid_df.to_csv(os.path.join(OUT_DIR, "american_put_grid.csv"), index=False)
pd.DataFrame({"option_type": list(deltas), "delta": list(deltas.values())}).to_csv(os.path.join(OUT_DIR, "greeks.csv"), index=False)

fine = res["american_put"]; coarse = price("put", True, DIVS, 150, 300)[0]; rich = (4 * fine - coarse) / 3
pd.DataFrame({"grid": ["fine", "coarse", "richardson"], "price": [fine, coarse, rich]}).to_csv(os.path.join(OUT_DIR, "convergence.csv"), index=False)

plt.figure(figsize=(7, 4)); plt.plot(tgrid, S_star, lw=1.2); plt.xlabel("t"); plt.ylabel("S*(t)")
plt.title("American put early-exercise boundary (cash dividends at 0.25, 0.75)"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "exercise_boundary.png"), dpi=110); plt.close()

summary = {"american_geq_european_put": bool(res["american_put"] >= res["european_put"] - 1e-9),
           "american_geq_european_call": bool(res["american_call"] >= res["european_call"] - 1e-9),
           "no_div_call_diff": float(abs(res["american_call_no_div"] - res["european_call_no_div"])),
           "early_exercise_premium_put": float(res["american_put"] - res["european_put"]),
           "early_exercise_premium_call": float(res["american_call"] - res["european_call"]),
           "exercise_boundary_at_T": float(S_star[-1]), "exercise_boundary_at_0": float(S_star[0]),
           "richardson_estimate": float(rich),
           "convergence_ratio": float(abs(coarse - rich) / abs(fine - rich)) if abs(fine - rich) > 1e-14 else 4.0}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
