# memory-solution
# unit: t1-creditmetrics-portfolio-var | category: credit | tags: creditmetrics, credit-var, expected-shortfall, gaussian-copula, single-factor, rating-migration, monte-carlo, thresholds
# deliverables: portfolio_loss.json, loss_by_rating.csv, migration_stats.json, summary.json
# Thresholds per initial rating from the right tail: c_j = sum_{k>=j} p_k (Default first), t_j = Phi^{-1}(c_j);
#   Z < t_Default -> default, ..., Z >= t_AA -> AAA, i.e. new = 6 - #(ascending thresholds <= Z) via searchsorted.
# Z_i = sqrt(rho) M + sqrt(1-rho) eps_i, rho = 0.2; 200,000 sims in chunks; loss = 0.6 * #defaults (millions).
# VaR = np.quantile(loss, p) (discrete losses -> multiples of 0.6); ES = mean(loss[loss >= VaR]).
# Per-rating mean_loss is the category TOTAL loss per simulation (n_bonds * PD * LGD), prob_default per bond-sim pair.
# Migration rates over all bond-sim pairs: upgrade (better rating), downgrade (worse, not default), default.
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "Default"]
TM = np.array([[90.00, 8.00, 1.50, 0.30, 0.10, 0.05, 0.05],
               [1.00, 88.00, 8.00, 2.00, 0.70, 0.20, 0.10],
               [0.10, 2.00, 89.00, 6.00, 2.00, 0.70, 0.20],
               [0.05, 0.30, 4.00, 86.00, 6.00, 2.50, 1.15],
               [0.03, 0.10, 0.50, 5.00, 80.00, 10.00, 4.37],
               [0.01, 0.05, 0.20, 0.50, 6.00, 78.00, 15.24]]) / 100.0
PORTFOLIO = {"AAA": 20, "AA": 30, "A": 25, "BBB": 15, "BB": 10}
FACE, REC, RHO, N_SIM, SEED = 1.0, 0.40, 0.20, 200_000, 42
LGD = (1 - REC) * FACE

init = np.concatenate([[RATINGS.index(r)] * n for r, n in PORTFOLIO.items()])     # (100,)
n_b = len(init)
# ascending thresholds per initial rating: t[k] = Phi^{-1}(P(new rating index >= 6-k)), k = 0..5
thr = np.full((len(TM), len(RATINGS) - 1), 0.0)
for i, row in enumerate(TM):
    rc = np.cumsum(row[::-1])[:-1]                 # P(Default), P(>=B), ..., P(>=AA)
    thr[i] = norm.ppf(rc)                          # ascending in Z
bond_thr = thr[init]                               # (100, 6)

rng = np.random.default_rng(SEED)
losses, defaults_by_bond, up, down = np.empty(N_SIM), np.zeros(n_b), 0, 0
CH = 20_000
for s in range(0, N_SIM, CH):
    m = min(CH, N_SIM - s)
    M = rng.standard_normal((m, 1)); eps = rng.standard_normal((m, n_b))
    Z = math.sqrt(RHO) * M + math.sqrt(1 - RHO) * eps
    n_below = (Z[:, :, None] >= bond_thr[None, :, :]).sum(axis=2)          # thresholds <= Z
    new = (len(RATINGS) - 1) - n_below                                       # 0 = AAA ... 6 = Default
    d = new == len(RATINGS) - 1
    losses[s:s + m] = LGD * d.sum(axis=1)
    defaults_by_bond += d.sum(axis=0)
    up += int((new < init[None, :]).sum()); down += int(((new > init[None, :]) & ~d).sum())

q = lambda p: float(np.quantile(losses, p))
v95, v99, v999 = q(0.95), q(0.99), q(0.999)
es = lambda v: float(losses[losses >= v].mean())
json.dump({"mean_loss": float(losses.mean()), "std_loss": float(losses.std(ddof=1)), "var_95": v95, "var_99": v99,
           "var_999": v999, "es_95": es(v95), "es_99": es(v99)}, open(os.path.join(OUT_DIR, "portfolio_loss.json"), "w"), indent=2)

rows = []
for r, n in PORTFOLIO.items():
    mask = init == RATINGS.index(r)
    pd_r = defaults_by_bond[mask].sum() / (n * N_SIM)
    rows.append({"initial_rating": r, "n_bonds": n, "mean_loss": float(LGD * defaults_by_bond[mask].sum() / N_SIM),
                 "prob_default": float(pd_r)})
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "loss_by_rating.csv"), index=False)

pairs = n_b * N_SIM
json.dump({"upgrade_rate": up / pairs, "downgrade_rate": down / pairs, "default_rate": float(defaults_by_bond.sum() / pairs)},
          open(os.path.join(OUT_DIR, "migration_stats.json"), "w"), indent=2)
json.dump({"task": "CreditMetrics portfolio credit VaR via single-factor Gaussian copula Monte Carlo",
           "n_bonds": int(n_b), "n_simulations": N_SIM, "correlation": RHO, "recovery_rate": REC,
           "mean_loss": float(losses.mean()), "var_99": v99, "es_99": es(v99)},
          open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
