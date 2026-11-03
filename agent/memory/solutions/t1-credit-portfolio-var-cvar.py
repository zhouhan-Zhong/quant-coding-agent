# memory-solution
# unit: t1-credit-portfolio-var-cvar | category: credit | tags: credit-var, cvar, gaussian-copula, student-t-copula, multi-factor, euler-allocation, vasicek, gordy-granularity, wrong-way-risk, hhi, monte-carlo
# deliverables: var_cvar_results.csv, wrong_way_results.csv, marginal_contributions.csv, loss_statistics.csv, concentration_risk.json, summary.json, analytical_benchmark.csv, sector_risk_decomposition.csv, granularity_adjustment.csv
# Multi-factor Merton: A_i = sqrt(rho_i) M_{s(i)} + sqrt(1-rho_i) eps_i, M = Z @ chol(C)^T over the sector correlation matrix;
#   default iff A_i < Phi^-1(PD_i); loss = sum EAD_i LGD_i 1{default}. Simulate in chunks of 10k sims (990 x 100k matrix).
# Student-t (nu=5): same normals divided by sqrt(W), W ~ chi2(nu)/nu per sim (one W for factors AND idiosyncratics),
#   thresholds t_nu^-1(PD_i) -> marginal PDs preserved (MC mean ~ analytic EL). Seed+1000 for the t run.
# VaR = np.quantile(L, a); CVaR = mean(L[L >= VaR]); Euler CVaR_i = mean(L_i | L >= VaR_99) (sums exactly to CVaR).
# Wrong-way: rho + 0.15 (cap .999) and LGD = 1 - recovery_stressed for Financials/Energy, same seed as base.
# Vasicek VaR_a = sum EAD LGD Phi(h_i), h_i = (Phi^-1 PD + sqrt(rho) Phi^-1 a)/sqrt(1-rho) (PD>0 only, no EL subtraction);
#   Gordy GA = M sigma^2 / (2 C'), sigma^2 = sum w^2 LGD^2 Phi(h)(1-Phi(h)), C' = sum w LGD phi(h) sqrt(rho)/(sqrt(1-rho) phi(Phi^-1 a)).
import os, json
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import norm

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

cfg = json.load(open(os.path.join(DATA_DIR, "config.json")))
port = pd.read_csv(os.path.join(DATA_DIR, "portfolio.csv"))
C = pd.read_csv(os.path.join(DATA_DIR, "sector_correlations.csv"), index_col=0)
assert np.allclose(C.values, C.values.T) and np.linalg.eigvalsh(C.values).min() > -1e-10
NS, SEED, ALPHAS, NU = int(cfg["n_simulations"]), int(cfg["seed"]), cfg["confidence_levels"], int(cfg["t_copula_df"])
n = len(port)
ead, pdv, rho = port["exposure_usd"].to_numpy(), port["pd_1y"].to_numpy(), port["basel_correlation"].to_numpy()
lgd, lgd_s = 1.0 - port["recovery_rate"].to_numpy(), 1.0 - port["recovery_stressed"].to_numpy()
sec_idx = np.array([list(C.index).index(s) for s in port["factor_sector"]])
Lc = np.linalg.cholesky(C.values)
M_tot, w = ead.sum(), ead / ead.sum()

def simulate(seed, rho_v, lgd_v, nu=None, keep=False):
    rng = np.random.default_rng(seed)
    thr = norm.ppf(pdv) if nu is None else stats.t.ppf(pdv, nu)
    ll = ead * lgd_v
    losses, D = np.empty(NS), (np.empty((NS, n), dtype=bool) if keep else None)
    for s in range(0, NS, 10_000):
        m = min(10_000, NS - s)
        Mf = rng.standard_normal((m, Lc.shape[0])) @ Lc.T                  # correlated sector factors
        eps = rng.standard_normal((m, n))
        A = np.sqrt(rho_v) * Mf[:, sec_idx] + np.sqrt(1 - rho_v) * eps
        if nu is not None:
            A /= np.sqrt(rng.chisquare(nu, size=(m, 1)) / nu)              # shared mixing variable per sim
        d = A < thr
        losses[s:s + m] = d @ ll
        if keep: D[s:s + m] = d
    return losses, D

def var_cvar(L, a):
    v = float(np.quantile(L, a)); return v, float(L[L >= v].mean())

L_g, D_g = simulate(SEED, rho, lgd, keep=True)
L_t, _ = simulate(SEED + 1000, rho, lgd, nu=NU)
ww = port["factor_sector"].isin(cfg["wrong_way_sectors"]).to_numpy()
rho_ww = np.where(ww, np.minimum(rho + cfg["wrong_way_correlation_boost"], 0.999), rho)
L_w, _ = simulate(SEED, rho_ww, np.where(ww, lgd_s, lgd))

vc = [{"copula": c, "confidence_level": a, "VaR": round(v, 2), "CVaR": round(e, 2)}
      for c, L in (("gaussian", L_g), ("student_t", L_t)) for a in ALPHAS for v, e in [var_cvar(L, a)]]
pd.DataFrame(vc).to_csv(os.path.join(OUT_DIR, "var_cvar_results.csv"), index=False)
wwr = [{"scenario": c, "confidence_level": a, "VaR": round(v, 2), "CVaR": round(e, 2)}
       for c, L in (("base", L_g), ("wrong_way", L_w)) for a in ALPHAS for v, e in [var_cvar(L, a)]]
pd.DataFrame(wwr).to_csv(os.path.join(OUT_DIR, "wrong_way_results.csv"), index=False)
g = {(r["copula"], r["confidence_level"]): r for r in vc}
wq = {(r["scenario"], r["confidence_level"]): r for r in wwr}

# Euler allocation of Gaussian CVaR_99: E[L_i | L >= VaR_99]
v99, cvar99 = var_cvar(L_g, 0.99)
tail = L_g >= v99
mc_i = D_g[tail].mean(axis=0) * ead * lgd
mc = pd.DataFrame({"obligor_id": port["obligor_id"], "issuer": port["issuer"], "sector": port["factor_sector"],
                   "exposure_usd": ead, "exposure_weight": w.round(6), "marginal_cvar_99": mc_i,
                   "pct_contribution": mc_i / cvar99 * 100}).sort_values("marginal_cvar_99", ascending=False)
mc.round({"exposure_usd": 2, "marginal_cvar_99": 2, "pct_contribution": 2}).to_csv(os.path.join(OUT_DIR, "marginal_contributions.csv"), index=False)

EL = float((pdv * ead * lgd).sum())
ls = [{"copula": c, "mean_loss": round(L.mean(), 2), "std_loss": round(L.std(ddof=1), 2),
       "skewness": round(float(stats.skew(L)), 4), "kurtosis": round(float(stats.kurtosis(L)), 4),
       "max_loss": round(L.max(), 2), "pct_zero_loss": round(float((L == 0).mean() * 100), 2),
       "expected_loss_analytical": round(EL, 2)} for c, L in (("gaussian", L_g), ("student_t", L_t))]
pd.DataFrame(ls).to_csv(os.path.join(OUT_DIR, "loss_statistics.csv"), index=False)

sec = port.groupby("factor_sector")["exposure_usd"].sum()
sec_w = sec / M_tot
top_e = port.assign(weight=w).nlargest(5, "exposure_usd")
json.dump({"obligor_hhi": float((w ** 2).sum()), "sector_hhi": float((sec_w ** 2).sum()),
           "top_5_exposures": [{"obligor_id": int(r.obligor_id), "issuer": r.issuer, "weight": round(float(r.weight), 6)} for r in top_e.itertuples()],
           "top_5_marginal_cvar": [{"obligor_id": int(r.obligor_id), "issuer": r.issuer, "marginal_cvar_99": round(float(r.marginal_cvar_99), 2)} for r in mc.head(5).itertuples()],
           "sector_weights": {k: round(float(v), 6) for k, v in sec_w.items()}},
          open(os.path.join(OUT_DIR, "concentration_risk.json"), "w"), indent=2)

# Vasicek single-factor VaR and Gordy granularity adjustment (PD > 0 obligors only)
pm = pdv > 0
def vasicek(a):
    h = (norm.ppf(pdv[pm]) + np.sqrt(rho[pm]) * norm.ppf(a)) / np.sqrt(1 - rho[pm])
    V = float((ead[pm] * lgd[pm] * norm.cdf(h)).sum())
    s2 = float((w[pm] ** 2 * lgd[pm] ** 2 * norm.cdf(h) * (1 - norm.cdf(h))).sum())
    dh = np.sqrt(rho[pm]) / (np.sqrt(1 - rho[pm]) * norm.pdf(norm.ppf(a)))
    Cp = float((w[pm] * lgd[pm] * norm.pdf(h) * dh).sum())
    return V, M_tot * s2 / (2 * Cp)
vas = {a: vasicek(a) for a in ALPHAS}
pd.DataFrame([{"confidence_level": a, "mc_var_gaussian": g[("gaussian", a)]["VaR"], "vasicek_var": round(vas[a][0], 2),
               "analytical_el": round(EL, 2), "mc_mean_loss": round(L_g.mean(), 2), "el_ratio": round(L_g.mean() / EL, 6)}
              for a in ALPHAS]).to_csv(os.path.join(OUT_DIR, "analytical_benchmark.csv"), index=False)
pd.DataFrame([{"confidence_level": a, "vasicek_var": round(vas[a][0], 2), "granularity_adjustment": round(vas[a][1], 2),
               "gordy_var": round(vas[a][0] + vas[a][1], 2), "mc_var_gaussian": g[("gaussian", a)]["VaR"],
               "gordy_to_mc_ratio": round((vas[a][0] + vas[a][1]) / g[("gaussian", a)]["VaR"], 6)}
              for a in ALPHAS]).to_csv(os.path.join(OUT_DIR, "granularity_adjustment.csv"), index=False)

sr = mc.groupby("sector").agg(n_obligors=("obligor_id", "size"), total_exposure=("exposure_usd", "sum"),
                              marginal_cvar_99=("marginal_cvar_99", "sum")).reset_index()
sr["exposure_weight"] = sr["total_exposure"] / M_tot
sr["pct_of_total_cvar"] = sr["marginal_cvar_99"] / cvar99 * 100
sr["risk_weight_ratio"] = sr["pct_of_total_cvar"] / (sr["exposure_weight"] * 100)
sr[["sector", "n_obligors", "total_exposure", "exposure_weight", "marginal_cvar_99", "pct_of_total_cvar", "risk_weight_ratio"]
   ].round(6).to_csv(os.path.join(OUT_DIR, "sector_risk_decomposition.csv"), index=False)

json.dump({"n_obligors": int(n), "total_exposure_usd": round(float(M_tot), 2), "n_simulations": NS,
           "gaussian_var_99": g[("gaussian", 0.99)]["VaR"], "gaussian_cvar_99": g[("gaussian", 0.99)]["CVaR"],
           "student_t_var_99": g[("student_t", 0.99)]["VaR"], "student_t_cvar_99": g[("student_t", 0.99)]["CVaR"],
           "wrong_way_var_99": wq[("wrong_way", 0.99)]["VaR"], "wrong_way_cvar_99": wq[("wrong_way", 0.99)]["CVaR"],
           "var_ratio_t_vs_gaussian_99": round(g[("student_t", 0.99)]["VaR"] / g[("gaussian", 0.99)]["VaR"], 6),
           "wrong_way_cvar_increase_pct": round((wq[("wrong_way", 0.99)]["CVaR"] / cvar99 - 1) * 100, 4),
           "obligor_hhi": float((w ** 2).sum()), "max_exposure_weight": float(w.max()),
           "mean_portfolio_pd": float((w * pdv).sum()), "analytical_el": round(EL, 2), "mc_el_ratio": round(float(L_g.mean() / EL), 6),
           "vasicek_var_99": round(vas[0.99][0], 2), "vasicek_to_mc_ratio_99": round(vas[0.99][0] / v99, 6),
           "marginal_cvar_sum_check": round(float(mc_i.sum() / cvar99), 6),
           "gordy_var_99": round(vas[0.99][0] + vas[0.99][1], 2), "granularity_adjustment_99": round(vas[0.99][1], 2)},
          open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
