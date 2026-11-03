# memory-solution
# unit: t1-credit-migration-matrix | category: credit | tags: transition-matrix, cerep, rating-migration, cumulative-pd, markov-test, chi-squared, generator-matrix, logm, cohort-default-rates
# deliverables: average_transition_matrix.csv, cumulative_default_probs.csv, markov_test.csv, generator_matrix.csv, cohort_sizes.csv, transition_matrix_deviation.csv, cohort_default_rates.csv, summary.json
# CEREP files are ';'-separated; map CCC/CC/C -> CCC, R/SD/D -> Default, drop NR + Withdrawals columns (censored, not redistributed).
# Hidden-default correction per year & fine BOP rating: n_defaults (Default-Rates file) - visible (R+SD+D cols), clipped >= 0,
#   added to the Default column; cohort size = corrected row sum (so total_transitions == sum of cohort sizes).
# Default BOP row forced absorbing. Pbar = pooled counts / pooled row sums; PD(h) = (Pbar^h)[:, Default].
# Markov: per BOP rating, cohorts x destinations table, keep columns with total >= 5 and cohorts with > 0 issuers,
#   chi2 = sum (O-E)^2/E, df = (R-1)(C-1); df = 0 -> stat 0, p = 1. Generator: real(logm(P)), clip negative off-diag, diag = -rowsum.
import os, json, math
import numpy as np, pandas as pd
from scipy.linalg import logm
from scipy.stats import chi2

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

cfg = json.load(open(os.path.join(DATA_DIR, "config.json")))
R = cfg["ratings"]; ND = cfg["n_non_default"]; H = cfg["projection_horizons"]; ALPHA = cfg["significance_level"]
YEARS = list(range(2018, 2025))
MAP = {"AAA": "AAA", "AA": "AA", "A": "A", "BBB": "BBB", "BB": "BB", "B": "B", "CCC": "CCC", "CC": "CCC", "C": "CCC",
       "R": "Default", "SD": "Default", "D": "Default"}
DEF_COLS = ["R", "SD", "D"]

counts = {}
for y in YEARS:
    tm = pd.read_csv(os.path.join(DATA_DIR, f"Transition-Matrix-{y}.csv"), sep=";", index_col=0)
    dr = pd.read_csv(os.path.join(DATA_DIR, f"Default-Rates-{y}.csv")).set_index("rating")["n_defaults"]
    C = np.zeros((len(R), len(R)))
    for bop, row in tm.iterrows():
        if bop not in MAP:
            continue                                            # NR start: censored
        i = R.index(MAP[bop])
        for eop, v in row.items():
            if eop in MAP:
                C[i, R.index(MAP[eop])] += float(v)             # NR / Withdrawals columns dropped
        hidden = max(float(dr.get(bop, 0)) - float(row[DEF_COLS].sum()), 0.0)
        C[i, -1] += hidden                                      # defaults hidden behind withdrawals
    C[-1, :] = 0.0; C[-1, -1] = 1.0                             # absorbing default (placeholder count)
    counts[y] = C

pooled = sum(counts.values())
pooled[-1, :] = 0.0; pooled[-1, -1] = 1.0
P = pooled / pooled.sum(axis=1, keepdims=True)
tm_df = pd.DataFrame(np.round(P, 6), columns=R); tm_df.insert(0, "rating", R)
tm_df.to_csv(os.path.join(OUT_DIR, "average_transition_matrix.csv"), index=False)

cum = {f"horizon_{h}": np.linalg.matrix_power(P, h)[:ND, -1] for h in H}
cum_df = pd.DataFrame({"rating": R[:ND], **{k: np.round(v, 6) for k, v in cum.items()}})
cum_df.to_csv(os.path.join(OUT_DIR, "cumulative_default_probs.csv"), index=False)

mrows = []
for i in range(ND):
    tab = np.array([counts[y][i] for y in YEARS])
    tab = tab[:, tab.sum(axis=0) >= 5]                          # drop sparse destination columns
    tab = tab[tab.sum(axis=1) > 0]                              # drop empty cohorts
    df_ = (tab.shape[0] - 1) * (tab.shape[1] - 1) if tab.size else 0
    if df_ <= 0:
        stat, pv = 0.0, 1.0
    else:
        E = np.outer(tab.sum(axis=1), tab.sum(axis=0)) / tab.sum()
        stat = float(np.sum((tab - E) ** 2 / E)); pv = float(chi2.sf(stat, df_))
    mrows.append({"rating": R[i], "chi2_stat": stat, "df": int(max(df_, 0)), "p_value": pv, "reject_H0": bool(pv < ALPHA)})
markov = pd.DataFrame(mrows); markov.to_csv(os.path.join(OUT_DIR, "markov_test.csv"), index=False)

Q = np.real(logm(P))
Q[-1, :] = 0.0
off = ~np.eye(len(R), dtype=bool)
Q[off] = np.maximum(Q[off], 0.0)
np.fill_diagonal(Q, 0.0); np.fill_diagonal(Q, -Q.sum(axis=1))
Q_df = pd.DataFrame(np.round(Q, 6), columns=R); Q_df.insert(0, "rating", R)
Q_df.to_csv(os.path.join(OUT_DIR, "generator_matrix.csv"), index=False)

sizes = pd.DataFrame([{"year": y, **{R[i]: int(round(counts[y][i].sum())) for i in range(ND)}} for y in YEARS])
sizes.to_csv(os.path.join(OUT_DIR, "cohort_sizes.csv"), index=False)

ref = pd.read_csv(os.path.join(DATA_DIR, "historical_transition_matrix.csv")).set_index("rating")[R].astype(float) / 100.0
ref = ref.div(ref.sum(axis=1), axis=0).loc[R].to_numpy()
D = P[:ND] - ref[:ND]
dev_df = pd.DataFrame(np.round(D, 6), columns=R); dev_df.insert(0, "rating", R[:ND])
dev_df.to_csv(os.path.join(OUT_DIR, "transition_matrix_deviation.csv"), index=False)

cdr = []
for y in YEARS:
    C = counts[y][:ND]
    tot, dft = C.sum(), C[:, -1].sum()
    cdr.append({"year": y, "total_issuers": int(round(tot)), "total_defaults": int(round(dft)),
                "default_rate": round(float(dft / tot), 6), **{f"{R[i]}_defaults": int(round(C[i, -1])) for i in range(ND)}})
cdr = pd.DataFrame(cdr); cdr.to_csv(os.path.join(OUT_DIR, "cohort_default_rates.csv"), index=False)

k = int(np.argmax(np.abs(D))); fi, fj = divmod(k, len(R))
pd1 = cum["horizon_1"]; pk, lo = int(cdr["default_rate"].idxmax()), int(cdr["default_rate"].idxmin())
json.dump({"total_transitions": int(round(pooled[:ND].sum())), "n_cohorts": len(YEARS),
           "avg_default_rate_1y": float(pd1.mean()), "avg_default_rate_5y": float(cum["horizon_5"].mean()),
           "investment_grade_1y_pd": float(pd1[:4].mean()), "speculative_grade_1y_pd": float(pd1[4:].mean()),
           "n_markov_rejections": int(markov["reject_H0"].sum()), "generator_max_offdiag": float(Q[off].max()),
           "generator_min_diag": float(np.diag(Q).min()), "max_absolute_deviation": float(np.abs(D).max()),
           "max_deviation_cell": f"{R[fi]}->{R[fj]}", "mean_absolute_deviation": float(np.abs(D).mean()),
           "peak_default_year": int(cdr.loc[pk, "year"]), "peak_default_rate": float(cdr.loc[pk, "default_rate"]),
           "lowest_default_year": int(cdr.loc[lo, "year"]), "lowest_default_rate": float(cdr.loc[lo, "default_rate"])},
          open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
