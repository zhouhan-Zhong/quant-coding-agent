# memory-solution
# unit: t1-pca-factor-portfolio | category: portfolio | tags: pca, sklearn, factor-neutral, hedge, lstsq, explained-variance, sharpe, factor-exposure
# deliverables: results.json
# R (T, N) with R[:, i] = returns of tickers[i]; sklearn PCA(n_components=n_factors).fit(R) directly on returns
#   (PCA centres internally; components_ rows = loadings, explained_variance_ratio_ sorted descending).
# Exposures e = components_ @ w. Hedge h = min-norm lstsq solution of [components_; 1^T] h = [-e; 0]
#   (dollar-neutral hedge, so neutral weights sum to the target sum). w_neutral = w + h; neutral exposures ~ 0.
# Sharpe = mean/std(ddof=0)*sqrt(12) on monthly R @ w; residual_r2 = 1 - var(R w_neutral)/var(R w_target).
import os, json, math
import numpy as np
from sklearn.decomposition import PCA

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

p = json.load(open(os.path.join(DATA_DIR, "params.json")))
tickers, k = p["tickers"], int(p["n_factors"])
R = np.array([p["monthly_returns"][t] for t in tickers], dtype=float).T      # (T, N)
w = np.array([p["target_weights"][t] for t in tickers], dtype=float)
N = len(tickers)
pcs = [f"PC{i + 1}" for i in range(k)]

pca = PCA(n_components=k).fit(R)
comp, evr = pca.components_, pca.explained_variance_ratio_                      # (k, N), (k,)
expo = comp @ w

A = np.vstack([comp, np.ones((1, N))])
b = np.append(-expo, [0.0])
h = np.linalg.lstsq(A, b, rcond=None)[0]
w_neu = w + h
expo_neu = comp @ w_neu

def sharpe(x):
    return float(np.mean(x) / np.std(x) * math.sqrt(12))

r_neu, r_tgt = R @ w_neu, R @ w
res = {"pca_explained_variance_ratio": [round(float(v), 4) for v in evr],
       "factor_loadings": {pc: {t: round(float(comp[i, j]), 6) for j, t in enumerate(tickers)} for i, pc in enumerate(pcs)},
       "target_factor_exposures": {pc: round(float(expo[i]), 6) for i, pc in enumerate(pcs)},
       "hedge_weights": {t: round(float(h[j]), 6) for j, t in enumerate(tickers)},
       "neutral_weights": {t: round(float(w_neu[j]), 6) for j, t in enumerate(tickers)},
       "neutral_factor_exposures": {pc: round(float(expo_neu[i]), 6) for i, pc in enumerate(pcs)},
       "portfolio_sharpe_neutral": round(sharpe(r_neu), 4),
       "portfolio_sharpe_target": round(sharpe(r_tgt), 4),
       "residual_r2": round(float(1.0 - np.var(r_neu) / np.var(r_tgt)), 6)}
json.dump(res, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
