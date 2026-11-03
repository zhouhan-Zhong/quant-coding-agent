# memory-solution
# unit: t1-lob-pc-signal | category: microstructure | tags: order-flow-imbalance, ofi, lob, pca, rolling-regression, walk-forward, information-coefficient, spearman, crypto, z-score
# deliverables: results.json
# weighted_ofi_i = w_bid_i * (bid_limit_i - bid_cancel_i - bid_market_i) - w_ask_i * (ask_limit_i - ask_cancel_i - ask_market_i), w = 1/(1+|distance_i|), 15 levels.
# target y_t = log(mid_{t+1}/mid_t) (NaN on the last row). Window 40: step t trains on rows t-40..t-1 (drop rows with NaN target),
#   z-score with the TRAINING mean/std (ddof=0, StandardScaler convention), sklearn PCA(1, random_state=0), flip PC1 so corr(PC1, ofi_0) >= 0,
#   OLS y = a + b*PC1, then transform row t with the same scaler/PCA/sign and predict. rolling_pc1_index[t] = PC1 of row t.
# ICs over all steps with a non-NaN realised forward return (59 pairs): scipy spearmanr / pearsonr.
import os, json, math
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from scipy import stats

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "ADA_1min.csv"))
mid = df["midpoint"].to_numpy(float)
y = np.r_[np.diff(np.log(mid)), np.nan]                                    # forward log return

def side_flow(side, i):
    flow = df[f"{side}_limit_notional_{i}"] - df[f"{side}_cancel_notional_{i}"] - df[f"{side}_market_notional_{i}"]
    return flow / (1.0 + df[f"{side}_distance_{i}"].abs())
X = np.column_stack([(side_flow("bids", i) - side_flow("asks", i)).to_numpy(float) for i in range(15)])

W = 40
pc1, pred = {}, {}
for t in range(W, len(df)):
    Xtr, ytr = X[t - W:t], y[t - W:t]
    keep = ~np.isnan(ytr)                                                  # skip training rows with a missing target
    Xtr, ytr = Xtr[keep], ytr[keep]
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)                             # ddof=0
    pca = PCA(n_components=1, random_state=0).fit((Xtr - mu) / sd)
    pc = pca.transform((Xtr - mu) / sd)[:, 0]
    sign = 1.0 if np.corrcoef(pc, Xtr[:, 0])[0, 1] >= 0 else -1.0
    pc *= sign
    beta = np.linalg.lstsq(np.column_stack([np.ones_like(pc), pc]), ytr, rcond=None)[0]   # OLS with intercept
    p_new = sign * pca.transform(((X[t] - mu) / sd)[None, :])[0, 0]
    pc1[t], pred[t] = float(p_new), float(beta[0] + beta[1] * p_new)

res = {}
for i in (0, 10, 20, 30):
    res[f"midpoint_return_fwd_index{i}"] = float(y[i])
    for d in (0, 2, 4, 6):
        res[f"weighted_ofi_{d}_index{i}"] = float(X[i, d])
for i in (40, 50, 60, 70):
    res[f"rolling_pc1_index{i}"] = pc1[i]
    res[f"predicted_return_index{i}"] = pred[i]
ts = [t for t in pred if not np.isnan(y[t])]
P, A = np.array([pred[t] for t in ts]), np.array([y[t] for t in ts])
res["spearman_ic"] = float(stats.spearmanr(P, A)[0])
res["pearson_ic"] = float(stats.pearsonr(P, A)[0])
json.dump(res, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
