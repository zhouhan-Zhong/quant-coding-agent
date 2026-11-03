# memory-solution
# unit: t1-fomc-tone-event-study | category: text-filings | tags: fomc, tone-lexicon, tfidf-novelty, event-study, treasury-yields, ar1-forecast, ols, newey-west, wild-bootstrap, leave-one-out, cooks-distance, leverage
# deliverables: event_panel.csv, influence.csv, diagnostics.csv, results.json
# tokens: lowercase, non [a-z] -> space, split; counts via exact set membership; tone = 1000*(h-d)/tokens
# tfidf: vocab = tokens in >= 2 statements, idf = ln((1+N)/(1+df)) + 1, tf = count/token_count; novelty = 1 - cos(v_i, mean(prev<=4 v))
# yields: valid rows only; e = first valid row with date >= meeting; offsets are row offsets (e-1, e+1, e+2, e-6, window e-63..e-1 ddof=1)
# AR(1) on bps changes delta_j, j = max(2, e-252)..e-1 regress delta_j on delta_{j-1}; expected = f1 + f2 two-step forecast from delta_e
# panel rounded to 6 dp BEFORE all regressions; HAC lag 2 Bartlett, raw sums; wild sign-flip enumerates 2^(n-1) vectors (first sign +1)
import os, json, re, itertools
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

st = pd.read_csv(os.path.join(DATA_DIR, "fomc_statements.csv")).sort_values("meeting_date").reset_index(drop=True)
lex = json.load(open(os.path.join(DATA_DIR, "tone_lexicon.json")))
hawk, dove = set(lex["hawkish"]), set(lex["dovish"])
ty = pd.read_csv(os.path.join(DATA_DIR, "treasury_yields.csv"))
ty["dgs2"] = pd.to_numeric(ty["dgs2"], errors="coerce"); ty["dgs10"] = pd.to_numeric(ty["dgs10"], errors="coerce")
ty = ty.dropna(subset=["dgs2", "dgs10"]).sort_values("date").reset_index(drop=True)
ydates = ty["date"].astype(str).tolist(); d2 = ty["dgs2"].to_numpy(float); d10 = ty["dgs10"].to_numpy(float)
curve = (d10 - d2) * 100.0; delta = np.full(len(d2), np.nan); delta[1:] = np.diff(d2) * 100.0

toks = [re.sub(r"[^a-z]", " ", t.lower()).split() for t in st["statement_text"]]
N = len(toks)
tone = np.array([1000.0 * (sum(w in hawk for w in t) - sum(w in dove for w in t)) / len(t) for t in toks])
dfreq = {}
for t in toks:
    for w in set(t): dfreq[w] = dfreq.get(w, 0) + 1
vocab = sorted(w for w, c in dfreq.items() if c >= 2); vi = {w: i for i, w in enumerate(vocab)}
idf = np.array([np.log((1 + N) / (1 + dfreq[w])) + 1 for w in vocab])
V = np.zeros((N, len(vocab)))
for i, t in enumerate(toks):
    for w in t:
        if w in vi: V[i, vi[w]] += 1.0 / len(t)
V *= idf

def ols(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; return b, y - X @ b

def hac_t(X, y, L=2):
    b, r = ols(X, y); XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * (r ** 2)[:, None]).T @ X
    for l in range(1, L + 1):
        G = (X[l:] * (r[l:] * r[:-l])[:, None]).T @ X[:-l]
        S += (1 - l / (L + 1)) * (G + G.T)
    return b / np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))

rows = []
for i in range(N):
    m = st.loc[i, "meeting_date"]; t = toks[i]
    prev = list(range(max(0, i - 4), i)); pc = len(prev)
    pmean = float(tone[prev].mean()) if pc else 0.0
    if pc:
        mv = V[prev].mean(axis=0); nov = 1.0 - float(V[i] @ mv / (np.linalg.norm(V[i]) * np.linalg.norm(mv)))
    else:
        nov = 0.0
    e = next(j for j, d in enumerate(ydates) if d >= m)
    win = curve[max(0, e - 63):e]; z = (curve[e - 1] - win.mean()) / win.std(ddof=1)
    js = np.arange(max(2, e - 252), e)
    (c0, phi), _ = ols(np.column_stack([np.ones(len(js)), delta[js - 1]]), delta[js])
    f1 = c0 + phi * delta[e]; f2 = c0 + phi * f1
    ch02 = (d2[e + 2] - d2[e]) * 100.0
    rows.append(dict(meeting_date=m, event_date=ydates[e], prior_count=pc, token_count=len(t),
                     hawkish_count=sum(w in hawk for w in t), dovish_count=sum(w in dove for w in t), tone_score=tone[i],
                     prior_tone_mean=pmean, tone_surprise=tone[i] - pmean, dgs2_event=d2[e], dgs10_event=d10[e],
                     dgs2_change_0_2_bps=ch02, curve_change_0_2_bps=curve[e + 2] - curve[e],
                     dgs2_change_m1_p1_bps=(d2[e + 1] - d2[e - 1]) * 100.0, tfidf_novelty=nov, pre_curve_level_bps=curve[e - 1],
                     pre_curve_z_63d=z, pre_dgs2_momentum_5d_bps=(d2[e - 1] - d2[e - 6]) * 100.0, ar1_phi_252d=phi,
                     expected_dgs2_change_0_2_bps=f1 + f2, abnormal_dgs2_change_0_2_bps=ch02 - (f1 + f2)))
panel = pd.DataFrame(rows).round(6)
panel.to_csv(os.path.join(OUT_DIR, "event_panel.csv"), index=False)

reg = panel[panel.prior_count == 4].reset_index(drop=True); n = len(reg)
x = reg.tone_surprise.to_numpy(); y1 = reg.dgs2_change_0_2_bps.to_numpy()
X1 = np.column_stack([np.ones(n), x]); b1, r1 = ols(X1, y1)
sse1 = float(r1 @ r1); se1 = np.sqrt(sse1 / (n - 2) * np.linalg.inv(X1.T @ X1)[1, 1])
X2 = np.column_stack([np.ones(n), x, reg.pre_curve_z_63d, reg.pre_dgs2_momentum_5d_bps, reg.tfidf_novelty,
                      x * reg.pre_curve_z_63d, x * reg.tfidf_novelty]); y2 = reg.abnormal_dgs2_change_0_2_bps.to_numpy()
b2, r2 = ols(X2, y2); fit2 = X2 @ b2; t2 = hac_t(X2, y2); p = X2.shape[1]
# wild sign-flip p-value (beta_tone_star is linear in y_star)
h = np.linalg.pinv(X2)[1]
signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n - 1))); signs = np.column_stack([np.ones(len(signs)), signs])
beta_star = h @ fit2 + signs @ (h * r2)
wild_p = float(np.mean(np.abs(beta_star) >= abs(b2[1]) - 1e-12))      # tolerance so the identity sign vector counts
# leave-one-out influence
inf = []
for i in range(n):
    keep = np.arange(n) != i; bl = ols(X2[keep], y2[keep])[0][1]
    inf.append(dict(excluded_meeting_date=reg.meeting_date[i], beta_tone_leave_one_out=bl, delta_beta_tone=bl - b2[1], abs_delta_beta_tone=abs(bl - b2[1])))
inf = pd.DataFrame(inf).sort_values(["abs_delta_beta_tone", "excluded_meeting_date"], ascending=[False, True]).reset_index(drop=True)
inf.to_csv(os.path.join(OUT_DIR, "influence.csv"), index=False)
# diagnostics
sse2 = float(r2 @ r2); mse = sse2 / (n - p); lev = np.einsum("ij,jk,ik->i", X2, np.linalg.inv(X2.T @ X2), X2)
cooks = r2 ** 2 / (p * mse) * lev / (1 - lev) ** 2
mse_i = (sse2 - r2 ** 2 / (1 - lev)) / (n - p - 1); stud = r2 / np.sqrt(mse_i * (1 - lev))
diag = pd.DataFrame(dict(meeting_date=reg.meeting_date, leverage=lev, externally_studentized_resid=stud, cooks_distance=cooks))
diag = diag.sort_values(["cooks_distance", "meeting_date"], ascending=[False, True]).reset_index(drop=True)
diag.to_csv(os.path.join(OUT_DIR, "diagnostics.csv"), index=False)

imax = int(panel.tone_surprise.idxmax()); nmax = int(panel.tfidf_novelty.idxmax())
res = {"meta": {"num_events": int(N), "regression_events": int(n), "first_event_date": panel.event_date.iloc[0], "last_event_date": panel.event_date.iloc[-1]},
       "tone": {"mean_score": float(panel.tone_score.mean()), "mean_surprise": float(panel.tone_surprise.mean()),
                "max_surprise_date": panel.meeting_date[imax], "max_surprise": float(panel.tone_surprise[imax])},
       "novelty": {"mean_tfidf_novelty": float(panel.tfidf_novelty.mean()), "max_tfidf_novelty_date": panel.meeting_date[nmax],
                   "max_tfidf_novelty": float(panel.tfidf_novelty[nmax])},
       "regression": {"intercept": float(b1[0]), "beta_tone_surprise": float(b1[1]), "t_stat_beta": float(b1[1] / se1),
                      "t_stat_beta_nw_lag2": float(hac_t(X1, y1)[1]), "durbin_watson": float((np.diff(r1) ** 2).sum() / sse1),
                      "r_squared": float(1 - sse1 / ((y1 - y1.mean()) ** 2).sum())},
       "state_adjusted": {"intercept": float(b2[0]), "beta_tone_surprise": float(b2[1]), "beta_pre_curve_z_63d": float(b2[2]),
                          "beta_pre_dgs2_momentum_5d_bps": float(b2[3]), "beta_tfidf_novelty": float(b2[4]),
                          "beta_tone_curve_interaction": float(b2[5]), "beta_tone_novelty_interaction": float(b2[6]),
                          "t_stat_tone_nw_lag2": float(t2[1]), "t_stat_tone_novelty_interaction_nw_lag2": float(t2[6]),
                          "wild_sign_p_value_tone": wild_p, "r_squared": float(1 - sse2 / ((y2 - y2.mean()) ** 2).sum())},
       "influence": {"max_influence_date": inf.excluded_meeting_date[0], "max_abs_delta_beta_tone": float(inf.abs_delta_beta_tone[0]),
                     "mean_abs_delta_beta_tone": float(inf.abs_delta_beta_tone.mean())},
       "diagnostics": {"max_cooks_date": diag.meeting_date[0], "max_cooks_distance": float(diag.cooks_distance[0]),
                       "max_abs_studentized_resid": float(np.abs(stud).max()), "high_leverage_count": int((lev > 2 * p / n).sum())}}
json.dump(res, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
