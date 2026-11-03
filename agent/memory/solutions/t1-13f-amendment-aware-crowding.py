# memory-solution
# unit: t1-13f-amendment-aware-crowding | category: text-filings | tags: 13f, sec-flat-files, tsv, amendments, restatement, new-holdings, cusip, holdings-cleaning, hhi, turnover, overlap, crowding
# deliverables: filing_resolution.csv, effective_holdings.csv, manager_metrics.csv, turnover.csv, overlap_latest.csv, crowded_securities_latest.csv, cleaning_audit.csv, summary.json
# eligible = SUBMISSION x COVERPAGE x SUMMARYPAGE in the target (CIK, period) universe, 13F-HR or 13F-HR/A, REPORTTYPE '13F HOLDINGS REPORT', PERIODOFREPORT == REPORTCALENDARORQUARTER
# per (CIK, period) apply filings sorted by (FILING_DATE, AMENDMENTNO, ACCESSION): original -> replace_initial; RESTATEMENT -> replace_restatement (state := its rows); NEW HOLDINGS -> append_new_holdings
# cleaning in sequence, each rejected row counted once: blank CUSIP -> SSHPRNAMTTYPE != SH -> PUTCALL non-blank -> VALUE<=0 or SSHPRNAMT<=0; then sum duplicate CUSIPs (VALUE, shares, votes)
# labels upper-cased + whitespace-collapsed, canonical = most frequent (tie -> shortest); VALUE is dollars; weight = VALUE / book total
# top = (weight desc, VALUE desc, CUSIP asc); hhi = sum w^2; turnover = 0.5 sum|w1 - w0| over the union; overlap = sum min(w_i, w_j); crowding sort (holders desc, value desc, CUSIP asc)
import os, json
from itertools import combinations
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)


def rd(name):
    return pd.read_csv(os.path.join(DATA_DIR, name), sep="\t", dtype=str, keep_default_na=False)


def canon(s):  # representative label: most frequent, then shortest, then alphabetical
    c = s.value_counts()
    return min(c.index, key=lambda x: (-c[x], len(x), x))


sub, cov, summ, info = rd("SUBMISSION.tsv"), rd("COVERPAGE.tsv"), rd("SUMMARYPAGE.tsv"), rd("INFOTABLE.tsv")
for df in (sub, cov, summ, info):
    for c in df.columns:
        df[c] = df[c].str.strip()
for df, cols in [(sub, ["ACCESSION_NUMBER", "SUBMISSIONTYPE"]), (cov, ["ACCESSION_NUMBER", "ISAMENDMENT", "AMENDMENTTYPE", "REPORTTYPE"]),
                 (summ, ["ACCESSION_NUMBER"]), (info, ["ACCESSION_NUMBER", "CUSIP", "SSHPRNAMTTYPE", "PUTCALL", "NAMEOFISSUER", "TITLEOFCLASS"])]:
    for c in cols:
        df[c] = df[c].str.split().str.join(" ").str.upper()
NUM = ["VALUE", "SSHPRNAMT", "VOTING_AUTH_SOLE", "VOTING_AUTH_SHARED", "VOTING_AUTH_NONE"]
for c in NUM:
    info[c] = pd.to_numeric(info[c], errors="coerce").fillna(0.0)
mgr = pd.read_csv(os.path.join(DATA_DIR, "target_managers.csv"), dtype={"CIK": str})
per = pd.read_csv(os.path.join(DATA_DIR, "target_periods.csv")).sort_values("period_order")
mgr["CIK"] = mgr["CIK"].str.strip().str.zfill(10)
sub["CIK"] = sub["CIK"].str.zfill(10)
per["period_end"] = pd.to_datetime(per["period_end"])
sec_date = lambda s: pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
sub["FILING_DATE"], sub["PERIODOFREPORT"], cov["REPORTCALENDARORQUARTER"] = sec_date(sub["FILING_DATE"]), sec_date(sub["PERIODOFREPORT"]), sec_date(cov["REPORTCALENDARORQUARTER"])
cov["AMENDMENTNO"] = pd.to_numeric(cov["AMENDMENTNO"], errors="coerce").fillna(0).astype(int)

# ---- eligible holdings reports + summary-page reconciliation
f = sub.merge(cov, on="ACCESSION_NUMBER").merge(summ, on="ACCESSION_NUMBER", how="left")
f = f[f["SUBMISSIONTYPE"].isin(["13F-HR", "13F-HR/A"]) & (f["REPORTTYPE"] == "13F HOLDINGS REPORT") & (f["PERIODOFREPORT"] == f["REPORTCALENDARORQUARTER"])]
f = f.merge(mgr, on="CIK").merge(per, left_on="PERIODOFREPORT", right_on="period_end")
agg = info.groupby("ACCESSION_NUMBER").agg(rows_in_filing=("VALUE", "size"), value_total_in_filing=("VALUE", "sum")).reset_index()
f = f.merge(agg, on="ACCESSION_NUMBER", how="left").fillna({"rows_in_filing": 0, "value_total_in_filing": 0.0})
f["summarypage_entry_total"] = pd.to_numeric(f["TABLEENTRYTOTAL"], errors="coerce")
f["summarypage_value_total"] = pd.to_numeric(f["TABLEVALUETOTAL"], errors="coerce")
f["summarypage_row_diff"] = f["rows_in_filing"] - f["summarypage_entry_total"]
f["summarypage_value_diff"] = f["value_total_in_filing"] - f["summarypage_value_total"]
amend = f["ISAMENDMENT"].eq("Y") | f["AMENDMENTNO"].gt(0)
f["action"] = np.where(~amend, "replace_initial", np.where(f["AMENDMENTTYPE"].str.contains("NEW HOLDINGS"), "append_new_holdings", "replace_restatement"))
f = f.sort_values(["manager_order", "period_order", "FILING_DATE", "AMENDMENTNO", "ACCESSION_NUMBER"]).reset_index(drop=True)
f["CIK"] = f["CIK"].astype(int)
ID = ["manager_order", "manager_label", "CIK", "period_order", "period_end"]
fr = f[ID + ["ACCESSION_NUMBER", "FILING_DATE", "ISAMENDMENT", "AMENDMENTNO", "AMENDMENTTYPE", "action", "rows_in_filing", "value_total_in_filing",
             "summarypage_entry_total", "summarypage_value_total", "summarypage_row_diff", "summarypage_value_diff"]].copy()
for c in ["period_end", "FILING_DATE"]:
    fr[c] = fr[c].dt.strftime("%Y-%m-%d")
fr.to_csv(os.path.join(OUT_DIR, "filing_resolution.csv"), index=False)

# ---- effective raw state per manager-period -> sequential cleaning -> CUSIP aggregation
by_acc = {a: g for a, g in info.groupby("ACCESSION_NUMBER", sort=False)}
books, audit = [], []
for (mo, po), g in f.groupby(["manager_order", "period_order"], sort=True):
    parts = []
    for _, r in g.iterrows():
        rows = by_acc.get(r["ACCESSION_NUMBER"], info.iloc[:0])
        parts = parts + [rows] if r["action"] == "append_new_holdings" else [rows]
    raw = pd.concat(parts, ignore_index=True)
    ids = dict(manager_order=mo, manager_label=g["manager_label"].iloc[0], CIK=g["CIK"].iloc[0], period_order=po,
               period_end=g["period_end"].iloc[0].strftime("%Y-%m-%d"))
    m1 = raw["CUSIP"] != ""
    m2 = m1 & (raw["SSHPRNAMTTYPE"] == "SH")
    m3 = m2 & (raw["PUTCALL"] == "")
    m4 = m3 & (raw["VALUE"] > 0) & (raw["SSHPRNAMT"] > 0)
    audit.append(dict(ids, raw_effective_rows=len(raw), kept_rows=int(m4.sum()), filtered_blank_cusip_rows=int((~m1).sum()),
                      filtered_nonsh_rows=int((m1 & ~m2).sum()), filtered_putcall_rows=int((m2 & ~m3).sum()), filtered_nonpositive_rows=int((m3 & ~m4).sum())))
    b = raw[m4].groupby("CUSIP", sort=False).agg(issuer_name_canonical=("NAMEOFISSUER", canon), titleofclass_canonical=("TITLEOFCLASS", canon),
                                                 **{c: (c, "sum") for c in NUM}).reset_index()
    b["weight"] = b["VALUE"] / b["VALUE"].sum()
    books.append(b.assign(**ids))
H = pd.concat(books, ignore_index=True).sort_values(["manager_order", "period_order", "weight", "CUSIP"], ascending=[True, True, False, True])
H = H[ID + ["CUSIP", "issuer_name_canonical", "titleofclass_canonical", "VALUE", "SSHPRNAMT", "weight", "VOTING_AUTH_SOLE", "VOTING_AUTH_SHARED", "VOTING_AUTH_NONE"]]
H.to_csv(os.path.join(OUT_DIR, "effective_holdings.csv"), index=False)
audit = pd.DataFrame(audit)
audit.to_csv(os.path.join(OUT_DIR, "cleaning_audit.csv"), index=False)

# ---- manager concentration
mm = []
for (mo, po), b in H.groupby(["manager_order", "period_order"], sort=True):
    top = b.sort_values(["weight", "VALUE", "CUSIP"], ascending=[False, False, True]).iloc[0]
    hhi = float((b["weight"] ** 2).sum())
    mm.append(dict(b.iloc[0][ID], num_positions=len(b), total_value=float(b["VALUE"].sum()), top_cusip=top["CUSIP"], top_issuer_name=top["issuer_name_canonical"],
                   top_weight=float(top["weight"]), hhi=hhi, effective_num_names=1.0 / hhi))
pd.DataFrame(mm).to_csv(os.path.join(OUT_DIR, "manager_metrics.csv"), index=False)

# ---- consecutive-period turnover (target_periods order)
P = list(zip(per["period_order"], per["period_end"].dt.strftime("%Y-%m-%d")))
to = []
for (mo, lab, cik), g in H.groupby(["manager_order", "manager_label", "CIK"], sort=True):
    for (p0, e0), (p1, e1) in zip(P[:-1], P[1:]):
        u = pd.concat([g[g["period_order"] == p0].set_index("CUSIP")["weight"].rename("a"),
                       g[g["period_order"] == p1].set_index("CUSIP")["weight"].rename("b")], axis=1).fillna(0.0)
        to.append(dict(manager_order=mo, manager_label=lab, CIK=cik, period_order_prev=p0, period_end_prev=e0, period_order_curr=p1, period_end_curr=e1,
                       turnover=float(0.5 * (u["b"] - u["a"]).abs().sum()), new_names_count=int(((u["a"] == 0) & (u["b"] > 0)).sum()),
                       exited_names_count=int(((u["a"] > 0) & (u["b"] == 0)).sum()), common_names_count=int(((u["a"] > 0) & (u["b"] > 0)).sum())))
to = pd.DataFrame(to)
to.to_csv(os.path.join(OUT_DIR, "turnover.csv"), index=False)

# ---- latest-period overlap and crowding
lp, le = P[-1]
L = H[H["period_order"] == lp]
labels = dict(zip(mgr["manager_order"], mgr["manager_label"]))
W = L.pivot(index="CUSIP", columns="manager_order", values="weight").fillna(0.0)
ov = pd.DataFrame([dict(manager_i_order=i, manager_i_label=labels[i], manager_j_order=j, manager_j_label=labels[j], period_end=le,
                        weighted_overlap=float(np.minimum(W[i], W[j]).sum()), common_names_count=int(((W[i] > 0) & (W[j] > 0)).sum()))
                   for i, j in combinations(sorted(W.columns), 2)])
ov.to_csv(os.path.join(OUT_DIR, "overlap_latest.csv"), index=False)
cr = L.sort_values("manager_order").groupby("CUSIP", sort=False).agg(
    issuer_name_canonical=("issuer_name_canonical", canon), titleofclass_canonical=("titleofclass_canonical", canon), holder_count=("manager_label", "size"),
    total_value=("VALUE", "sum"), avg_holder_weight=("weight", "mean"), max_holder_weight=("weight", "max"), holder_list=("manager_label", "|".join)).reset_index()
cr = cr.sort_values(["holder_count", "total_value", "CUSIP"], ascending=[False, False, True]).reset_index(drop=True)
cr.insert(0, "rank", np.arange(1, len(cr) + 1))
cr.insert(1, "period_end", le)
cr = cr[["rank", "period_end", "CUSIP", "issuer_name_canonical", "titleofclass_canonical", "holder_count", "total_value", "avg_holder_weight", "max_holder_weight", "holder_list"]]
cr.to_csv(os.path.join(OUT_DIR, "crowded_securities_latest.csv"), index=False)

# ---- summary + self-checks
files = ["filing_resolution.csv", "effective_holdings.csv", "manager_metrics.csv", "turnover.csv", "overlap_latest.csv", "crowded_securities_latest.csv", "cleaning_audit.csv"]
w_ok = bool(np.allclose(H.groupby(["manager_order", "period_order"])["weight"].sum(), 1.0, atol=1e-10, rtol=0))
sp_ok = bool((fr["summarypage_row_diff"] == 0).all() and (fr["summarypage_value_diff"].abs() <= 0.5 * fr["rows_in_filing"]).all())  # <= $0.5 rounding per row
files_ok = all(os.path.exists(os.path.join(OUT_DIR, x)) for x in files)
audit_ok = bool((audit["raw_effective_rows"] == audit[[c for c in audit.columns if c.startswith("kept") or c.startswith("filtered")]].sum(axis=1)).all())
it, io_ = to["turnover"].idxmax(), ov["weighted_overlap"].idxmax()
summary = {"num_managers": int(len(mgr)), "num_periods": int(len(per)), "latest_period_end": le, "num_applied_filings": int(len(fr)),
           "num_clean_positions": int(len(H)), "num_unique_latest_cusips": int(cr["CUSIP"].nunique()),
           "num_append_new_holdings_filings": int((fr["action"] == "append_new_holdings").sum()), "num_restatement_filings": int((fr["action"] == "replace_restatement").sum()),
           "num_filtered_nonsh_rows": int(audit["filtered_nonsh_rows"].sum()), "num_filtered_putcall_rows": int(audit["filtered_putcall_rows"].sum()),
           "max_summarypage_abs_value_diff": float(fr["summarypage_value_diff"].abs().max()),
           "max_turnover_manager": to.loc[it, "manager_label"], "max_turnover_value": float(to.loc[it, "turnover"]),
           "max_overlap_pair": [ov.loc[io_, "manager_i_label"], ov.loc[io_, "manager_j_label"]], "max_overlap_value": float(ov.loc[io_, "weighted_overlap"]),
           "top_crowded_cusip": str(cr.loc[0, "CUSIP"]), "top_crowded_holder_count": int(cr.loc[0, "holder_count"]),
           "weights_sum_check_passed": w_ok, "summarypage_checks_passed": sp_ok, "all_required_files_written": bool(files_ok),
           "all_checks_passed": bool(w_ok and sp_ok and files_ok and audit_ok)}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
