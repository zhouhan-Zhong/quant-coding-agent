# memory-solution
# unit: t1-brinson-sector-attribution | category: portfolio | tags: brinson-fachler, attribution, allocation-selection-interaction, sector-etfs, weight-drift, quarterly-rebalance, cash-drag, monthly-returns
# deliverables: results.json
# monthly ETF returns from month-end (last trading day) closes, Dec-2023 anchor -> 12 returns Jan..Dec 2024
# benchmark R^b_m = sum_i wb_i(quarter) * R_i ; portfolio ETF for sector i from portfolio_etfs; cash return = annual_rate/12
# portfolio weights: reset to targets in Jan/Apr/Jul/Oct, otherwise drift w_i <- w_i(1+r_i)/sum_j w_j(1+r_j) (incl. CASH)
# BF effects with beginning-of-month weights: alloc=(wp-wb)(Rb_i-Rb), sel=wb(Rp_i-Rb_i), inter=(wp-wb)(Rp_i-Rb_i); CASH wb=0 -> alloc only
# annual returns are compounded; totals/quarters/sectors are plain sums of monthly effects
import os, json
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)

params = json.load(open(os.path.join(DATA_DIR, "params.json")))
px = pd.read_csv(os.path.join(DATA_DIR, "sector_etfs.csv"), parse_dates=["Date"]).sort_values("Date").set_index("Date")
cash = pd.read_csv(os.path.join(DATA_DIR, "cash_rates.csv")).set_index("month")["annual_rate"]

me = px.groupby(px.index.to_period("M")).last()                     # month-end closes, indexed by period
me = me.loc["2023-12":"2024-12"]
rets = (me / me.shift(1) - 1.0).loc["2024-01":"2024-12"]            # 12 monthly simple returns per ETF
months = rets.index
sectors = list(params["portfolio_etfs"].keys())                     # 11 equity sectors (benchmark ETF names)
pmap = params["portfolio_etfs"]
wp_target = params["portfolio_weights"]
holdings = sectors + ["CASH"]

wp = None
m_alloc, m_sel, m_inter, m_rp, m_rb = [], [], [], [], []
sec_alloc = {s: [] for s in holdings}; sec_sel = {s: 0.0 for s in sectors}; sec_inter = {s: 0.0 for s in sectors}
snapshot = None
for k, per in enumerate(months):
    q = f"Q{(per.month - 1) // 3 + 1}"
    wb = params["benchmark_weights"][q]
    if per.month in (1, 4, 7, 10):                                  # quarterly rebalance to targets
        wp = {h: float(wp_target[h]) for h in holdings}
    if per.month == 3:
        snapshot = dict(wp)
    r = rets.loc[per]
    rb_i = {s: float(r[s]) for s in sectors}
    rp_i = {s: float(r[pmap[s]]) for s in sectors}
    rp_i["CASH"] = float(cash[str(per)]) / 12.0
    Rb = sum(wb[s] * rb_i[s] for s in sectors)
    Rp = sum(wp[h] * rp_i[h] for h in holdings)
    a = s_ = i_ = 0.0
    for s in sectors:
        al = (wp[s] - wb[s]) * (rb_i[s] - Rb)
        se = wb[s] * (rp_i[s] - rb_i[s])
        it = (wp[s] - wb[s]) * (rp_i[s] - rb_i[s])
        a += al; s_ += se; i_ += it
        sec_alloc[s].append(al); sec_sel[s] += se; sec_inter[s] += it
    al_c = wp["CASH"] * (rp_i["CASH"] - Rb)                         # cash: wb = 0
    a += al_c; sec_alloc["CASH"].append(al_c)
    m_alloc.append(a); m_sel.append(s_); m_inter.append(i_); m_rp.append(Rp); m_rb.append(Rb)
    grow = {h: wp[h] * (1.0 + rp_i[h]) for h in holdings}           # drift weights to next month start
    tot = sum(grow.values())
    wp = {h: grow[h] / tot for h in holdings}

port_ret = float(np.prod(1.0 + np.array(m_rp)) - 1.0)
bench_ret = float(np.prod(1.0 + np.array(m_rb)) - 1.0)
res = {
    "portfolio_return": port_ret, "benchmark_return": bench_ret, "active_return": port_ret - bench_ret,
    "total_allocation_effect": float(sum(m_alloc)), "total_selection_effect": float(sum(m_sel)),
    "total_interaction_effect": float(sum(m_inter)),
    "monthly_allocation": m_alloc, "monthly_selection": m_sel, "monthly_interaction": m_inter,
    **{f"q{j + 1}_allocation": float(sum(m_alloc[3 * j:3 * j + 3])) for j in range(4)},
    "sector_allocation": {h: float(sum(sec_alloc[h])) for h in holdings},
    "sector_selection": {s: sec_sel[s] for s in sectors},
    "sector_interaction": {s: sec_inter[s] for s in sectors},
    "monthly_sector_allocation": {h: sec_alloc[h] for h in holdings},
    "weight_snapshot_march": snapshot,
}
json.dump(res, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
