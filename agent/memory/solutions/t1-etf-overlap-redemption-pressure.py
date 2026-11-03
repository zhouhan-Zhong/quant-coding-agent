# memory-solution
# unit: t1-etf-overlap-redemption-pressure | category: portfolio | tags: etf, holdings, overlap, redemption, liquidity, concentration, hhi, excel, openpyxl, spdr, scenario-analysis
# deliverables: fund_snapshot.csv, curated_universe.csv, scenario_pressure.csv, scenario_summary.csv, fund_scenario_bridge.csv, pressure_decomposition.csv, pressure_rankings.csv, summary.json
# Holdings xlsx: header row = row whose first cell is "Name"; Weight is in percent (/100); top 30 by (weight desc, ticker asc).
# Product xlsx: header row contains "Ticker"/"NAV"; parse "$653,201.93 M" -> 653201930000.0, "0.02%" -> 0.0002 (Decimal, exact);
#   premium_discount is the REPORTED %, not (close-nav)/nav. Exposures outside a fund's top 30 count as zero (shares, weight, dollars).
# implied_price = TNA * w / shares_held (blank if not held); holding $ = shares * implied; reconciled = combined $ / combined shares.
# Scenario: sell_shares = shares_held * fund pct, sell $ = sell_shares * implied price; pct_holdings = combined sell shares / combined shares.
# Rank sell-% on the exact float values (ties -> sell $ desc, ticker asc): float noise (0.0030000000000000005) decides the reference ranks.
# Bridge: redeemed shares = shares_out * pct; NAV $ / market $ = shares * nav / close; capture = top30 sell $ / NAV $; transfer bps = 1e4*(mkt-nav)/nav.
import os, json, math
from decimal import Decimal
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
FUNDS = ["SPY", "XLK"]
out = lambda df, name: df.to_csv(os.path.join(OUT_DIR, name), index=False)

def num(v):                                           # "$653,201.93 M" / "0.02%" / "24,799,876" -> float
    if isinstance(v, (int, float, np.integer, np.floating)): return float(v)
    s = str(v).strip().replace("$", "").replace(",", "")
    if s in ("", "-", "nan"): return float("nan")
    mult = Decimal(1)
    if s.endswith("%"): s, mult = s[:-1], Decimal("0.01")
    elif s[-1] in "KMB": s, mult = s[:-1], Decimal(10) ** {"K": 3, "M": 6, "B": 9}[s[-1]]
    return float(Decimal(s.strip()) * mult)

def holdings(fund):
    d = pd.read_excel(os.path.join(DATA_DIR, f"{fund.lower()}_holdings.xlsx"), header=None, engine="openpyxl")
    h = d.index[d[0].astype(str).str.strip() == "Name"][0]
    t = d.iloc[h + 1:].copy(); t.columns = [str(c).strip() for c in d.iloc[h]]
    t["w"] = pd.to_numeric(t["Weight"], errors="coerce") / 100.0
    t["Ticker"] = t["Ticker"].astype(str).str.strip()
    t = t[t["w"].notna() & ~t["Ticker"].isin(["-", "", "nan"])]
    t = t.sort_values(["w", "Ticker"], ascending=[False, True]).head(30)
    return t.assign(shares=t["Shares Held"].astype(float), name=t["Name"].astype(str).str.strip(), sector=t["Sector"].astype(str).str.strip())

pdx = pd.read_excel(os.path.join(DATA_DIR, "spdr_product_data.xlsx"), header=None, engine="openpyxl")
hr = next(i for i in range(len(pdx)) if {"Ticker", "NAV"} <= set(pdx.iloc[i].astype(str).str.strip()))
prod = pdx.iloc[hr + 1:].copy(); prod.columns = [str(c).strip() for c in pdx.iloc[hr]]
prod = prod[prod["Ticker"].astype(str).str.strip().isin(FUNDS)].set_index("Ticker")
H = {f: holdings(f) for f in FUNDS}

snap, F = [], {}
for f in FUNDS:
    p, t = prod.loc[f], H[f]
    F[f] = dict(close=num(p["Closing Price"]), nav=num(p["NAV"]), so=num(p["Shares Outstanding"]), vol=num(p["Exchange Volume (shares)"]),
                tna=num(p["Total Net Assets"]), cash=num(p["Net Cash Amount"]))
    asof = pd.to_datetime(str(p["As of**"]).strip(), format="%b %d %Y").strftime("%Y-%m-%d")
    snap.append(dict(ticker=f, fund_name=str(p["Name"]).strip(), as_of_date=asof, closing_price=F[f]["close"], nav=F[f]["nav"],
                     premium_discount=num(p["Premium/Discount"]), shares_outstanding=F[f]["so"], exchange_volume_shares=int(F[f]["vol"]),
                     share_turnover=F[f]["vol"] / F[f]["so"], total_net_assets=F[f]["tna"], net_cash_amount=F[f]["cash"],
                     cash_buffer_bps=1e4 * F[f]["cash"] / F[f]["tna"], top10_weight_sum=float(t["w"].head(10).sum()),
                     top30_weight_sum=float(t["w"].sum()), top30_constituent_hhi=float((t["w"] ** 2).sum()),
                     top_weight_ticker=t["Ticker"].iloc[0], top_weight_pct=float(t["w"].iloc[0])))
snap = pd.DataFrame(snap).sort_values("ticker"); out(snap, "fund_snapshot.csv")

# ---- curated universe (union of the two top-30 lists)
u = pd.DataFrame(index=sorted(set(H["SPY"]["Ticker"]) | set(H["XLK"]["Ticker"])))
for f in FUNDS:
    t, k = H[f].set_index("Ticker"), f.lower()
    u[f"in_{k}"] = u.index.isin(t.index).astype(int)
    u[f"shares_held_{k}"] = t["shares"].reindex(u.index).fillna(0.0)
    u[f"weight_{k}"] = t["w"].reindex(u.index).fillna(0.0)
    u[f"implied_price_{k}"] = (F[f]["tna"] * t["w"] / t["shares"]).reindex(u.index)          # NaN when not held
    u[f"holding_dollars_{k}"] = (u[f"shares_held_{k}"] * u[f"implied_price_{k}"]).fillna(0.0)
    for c in ("name", "sector"):
        u[c] = u.get(c, pd.Series(index=u.index, dtype=object)).fillna(t[c].reindex(u.index))
u["overlap_flag"] = (u["in_spy"] & u["in_xlk"]).astype(int)
u["combined_shares_held"] = u["shares_held_spy"] + u["shares_held_xlk"]
u["weight_sum"] = u["weight_spy"] + u["weight_xlk"]; u["overlap_weight_min"] = np.minimum(u["weight_spy"], u["weight_xlk"])
u["combined_holding_dollars"] = u["holding_dollars_spy"] + u["holding_dollars_xlk"]
u["reconciled_price"] = u["combined_holding_dollars"] / u["combined_shares_held"]
u["xlk_share_of_combined_dollars"] = u["holding_dollars_xlk"] / u["combined_holding_dollars"]
u = u.rename_axis("ticker").reset_index().sort_values(["overlap_flag", "combined_holding_dollars", "ticker"], ascending=[False, False, True])
u["category"] = np.where(u["overlap_flag"] == 1, "overlap", np.where(u["in_spy"] == 1, "spy_only", "xlk_only"))
UCOLS = ["ticker", "name", "sector", "in_spy", "in_xlk", "overlap_flag", "shares_held_spy", "shares_held_xlk", "combined_shares_held",
         "weight_spy", "weight_xlk", "weight_sum", "overlap_weight_min", "implied_price_spy", "implied_price_xlk", "reconciled_price",
         "holding_dollars_spy", "holding_dollars_xlk", "combined_holding_dollars", "xlk_share_of_combined_dollars"]
out(u[UCOLS], "curated_universe.csv")

# ---- scenarios
sc = pd.read_csv(os.path.join(DATA_DIR, "redemption_scenarios.csv"))
press, summ, bridge, decomp, ranks, winners = [], [], [], [], [], {}
for _, s in sc.iterrows():
    name, pct = s["scenario"], {"SPY": float(s["spy_redemption_pct"]), "XLK": float(s["xlk_redemption_pct"])}
    g = u.copy(); g.insert(0, "scenario", name)
    for f in FUNDS:
        k = f.lower()
        g[f"sell_shares_{k}"] = g[f"shares_held_{k}"] * pct[f]
        g[f"sell_dollars_{k}"] = (g[f"sell_shares_{k}"] * g[f"implied_price_{k}"]).fillna(0.0)
    g["combined_sell_shares"] = g["sell_shares_spy"] + g["sell_shares_xlk"]
    g["combined_sell_dollars"] = g["sell_dollars_spy"] + g["sell_dollars_xlk"]
    g["combined_sell_as_pct_combined_holdings"] = g["combined_sell_shares"] / g["combined_shares_held"]
    g["combined_sell_as_pct_combined_dollars"] = g["combined_sell_dollars"] / g["combined_holding_dollars"]
    g = g.sort_values(["combined_sell_dollars", "ticker"], ascending=[False, True]).reset_index(drop=True)
    g["rank_combined_sell_dollars"] = np.arange(1, len(g) + 1)
    order = g.sort_values(["combined_sell_as_pct_combined_holdings", "combined_sell_dollars", "ticker"], ascending=[False, False, True]).index
    g.loc[order, "rank_sell_pct_combined_holdings"] = np.arange(1, len(g) + 1)
    press.append(g)
    tot = float(g["combined_sell_dollars"].sum())
    B = {b: g[g["category"] == b] for b in ["overlap", "spy_only", "xlk_only"]}
    top, otop = g.iloc[0], B["overlap"].iloc[0]
    g["is_top_overlap_name"] = (g["ticker"] == otop["ticker"]).astype(int)
    ranks.append(g)
    summ.append(dict(scenario=name, total_combined_sell_dollars=tot, overlap_sell_dollars=float(B["overlap"]["combined_sell_dollars"].sum()),
                     spy_only_sell_dollars=float(B["spy_only"]["combined_sell_dollars"].sum()), xlk_only_sell_dollars=float(B["xlk_only"]["combined_sell_dollars"].sum()),
                     overlap_share_of_total=float(B["overlap"]["combined_sell_dollars"].sum()) / tot,
                     constituent_hhi=float(((g["combined_sell_dollars"] / tot) ** 2).sum()), top_ticker_by_dollars=top["ticker"],
                     top_ticker_sell_dollars=float(top["combined_sell_dollars"]), top_overlap_ticker=otop["ticker"],
                     top_overlap_sell_dollars=float(otop["combined_sell_dollars"]),
                     names_above_100m_sell_dollars=int((g["combined_sell_dollars"] >= 1e8).sum()),
                     names_above_250m_sell_dollars=int((g["combined_sell_dollars"] >= 2.5e8).sum())))
    for b, bg in B.items():
        decomp.append(dict(scenario=name, bucket=b, name_count=int((bg["combined_sell_dollars"] > 0).sum()), sell_shares=float(bg["combined_sell_shares"].sum()),
                           sell_dollars=float(bg["combined_sell_dollars"].sum()), share_of_total_sell_dollars=float(bg["combined_sell_dollars"].sum()) / tot,
                           largest_ticker=bg["ticker"].iloc[0] if len(bg) else "", largest_ticker_sell_dollars=float(bg["combined_sell_dollars"].iloc[0]) if len(bg) else 0.0))
    for f in FUNDS:
        k, Ff = f.lower(), F[f]
        rs = Ff["so"] * pct[f]; navd, mktd = rs * Ff["nav"], rs * Ff["close"]
        t30 = float(g[f"sell_dollars_{k}"].sum()); ov = float(g.loc[g["overlap_flag"] == 1, f"sell_dollars_{k}"].sum())
        bridge.append(dict(scenario=name, ticker=f, redemption_pct=pct[f], redeemed_fund_shares=rs, redeemed_nav_dollars=navd, redeemed_market_dollars=mktd,
                           redeemed_as_pct_exchange_volume=rs / Ff["vol"], top30_sell_dollars=t30, top30_capture_ratio=t30 / navd, overlap_sell_dollars=ov,
                           fund_specific_sell_dollars=float(g.loc[g["overlap_flag"] == 0, f"sell_dollars_{k}"].sum()),
                           overlap_share_of_top30_sell=ov / t30 if t30 else 0.0, premium_discount_transfer_bps=1e4 * (mktd - navd) / navd))

PCOLS = ["scenario", "ticker", "name", "sector", "overlap_flag", "sell_shares_spy", "sell_shares_xlk", "combined_sell_shares", "sell_dollars_spy",
         "sell_dollars_xlk", "combined_sell_dollars", "combined_sell_as_pct_combined_holdings", "combined_sell_as_pct_combined_dollars"]
out(pd.concat(press)[PCOLS], "scenario_pressure.csv")
summ, bridge = pd.DataFrame(summ), pd.DataFrame(bridge)
out(summ, "scenario_summary.csv"); out(bridge, "fund_scenario_bridge.csv"); out(pd.DataFrame(decomp), "pressure_decomposition.csv")
rk = pd.concat(ranks)
rk["rank_sell_pct_combined_holdings"] = rk["rank_sell_pct_combined_holdings"].astype(int)
out(rk[["scenario", "rank_combined_sell_dollars", "rank_sell_pct_combined_holdings", "ticker", "name", "category", "combined_sell_dollars",
        "combined_sell_as_pct_combined_holdings", "overlap_weight_min", "is_top_overlap_name"]], "pressure_rankings.csv")

best = lambda df, col, keys: df.sort_values([col] + keys, ascending=[False] + [True] * len(keys)).iloc[0]
ext = {}
for key, col in [("largest_redeemed_nav_dollars", "redeemed_nav_dollars"), ("highest_redeemed_as_pct_exchange_volume", "redeemed_as_pct_exchange_volume"),
                 ("highest_top30_capture_ratio", "top30_capture_ratio")]:
    r = best(bridge, col, ["scenario", "ticker"]); ext[f"{key}_scenario"], ext[f"{key}_ticker"] = r["scenario"], r["ticker"]
for _, s in summ.iterrows():
    winners[s["scenario"]] = dict(top_ticker_by_dollars=s["top_ticker_by_dollars"], top_overlap_ticker=s["top_overlap_ticker"],
                                  largest_redeemed_nav_dollars_ticker=best(bridge[bridge["scenario"] == s["scenario"]], "redeemed_nav_dollars", ["ticker"])["ticker"],
                                  overlap_share_of_total=float(s["overlap_share_of_total"]), names_above_100m_sell_dollars=int(s["names_above_100m_sell_dollars"]))
summary = {
    "as_of_date": snap["as_of_date"].iloc[0], "funds": FUNDS, "top30_counts": {f: int(len(H[f])) for f in FUNDS},
    "universe_size": int(len(u)), "overlap_name_count": int(u["overlap_flag"].sum()),
    "metric_winners": {k: best(snap, c, ["ticker"])["ticker"] for k, c in [("largest_total_net_assets", "total_net_assets"),
                       ("highest_cash_buffer_bps", "cash_buffer_bps"), ("highest_top30_hhi", "top30_constituent_hhi"), ("largest_top30_weight_sum", "top30_weight_sum")]},
    "largest_constituent_by_combined_holding_dollars": best(u, "combined_holding_dollars", ["ticker"])["ticker"],
    "largest_overlap_weight_min_ticker": best(u, "overlap_weight_min", ["ticker"])["ticker"],
    "fund_bridge_extremes": ext, "scenario_winners": winners,
}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
