# memory-solution
# unit: t1-form4-cross-sectional-sale-pressure | category: text-filings | tags: form-4, xml, elementtree, insider-sales, ownership-rollforward, footnotes, price-range, sale-pressure, cross-sectional-rank, hhi
# deliverables: transactions_normalized.csv, ownership_rollforward.csv, entity_day_sale_pressure.csv, issuer_pressure_scenarios.csv, cross_sectional_rankings.csv, pressure_decomposition.csv, summary.json
# sale ledger = nonDerivativeTransaction rows with code S only (drop same-day C conversion / F tax-withholding rows); nonDerivativeHolding rows = static buckets
# entity: D -> "Direct holdings"; else natureOfOwnership text; "See footnote" -> name from its footnote ("Trustee of The X u/a/d" / "held by the X")
# ownership_footnotes = footnote numbers under postTransactionAmounts + ownershipNature; price band = price footnote "ranging from $a to $b", bps = 1e4 (b-a)/mid
# bucket inventory before trading = first sale's shares + shares_following (per entity-day; per filing for the rollforward); ending = last shares_following
# liquidation = 1e4 (.55 pct_inv + .25 sh/adv20 + .20 $/(adv20*close)); blended = 1e4 (.24 sh/dayvol + .20 sh/adv20 + .12 $/adv20$ + .24 sh/inventory + .12 sh/float + .08 HHI)
# ranks: rank(method='min', ascending=False); winners by (metric desc, shares_sold desc, ticker asc); filing inventory = sum over its entity-days
import os, re, json, glob
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
KEY = ["filing_id", "transaction_date"]


def txt(el, path):
    x = el.find(path)
    return (x.text or "").strip() if x is not None else ""


def fns(el, path):  # footnote numbers attached anywhere under el/path
    x = el.find(path)
    return [int(re.sub(r"\D", "", f.get("id"))) for f in x.iter("footnoteId")] if x is not None else []


def num(s):
    v = float(s)
    return int(v) if v.is_integer() else v


def entity(el, notes):
    form, nature = txt(el, "ownershipNature/directOrIndirectOwnership/value"), txt(el, "ownershipNature/natureOfOwnership/value")
    if form == "D" and (not nature or nature.lower().startswith("direct")):
        return "Direct holdings"
    if re.fullmatch(r"see footnotes?", nature, re.I):
        for n in fns(el, "ownershipNature/natureOfOwnership"):
            t = notes.get(n, "")
            m = re.search(r"Trustee of (?:the )?(.+?)(?: u/a/d|,|\.)", t, re.I) or re.search(r"held by (?:the )?(.+?)(?: u/a/d|,|\.|$)", t, re.I)
            if m:
                return m.group(1).strip()
    return nature


tx_rows, hold_rows, meta = [], [], {}
for path in sorted(glob.glob(os.path.join(DATA_DIR, "form4_*.xml"))):
    fid = re.search(r"form4_(\w+)\.xml$", path).group(1)
    root = ET.parse(path).getroot()
    notes = {int(re.sub(r"\D", "", f.get("id"))): " ".join((f.text or "").split()) for f in root.iter("footnote")}
    info = dict(filing_id=fid, issuer_name=txt(root, "issuer/issuerName"), ticker=txt(root, "issuer/issuerTradingSymbol").upper(),
                reporting_owner_name=txt(root, "reportingOwner/reportingOwnerId/rptOwnerName"))
    meta[fid] = dict(info, report_date=txt(root, "periodOfReport"))
    plan = txt(root, "aff10b5One").lower() in ("1", "true")
    k = 0
    for el in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        if txt(el, "transactionCoding/transactionCode").upper() != "S":
            continue
        k += 1
        sh, px = num(txt(el, "transactionAmounts/transactionShares/value")), float(txt(el, "transactionAmounts/transactionPricePerShare/value"))
        pfn, lo, hi = None, np.nan, np.nan
        for n in fns(el, "transactionAmounts/transactionPricePerShare"):
            m = re.search(r"ranging from \$\s*(\d[\d,]*(?:\.\d+)?) to \$\s*(\d[\d,]*(?:\.\d+)?)", notes.get(n, ""))
            if m:
                pfn, lo, hi = n, float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
                break
        own = sorted(set(fns(el, "postTransactionAmounts") + fns(el, "ownershipNature")))
        tx_rows.append(dict(info, transaction_index=k, transaction_date=txt(el, "transactionDate/value")[:10], security_title=txt(el, "securityTitle/value"),
                            transaction_code="S", shares=sh, price_per_share=px, gross_value=sh * px,
                            acquired_disposed_code=txt(el, "transactionAmounts/transactionAcquiredDisposedCode/value"),
                            shares_following_transaction=num(txt(el, "postTransactionAmounts/sharesOwnedFollowingTransaction/value")),
                            ownership_form=txt(el, "ownershipNature/directOrIndirectOwnership/value"), ownership_entity=entity(el, notes),
                            price_footnote_number=pfn, price_band_low=lo, price_band_high=hi, price_band_width_bps=1e4 * (hi - lo) / ((hi + lo) / 2),
                            ownership_footnotes=";".join(map(str, own)),
                            rule_10b5_1_flag=bool(plan or any("10b5-1" in notes.get(n, "") for n in fns(el, ".")))))
    for el in root.findall("nonDerivativeTable/nonDerivativeHolding"):
        own = sorted(set(fns(el, "postTransactionAmounts") + fns(el, "ownershipNature")))
        hold_rows.append(dict(filing_id=fid, ticker=info["ticker"], report_date=meta[fid]["report_date"], ownership_entity=entity(el, notes),
                              ownership_form=txt(el, "ownershipNature/directOrIndirectOwnership/value"), ownership_footnotes=";".join(map(str, own)),
                              shares=num(txt(el, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"))))

tx = pd.DataFrame(tx_rows)
tx["price_footnote_number"] = tx["price_footnote_number"].astype("Int64")
tx.to_csv(os.path.join(OUT_DIR, "transactions_normalized.csv"), index=False)

# ---- ownership rollforward: seller buckets (filing-level) + static-only holding buckets
g = tx.groupby(["filing_id", "ownership_entity"], sort=False)
sel = g.agg(ticker=("ticker", "first"), ownership_form=("ownership_form", "first"), ownership_footnotes=("ownership_footnotes", "first"),
            f_sh=("shares", "first"), f_post=("shares_following_transaction", "first"), sold_on_report_date=("shares", "sum"),
            ending_shares=("shares_following_transaction", "last")).reset_index()
sel["starting_shares"] = sel["f_sh"] + sel["f_post"]
sel["report_date"] = sel["filing_id"].map(lambda f: meta[f]["report_date"])
sel["is_seller"], sel["is_static_holding_only"] = True, False
hold = pd.DataFrame(hold_rows)
seller_keys = set(zip(sel["filing_id"], sel["ownership_entity"]))
st = hold[[(f, e) not in seller_keys for f, e in zip(hold["filing_id"], hold["ownership_entity"])]].copy()
st["starting_shares"] = st["ending_shares"] = st["shares"]
st["sold_on_report_date"], st["is_seller"], st["is_static_holding_only"] = 0, False, True
ROLL = ["filing_id", "ticker", "report_date", "ownership_entity", "ownership_form", "ownership_footnotes", "starting_shares",
        "sold_on_report_date", "ending_shares", "pct_reported_holdings_sold", "is_seller", "is_static_holding_only"]
roll = pd.concat([sel, st], ignore_index=True)
roll["pct_reported_holdings_sold"] = roll["sold_on_report_date"] / roll["starting_shares"]
roll[ROLL].to_csv(os.path.join(OUT_DIR, "ownership_rollforward.csv"), index=False)
static = roll[roll["is_static_holding_only"]]

# ---- entity-day buckets
mc = pd.read_csv(os.path.join(DATA_DIR, "market_context.csv"), dtype={"transaction_date": str})
ed = tx.groupby(KEY + ["ownership_entity"], sort=False).agg(
    ticker=("ticker", "first"), issuer_name=("issuer_name", "first"), sale_transactions=("shares", "size"), shares_sold=("shares", "sum"),
    gross_proceeds=("gross_value", "sum"), f_sh=("shares", "first"), f_post=("shares_following_transaction", "first"),
    ending_shares_after_last_trade=("shares_following_transaction", "last")).reset_index()
ed["inventory_before_trading"] = ed["f_sh"] + ed["f_post"]
ed = ed.merge(mc.drop(columns="ticker"), on=KEY, how="left")
ed["weighted_average_sale_price"] = ed["gross_proceeds"] / ed["shares_sold"]
ed["pct_inventory_sold"] = ed["shares_sold"] / ed["inventory_before_trading"]
ed["shares_sold_over_adv20"] = ed["shares_sold"] / ed["adv20_shares"]
ed["dollars_sold_over_adv20_dollar_volume"] = ed["gross_proceeds"] / (ed["adv20_shares"] * ed["close_price"])
ed["liquidation_score"] = 1e4 * (0.55 * ed["pct_inventory_sold"] + 0.25 * ed["shares_sold_over_adv20"] + 0.20 * ed["dollars_sold_over_adv20_dollar_volume"])
ed["frac"] = ed["shares_sold"] / ed.groupby(KEY)["shares_sold"].transform("sum")
ed["frac2"] = ed["frac"] ** 2
ed[["filing_id", "ticker", "issuer_name", "transaction_date", "ownership_entity", "inventory_before_trading", "sale_transactions", "shares_sold",
    "gross_proceeds", "weighted_average_sale_price", "ending_shares_after_last_trade", "pct_inventory_sold", "shares_sold_over_adv20",
    "dollars_sold_over_adv20_dollar_volume", "liquidation_score"]].to_csv(os.path.join(OUT_DIR, "entity_day_sale_pressure.csv"), index=False)

# ---- issuer-day scenarios
iss = ed.groupby(KEY, sort=False).agg(ticker=("ticker", "first"), issuer_name=("issuer_name", "first"), shares_sold=("shares_sold", "sum"),
                                      gross_proceeds=("gross_proceeds", "sum"), seller_entities=("ownership_entity", "nunique"),
                                      seller_concentration_hhi=("frac2", "sum"), max_entity_pct_of_day_sales=("frac", "max"),
                                      inv=("inventory_before_trading", "sum"), end=("ending_shares_after_last_trade", "sum")).reset_index()
iss = iss.merge(mc.drop(columns="ticker"), on=KEY, how="left")
iss["weighted_average_sale_price"] = iss["gross_proceeds"] / iss["shares_sold"]
for col, den in [("sale_shares_over_day_volume", "daily_volume_shares"), ("sale_shares_over_adv20", "adv20_shares"), ("sale_shares_over_float", "float_shares"),
                 ("sale_shares_over_outstanding", "shares_outstanding"), ("sale_shares_over_reported_inventory", "inv")]:
    iss[col] = iss["shares_sold"] / iss[den]
iss["sale_dollars_over_adv20_dollar_volume"] = iss["gross_proceeds"] / (iss["adv20_shares"] * iss["close_price"])
iss["blended_pressure_score"] = 1e4 * (0.24 * iss["sale_shares_over_day_volume"] + 0.20 * iss["sale_shares_over_adv20"]
                                       + 0.12 * iss["sale_dollars_over_adv20_dollar_volume"] + 0.24 * iss["sale_shares_over_reported_inventory"]
                                       + 0.12 * iss["sale_shares_over_float"] + 0.08 * iss["seller_concentration_hhi"])
iss[["filing_id", "ticker", "issuer_name", "transaction_date", "shares_sold", "gross_proceeds", "weighted_average_sale_price", "seller_entities",
     "seller_concentration_hhi", "max_entity_pct_of_day_sales", "sale_shares_over_day_volume", "sale_shares_over_adv20",
     "sale_dollars_over_adv20_dollar_volume", "sale_shares_over_float", "sale_shares_over_outstanding", "sale_shares_over_reported_inventory",
     "blended_pressure_score"]].to_csv(os.path.join(OUT_DIR, "issuer_pressure_scenarios.csv"), index=False)

# ---- cross-sectional ranks (1 = most pressure)
METRICS = ["shares_sold", "gross_proceeds", "sale_shares_over_day_volume", "sale_shares_over_adv20", "sale_shares_over_float",
           "sale_shares_over_reported_inventory", "blended_pressure_score"]
rk = iss[["filing_id", "ticker", "transaction_date"]].copy()
for m in METRICS:
    rk["rank_" + m] = iss[m].rank(method="min", ascending=False).astype(int)
R = rk[["rank_" + m for m in METRICS]]
rk["best_rank"], rk["worst_rank"], rk["top_metric_count"] = R.min(axis=1), R.max(axis=1), (R == 1).sum(axis=1)
rk["rank_spread"] = rk["worst_rank"] - rk["best_rank"]
rk = rk[["filing_id", "ticker", "transaction_date"] + list(R.columns) + ["best_rank", "worst_rank", "rank_spread", "top_metric_count"]]
rk.to_csv(os.path.join(OUT_DIR, "cross_sectional_rankings.csv"), index=False)

# ---- decomposition per issuer-day
dec_rows = []
for _, r in iss.iterrows():
    t = tx[(tx["filing_id"] == r["filing_id"]) & (tx["transaction_date"] == r["transaction_date"])]
    e = ed[(ed["filing_id"] == r["filing_id"]) & (ed["transaction_date"] == r["transaction_date"])].sort_values(["shares_sold", "ownership_entity"], ascending=[False, True])
    s = static[static["filing_id"] == r["filing_id"]]
    tot, d, top = t["shares"].sum(), t.loc[t["ownership_form"] == "D", "shares"].sum(), e.iloc[0]
    cov = t.loc[t["price_footnote_number"].notna(), "shares"].sum()
    dec_rows.append(dict(filing_id=r["filing_id"], ticker=r["ticker"], transaction_date=r["transaction_date"], direct_sale_shares=d,
                         indirect_sale_shares=t.loc[t["ownership_form"] == "I", "shares"].sum(), direct_sale_pct=d / tot,
                         indirect_sale_pct=t.loc[t["ownership_form"] == "I", "shares"].sum() / tot, top_entity_name=top["ownership_entity"],
                         top_entity_sale_shares=top["shares_sold"], top_entity_sale_pct=top["shares_sold"] / tot,
                         tail_entity_sale_shares=tot - top["shares_sold"], tail_entity_sale_pct=(tot - top["shares_sold"]) / tot,
                         price_range_covered_shares=cov, price_range_coverage_pct=cov / tot, reported_inventory_before_trading=r["inv"],
                         seller_ending_shares=r["end"], static_only_ending_shares=s["ending_shares"].sum(), direct_seller_flag=bool(d > 0),
                         static_holding_only_entities="|".join(s["ownership_entity"])))
pd.DataFrame(dec_rows).to_csv(os.path.join(OUT_DIR, "pressure_decomposition.csv"), index=False)

# ---- summary
winners = {}
for m in METRICS:
    w = iss.sort_values([m, "shares_sold", "ticker"], ascending=[False, False, True]).iloc[0]
    winners[m] = {"ticker": w["ticker"], "transaction_date": w["transaction_date"], "value": float(w[m])}
sp = rk.sort_values(["rank_spread", "best_rank", "ticker"], ascending=[False, True, True]).iloc[0]
big = static.sort_values("ending_shares", ascending=False).iloc[0]
per = {}
for fid, m in meta.items():
    t, e, i, s = tx[tx["filing_id"] == fid], ed[ed["filing_id"] == fid], iss[iss["filing_id"] == fid], static[static["filing_id"] == fid]
    ent = t.groupby("ownership_entity")["shares"].sum().reset_index().sort_values(["shares", "ownership_entity"], ascending=[False, True]).iloc[0]
    pk = i.sort_values("blended_pressure_score", ascending=False).iloc[0]
    per[m["ticker"]] = {"reporting_owner_name": m["reporting_owner_name"], "report_date": m["report_date"], "sale_dates": sorted(t["transaction_date"].unique().tolist()),
                        "total_sale_transactions": int(len(t)), "total_shares_sold": int(t["shares"].sum()), "total_gross_proceeds": float(t["gross_value"].sum()),
                        "seller_entities": int(t["ownership_entity"].nunique()), "direct_seller_flag": bool((t["ownership_form"] == "D").any()),
                        "static_holding_only_entities": s["ownership_entity"].tolist(), "top_entity_name": ent["ownership_entity"],
                        "top_entity_sale_shares": int(ent["shares"]), "peak_blended_pressure_date": pk["transaction_date"],
                        "peak_blended_pressure_score": float(pk["blended_pressure_score"]), "static_only_ending_shares": int(s["ending_shares"].sum()),
                        "reported_inventory_before_trading": int(e["inventory_before_trading"].sum())}
summary = {"filing_ids": list(meta), "tickers": [m["ticker"] for m in meta.values()], "issuer_day_count": int(len(iss)),
           "total_non_derivative_sale_transactions": int(len(tx)), "total_shares_sold": int(tx["shares"].sum()),
           "total_gross_sale_proceeds": float(tx["gross_value"].sum()), "metric_winners": winners,
           "highest_rank_spread": {"ticker": sp["ticker"], "transaction_date": sp["transaction_date"], "rank_spread": int(sp["rank_spread"])},
           "largest_static_holding_entity": big["ownership_entity"], "largest_static_holding_shares": int(big["ending_shares"]), "per_filing": per}
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
