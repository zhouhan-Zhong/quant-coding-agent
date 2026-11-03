# memory-solution
# unit: t1-sec-10k-report-long | category: text-filings | tags: sec, 10-k, inline-xbrl, ixbrl, xhtml-tables, lxml, zip, dei, us-gaap, scale-sign, year-over-year, walmart
# deliverables: results.json
# the -xbrl.zip holds an INLINE XBRL .htm (valid XHTML): parse with lxml.etree; facts = ix:nonFraction/ix:nonNumeric
#   (value = text * 10**scale, negated when sign="-"); contexts in ix:resources give period + explicit dimensions
# dei facts: DocumentType, DocumentAnnualReport (ballot box "☒" = True), DocumentPeriodEndDate ("January 31, 2025" -> ISO),
#   EntityRegistrantName (verbatim), EntityPublicFloat; Security12bTitle <-> TradingSymbol share the same contextRef
# us-gaap (no dims, current FY): net sales = RevenueFromContractWithCustomerExcludingAssessedTax, CostOfRevenue, OperatingIncomeLoss,
#   NetCashProvidedByUsedIn{Operating,Investing}Activities, TreasuryStockSharesAcquired / AverageCostPerShare, PaymentsForRepurchaseOfCommonStock
# untagged items from <table> rows (label regex -> numbers; "(x)" negative; header "(Amounts in millions)" -> x1e6):
#   square feet (totals in thousands per footnote; averages raw), executive-officer table, capex allocation, Sam's Club incl/excl fuel,
#   Item 7A short-term borrowings variable-rate total + rate, share-based comp by award type; first numeric column = latest FY
import os, re, json, glob, zipfile
from datetime import datetime
from lxml import etree

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
IX, XBRLI, XDI, H = ("{http://www.xbrl.org/2013/inlineXBRL}", "{http://www.xbrl.org/2003/instance}",
                     "{http://xbrl.org/2006/xbrldi}", "{http://www.w3.org/1999/xhtml}")
NUMCELL = re.compile(r"^\(?-?[\d,]*\.?\d+\)?$")


def clean(s):
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def num(c):
    v = float(c.strip("()").replace(",", ""))
    return -v if c.startswith("(") else v


class Report:
    def __init__(self, zpath):
        z = zipfile.ZipFile(zpath); names = z.namelist()
        xsd = [n for n in names if n.endswith(".xsd")]
        main = [n for n in names if xsd and n == xsd[0][:-4] + ".htm"] or sorted((n for n in names if n.endswith(".htm")), key=lambda n: -z.getinfo(n).file_size)
        self.root = etree.fromstring(z.read(main[0]), etree.XMLParser(huge_tree=True, recover=True))
        self.ctx = {}
        for c in self.root.iter(XBRLI + "context"):
            per = {etree.QName(x).localname: x.text.strip() for x in c.find(XBRLI + "period")}
            self.ctx[c.get("id")] = dict(per, dims={m.get("dimension"): m.text.strip() for m in c.iter(XDI + "explicitMember")})
        self.facts = [f for f in self.root.iter(IX + "nonFraction", IX + "nonNumeric")]
        self.end = self.iso(self.text("dei:DocumentPeriodEndDate"))
        self.tables = []
        for t in self.root.iter(H + "table"):
            rows = [[c for c in (clean("".join(td.itertext())) for td in tr if etree.QName(td).localname in ("td", "th"))
                     if c not in ("", "$", "%", ")")] for tr in t.iter(H + "tr")]
            rows = [r for r in rows if r]
            if rows:
                head = " ".join(" ".join(r) for r in rows[:3]).lower()
                mult = 1e9 if "in billions" in head else 1e6 if "in millions" in head else 1e3 if "in thousands" in head else 1.0
                self.tables.append((rows, mult, " | ".join(" ; ".join(r) for r in rows)))
        self.fulltext = clean(" ".join(self.root.itertext()))

    @staticmethod
    def iso(s):
        return datetime.strptime(clean(s), "%B %d, %Y").strftime("%Y-%m-%d")

    def text(self, name, ctx=None):
        return next(clean("".join(f.itertext())) for f in self.facts if f.get("name") == name and (ctx is None or f.get("contextRef") == ctx))

    def value(self, name):  # numeric fact, no dimensions, current fiscal period (duration ending at / instant at period end)
        for f in self.facts:
            c = self.ctx.get(f.get("contextRef"), {})
            if f.get("name") == name and not c.get("dims") and (c.get("endDate") or c.get("instant")) == self.end:
                v = float(clean("".join(f.itertext())).replace(",", "")) * 10 ** int(f.get("scale") or 0)
                return -v if f.get("sign") == "-" else v
        raise KeyError(name)

    def title_for_symbol(self, sym):
        ctx = next(f.get("contextRef") for f in self.facts if f.get("name") == "dei:TradingSymbol" and clean("".join(f.itertext())) == sym)
        return self.text("dei:Security12bTitle", ctx)

    def row(self, label, table_has=(), after=None):  # (numbers, multiplier) of the first matching table row
        for rows, mult, txt in self.tables:
            if not all(re.search(p, txt, re.I) for p in table_has):
                continue
            armed = after is None
            for r in rows:
                if not armed:
                    armed = bool(re.match(after, r[0], re.I)); continue
                vals = [num(c) for c in r[1:] if NUMCELL.match(c)]
                if re.match(label, r[0], re.I) and vals:
                    return vals, mult
        raise KeyError(label)


def extract(rep):
    out = {"DocumentType": rep.text("dei:DocumentType"),
           "DocumentAnnualReport": rep.text("dei:DocumentAnnualReport") in ("☒", "true", "Yes", "yes", "X", "x"),
           "DocumentPeriodEndDate": rep.end, "EntityRegistrantName": rep.text("dei:EntityRegistrantName"),
           "Security12bTitle_for_TradingSymbol__WMT26A": rep.title_for_symbol("WMT26A"),
           "Security12bTitle_for_TradingSymbol__WMT35A": rep.title_for_symbol("WMT35A")}
    out["aggregate_market_value_of_the_voting_common_stock_of_the_registrant_held_by_non_affiliates_of_the_registrant"] = \
        int(round(float(rep.text("dei:EntityPublicFloat").replace(",", "").replace("$", "")) *
                  10 ** int(next(f.get("scale") or 0 for f in rep.facts if f.get("name") == "dei:EntityPublicFloat"))))
    for rows, mult, txt in rep.tables:   # executive officers: Name | Business Experience | Since | Age
        off = next((r for r in rows if len(r) >= 3 and "Corporate Affairs" in r[1]), None)
        if re.search(r"\bName\b", txt) and re.search(r"\bAge\b", txt) and off:
            out["Corporate_Affairs_Officier"], out["Corporate_Affairs_Officier_Age"] = off[0], int(off[-1]); break
    sqft_mult = 1000 if re.search(r"square feet reported in thousands", rep.fulltext, re.I) else 1
    tot, avg = ("Total Company", r"square feet"), (r"Average", r"Supercenters")
    for key, lab in [("Walmart_U.S._Supercenters", r"Supercenters"), ("Walmart_U.S._Discount_Stores", r"Discount Stores"),
                     ("Walmart_U.S._Neighborhood_Markets", r"Neighborhood Markets"), ("Sam's_Club_U.S.", r"Sam.s Club U\.S\."),
                     ("Walmart_International_Retail", r"Retail$"), ("Walmart_International_Wholesale", r"Wholesale$")]:
        out[key + "_Total_Square_Feet"] = int(rep.row(lab, tot)[0][1] * sqft_mult)        # [unit count, total sq ft, ...]
        if key.startswith("Walmart_U.S."):
            out[key + "_Average_Square_Feet"] = int(rep.row(lab, avg)[0][-1])               # last column = average
    net_sales = rep.value("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"); cogs = rep.value("us-gaap:CostOfRevenue")
    out["Margin_Net_sales"] = int(round(net_sales))
    out["Gross_profit_as_a_percentage_of_cose_of_sales"] = round((net_sales - cogs) / cogs * 100, 1)
    out["Margin_Operating_income"] = int(round(rep.value("us-gaap:OperatingIncomeLoss")))
    for key, lab in [("new_stores_and_clubs_including_expansions_and_relocations", r"New stores and clubs"),
                     ("Walmart_International", r"Walmart International$")]:
        v, m = rep.row(lab, (r"Allocation of Capital Expenditures",))
        out["Allocation_of_Capital_Expenditures_to_" + key] = int(round(v[0] * m))
    fuel = (r"Including Fuel", r"Excluding Fuel")
    inc, m = rep.row(r"Operating income$", fuel, after=r"Including Fuel"); exc, _ = rep.row(r"Operating income$", fuel, after=r"Excluding Fuel")
    out["Sam's_Club_U.S._Segment_Operating_income_Including_Fuel"] = int(round(inc[0] * m))
    out["Sam's_Club_U.S._Segment_Operating_income_Excluding_Fuel"] = int(round(exc[0] * m))
    out["Sam's_Club_U.S._Segment_Operating_income_Fuel_Only"] = int(round((inc[0] - exc[0]) * m))
    out["Liquidity_Net_Cash_Provided_by_Operating_Activities"] = int(round(rep.value("us-gaap:NetCashProvidedByUsedInOperatingActivities")))
    out["Liquidity_Net_Cash_Used_in_Investing_Activities"] = int(round(rep.value("us-gaap:NetCashProvidedByUsedInInvestingActivities")))
    out["Total_number_of_shares_repurchased"] = int(round(rep.value("us-gaap:TreasuryStockSharesAcquired")))
    out["Average_price_paid_per_share_repurchased"] = float(rep.value("us-gaap:TreasuryStockAcquiredAverageCostPerShare"))
    out["Total_amount_paid_for_share_repurchases"] = int(round(rep.value("us-gaap:PaymentsForRepurchaseOfCommonStock")))
    mr = (r"Expected Maturity Date", r"Short-term borrowings")             # Item 7A market-risk table, "Total" = last column
    v, m = rep.row(r"Variable rate$", mr, after=r"Short-term borrowings")
    out["Liabilities_Short-term_borrowings_Variable_rate_Total"] = int(round(v[-1] * m))
    out["Liabilities_Short-term_borrowings_weighted-average_interest_rate_in_percentage_Total"] = float(
        rep.row(r"Weighted-average interest rate", mr, after=r"Short-term borrowings")[0][-1])
    v, m = rep.row(r"Restricted stock and performance-based restricted stock units$")
    out["Shareholders'_Equity_Restricted_stock_and_performance-based_restricted_stock_units"] = int(round(v[0] * m))
    return out


zips = sorted(glob.glob(os.path.join(DATA_DIR, "*-xbrl.zip")))              # accession ...-25-... (A) before ...-26-... (B)
A, B = extract(Report(zips[0])), extract(Report(zips[1]))
res = {}
for k in A:                                                                  # B - A for every numeric field except the age
    res[k + "__inA"], res[k + "__inB"] = A[k], B[k]
    if isinstance(A[k], (int, float)) and not isinstance(A[k], bool) and k != "Corporate_Affairs_Officier_Age":
        d = B[k] - A[k]
        res[k + "__BoverA"] = round(d, 6) if isinstance(d, float) else int(d)
json.dump(res, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2, ensure_ascii=False)
