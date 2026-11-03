# memory-solution
# unit: t1-sec-8k-event-alpha | category: text-filings | tags: sec, 8-k, event-driven, html-parsing, regex, guidance, executive-departure, debt-issuance, alpha-score, params-json-family
# deliverables: filing_signals.csv, results.json, solution.json
# html -> text via the shipped html_to_text.py (import from DATA_DIR); item numbers = line-start "Item d.dd" headings (set, sorted)
# event date = "Date of Report ...: Month d, yyyy" on the cover; ticker = first universe ticker appearing as a word in the cover text
# classification: Item 5.02 + resign/step down -> executive_departure (CEO flag if the resignation sentence names the CEO);
#   "principal amount" of notes -> debt_financing (initial principal = total issued minus the option/greenshoe amount it "includes");
#   otherwise guidance: revenue sentence with "$a billion to $b billion" / "$a-b billion" (midpoint) or "$x billion, plus or minus" (point)
# score = base * severity * liquidity; severity guidance = 1 + min(|pct|, cap)/denominator; CEO 1.25 / CFO 1.10; debt >= 1000 -> 1.2
import os, re, sys, json, glob
import numpy as np, pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, DATA_DIR)
from html_to_text import html_to_text

P = json.load(open(os.path.join(DATA_DIR, "params.json")))
uni = pd.read_csv(os.path.join(DATA_DIR, "universe.csv"))
bucket = dict(zip(uni["ticker"], uni["liquidity_bucket"]))
UNIT = {"billion": 1000.0, "million": 1.0}
NUM = r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million)?"


def money(num, unit):  # -> millions USD ("$1,725,000,000" without a unit word is raw dollars)
    v = float(num.replace(",", ""))
    return v * UNIT[unit.lower()] if unit else (v / 1e6 if v >= 1e5 else v)


def guidance_midpoint(text):
    """Midpoint (musd) of the forward revenue guidance sentence, or None."""
    for para in text.split("\n"):
        if not re.search(r"revenue", para, re.I) or not re.search(r"expect|forecast|guidance|outlook", para, re.I):
            continue
        m = re.search(r"\$\s*(\d+(?:\.\d+)?)\s*(billion|million)?\s*(?:to|-|–)\s*\$?\s*(\d+(?:\.\d+)?)\s*(billion|million)", para, re.I)
        if m:
            u = m.group(4)
            return 0.5 * (money(m.group(1), m.group(2) or u) + money(m.group(3), u))
        m = re.search(r"\$\s*(\d+(?:\.\d+)?)\s*(billion|million)\s*,?\s*(?:plus or minus|\+/-|±)", para, re.I)
        if m:
            return money(m.group(1), m.group(2))
    return None


def debt_initial_principal(text):
    """Total principal issued minus the additional/option (greenshoe) amount that the issue 'includes'."""
    tot = re.search(NUM + r"\s*(?:aggregate\s+)?principal amount", text, re.I)
    if not tot:
        return None
    total = money(tot.group(1), tot.group(2))
    add = re.search(r"additional\s+" + NUM + r"\s*(?:aggregate\s+)?principal amount", text, re.I)
    if add and re.search(r"include\w*\s+\$\s*" + re.escape(add.group(1)), text, re.I):
        total -= money(add.group(1), add.group(2))
    return total


rows, inter = [], {}
for path in sorted(glob.glob(os.path.join(DATA_DIR, "filings", "*.html"))):
    fid = os.path.splitext(os.path.basename(path))[0]
    text = html_to_text(open(path, encoding="utf-8", errors="replace").read())
    items = sorted(set(re.findall(r"^\s*Item\s+(\d+\.\d{2})", text, re.M)))
    m = re.search(r"Date of [Rr]eport[^:\n]*:\s*([A-Z][a-z]+ \d{1,2}, \d{4})", text)
    event_date = pd.to_datetime(m.group(1)).strftime("%Y-%m-%d") if m else ""
    cover = text.split("Item ", 1)[0]
    ticker = next((t for t in bucket if re.search(r"\b%s\b" % re.escape(t), cover)), None)
    if ticker is None:
        ticker = next(t for t in bucket if re.search(r"\b%s\b" % re.escape(t), text))
    liq = bucket[ticker]

    body = text.split("SIGNATURE", 1)[0]  # the 8-K body, before signatures / exhibits
    dep = re.search(r"[^.\n]*\b(resign\w*|step\w* down|retire\w*)\b[^.\n]*", body, re.I)
    if "5.02" in items and dep:
        etype = "executive_departure"
        is_ceo = bool(re.search(r"Chief Executive Officer|\bCEO\b", dep.group(0)))
        role = "Chief Executive Officer" if is_ceo else "Chief Financial Officer"
        sev = float(P["executive_departure_multipliers"][role])
        inter[f"ceo_departure_flag_{fid}"] = float(is_ceo)
    elif re.search(r"principal amount", body, re.I) and re.search(r"\bnotes?\b", body, re.I):
        etype = "debt_financing"
        amt = debt_initial_principal(body)
        sev = float(P["debt_large_issue_multiplier"] if amt >= P["debt_large_issue_threshold_musd"] else P["debt_small_issue_multiplier"])
        inter[f"debt_amount_musd_{fid}"] = float(amt)
    else:
        cons = float(P["analyst_consensus_revenue_musd"][fid])
        mid = guidance_midpoint(text)
        pct = (mid - cons) / cons * 100.0
        etype = "guidance_raise" if pct > 0 else "guidance_cut"
        sev = 1.0 + min(abs(pct), float(P["guidance_cap_pct"])) / float(P["guidance_denominator"])
        inter[f"guidance_change_pct_{fid}"] = float(pct)
    base = float(P["base_scores"][etype]); lm = float(P["liquidity_multipliers"][liq])
    score = base * sev * lm
    sig = "long" if score >= P["long_threshold"] else ("short" if score <= P["short_threshold"] else "hold")
    rows.append({"filing_id": fid, "ticker": ticker, "event_date": event_date, "event_type": etype, "item_numbers": "|".join(items),
                 "liquidity_bucket": liq, "base_score": base, "severity_multiplier": float(sev), "liquidity_multiplier": lm,
                 "event_alpha_score": float(score), "signal": sig})

df = pd.DataFrame(rows).sort_values("filing_id").reset_index(drop=True)
df.to_csv(os.path.join(OUT_DIR, "filing_signals.csv"), index=False)

s = df["event_alpha_score"]
results = {"num_filings": int(len(df)), "num_long_signals": int((df["signal"] == "long").sum()),
           "num_short_signals": int((df["signal"] == "short").sum()), "mean_alpha_score": float(s.mean()),
           "top_long_ticker": str(df.loc[s.idxmax(), "ticker"]), "top_short_ticker": str(df.loc[s.idxmin(), "ticker"]),
           "portfolio_net_score": float(s.sum()), "portfolio_gross_score": float(s.abs().sum())}
json.dump(results, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

for et in P["base_scores"]:
    inter[f"num_{et}"] = float((df["event_type"] == et).sum())
inter.update({"mean_severity_multiplier": float(df["severity_multiplier"].mean()),
              "mean_liquidity_multiplier": float(df["liquidity_multiplier"].mean()),
              "mean_alpha_score": results["mean_alpha_score"], "portfolio_net_score": results["portfolio_net_score"],
              "portfolio_gross_score": results["portfolio_gross_score"]})
json.dump({"intermediates": {k: {"value": v} for k, v in inter.items()}, "deliverables": results},
          open(os.path.join(OUT_DIR, "solution.json"), "w"), indent=2)
