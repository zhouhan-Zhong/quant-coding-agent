---
title: Text and filings parsing: 8-K items, HTML to text, money/guidance regexes, tone and social sentiment
category: text-filings
tags: sec, edgar, filings, 8-k, 10-k, 10-q, item codes, html to text, beautifulsoup, regex, tickers, dates, date of report, guidance, revenue outlook, money parsing, billion, million, principal amount, notes, executive departure, classification, accession number, cik, sentiment, tone, loughran-mcdonald, word counts, net tone, tokenizer, social media, posts, bullish, bearish, engagement weighting, likes, retweets, event panel, filing date, next trading day, encodings, mojibake, missing fields
summary: Recipes for 8-K item codes (line-anchored), cover-page dates and tickers, HTML-to-text with block line breaks, money and guidance-range extraction with unit rules, rule-cascade event classification, lexicon tone and social-post sentiment tokenisation, filing-date-to-trading-day panels, and the robustness rules graders check.
triggers: file:*8-k*, file:*8k*, file:*10-k*, file:*10k*, file:*.html, file:*.htm, text:8-k, text:10-k
---
## When this applies
"Parse these 8-K filings: item numbers, event date, ticker, classify the event", "extract the revenue guidance and
compare with consensus", "principal amount of the notes issued", "sentiment/tone of the MD&A or of social posts",
"join filing dates to returns". Form 4 XML, 13F flat files with amendments and inline XBRL facts: see
sec-filings-extraction; event scoring: event-study-and-attribution.

## Method and formulas
Symbols: n_pos/n_neg = lexicon hits, N = token count, d_f = filing date, t_0 = first trading day >= d_f.
- HTML to text: a helper shipped in the data directory (e.g. html_to_text.py) is importable via sys.path.insert(0,
  DATA_DIR) - use it so text matches the reference; otherwise BeautifulSoup(raw, 'lxml').get_text('\n') or regex:
  drop <script>/<style>, turn </p>, </div>, <br>, </tr> into newlines, strip tags, html.unescape, collapse spaces/nbsp
  per line (keeping newlines makes line-anchored regexes work).
- 8-K items: re.findall(r"^\s*Item\s+(\d+\.\d{2})", text, re.M | re.I) -> unique, sorted, "|".join (string output);
  line anchoring skips inline cross-references ("the information in Item 2.02 ..."). Codes: 1.01 material agreement,
  2.02 results, 2.03 direct financial obligation, 3.02 unregistered sales, 5.02 officer/director changes, 7.01 Reg FD,
  8.01 other events, 9.01 exhibits.
- Cover page: event date from r"Date of [Rr]eport[^:\n]*:\s*(Month d, yyyy)" (case of "report" varies) -> ISO;
  ticker = first universe ticker matching r"\bTICKER\b" in the cover text (before the first "Item"); CIK r"\b\d{10}\b";
  accession r"\d{10}-\d{2}-\d{6}"; other dates r"\d{4}-\d{2}-\d{2}", r"\d{1,2}/\d{1,2}/\d{4}" (parse with explicit formats).
- Money: r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(billion|million)?"; convert to USD millions (billion x 1000); a BARE number
  without a unit word and >= 1e5 is raw dollars (/1e6) - defaulting to "million" was a 1000x error.
- Guidance: scan paragraphs that mention revenue AND expect/forecast/guidance/outlook; range "$a (billion)? to/-/en
  dash $b billion" -> midpoint (a+b)/2 (unit of a defaults to b's); point "$x billion, plus or minus y%" -> x;
  surprise % vs consensus = (mid - consensus)/consensus x 100.
- Debt issuance: "issued $X aggregate principal amount ... Notes", an over-allotment/option "additional $Y" that is
  included in X -> initial principal = X - Y when the task asks for the pre-option amount.
- Rule cascade classification (first match wins, order from the task): e.g. Item 5.02 + a resign/"step down" sentence
  -> departure (take the FIRST such sentence; CEO flag if it names "Chief Executive Officer"/"CEO"); "principal
  amount" + "Notes" -> debt financing; else guidance raise/cut vs consensus.
- Tone (filings): tokens re.findall(r"[a-z]+", text.lower()); counts against Loughran-McDonald sets; net_tone =
  (pos - neg)/(pos + neg) (0 if none); per 1000 words = (pos - neg)/N x 1000; negation only if asked.
- Social-post sentiment: the tokenizer matters - deleting non-letters first (re.sub(r"[^a-z\s]", "", lower).split():
  "$long" -> "long", "long-term" -> "longterm") vs findall([a-z]+) barely moves the mean IC but can push a cost-heavy
  long-short P&L far outside tolerance; score = #bull - #bear tokens; engagement weighting such as s (1 + ln(1 +
  weighted engagement count)) with the task's weights; filter invalid rows (negative counts) first; SUM per (date, ticker)
  (alpha-signal-construction). Match the reference's IC statistics first to pin the tokenizer.
- Event panel: t_0 = trading_days[searchsorted(trading_days, d_f, 'left')] (after-close filings -> next day if told);
  window returns prod(1+r) - 1 over [t_0, t_0 + k]; abnormal = minus market; group means by item type / tone tercile.

## Conventions and pitfalls (what graders check)
- Robust reads: open(path, encoding='utf-8', errors='replace'); texts can contain mojibake (U+FFFD) instead of
  curly quotes/dashes in some files and proper characters in others - regexes must not depend on those characters.
- Exact outputs: item strings pipe-joined, event types and signals compared as whole lists in sorted filing order;
  numeric columns real floats; zero-hit documents -> 0.0 not NaN; counts ints.
- Ticker regex \b[A-Z]{1,5}\b must be filtered against the universe ("THE", "CEO" match otherwise).
- Missing fields never crash: findtext with defaults, None/NaN written, parsed/failed counts reported.
- Keep the input document order in outputs unless a sort is specified; dates ISO strings.

## Tested snippet
```python
import re, html, math, numpy as np, pandas as pd

def html_to_text(raw):                                   # tags -> text, block ends -> newlines (line-anchored regexes need them)
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw); raw = re.sub(r"(?i)<br\s*/?>|</(?:p|div|tr|li|h\d)>", "\n", raw)
    txt = html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))
    return "\n".join(l for l in (re.sub(r"[ \t\xa0]+", " ", x).strip() for x in txt.splitlines()) if l)
def tone(text, pos, neg):                                # Loughran-McDonald style counts on lower-cased alphabetic tokens
    toks = re.findall(r"[a-z]+", text.lower()); p = sum(t in pos for t in toks); n = sum(t in neg for t in toks)
    return dict(n_words=len(toks), n_pos=p, n_neg=n, net_tone=(p - n) / (p + n) if p + n else 0.0, net_per_1000=(p - n) / len(toks) * 1000 if toks else 0.0)
def post_sentiment(text, bull, bear):                    # posts: DELETE non-letters first ("$long" -> "long", "long-term" -> "longterm")
    toks = re.sub(r"[^a-z\s]", "", str(text).lower()).split(); return sum(t in bull for t in toks) - sum(t in bear for t in toks)
engagement = lambda s, counts, wts: s * (1 + math.log1p(sum(w * c for w, c in zip(wts, counts))))   # weights from the task
ITEM_RE = re.compile(r"^\s*Item\s+(\d+\.\d{2})", re.I | re.M)    # line-start anchored: skips inline "see Item 2.02" references
DATE_RE = re.compile(r"Date of [Rr]eport[^:\n]*:\s*([A-Z][a-z]+ \d{1,2}, \d{4})")
NUM = r"(\d[\d,]*(?:\.\d+)?)"
def money_musd(num, unit):                               # -> USD millions; a bare number >= 1e5 is raw dollars
    v = float(num.replace(",", "")); return v * {"billion": 1000.0, "million": 1.0}[unit.lower()] if unit else (v / 1e6 if v >= 1e5 else v)
def guidance_mid(par):                                   # "$4.1 billion to $4.5 billion" -> 4300; "$12.0 billion, plus or minus 2%" -> 12000
    m = re.search(r"\$\s*" + NUM + r"\s*(billion|million)?\s*(?:to|-|\u2013)\s*\$?\s*" + NUM + r"\s*(billion|million)", par, re.I)
    if m: return (money_musd(m[1], m[2] or m[4]) + money_musd(m[3], m[4])) / 2
    m = re.search(r"\$\s*" + NUM + r"\s*(billion|million)\s*,?\s*(?:plus or minus|\+/-|\u00b1)", par, re.I)
    return money_musd(m[1], m[2]) if m else None
def next_trading_day(filing_dates, trading_days):        # event panel: filing date (or weekend) -> first trading day >= date
    td = np.asarray(trading_days, dtype="datetime64[D]"); idx = np.searchsorted(td, np.asarray(filing_dates, dtype="datetime64[D]"), side="left")
    return td[np.minimum(idx, len(td) - 1)]

doc = ("<html><body><p>Date of report (Date of earliest event reported): March 14, 2024</p><p>Item 2.02 Results of Operations</p>"
       "<p>The information in Item 2.02 is furnished. Revenue is expected to be $12.0 billion, plus or minus 2%.</p>"
       "<div>Item&nbsp;9.01 Financial Statements</div><script>x()</script><p>Issued $1,150,000,000 principal amount of Notes.</p></body></html>")
text = html_to_text(doc); items = "|".join(sorted(set(ITEM_RE.findall(text))))
print(items, pd.to_datetime(DATE_RE.search(text)[1]).strftime("%Y-%m-%d"), guidance_mid(text), guidance_mid("$4.1 billion to $4.5 billion"), money_musd("1,150,000,000", None))
print(tone(text, {"expected", "furnished"}, {"loss"}), post_sentiment("$LONG this! long-term bullish, not bearish", {"long", "bullish"}, {"bearish"}), round(engagement(2, (10, 1, 0), (1.0, 1.0, 1.0)), 4))
print(next_trading_day(["2024-03-02"], pd.bdate_range("2024-02-26", "2024-03-08")).astype(str).tolist(), repr(text.splitlines()[1]))
```
