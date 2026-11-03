---
title: Earnings-call and 10-K text analytics: lexicon sentiment (Loughran-McDonald), speaker segmentation, risk-factor extraction, event alignment
category: text-filings
tags: earnings call, transcript, prepared remarks, q&a, management, analyst, speaker type, speaker, sentiment score, sentiment analysis, lexicon, dictionary, loughran-mcdonald, lm dictionary, positive words, negative words, uncertainty words, litigious, net sentiment, tone, polarity, word count, tokenization, tokens, stopwords, boilerplate, forward-looking statements, safe harbor, 10-k, item 1a, risk factors, risk categories, classification, keyword rules, closed label set, section extraction, regex, html to text, event alignment, trading day, next trading day, abnormal return, car, look-ahead, readability, fog index, numeric extraction, millions, billions, finbert
summary: Token/lexicon scoring conventions (alphabetic tokens, bounded polarity, per-1000-word tone), speaker-type aggregates with boilerplate removal, robust Item 1A extraction (last heading match, risk splitting, closed-vocabulary keyword classification), event-date-to-trading-day alignment and the range/type invariants graders assert.
---
## When this applies
"Score the sentiment of earnings-call segments / transcripts with a word list", "management vs analyst tone", "extract
the Risk Factors section (Item 1A) and classify each risk", "count positive/negative/uncertainty words", "align text
events to stock returns (event study)", "parse dollar amounts in millions/billions from text", JSON/CSV/TXT corpora
of segments with fields like company, quarter, speaker_type, text.

## Method and formulas
- Tokens: re.findall(r"[a-z]+", text.lower()) (hyphenated/apostrophe variants only if the task defines them); no
  stemming/lemmatisation unless told; stopwords removed only when the task lists them. n_words = len(tokens).
- Lexicon hits: pos = count of tokens in the positive set, neg likewise (sets lower-cased; the Loughran-McDonald lists
  ship as CSV/TXT columns - load exactly the file given, every word, not a sample). Scores: polarity = (pos - neg)/(pos
  + neg) (0 when pos + neg = 0, or NaN if the task says), net tone per 1000 words = (pos - neg)/n_words x 1000, pos_ratio
  = pos/n_words, uncertainty_ratio = unc/n_words. Bounded scores in [-1, 1]; clip only if the formula can exceed it.
- Negation ("not good") is NOT part of the LM standard; apply a window rule only when the task specifies it.
- Speaker segmentation: fields speaker_type in {management, analyst, operator}; score prepared remarks and Q&A
  separately when asked; aggregates per (company, quarter[, speaker_type]) = pooled counts ((sum pos - sum neg)/(sum
  pos + sum neg)) or the mean of segment scores - the task's wording ("average segment score" vs "overall tone")
  decides; management_minus_analyst = mgmt - analyst. Operator lines are excluded from both.
- Boilerplate: remove safe-harbor/forward-looking-statement paragraphs by regex before counting when the task asks
  ("exclude disclaimers"); otherwise keep everything - do not silently filter.
- 10-K Item 1A: regex r"item\s*1a\.?\s*risk\s*factors(.*?)item\s*1b" with re.I | re.S; the table of contents matches
  first, so take the LAST (or the longest) span; fall back to "Item 2" when 1B is absent. Split risks on sub-headings
  (short lines without a final period, bold/italic markers, or numbered lines) or paragraphs as the task defines; drop
  empty/very short fragments (< 4 words) and state the rule.
- Classification with a CLOSED label set: keyword rules per label; first-match in the task's priority order (or
  max-count with deterministic ties = earlier label); default label ("other") only if it exists in the declared set.
  Counts per (company, label) as ints; include zero rows when the template shows every label.
- Readability: Gunning fog = 0.4 x (words/sentences + 100 x complex/words), complex = >= 3 syllables (vowel groups);
  sentences split on [.!?]+.
- Event alignment: event day 0 = first trading day >= event date (searchsorted on the return index); announcements
  after the close -> next trading day when the task says; window returns (-1, 0, +1, +2) relative to day 0; abnormal
  return = r - r_market (or market-model alpha + beta from an estimation window ending before the event); CAR = sum over
  the window; group means by event type / sentiment tercile. Predictive setups require event_date < return window start.
- Numbers in text: "$1.2 billion" -> 1.2e9, "(in millions)" table scale -> x 1e6, "(123)" -> -123, "12.5%" -> 0.125
  unless the template shows percent; keep the sign and the unit the template uses.
- HTML inputs: BeautifulSoup(html, "lxml").get_text(" ") then collapse whitespace; strip <table> only when told.

## Conventions and pitfalls (what graders check)
- Invariants asserted: scores in the declared range ([-1, 1] or [0, 1]), counts >= 0 ints, labels drawn only from the
  declared set, every input segment/company present once in the output (dedupe by (company, quarter) if the corpus
  repeats), event dates precede the return windows used for prediction, n_words > 0 for scored segments, aggregates
  recomputable from the per-segment rows.
- Sorting: company then quarter (chronological "2023Q4" < "2024Q1" works lexicographically when zero-padded; convert
  "Q1 2024" formats explicitly); keep the input order when the task says "in the order given".
- Case: match lexicon words case-insensitively; keep company names/tickers as written in outputs.
- Encoding: open text with encoding="utf-8", errors="ignore"; curly quotes/dashes can hide inside tokens - normalise
  with unicodedata.normalize("NFKC") only if the task mentions it.
- Empty text: polarity 0 vs NaN is a frequent fail - follow the task; never write NaN into JSON.
- Do not invent labels, do not merge categories, do not drop risks that match no rule (they go to the default label).
- Percent vs decimal for ratios (pos_ratio 0.012 vs 1.2): the template's example values show which.
- Output hygiene: JSON keys exactly as the template; CSV one row per (segment) or per (company, quarter, speaker_type)
  as specified; floats with full precision; speaker_type values in the input's spelling.

## Tested snippet
```python
import re, numpy as np, pandas as pd
pos_words = {"growth", "strong", "improve", "improved", "record", "confident", "exceed", "exceeded", "robust", "momentum"}
neg_words = {"decline", "declined", "weak", "weakness", "loss", "losses", "challenging", "headwind", "headwinds", "uncertain", "miss"}
tok = lambda s: re.findall(r"[a-z]+", s.lower())            # alphabetic tokens; no stemming unless the task says so
def lm_score(text):
    w = tok(text); p = sum(t in pos_words for t in w); n = sum(t in neg_words for t in w)
    return dict(n_words=len(w), pos=p, neg=n, polarity=(p - n) / (p + n) if p + n else 0.0, net_per_1k=(p - n) / len(w) * 1000 if w else 0.0)
segs = [dict(company="ACME", quarter="2024Q1", speaker_type="management", text="We delivered record revenue and strong margin growth; momentum continued."),
        dict(company="ACME", quarter="2024Q1", speaker_type="analyst", text="Given the weak macro and headwinds, is the guidance uncertain?"),
        dict(company="ACME", quarter="2024Q1", speaker_type="management", text="This call contains forward-looking statements under the safe harbor provisions."),
        dict(company="ACME", quarter="2024Q1", speaker_type="management", text="Costs declined and we are confident we will exceed targets.")]
df = pd.DataFrame(segs)
df = df[~df["text"].str.contains(r"forward-looking statements|safe harbor", case=False, regex=True)].copy()   # drop disclaimer boilerplate
df = pd.concat([df, df["text"].apply(lambda s: pd.Series(lm_score(s)))], axis=1)
g = df.groupby(["company", "quarter", "speaker_type"])[["pos", "neg"]].sum()
agg = ((g["pos"] - g["neg"]) / (g["pos"] + g["neg"]).clip(lower=1)).unstack()                 # pooled polarity per speaker type
agg["mgmt_minus_analyst"] = agg["management"] - agg["analyst"]
print(df[["speaker_type", "n_words", "pos", "neg", "polarity"]].to_string(index=False)); print(agg.round(4).to_string())
tenk = ("Item 1A. Risk Factors 12  Item 1B. Unresolved Staff Comments 20 ... ITEM 1A. RISK FACTORS\n"
        "Our business depends on a few large customers.\nWe face intense competition and pricing pressure.\n"
        "Changes in interest rates could increase our borrowing costs.\nCyber attacks may disrupt our systems.\n"
        "New regulations may increase compliance costs.\nITEM 1B. UNRESOLVED STAFF COMMENTS\nNone.")
spans = list(re.finditer(r"item\s*1a\.?\s*risk\s*factors(.*?)item\s*1b", tenk, re.I | re.S))
body = spans[-1].group(1) if spans else ""                                                      # LAST match skips the table of contents
risks = [ln.strip() for ln in body.splitlines() if len(ln.split()) >= 4]
labels = ["market", "credit", "cyber", "regulatory", "competition", "operational", "other"]    # closed label set, priority order
rules = {"market": r"interest rate|currency|commodity|equity price", "credit": r"borrow|counterpart|default", "cyber": r"cyber|data breach|security",
         "regulatory": r"regulat|compliance|legislat", "competition": r"compet|pricing pressure", "operational": r"customer|supplier|system|disrupt"}
classify = lambda r: next((lab for lab in labels if lab in rules and re.search(rules[lab], r, re.I)), "other")
out = pd.DataFrame({"risk": risks, "label": [classify(r) for r in risks]})
print(out.to_string(index=False)); print(set(out["label"]) <= set(labels), int(len(out)), "risks;", out["label"].value_counts().to_dict())
```
