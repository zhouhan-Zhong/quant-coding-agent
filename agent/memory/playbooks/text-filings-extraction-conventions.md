---
title: Text and filings extraction conventions: HTML/XML to text, number parsing with scale words and parentheses, table-row anchoring, date fields of SEC filings, item codes, de-duplicating amendments, scoring text with word lists, joining filings to market data
category: text-filings
tags: sec filings, edgar, 10-k, 10-q, 8-k, form 4, 13f, html, xml, beautifulsoup, lxml, text extraction, regex, number parsing, in thousands, in millions, parentheses negative, percent, table, row label, anchor, fiscal year, period of report, filing date, accession number, amendment, /a, restatement, item 2.02, item 7.01, item 9.01, word list, loughran mcdonald, tone, sentiment score, tf, normalisation, tokenisation, stop words, join to returns, next trading day, after close, event date
summary: Extraction tasks fail on mechanics, not finance: parse HTML with a real parser and take the text of table rows anchored by the row label the task names (not the first number on the page); convert "(1,234)" to -1234, "1,234.5" to 1234.5, apply the table's scale word ("in millions") once, strip footnote marks; the period of report, filing date and accession number are different fields; amendments (…/A) replace the original for the same period unless the task says to keep both; 8-K item codes are the "Item x.yy" headings present in the filing; word-list scores count token hits over the task's token definition (lowercased alphabetic tokens, with or without stop words) divided by the stated denominator; filings join to prices on the first trading day on or after the filing date (next day if after 16:00 ET or when the task says so).
triggers: file:*.htm, file:*.html, file:*.xml, text:8-k, text:10-k, text:form 4, text:13f, text:xbrl, text:edgar
---
## When this applies
"Extract <metric> for each company from the 10-K HTML", "parse the 8-K items and classify", "Form 4 insider sales table",
"13F holdings with amendments", "tone / sentiment score of the MD&A or calls", "join the filings to the return series".

## Parsing mechanics (do these literally; a degenerate output with zeros means one of them failed)
- Load with BeautifulSoup(html, "lxml") (fallback "html.parser"); for XML filings use lxml.etree or BeautifulSoup
  "xml".  Get text with get_text(" ", strip=True) and collapse whitespace; keep tables as rows: for each <tr>, the list
  of cell texts.
- Anchoring a value: find the row whose first cell (label) matches the task's label (case-insensitive, punctuation
  stripped, e.g. "Total revenues" vs "Total revenue"); take the column the task names (current year = first numeric cell
  unless the header says otherwise).  Never take "the first number in the document".
- Number parsing: remove $, commas, footnote marks and non-breaking spaces; "(1,234)" or "−1,234" -> -1234; "—" / "-" /
  "" -> missing; percentages keep the % flag; then apply the scale stated in the table header ("in thousands" x 1e3,
  "in millions" x 1e6) exactly once; values the task wants "as reported" are not scaled.  Dates: parse the filing's own
  format (e.g. "February 3, 2024") to ISO.
- Fields: periodOfReport / "fiscal year ended" (the period), filingDate / acceptance date (when filed), accession number
  (0001234567-24-000012; the CIK is the first 10 digits); do not confuse them.  Multiple filings per company: pick the one
  the task's rule names (latest period, or the filing within the window).
- Amendments: form "10-K/A" / "13F-HR/A" replaces the original for the same period when the task says "amendment-aware";
  a 13F amendment can be a RESTATEMENT (replace all holdings) or NEW HOLDINGS (add) - the cover page says which.
  De-duplicate on (CIK, period, form) keeping the latest filing date.
- 8-K items: regex `Item\\s+(\\d\\.\\d{2})` over the text; the item list is all distinct codes in document order joined by
  the task's separator; event type from the item codes and keywords exactly as the task's mapping says.
- Form 4: transaction code (S = sale, P = purchase), shares and price per transaction row (transactionAmounts), direct /
  indirect ownership, footnote references; aggregate per (insider, day) as the task says (sum of shares x price for
  value; weighted average price).
- Degenerate check: if every extracted value is 0 / missing, the anchor or the scale word was not found - print the
  candidate rows before giving up.

## Scoring text
- Tokens: lowercase, alphabetic (regex [a-z]+ or the task's rule), optional stop-word removal; counts of word-list hits
  (Loughran-McDonald positive / negative / uncertainty, or the task's list, matched as whole tokens, inflections only if
  the list has them).
- Score = (pos - neg) / (pos + neg) or (pos - neg) / total tokens or per-1000 words - the task names one; a sign flip
  of the mean score means the lists or the formula were swapped.  Weighted scores: weights as stated (e.g. by
  followers, by filing size), weighted mean = sum w s / sum w.
- Number of valid documents = documents with at least one token after cleaning (or the task's rule); report it.

## Joining to market data
- Event date = filing date mapped to the first trading day >= date (next trading day when the timestamp is after the
  close or the task says "after-hours"); returns on that day are close-to-close.  Companies without price data are
  dropped and counted.
- Signals: thresholds applied to the score exactly as stated (>, >=); "top long / top short ticker" by the score with
  the task's tie-break; portfolio scores = sum over signals (net = long minus short, gross = long plus short).

## Invariants worth printing
- No extracted metric is zero / missing for a company that has the table; counts of filings per form reconcile with the
  input list; amendments de-duplicated (one filing per CIK-period); event dates are trading days; mean score sign and
  magnitude consistent with the counts (pos > neg -> positive).
