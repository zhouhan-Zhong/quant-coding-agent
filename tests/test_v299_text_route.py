"""Unit tests for agent 2.9.9: the text route (AGENT_TEXT_ROUTE) - document routing, the code-only DOCUMENT PROBE
(inline XBRL, item codes, small-input values, keyword tables and sentences, XML structure and footnotes, helper code),
EXTRACT line checks, forced degenerate checks and the unchanged behaviour of units without documents."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))
sys.path.insert(0, str(HERE))

import agent  # noqa: E402
from test_v22_features import FakeHouse, _solver  # noqa: E402

bs4 = pytest.importorskip("bs4")

KIT = HERE.parent / "kit" / "units"

HTML = """<html><body>
<ix:header><ix:hidden>junk-header 0000999 2025 FY false http://fasb.org/us-gaap/2024#X</ix:hidden></ix:header>
<p>FORM 8-K. Date of Report (Date of earliest event reported): May 22, 2024. ACME CORP.</p>
<p>Trading Symbol(s) Common Stock ACME The Nasdaq Global Select Market</p>
<p><ix:nonNumeric name="dei:DocumentType" contextRef="c-1">8-K</ix:nonNumeric></p>
<p>Item 2.02 Results of Operations. Item 9.01 Financial Statements.</p>
<p>Outlook for the second quarter: Revenue is expected to be $28.0 billion, plus or minus 2%.</p>
<p>The weather was pleasant on 12 occasions.</p>
<table><tr><td>Segment</td><td>Square feet</td></tr><tr><td>Supercenters</td><td>632,577</td></tr></table>
<table><tr><td>Cash</td><td>9,037</td></tr></table>
</body></html>"""

XML = """<?xml version="1.0"?><ownershipDocument><issuer><issuerTradingSymbol>ACME</issuerTradingSymbol></issuer>
<nonDerivativeTable>
 <nonDerivativeTransaction><securityTitle><value>Common Stock</value></securityTitle><transactionCoding><transactionCode>S</transactionCode></transactionCoding>
  <transactionAmounts><transactionShares><value>100</value></transactionShares><transactionPricePerShare><value>10.5</value><footnoteId id="F1"/></transactionPricePerShare></transactionAmounts>
  <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature></nonDerivativeTransaction>
 <nonDerivativeTransaction><securityTitle><value>Common Stock</value></securityTitle><transactionCoding><transactionCode>S</transactionCode></transactionCoding>
  <transactionAmounts><transactionShares><value>50</value></transactionShares><transactionPricePerShare><value>10.9</value></transactionPricePerShare></transactionAmounts>
  <ownershipNature><directOrIndirectOwnership><value>I</value></directOrIndirectOwnership></ownershipNature></nonDerivativeTransaction>
 <nonDerivativeTransaction><securityTitle><value>Common Stock</value></securityTitle><transactionCoding><transactionCode>F</transactionCode></transactionCoding>
  <transactionAmounts><transactionShares><value>7</value></transactionShares></transactionAmounts>
  <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature></nonDerivativeTransaction>
</nonDerivativeTable>
<footnotes><footnote id="F1">Weighted average price; sold at prices ranging from $10.20 to $10.80.</footnote></footnotes>
</ownershipDocument>"""

INSTR = ("Extract the `ticker`, the `event_date` from the \"Date of Report\" line and the revenue guidance midpoint. "
         "Report `Supercenters_Total_Square_Feet` and the Form 4 sale transactions. Write f0.json with "
         '"guidance_change_pct": <float>.')


def _text_unit(tmp_path: Path, with_docs: bool = True) -> agent.Unit:
    data = tmp_path / "input" / "environment" / "data"
    (data / "filings").mkdir(parents=True, exist_ok=True)
    (data / "universe.csv").write_text("ticker,bucket\nACME,Large\nZZZ,Small\n")
    if with_docs:
        (data / "filings" / "filing_01.html").write_text(HTML)
        (data / "form4_acme.xml").write_text(XML)
        with zipfile.ZipFile(data / "report.zip", "w") as z:
            z.writestr("acme-2025.htm", HTML)
            z.writestr("acme-2025_cal.xml", "<linkbase/>")
        (data / "html_to_text.py").write_text('"""Convert filing HTML to plain text."""\ndef html_to_text(path):\n    return ""\n')
    files = [agent.DataFile(p, str(p.relative_to(data)), p.stat().st_size) for p in sorted(data.rglob("*")) if p.is_file()]
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return agent.Unit(task_dir=tmp_path / "input", out_dir=out, unit_id="t1-test-text", instruction=INSTR, data_root=data, data_files=files)


def test_text_route_docs_finds_html_xml_and_zip_members(tmp_path):
    u = _text_unit(tmp_path)
    docs = agent.text_route_docs(u)
    names = sorted(d["name"] for d in docs)
    assert names == ["filings/filing_01.html", "form4_acme.xml", "report.zip:acme-2025.htm", "report.zip:acme-2025_cal.xml"]
    assert agent.text_route_docs(_text_unit(tmp_path / "plain", with_docs=False)) == []


def test_keywords_and_small_input_values(tmp_path):
    kws = agent.task_keywords(INSTR)
    assert {"ticker", "date", "supercenters", "square", "feet", "guidance"} <= set(kws) and "the" not in kws and "report" not in kws
    assert agent.small_input_values(_text_unit(tmp_path)) == ["ACME", "ZZZ"]


def test_html_probe_shows_facts_items_values_sentences_tables():
    kws = agent.task_keywords(INSTR)
    out = agent.html_probe("f.html", HTML, kws, ["ACME"], 4000)
    assert "dei:DocumentType contextRef=c-1" in out and "'8-K'" in out
    assert "Item codes in the text (document order): 2.02, 9.01" in out
    assert "input value 'ACME' occurs" in out and "Trading Symbol" in out
    assert "sentence: Revenue is expected to be $28.0 billion, plus or minus 2%." in out   # "guidance" activates "expected"
    assert "pleasant" not in out and "junk-header" not in out
    assert "Supercenters | 632,577" in out


def test_xml_probe_shows_codes_records_and_footnotes():
    out = agent.xml_probe("form4.xml", XML, 3000)
    assert "transactionCode: S x2, F x1" in out and "directOrIndirectOwnership/value: D x2, I x1" in out
    assert "first <nonDerivativeTransaction>: securityTitle=Common Stock" in out and "transactionShares=100" in out
    assert "footnote F1: Weighted average price; sold at prices ranging from $10.20 to $10.80." in out
    assert "could not be parsed" in agent.xml_probe("bad.xml", "<a><b></a>", 500)


def test_text_probe_assembles_documents_zip_and_helpers(tmp_path):
    u = _text_unit(tmp_path)
    pr = agent.text_probe(u, agent.text_route_docs(u))
    assert pr.startswith("DOCUMENT PROBE") and "identifier values of small inputs: ACME, ZZZ" in pr
    assert "read with zipfile.ZipFile(path).read('acme-2025.htm')" in pr and "linkbase / schema members" in pr and "acme-2025_cal.xml" in pr
    assert "== form4_acme.xml" in pr and "== filings/filing_01.html" in pr
    assert "html_to_text.py: Convert filing HTML to plain text.; functions: html_to_text(path)" in pr


def test_extract_lines_and_issues():
    lines = agent.parse_extract_lines("EXTRACT ticker ACME | Trading Symbol(s) Common Stock ACME\nEXTRACT date MISSING | Date of Report\n"
                                      "noise\nEXTRACT revenue 28000 | Revenue is expected to be $28.0 billion")
    assert lines[0] == ("ticker", "ACME", "Trading Symbol(s) Common Stock ACME") and lines[1][1] == "MISSING"
    doc = "trading symbol(s) common stock acme the nasdaq revenue is expected to be $28.0 billion"
    hard, soft = agent.extract_issues(lines, doc)
    assert len(hard) == 1 and "extraction failed for 1 of 3 field(s): date" in hard[0] and not soft
    zeros = [("a", "0", ""), ("b", "0.0", ""), ("c", "", ""), ("d", "5", "")]
    hard, soft = agent.extract_issues(zeros, "")
    assert any("3 of 4 extracted values are empty or 0" in h for h in hard)
    hard, soft = agent.extract_issues([("a", "0", ""), ("b", "3", ""), ("c", "4", ""), ("d", "5", "")], "")
    assert not hard and "extracted value 0 for a" in soft[0]
    hard, soft = agent.extract_issues([("a", "3", "this sentence is nowhere in the documents")], doc)
    assert not hard and "does not occur in the documents" in soft[0]
    assert agent.extract_issues([], doc) == ([], ["the script printed no EXTRACT lines (print one per value read from a document)"])


def _routed_solver(tmp_path, **flags):
    s = _solver(tmp_path, n_deliv=1, text_route=True, **flags)
    s.unit = _text_unit(tmp_path / "u")
    s.house = FakeHouse([], s.cfg)
    s.text_docs = agent.text_route_docs(s.unit)
    s.text_routed = True
    return s


def test_routed_run_fails_on_empty_extraction_and_passes_when_fixed(tmp_path):
    s = _routed_solver(tmp_path)
    w = "import json, os\njson.dump({'ticker': 'ACME', 'n': 3.5, 'x': 1.25}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))\n"
    val, _, fb = s.run_attempt(w + "print('EXTRACT ticker MISSING | Trading Symbol')", "a1")
    assert not val.ok and "extraction failed for 1 of 1 field(s): ticker" in fb
    val, _, fb = s.run_attempt(w + "print('EXTRACT ticker ACME | Trading Symbol(s) Common Stock ACME')", "a2")
    assert val.ok, fb
    assert "EXTRACT <field> <value>" in s._code_notes()


def test_routed_units_get_degenerate_checks_even_without_the_flag(tmp_path):
    s = _routed_solver(tmp_path)
    assert not s.cfg.degen
    code = "import json, os\njson.dump({'a': 0, 'b': 0, 'c': 0, 'd': ''}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))\nprint('EXTRACT a 7 | x')"
    val, _, fb = s.run_attempt(code, "a1")
    assert not val.ok and "degenerate:" in fb


def test_units_without_documents_are_unchanged(tmp_path):
    s = _solver(tmp_path, n_deliv=1, text_route=True)
    s.house = FakeHouse([], s.cfg)
    assert agent.text_route_docs(s.unit) == [] and not s.text_routed and s._code_notes() == ""
    code = "import json, os\njson.dump({'a': 0, 'b': 0, 'c': 0}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))"
    val, _, _ = s.run_attempt(code, "a1")
    assert val.ok   # no degenerate check, no EXTRACT check: identical to the release behaviour


def test_config_parses_text_route(monkeypatch):
    monkeypatch.setenv("AGENT_TEXT_ROUTE", "1")
    assert agent.Config.from_env().text_route


def test_text_playbook_is_triggered_by_html_inputs():
    import memory
    idx = memory.MemoryIndex(HERE.parent / "agent" / "memory")
    hits = {d.id for d in idx.triggered("Parse the filings.", ["filing_01.html"], unit_id="t1-new", task_dir_name="t1-new")}
    assert "text-filings-extraction-conventions" in hits


@pytest.mark.skipif(not (KIT / "t1-sec-8k-event-alpha").is_dir(), reason="public kit not present")
def test_probe_on_the_public_filing_units():
    cfg = agent.Config(endpoint="", memory=False, ignore_checks=True)
    expect = {"t1-sec-8k-event-alpha": ["input value 'NVDA' occurs", "Revenue is expected to be $28.0 billion", "Item codes in the text (document order): 2.02, 7.01, 9.01"],
              "t1-sec-10k-report-long": ["dei:DocumentType contextRef=c-1 scale= = '10-K'", "Supercenters | 3,559 | 632,577"],
              "t1-form4-cross-sectional-sale-pressure": ["transactionCode: S x8, C x1, F x1", "prices ranging from $106.73 to $107.73"]}
    import tempfile
    for u, needles in expect.items():
        with tempfile.TemporaryDirectory() as td:
            unit = agent.load_unit(KIT / u, Path(td), cfg)
            pr = agent.text_probe(unit, agent.text_route_docs(unit))
            for nd in needles:
                assert nd in pr, (u, nd)
