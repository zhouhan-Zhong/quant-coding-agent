"""Unit tests for agent 2.9.10: the referential-integrity check of identifier columns (AGENT_KEY_CHECK)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))
sys.path.insert(0, str(HERE))

import agent  # noqa: E402
from test_v22_features import FakeHouse, _solver  # noqa: E402


def _inputs(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    (d / "universe.csv").write_text('ticker,company_name,bucket\nNVDA,"NVIDIA Corporation",Large\nINTC,Intel,Mid\nRIVN,Rivian,Mid\n')
    (d / "prices.csv").write_text("date,ticker,close,note\n2024-01-02,NVDA,1.5,\n2024-01-02,INTC,2.5,x\n2024-01-03,NVDA,1.6,\n")
    (d / "quotes_a.csv").write_text("tenor_label,rate\n1M,0.05\n3M,0.051\n")
    (d / "quotes_b.csv").write_text("tenor_label,rate\n4Y,0.04\n25Y,0.045\n")
    return sorted(d.iterdir())


def test_input_key_sets_picks_identifier_columns(tmp_path):
    keys = agent.input_key_sets(_inputs(tmp_path))
    assert keys["ticker"][1] == {"NVDA", "INTC", "RIVN"} and keys["ticker"][2] is False
    assert keys["bucket"][1] == {"Large", "Mid"}
    assert "date" not in keys and "close" not in keys and "rate" not in keys   # dates and numbers are not identifiers
    assert keys["tenor_label"][1] == {"1M", "3M", "4Y", "25Y"}                # same column in two inputs -> union
    assert "note" not in keys                                                  # a single distinct value


def test_key_issues_flags_foreign_values_and_empties(tmp_path):
    keys = agent.input_key_sets(_inputs(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    (out / "signals.csv").write_text("filing_id,ticker,score\nf1,Name,1.0\nf2,INTC,2.0\nf3,,3.0\nf4,per,0.5\n")
    (out / "results.json").write_text(json.dumps({"top_long_ticker": "N", "top_short_ticker": "INTC", "n": 4,
                                                  "by_tenor": [{"tenor_label": "2024-03-22"}]}))
    hard, soft = agent.key_issues([out / "signals.csv", out / "results.json"], keys, "Write signals.csv, one row per filing.")
    text = "\n".join(hard)
    assert "signals.csv column ticker: 2 value(s) are not values of the input prices.csv:ticker / universe.csv:ticker: ['Name', 'per']" in text
    assert "signals.csv column ticker: 1 of 4 value(s) are empty" in text
    assert "results.json:top_long_ticker: 1 value(s)" in text and "top_short_ticker" not in text
    assert "results.json:by_tenor[0].tenor_label" in text
    assert soft == []


def test_key_issues_allows_task_literals_case_and_clean_outputs(tmp_path):
    keys = agent.input_key_sets(_inputs(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.csv").write_text("ticker,w\nNVDA,0.5\nINTC,0.5\nTOTAL,1.0\nportfolio,1.0\n")
    hard, soft = agent.key_issues([out / "a.csv"], keys, 'Add a TOTAL row and a row with ticker "portfolio".')
    assert hard == [] and soft == []
    (out / "b.csv").write_text("ticker,w\nnvda,0.5\nINTC,0.5\n")
    hard, soft = agent.key_issues([out / "b.csv"], keys, "")
    assert hard == [] and len(soft) == 1 and "letter case" in soft[0]
    # an input column with blanks allows blanks in the output
    (out / "c.csv").write_text("ticker,note\nNVDA,\nINTC,x\n")
    keys2 = dict(keys, note=("prices.csv:note", {"x", "y"}, True))
    assert agent.key_issues([out / "c.csv"], keys2, "")[0] == []


def test_key_check_blocks_then_softens(tmp_path, capsys):
    s = _solver(tmp_path, n_deliv=1, key_check=True)
    s.house = FakeHouse([], s.cfg)
    data = s.unit.data_root
    (data / "universe.csv").write_text("ticker,bucket\nNVDA,Large\nINTC,Mid\n")
    s.unit.data_files = [agent.DataFile(p, p.name, p.stat().st_size) for p in sorted(data.iterdir())]
    s.deliverables = [agent.Deliverable("f0.csv", "csv", [], [], "test")]
    code = ("import os\nopen(os.path.join(os.environ['OUTPUT_DIR'], 'f0.csv'), 'w').write('ticker,score\\nName,1.0\\nINTC,2.0\\n')\n")
    for i in range(2):
        val, _, fb = s.run_attempt(code, f"a{i}")
        assert not val.ok and "not values of the input universe.csv:ticker: ['Name']" in fb
    val, _, _ = s.run_attempt(code, "a2")
    assert val.ok and any("repeated" in w for w in s.soft_warnings)
    out = capsys.readouterr().out
    assert "key check: 1 identifier column(s) in the inputs: ['ticker']" in out or "identifier column(s)" in out
    good = ("import os\nopen(os.path.join(os.environ['OUTPUT_DIR'], 'f0.csv'), 'w').write('ticker,score\\nNVDA,1.0\\nINTC,2.0\\n')\n")
    val, _, _ = s.run_attempt(good, "a3")
    assert val.ok


def test_key_check_off_by_default_and_flag_parses(monkeypatch):
    assert agent.Config(endpoint="x").key_check is False
    monkeypatch.setenv("AGENT_KEY_CHECK", "1")
    assert agent.Config.from_env().key_check
