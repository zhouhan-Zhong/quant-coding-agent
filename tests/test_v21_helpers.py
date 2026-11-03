"""Unit tests for the agent 2.1 helpers: script fingerprints, edit blocks, output comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))

import agent  # noqa: E402

CODE = "import json\n\ndef main():\n    x = 1\n    y = 2   \n    print(x + y)\n\nmain()\n"


def test_fingerprint_ignores_blank_lines_and_trailing_space():
    other = CODE.replace("\n\n", "\n").replace("y = 2   ", "y = 2")
    assert agent.code_fingerprint(CODE) == agent.code_fingerprint(other)
    assert agent.code_fingerprint(CODE) != agent.code_fingerprint(CODE.replace("y = 2", "y = 3"))


def test_edit_blocks_apply_exact_and_whitespace_tolerant():
    reply = ("<<<<<<< SEARCH\n    x = 1\n=======\n    x = 10\n>>>>>>> REPLACE\n"
             "<<<<<<< SEARCH\n    y = 2\n=======\n    y = 20\n>>>>>>> REPLACE")
    out = agent.apply_edit_blocks(CODE, reply)
    assert out is not None and "x = 10" in out and "y = 20" in out and "print(x + y)" in out


def test_edit_blocks_reject_missing_search_and_plain_text():
    assert agent.apply_edit_blocks(CODE, "<<<<<<< SEARCH\nz = 9\n=======\nz = 1\n>>>>>>> REPLACE") is None
    assert agent.apply_edit_blocks(CODE, "no edits here") is None


def test_edit_block_may_delete_lines():
    out = agent.apply_edit_blocks(CODE, "<<<<<<< SEARCH\n    print(x + y)\n=======\n>>>>>>> REPLACE")
    assert out is not None and "print(x + y)" not in out


def test_compare_outputs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "r.json").write_text(json.dumps({"price": 10.0, "nested": {"vol": 0.2}, "n": 5}))
    (b / "r.json").write_text(json.dumps({"price": 10.00001, "nested": {"vol": 0.25}, "n": 5}))
    (a / "t.csv").write_text("date,v\n2020-01-01,1.0\n2020-01-02,2.0\n")
    (b / "t.csv").write_text("date,v\n2020-01-01,1.0\n2020-01-02,2.5\n")
    (a / "s.csv").write_text("v\n1\n2\n")
    (b / "s.csv").write_text("v\n1\n")
    dl = [agent.Deliverable(f, f.rsplit(".", 1)[1], [], [], "test") for f in ("r.json", "t.csv", "s.csv", "missing.json")]
    diffs = agent.compare_outputs(a, b, dl)
    text = "\n".join(diffs)
    assert "nested.vol" in text and "price" not in text
    assert "t.csv column v: 1/2 rows differ" in text
    assert "s.csv: shape" in text
    assert len(diffs) == 3
