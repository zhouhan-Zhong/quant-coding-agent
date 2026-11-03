"""Unit tests for agent/memory.py and agent/selfcheck.py (run: python -m pytest tests/ -q)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))

import memory  # noqa: E402
import selfcheck  # noqa: E402

MEM = HERE.parent / "agent" / "memory"


@pytest.fixture(scope="module")
def index():
    return memory.MemoryIndex(MEM)


def test_library_loads_and_is_canary_free(index):
    kinds = {}
    for d in index.docs:
        kinds[d.kind] = kinds.get(d.kind, 0) + 1
    assert kinds.get("solution", 0) >= 30
    assert kinds.get("playbook", 0) >= 20
    assert kinds.get("note", 0) >= 3
    assert not index.dropped, index.dropped
    for d in index.docs:
        assert not memory._UUID_RE.search(d.text), d.id
        assert "canary" not in d.text.lower() or d.kind != "solution"
    assert any(d.pinned for d in index.docs)


def test_solution_headers_parsed(index):
    sols = [d for d in index.docs if d.kind == "solution"]
    for d in sols:
        assert d.unit_id.startswith("t1-"), d.id
        assert d.tags, d.id
        assert d.deliverables, d.id
        assert d.id == d.unit_id, (d.id, d.unit_id)
        assert d.unit_id in index.fingerprints, d.unit_id


def test_same_unit_excluded_by_id_and_content(index, tmp_path):
    sol = next(d for d in index.docs if d.kind == "solution")
    unit_dir = Path("/home/claude/track1-coding-public/units") / sol.unit_id
    if not unit_dir.is_dir():
        pytest.skip("public kit not available")
    instr = (unit_dir / "instruction.md").read_text()
    # by id
    sols, _ = index.retrieve(instr, unit_id=sol.unit_id, task_dir_name="whatever")
    assert sol.id not in [d.id for d, _ in sols]
    # by content (different id, lightly edited text)
    edited = instr.replace("0.05", "0.04").replace("252", "260")
    sols, _ = index.retrieve(edited, unit_id="t1-renamed", task_dir_name="t1-renamed")
    assert sol.id not in [d.id for d, _ in sols]
    # a genuinely different task keeps it available as a candidate when it ranks
    other = "Price a European call with Black-Scholes and write results.json with keys price and delta. " * 5
    sols, plays = index.retrieve(other, unit_id="t1-x", task_dir_name="t1-x")
    assert plays, "playbooks should be retrievable"


def test_shingle_similarity_separates_units(index):
    fps = index.fingerprints
    ids = sorted(fps)
    assert len(ids) >= 80
    # every unit is identical to itself and distinct from almost everything else
    for a in ids[:20]:
        assert memory.jaccard(fps[a], fps[a]) == 1.0
    close = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:] if memory.jaccard(fps[a], fps[b]) >= memory.SAME_UNIT_JACCARD]
    assert close == [], close


def test_plan_memory_parsing(index):
    sols = [(d, 10.0) for d in index.docs if d.kind == "solution"][:3]
    plays = [(d, 5.0) for d in index.docs if d.kind == "playbook"][:4]
    toc, labels = index.toc(sols, plays)
    assert "S1" in labels and "P1" in labels and "S1:" in toc
    picks = memory.parse_plan_memory({"memory": {"solutions": ["S2", "S1", "S3"], "playbooks": ["P4", "P1", "P2", "P3"]}}, labels)
    assert [d.id for d in picks if d.kind == "solution"] == [labels["S2"].id, labels["S1"].id]  # max 2
    assert len([d for d in picks if d.kind == "playbook"]) == 2  # max 2
    assert memory.parse_plan_memory({"memory": {"solutions": [], "playbooks": []}}, labels) == []
    assert memory.parse_plan_memory({"memory": "use S1 and P2"}, labels) == [labels["S1"], labels["P2"]]
    assert memory.parse_plan_memory(None, labels) == []
    picks = memory.default_picks(sols, plays)
    assert picks and picks[0].kind == "solution"


def test_render_respects_budget_and_marks_solutions(index):
    docs = index.pinned + [d for d in index.docs if d.kind == "solution"][:2] + [d for d in index.docs if d.kind == "playbook"][:2]
    pinned_len = sum(len(d.text) for d in index.pinned)
    txt = index.render(docs, 9000)
    assert len(txt) < 9000 + pinned_len + 800  # pinned material is added on top of the budget
    assert txt.count("```") % 2 == 0, "unbalanced code fences"
    assert txt.count("SOLUTION TO A DIFFERENT TASK") == 2
    assert "Essentials" in txt
    assert not memory._UUID_RE.search(txt)


def test_guid_documents_are_dropped(tmp_path):
    (tmp_path / "solutions").mkdir()
    (tmp_path / "solutions" / "t1-bad.py").write_text("# memory-solution\n# unit: t1-bad | category: x | tags: a\n# deliverables: a.json\n# 12345678-90ab-4cde-8f01-23456789abcd\nprint(1)\n")
    (tmp_path / "solutions" / "t1-good.py").write_text("# memory-solution\n# unit: t1-good | category: x | tags: a\n# deliverables: a.json\nprint(1)\n")
    idx = memory.MemoryIndex(tmp_path)
    assert [d.id for d in idx.docs] == ["t1-good"]
    assert len(idx.dropped) == 1


# ---------------------------------------------------------------------------------------------
# selfcheck
# ---------------------------------------------------------------------------------------------

INSTR = """
## Output
### File 1: `/app/output/calibration.json`
```json
{
    "n_returns": <int>,
    "sigma": <float>,
    "ticker": "<string>",
    "flag": <bool>,
    "n_maturities": 3
}
```
### File 2: `/app/output/prices.csv`
Columns: `T, K, price`

6 rows (2 maturities x 3 strikes), sorted by T then K ascending.
"""


def test_contract_parsing():
    c = selfcheck.parse_contract(INSTR)
    assert set(c.json_specs) == {"calibration.json"}
    assert c.csv_specs["prices.csv"].columns == ["T", "K", "price"]
    assert c.csv_specs["prices.csv"].n_rows == 6
    assert c.csv_specs["prices.csv"].sort_by == ["T", "K"]
    c2 = selfcheck.parse_contract(INSTR, ["calibration.json"])
    assert "prices.csv" not in c2.csv_specs


def test_check_outputs_hard_and_soft(tmp_path):
    c = selfcheck.parse_contract(INSTR)
    good = {"n_returns": 692, "sigma": 0.15, "ticker": "SPY", "flag": True, "n_maturities": 3}
    (tmp_path / "calibration.json").write_text(json.dumps(good))
    (tmp_path / "prices.csv").write_text("T,K,price\n0.5,90,1\n0.5,100,2\n0.5,110,3\n1,90,4\n1,100,5\n1,110,6\n")
    hard, soft = selfcheck.check_outputs(tmp_path, c)
    assert hard == [], hard
    bad = {"n_returns": "692", "sigma": "0.15", "ticker": 5, "flag": "true", "n_maturities": 4}
    (tmp_path / "calibration.json").write_text(json.dumps(bad))
    (tmp_path / "prices.csv").write_text("T,K,px\n0.5,90,1\n")
    hard, soft = selfcheck.check_outputs(tmp_path, c)
    joined = "\n".join(hard)
    assert "n_returns" in joined and "flag" in joined and "n_maturities" in joined and "ticker" in joined
    assert "missing columns ['price']" in joined
    assert "has 1 rows, the task says 6" in joined
    # integer-valued float for an int placeholder is only a soft warning
    (tmp_path / "calibration.json").write_text(json.dumps({**good, "n_returns": 692.0}))
    (tmp_path / "prices.csv").write_text("T,K,price\n0.5,90,1\n0.5,100,2\n0.5,110,3\n1,90,4\n1,100,5\n1,110,6\n")
    hard, soft = selfcheck.check_outputs(tmp_path, c)
    assert hard == [] and any("n_returns" in s for s in soft)


def test_dynamic_keys_and_arrays(tmp_path):
    instr = """
### `results.json`
```json
{
  "<ticker>": {"beta": <float>, "n": <int>},
  "summary": {"best": "<string>"}
}
```
### `events.json`
Save a JSON array with one object per event:
```json
[
  {"date": "<string>", "car": <float>}
]
```
"""
    c = selfcheck.parse_contract(instr)
    (tmp_path / "results.json").write_text(json.dumps({"AAPL": {"beta": 1.1, "n": 10}, "MSFT": {"beta": 0.9, "n": 12}, "summary": {"best": "AAPL"}}))
    (tmp_path / "events.json").write_text(json.dumps([{"date": "2024-01-02", "car": 0.1}, {"date": "2024-02-02", "car": "x"}]))
    hard, soft = selfcheck.check_outputs(tmp_path, c)
    assert any("events.json[1].car" in h for h in hard), hard
    assert not any("results.json" in h for h in hard), hard


@pytest.mark.skipif(not Path("/home/claude/track1-coding-public/units").is_dir(), reason="public kit not available")
def test_contract_parses_every_public_instruction():
    units = Path("/home/claude/track1-coding-public/units")
    n_json = n_csv = 0
    for u in sorted(units.iterdir()):
        instr = u / "instruction.md"
        if not instr.is_file():
            continue
        c = selfcheck.parse_contract(instr.read_text(encoding="utf-8", errors="replace"))
        n_json += len(c.json_specs)
        n_csv += len(c.csv_specs)
    assert n_json >= 60 and n_csv >= 60, (n_json, n_csv)


def test_section_aware_excerpt_keeps_conventions(index):
    pb = max((d for d in index.docs if d.kind == "playbook"), key=lambda d: len(d.text))
    ex = memory.excerpt_markdown(pb.text, 3000)
    assert len(ex) <= 3400
    assert "Conventions" in ex, ex[:300]
    assert ex.count("```") % 2 == 0
    sol = max((d for d in index.docs if d.kind == "solution"), key=lambda d: len(d.text))
    ex = memory.excerpt_code(sol.text, 4000)
    assert len(ex) <= 4100 and "middle of the script omitted" in ex
    assert ex.startswith("# memory-solution")


def test_pb_exclude_hides_unit_derived_playbooks(monkeypatch):
    import json as _json
    from pathlib import Path as _P
    root = _P(__file__).resolve().parent.parent / "agent" / "memory"
    excl = _json.loads((root / "pb_exclude.json").read_text())
    unit, docs = next(iter(excl.items()))
    instr = (root.parent.parent / "kit" / "units" / unit / "instruction.md")
    text = instr.read_text() if instr.exists() else " ".join(docs) * 20
    import memory as _m
    plain = _m.MemoryIndex(root)
    _, plays = plain.retrieve(text, unit_id=unit, task_dir_name=unit, n_playbooks=40)
    monkeypatch.setenv("AGENT_PB_EXCLUDE", "1")
    gated = _m.MemoryIndex(root)
    assert gated.pb_exclude and len(gated.docs) == len(plain.docs)
    _, plays2 = gated.retrieve(text, unit_id=unit, task_dir_name=unit, n_playbooks=40)
    ids, ids2 = {d.id for d, _ in plays}, {d.id for d, _ in plays2}
    assert not (ids2 & set(docs)) and (ids - ids2) <= set(docs)
