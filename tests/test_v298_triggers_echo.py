"""Unit tests for agent 2.9.8: trigger-pinned memory documents (AGENT_MEM_TRIGGERS) and the parameter echo check
(AGENT_PARAM_ECHO)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))
sys.path.insert(0, str(HERE))

import agent  # noqa: E402
import memory  # noqa: E402
from test_v22_features import FakeHouse, _solver  # noqa: E402


def _lib(tmp_path):
    root = tmp_path / "mem"
    (root / "notes").mkdir(parents=True)
    (root / "playbooks").mkdir()
    (root / "notes" / "family.md").write_text("---\ntitle: Family note\ncategory: meta\ntags: a\nsummary: s\ntriggers: file:params.json, text:solution.json intermediates\n---\n## body\nfamily text\n")
    (root / "playbooks" / "topic.md").write_text("---\ntitle: Topic\ncategory: risk\ntags: b\nsummary: s\ntriggers: text:copula\n---\n## body\ntopic text\n")
    (root / "playbooks" / "plain.md").write_text("---\ntitle: Plain\ncategory: risk\ntags: c\nsummary: s\n---\n## body\nplain text\n")
    (root / "pb_exclude.json").write_text(json.dumps({"t1-x": ["topic"]}))
    return memory.MemoryIndex(root)


def test_triggers_parse_and_match(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PB_EXCLUDE", "1")
    idx = _lib(tmp_path)
    fam = next(d for d in idx.docs if d.id == "family")
    assert fam.triggers == ["file:params.json", "text:solution.json intermediates"]
    hits = idx.triggered("Fit a copula to the returns.", ["returns.csv", "params.json"], unit_id="t1-y", task_dir_name="t1-y")
    assert sorted(d.id for d in hits) == ["family", "topic"]
    hits = idx.triggered("Nothing special here.", ["prices.csv"], unit_id="t1-y", task_dir_name="t1-y")
    assert hits == []
    # a document derived from the unit itself stays excluded
    hits = idx.triggered("Fit a copula.", ["params.json"], unit_id="t1-x", task_dir_name="t1-x")
    assert [d.id for d in hits] == ["family"]


def test_real_library_triggers_cover_the_family_notes():
    idx = memory.MemoryIndex(Path(__file__).resolve().parent.parent / "agent" / "memory")
    hits = {d.id for d in idx.triggered("Write solution.json with intermediates.", ["params.json", "glossary.json", "returns.csv"], unit_id="t1-new", task_dir_name="t1-new")}
    assert "params-verifier-family" in hits and "data-cleaning-report-and-count-conventions" not in hits
    assert "params-verifier-family" not in {d.id for d in idx.triggered("x", ["params.json", "a.csv"], unit_id="t1-new", task_dir_name="t1-new")}
    hits = {d.id for d in idx.triggered("Parse the 8-K filings.", ["filing_01.html"], unit_id="t1-new", task_dir_name="t1-new")}
    assert "text-and-filings-parsing" in hits


def test_param_echo_helpers():
    echo = agent.parse_param_lines("PARAM window 60\nnoise\nPARAM seed 42\nPARAM start 2020-01-01\nPARAM alpha 0.99\nPARAM n_paths 100000")
    assert echo == {"window": "60", "seed": "42", "start": "2020-01-01", "alpha": "0.99", "n_paths": "100000"}
    nums, dates = agent.stated_values("Use a 60-day window, seed 42, 99% confidence and 100,000 paths from 2020-01-01.")
    assert {60.0, 42.0, 99.0, 0.99, 100000.0} <= nums and dates == {"2020-01-01"}
    assert agent.unstated_params(echo, nums, dates) == []
    bad = agent.unstated_params({"window": "750", "seed": "42", "end": "2021-06-30"}, nums, dates)
    assert len(bad) == 2 and bad[0].startswith("window=750") and "end=2021-06-30" in bad[1]


def test_param_echo_adds_soft_warnings_without_blocking(tmp_path, capsys):
    s = _solver(tmp_path, n_deliv=1, param_echo=True)
    s.house = FakeHouse([], s.cfg)
    s.unit.instruction = "Use a 60-day window and seed 42."
    code = ("import json, os\nprint('PARAM window 90')\nprint('PARAM seed 42')\n"
            "json.dump({'v': 1.5}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))")
    val, _, _ = s.run_attempt(code, "a1")
    assert val.ok and any("window=90 is not a value stated" in w for w in s.soft_warnings)
    assert "param echo: 2 parameters, 1 not stated" in capsys.readouterr().out
    assert "PARAM <name> <value>" in s._code_notes()


def test_config_parses_298_flags(monkeypatch):
    monkeypatch.setenv("AGENT_MEM_TRIGGERS", "1")
    monkeypatch.setenv("AGENT_PARAM_ECHO", "1")
    c = agent.Config.from_env()
    assert c.mem_triggers and c.param_echo
