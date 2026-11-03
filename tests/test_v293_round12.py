"""Unit tests for agent 2.9.3 (round 12): degenerate-output detection (AGENT_DEGEN), the plan's conventions list carried
into code / review / vote prompts (AGENT_CONVENTIONS), INVARIANT lines as repair feedback (AGENT_INVARIANTS) and the
input-structure probe after degenerate outputs (AGENT_PROBE2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))
sys.path.insert(0, str(HERE))

import agent  # noqa: E402
from test_v22_features import FakeHouse, _solver  # noqa: E402

F = "```"


def _py(code):
    return (F + "python\n" + code + "\n" + F, "stop")


def _deliv(*files):
    return [agent.Deliverable(f, f.rsplit(".", 1)[-1], [], [], "t") for f in files]


# ---- degenerate outputs -----------------------------------------------------------------------------------------------
def test_degenerate_csv_columns(tmp_path):
    (tmp_path / "a.csv").write_text("date,qty,price,tag\n2020-01-01,0,1.5,x\n2020-01-02,0,1.7,x\n2020-01-03,0,1.9,x\n")
    iss = agent.degenerate_issues(tmp_path, _deliv("a.csv"))
    assert len(iss) == 1 and "qty (all zero)" in iss[0] and "price" not in iss[0] and "tag" not in iss[0]   # constant text is fine
    (tmp_path / "b.csv").write_text("x,y\n1,\n2,\n3,\n")
    iss = agent.degenerate_issues(tmp_path, _deliv("b.csv"))
    assert len(iss) == 1 and "y (all missing)" in iss[0]
    (tmp_path / "c.csv").write_text("x,name\n1,a\n2,b\n")
    assert agent.degenerate_issues(tmp_path, _deliv("c.csv")) == []


def test_degenerate_json_values(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"a": 0, "b": 0.0, "c": 0, "d": 1.5, "seed": 0}))
    iss = agent.degenerate_issues(tmp_path, _deliv("r.json"))
    assert len(iss) == 1 and "3 of 4 numeric values are exactly 0" in iss[0]
    (tmp_path / "s.json").write_text(json.dumps({"a": 1.2, "b": 0, "c": {"d": 3}, "ticker": ""}))
    iss = agent.degenerate_issues(tmp_path, _deliv("s.json"))
    assert len(iss) == 1 and "empty strings at ticker" in iss[0]
    (tmp_path / "t.json").write_text(json.dumps({"rows": [], "stats": {}, "n": 3}))
    iss = agent.degenerate_issues(tmp_path, _deliv("t.json"))
    assert len(iss) == 1 and "empty lists/objects at rows, stats" in iss[0]
    (tmp_path / "u.json").write_text(json.dumps({"a": 1.2, "b": 2.5, "c": 0}))
    assert agent.degenerate_issues(tmp_path, _deliv("u.json")) == []


def test_degenerate_files_named_in_the_task(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"a": 1.5, "b": 2.5}))
    instr = "Write results.json and monthly_results.csv. Inputs: prices.csv. See README.md. Also write plot.png."
    iss = agent.degenerate_issues(tmp_path, _deliv("results.json"), instr, {"prices.csv"})
    assert len(iss) == 1 and "monthly_results.csv" in iss[0] and "plot.png" in iss[0] and "prices.csv" not in iss[0] and "README" not in iss[0]
    (tmp_path / "monthly_results.csv").write_text("m,v\n1,2\n")
    (tmp_path / "plot.png").write_bytes(b"x")
    assert agent.degenerate_issues(tmp_path, _deliv("results.json"), instr, {"prices.csv"}) == []
    # a missing deliverable is reported by the shape validation, not twice
    assert agent.degenerate_issues(tmp_path, _deliv("results.json", "other.csv"), "other.csv is required", set()) == []


# ---- conventions and invariants helpers -------------------------------------------------------------------------------
def test_conventions_text_rendering():
    plan = {"conventions": [{"item": "sample window", "task_text": "use 2015-2020", "choice": "2015-01-01..2020-12-31 inclusive"},
                            {"item": "ddof", "task_text": "not stated", "choice": "1"}, {"item": "", "choice": "x"}, "junk"]}
    txt = agent.conventions_text(plan)
    assert txt.startswith("CONVENTIONS decided in the plan") and "- sample window: 2015-01-01..2020-12-31 inclusive  [task: use 2015-2020]" in txt
    assert "- ddof: 1  [not stated in the task; standard convention]" in txt and "junk" not in txt
    assert agent.conventions_text({"conventions": []}) == "" and agent.conventions_text(None) == ""


def test_parse_invariants():
    failed, ok = agent.parse_invariants("INVARIANT parity OK\nnoise\nINVARIANT pv_match FAIL 29363 vs 658385\n  INVARIANT mono fail x\n")
    assert ok == ["parity"] and failed == ["pv_match 29363 vs 658385", "mono x"]


# ---- solver integration -----------------------------------------------------------------------------------------------
def _zero_writer():
    return "import json, os\njson.dump({'a': 0, 'b': 0, 'c': 0, 'd': 0.0}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))"


def test_degen_fails_validation_and_probe2_adds_data_facts(tmp_path):
    s = _solver(tmp_path, n_deliv=1, degen=True, probe2=True)
    s.house = FakeHouse([], s.cfg)
    val, res, fb = s.run_attempt(_zero_writer(), "a1")
    assert not val.ok and any(i.startswith("degenerate:") for i in val.issues) and "4 of 4 numeric values are exactly 0" in fb
    assert "DATA FACTS" in fb and "prices.csv" in fb and s.probes_sent == 1
    s2 = _solver(tmp_path, n_deliv=1)   # flag off: the same output passes the shape checks
    s2.house = FakeHouse([], s2.cfg)
    val2, _, _ = s2.run_attempt(_zero_writer(), "a1")
    assert val2.ok


def test_invariant_failures_are_repair_feedback_then_soft(tmp_path):
    code = ("import json, os\njson.dump({'a': 1.5, 'b': 2.5}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))\n"
            "print('INVARIANT parity FAIL 49.2 > 0.01')\nprint('INVARIANT positive OK')")
    s = _solver(tmp_path, n_deliv=1, invariants=True)
    s.house = FakeHouse([], s.cfg)
    val, _, fb = s.run_attempt(code, "a1")
    assert not val.ok and "invariant failed: parity 49.2 > 0.01" in fb and s.inv_fails == {"parity": 1}
    val, _, _ = s.run_attempt(code + "\n# v2", "a2")
    assert not val.ok and s.inv_fails["parity"] == 2
    val, _, _ = s.run_attempt(code + "\n# v3", "a3")
    assert val.ok and any("failed repeatedly" in w for w in s.soft_warnings)


def test_conventions_flow_plan_code_review_and_vote(tmp_path):
    plan_reply = ("Plan text.\n```json\n" + json.dumps({"deliverables": [{"file": "f0.json", "format": "json", "required_keys": ["v"]}],
                  "conventions": [{"item": "window", "task_text": "2015-2020", "choice": "inclusive ends"}]}) + "\n```", "stop")
    code = _py("import json, os\njson.dump({'v': 2.5}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))")
    s = _solver(tmp_path, n_deliv=1, conventions=True, invariants=True, vote=True)
    s.cfg = agent.Config(endpoint="http://127.0.0.1:1", memory=False, selfcheck=False, do_plan=True, conventions=True, invariants=True, vote=True)
    s.house = FakeHouse([plan_reply, code, ("VERDICT: OK", "stop"), code], s.cfg)
    s.task_context = "task"
    pj = s.plan("task", with_memory=False)
    assert "conventions" in s.house.calls[0][2] and "sample window" in s.house.calls[0][2]
    s.plan_json = pj
    assert s.plan_json["conventions"][0]["choice"] == "inclusive ends"
    out = s.first_code("task", have_plan=True)
    req = s.house.calls[1][2]
    assert "CONVENTIONS decided in the plan" in req and "- window: inclusive ends  [task: 2015-2020]" in req and "INVARIANT <short_name> OK" in req
    val, _, _ = s.run_attempt(out, "a1")
    assert val.ok
    assert s.review() is None and "verify that the code implements EXACTLY these" in s.house.calls[2][2]
    s.vote(True)
    assert s.house.calls[3][0] == "vote1" and "CONVENTIONS decided in the plan" in s.house.calls[3][2] and s.vote_agreed


def test_config_parses_round12_flags(monkeypatch):
    for k in ("AGENT_DEGEN", "AGENT_CONVENTIONS", "AGENT_INVARIANTS", "AGENT_PROBE2"):
        monkeypatch.setenv(k, "1")
    c = agent.Config.from_env()
    assert c.degen and c.conventions and c.invariants and c.probe2


# ---- 2.9.4: the plan's JSON blocks are merged and parsed leniently ------------------------------------------------------
def test_extract_json_block_merges_blocks_and_repairs_slips():
    d = '```json\n{"deliverables": [{"file": "r.json", "format": "json", "required_keys": ["v"]}]}\n```'
    c = '```json\n{"conventions": [{"item": "ddof", "task_text": "not stated", "choice": "1",}],}\n```'   # trailing commas
    obj = agent.extract_json_block("plan text\n" + d + "\nmore\n" + c)
    assert obj["deliverables"][0]["file"] == "r.json" and obj["conventions"][0]["choice"] == "1"
    broken = '```json\n{"conventions": [ this is not json\n```'
    obj = agent.extract_json_block(d + "\n" + broken)
    assert obj["deliverables"] and "conventions" not in obj          # the broken second block costs nothing
    assert agent.extract_json_block("no json here") is None
    smart = '```json\n{“deliverables”: []}\n```'
    assert agent.extract_json_block(smart) == {"deliverables": []}


def test_conventions_note_asks_for_a_separate_block():
    assert "SECOND, separate ```json block" in agent.CONVENTIONS_PLAN_NOTE
