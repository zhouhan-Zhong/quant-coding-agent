"""Unit tests for agent 2.9.6: convention voting (AGENT_CONV_VOTE), the clause audit review (AGENT_CLAUSE_AUDIT),
Monte-Carlo rules and vote tolerance (AGENT_MC_RULES), full-effort plan reasoning (AGENT_PLAN_THINK) and the plan-time
data probe (AGENT_PROBE_PLAN)."""
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


def _conv(items):
    return ("```json\n" + json.dumps({"conventions": items}) + "\n```", "stop")


class KwHouse(FakeHouse):
    """FakeHouse that also records the thinking / low_effort keywords."""

    def chat(self, messages, *, thinking=None, max_tokens=None, label="call", temperature=None, low_effort=None):
        self.calls.append((label, [m["role"] for m in messages], messages[-1]["content"], thinking, low_effort))
        self.requests_made += 1
        return self.replies.pop(0)


# ---- convention voting -------------------------------------------------------------------------------------------------
def test_merge_conventions_majority_and_ties():
    plan = [{"item": "ddof", "task_text": "not stated", "choice": "0"}, {"item": "Sample window", "task_text": "2015-2020", "choice": "inclusive"}]
    r1 = [{"item": "DDOF", "task_text": "x", "choice": "1"}, {"item": "sample window", "task_text": "", "choice": "inclusive"}, {"item": "seed", "choice": "42"}]
    r2 = [{"item": "ddof", "task_text": "", "choice": "1"}]
    merged, multi, changed = agent.merge_conventions([plan, r1, r2])
    by = {agent._conv_key(c["item"]): c["choice"] for c in merged}
    assert by["ddof"] == "1" and by["sample window"] == "inclusive" and by["seed"] == "42"
    assert multi == 2 and changed == 1
    merged, _, changed = agent.merge_conventions([plan, [{"item": "ddof", "choice": "1"}]])   # 1-1 tie keeps the plan
    assert {c["item"]: c["choice"] for c in merged}["ddof"] == "0" and changed == 0


def test_conventions_vote_replaces_the_plan_list(tmp_path, capsys):
    s = _solver(tmp_path, n_deliv=1, conventions=True, conv_vote=2)
    s.house = FakeHouse([_conv([{"item": "ddof", "task_text": "", "choice": "1"}]), _conv([{"item": "ddof", "task_text": "", "choice": "1"}, {"item": "window", "choice": "60 days"}])], s.cfg)
    s.plan_json = {"conventions": [{"item": "ddof", "task_text": "not stated", "choice": "0"}]}
    s._conventions_vote("task")
    assert [c[0] for c in s.house.calls] == ["conv1", "conv2"] and "ONLY one ```json block" in s.house.calls[0][2]
    by = {c["item"]: c["choice"] for c in s.plan_json["conventions"]}
    assert by["ddof"] == "1" and by["window"] == "60 days"
    assert "3 reads -> 2 items (1 seen in >= 2 reads, 1 plan choices outvoted)" in capsys.readouterr().out


def test_conventions_vote_keeps_plan_when_reads_fail(tmp_path):
    s = _solver(tmp_path, n_deliv=1, conventions=True, conv_vote=1)
    s.house = FakeHouse([("no json here", "stop")], s.cfg)
    s.plan_json = {"conventions": [{"item": "ddof", "task_text": "", "choice": "0"}]}
    s._conventions_vote("task")
    assert s.plan_json["conventions"][0]["choice"] == "0"


# ---- clause audit ----------------------------------------------------------------------------------------------------------
def test_clause_audit_applies_edit_blocks_and_logs_counts(tmp_path, capsys):
    code = "import json, os\njson.dump({'v': 2.5}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))"
    s = _solver(tmp_path, n_deliv=1, clause_audit=True)
    reply = ("CLAUSE: 'v equals 5' -> json.dump -> MISMATCH: writes 2.5\nCLAUSE: 'write f0.json' -> json.dump -> OK\n"
             "<<<<<<< SEARCH\njson.dump({'v': 2.5}\n=======\njson.dump({'v': 5.0}\n>>>>>>> REPLACE", "stop")
    s.house = FakeHouse([reply], s.cfg)
    val, _, _ = s.run_attempt(code, "a1")
    assert val.ok
    new = s.review()
    assert new and "5.0" in new and "clause by clause" in s.house.calls[0][2]
    assert "clause audit: 2 clauses, 1 mismatches" in capsys.readouterr().out
    s.house = FakeHouse([("CLAUSE: 'x' -> f -> OK\nVERDICT: OK", "stop")], s.cfg)
    assert s.review() is None


# ---- Monte-Carlo rules ---------------------------------------------------------------------------------------------------------
def test_mc_rules_note_and_vote_tolerance(tmp_path):
    s = _solver(tmp_path, n_deliv=1, mc_rules=True, vote=True)
    s.house = FakeHouse([], s.cfg)
    s.mc_task = True
    assert "MONTE CARLO / SIMULATION RULES" in s._code_notes()
    s.mc_task = False
    assert s._code_notes() == ""
    # compare_outputs honours the tolerance
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "f0.json").write_text(json.dumps({"price": 10.0})); (b / "f0.json").write_text(json.dumps({"price": 10.3}))
    d = [agent.Deliverable("f0.json", "json", [], [], "t")]
    assert agent.compare_outputs(a, b, d) and not agent.compare_outputs(a, b, d, rtol=0.05)


def test_mc_vote_agreement_within_five_percent(tmp_path):
    def script(v):
        return _py("import json, os\njson.dump({'v': " + str(v) + "}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))")
    s = _solver(tmp_path, n_deliv=1, vote=True, mc_rules=True, vote_max=2)
    s.house = FakeHouse([script(1.02)], s.cfg)
    s.task_context = "task"
    s.mc_task = True
    val, _, _ = s.run_attempt(agent.extract_code(script(1.0)[0]), "a1")
    assert val.ok
    s.vote(True)
    assert s.vote_agreed and len(s.house.calls) == 1


# ---- plan with full reasoning -----------------------------------------------------------------------------------------------------
def test_plan_think_requests_full_effort_reasoning(tmp_path):
    s = _solver(tmp_path, n_deliv=1, plan_think=True, low_effort=True)
    s.cfg = agent.Config(endpoint="http://127.0.0.1:1", memory=False, selfcheck=False, do_plan=True, plan_think=True, low_effort=True)
    s.house = KwHouse([("plan\n```json\n{\"deliverables\": []}\n```", "stop")], s.cfg)
    s.plan("task")
    assert s.house.calls[0][3] is True and s.house.calls[0][4] is False


# ---- plan-time data probe (via solve's context building is exercised in the mock run; here the helper) --------------------------
def test_data_facts_available_for_probe_plan(tmp_path):
    s = _solver(tmp_path, n_deliv=1, probe_plan=True)
    facts = agent.data_facts(s.unit)
    assert "prices.csv" in facts and "close" in facts


def test_config_parses_round13b_flags(monkeypatch):
    monkeypatch.setenv("AGENT_CONV_VOTE", "5")
    for k in ("AGENT_CLAUSE_AUDIT", "AGENT_MC_RULES", "AGENT_PLAN_THINK", "AGENT_PROBE_PLAN"):
        monkeypatch.setenv(k, "1")
    c = agent.Config.from_env()
    assert c.conv_vote == 3 and c.clause_audit and c.mc_rules and c.plan_think and c.probe_plan


# ---- 2.9.7: robust convention parsing, group size cap, piece feedback ----------------------------------------------------
def test_parse_conventions_reply_fallbacks():
    good = '```json\n{"conventions": [{"item": "ddof", "task_text": "", "choice": "1"}]}\n```'
    assert agent.parse_conventions_reply(good)[0]["choice"] == "1"
    bare = 'Here you go:\n{"conventions": [{"item": "ddof", "task_text": "x", "choice": "1",}]}'
    assert agent.parse_conventions_reply(bare)[0]["item"] == "ddof"
    fenced = '```\n{"conventions": [{"item": "window", "task_text": "", "choice": "60"}]}\n```'
    assert agent.parse_conventions_reply(fenced)[0]["choice"] == "60"
    broken = '```json\n{"conventions": [{"item": "ddof", "task_text": "not stated", "choice": "1"}, {"item": "seed" "choice": "42"}]\n```'
    got = agent.parse_conventions_reply(broken)
    assert [c["item"] for c in got] == ["ddof", "seed"] and got[0]["task_text"] == "not stated" and got[1]["choice"] == "42"
    assert agent.parse_conventions_reply("nothing") == []


def test_plan_recovers_conventions_from_a_malformed_block(tmp_path):
    s = _solver(tmp_path, n_deliv=1, conventions=True)
    s.cfg = agent.Config(endpoint="http://127.0.0.1:1", memory=False, selfcheck=False, do_plan=True, conventions=True)
    reply = ('```json\n{"deliverables": [{"file": "f0.json", "format": "json", "required_keys": ["v"]}]}\n```\n'
             '```json\n{"conventions": [{"item": "ddof", "task_text": "", "choice": "1"} {"item": "window", "choice": "60"}]}\n```', "stop")
    s.house = FakeHouse([reply], s.cfg)
    pj = s.plan("task")
    assert pj["deliverables"] and [c["item"] for c in pj["conventions"]] == ["ddof", "window"]


def test_groups_are_capped_at_three_files_and_never_named_core(tmp_path):
    s = _solver(tmp_path, n_deliv=6, two_track=True, track_l_min=3)
    s.plan_json = {"groups": [{"name": "core", "files": [f"f{i}.json" for i in range(5)], "needs": ["v"]}]}
    gs = s._derive_groups()
    assert [g["files"] for g in gs] == [["f0.json", "f1.json", "f2.json"], ["f3.json", "f4.json"], ["f5.json"]]
    assert gs[0]["name"] == "g1" and gs[1]["name"] == "g1_2" and gs[2]["name"] == "g3"


def test_piece_feedback_hints_when_nothing_was_written(tmp_path):
    s = _solver(tmp_path, n_deliv=2, two_track=True)
    res = agent.RunResult(0, "", "", False, 0.5)
    fb = s._piece_feedback(res, ["missing deliverable: f0.json", "missing deliverable: f1.json"])
    assert "exited normally but wrote none" in fb
    assert "exited normally" not in s._piece_feedback(agent.RunResult(1, "", "Traceback\nKeyError: 'x'", False, 0.5), ["missing deliverable: f0.json"])
