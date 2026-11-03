"""Unit tests for agent 2.9.5 (round 13): the two-track router (AGENT_TWO_TRACK) and the L track - core script with
shared quantities in $CORE_DIR/core.pkl, one short script per deliverable group, incremental runs, self-checks."""
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


CORE = _py("import pickle, os\npickle.dump({'v': 2.5, 'w': 7}, open(os.path.join(os.environ['CORE_DIR'], 'core.pkl'), 'wb'))\n"
           "print('CORE v float 2.5')\nprint('CORE w int 7')\nprint('CORE DONE')")
CORE_BAD = _py("raise RuntimeError('no data')")


def _group(files, value="c['v']", extra=""):
    return _py("import pickle, os, json\nc = pickle.load(open(os.path.join(os.environ['CORE_DIR'], 'core.pkl'), 'rb'))\n"
               f"for f in {files!r}:\n    json.dump({{'v': {value}}}, open(os.path.join(os.environ['OUTPUT_DIR'], f), 'w'))\n"
               "print('INVARIANT positive OK')" + extra)


def _lsolver(tmp_path, replies, n_deliv=4, **flags):
    s = _solver(tmp_path, n_deliv=n_deliv, two_track=True, track_l_min=3, track_l_repairs=1, **flags)
    s.house = FakeHouse(list(replies), s.cfg)
    s.task_context = "task"
    s.plan_json = {"groups": [{"name": "prices", "files": ["f0.json", "f1.json"], "needs": ["v"]},
                              {"name": "risk", "files": ["f2.json", "nonsense.csv"], "needs": ["w"]}]}
    return s


def test_derive_groups_restricts_and_completes(tmp_path):
    s = _lsolver(tmp_path, [])
    gs = s._derive_groups()
    assert [g["files"] for g in gs] == [["f0.json", "f1.json"], ["f2.json"], ["f3.json"]] and gs[0]["needs"] == ["v"] and gs[2]["name"] == "g3"
    s.plan_json = None
    assert [g["files"] for g in s._derive_groups()] == [["f0.json", "f1.json"], ["f2.json", "f3.json"]]


def test_track_l_core_and_groups_write_everything(tmp_path):
    s = _lsolver(tmp_path, [CORE, _group(["f0.json", "f1.json"]), _group(["f2.json"], value="c['w']"), _group(["f3.json"])])
    assert s._track_l("") is True
    assert [c[0] for c in s.house.calls] == ["core", "groupprices", "grouprisk", "groupg3"]
    assert json.loads((s.out_dir / "f0.json").read_text()) == {"v": 2.5} and json.loads((s.out_dir / "f2.json").read_text()) == {"v": 7}
    assert s.best_ok and s.backup_ok and "# ===== core =====" in s.last_code and "# ===== risk =====" in s.last_code
    req = s.house.calls[1][2]
    assert "core.pkl" in req and "CORE v float 2.5" in req and "f0.json, f1.json" in req and "INVARIANT <name> OK" in req and "do not recompute" in req
    assert "CORE_DIR" in s.house.calls[0][2] and "SUMMARY" in s.house.calls[0][2]


def test_track_l_repairs_a_failing_group_and_keeps_other_files(tmp_path):
    bad = _py("raise ValueError('boom')")
    s = _lsolver(tmp_path, [CORE, _group(["f0.json", "f1.json"]), bad, _group(["f2.json"]), _group(["f3.json"])])
    assert s._track_l("") is True
    labels = [c[0] for c in s.house.calls]
    assert labels == ["core", "groupprices", "grouprisk", "grouprisk-fix1", "groupg3"]
    assert "boom" in s.house.calls[3][2] and (s.out_dir / "f0.json").is_file() and (s.out_dir / "f2.json").is_file() and s.best_ok


def test_track_l_flags_degenerate_and_outside_writes(tmp_path):
    outside = _group(["f0.json", "f1.json"], extra="\njson.dump({'x': 1}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f3.json'), 'w'))")
    zero = _group(["f2.json"], value="0")
    s = _lsolver(tmp_path, [CORE, outside, _group(["f0.json", "f1.json"]), zero, _group(["f3.json"])])
    assert s._track_l("") is True
    labels = [c[0] for c in s.house.calls]
    assert labels == ["core", "groupprices", "groupprices-fix1", "grouprisk", "groupg3"]
    assert "outside its group" in s.house.calls[2][2]
    # a single-value JSON is below the degenerate threshold (needs >= 3 numbers), so the zero group only fails the shape of nothing: it passes
    assert (s.out_dir / "f2.json").is_file()


def test_track_l_falls_back_when_the_core_cannot_be_built(tmp_path):
    s = _lsolver(tmp_path, [CORE_BAD, CORE_BAD, _group(["f0.json"])])
    assert s._track_l("") is False
    assert [c[0] for c in s.house.calls] == ["core", "core-fix1"] and "no data" in s.house.calls[1][2]


def test_track_l_invariant_failure_triggers_a_repair(tmp_path):
    failing = _group(["f0.json", "f1.json"], extra="\nprint('INVARIANT parity FAIL 49.2')")
    s = _lsolver(tmp_path, [CORE, failing, _group(["f0.json", "f1.json"]), _group(["f2.json"]), _group(["f3.json"])])
    assert s._track_l("") is True
    assert [c[0] for c in s.house.calls][:3] == ["core", "groupprices", "groupprices-fix1"] and "invariant failed: parity 49.2" in s.house.calls[2][2]


def test_plan_asks_for_groups_only_on_big_units(tmp_path):
    s = _solver(tmp_path, n_deliv=4, two_track=True, track_l_min=3)
    s.cfg = agent.Config(endpoint="http://127.0.0.1:1", memory=False, selfcheck=False, do_plan=True, two_track=True, track_l_min=3)
    s.house = FakeHouse([("plan\n```json\n{\"deliverables\": []}\n```", "stop")], s.cfg)
    s.plan("task")
    assert '"groups"' in s.house.calls[0][2]
    s2 = _solver(tmp_path, n_deliv=1, two_track=True, track_l_min=5)
    s2.cfg = agent.Config(endpoint="http://127.0.0.1:1", memory=False, selfcheck=False, do_plan=True, two_track=True, track_l_min=5)
    s2.house = FakeHouse([("plan\n```json\n{\"deliverables\": []}\n```", "stop")], s2.cfg)
    s2.plan("task")
    assert '"groups"' not in s2.house.calls[0][2]


def test_config_parses_two_track_flags(monkeypatch):
    monkeypatch.setenv("AGENT_TWO_TRACK", "1")
    monkeypatch.setenv("AGENT_TRACK_L_MIN", "7")
    monkeypatch.setenv("AGENT_TRACK_L_REPAIRS", "9")
    c = agent.Config.from_env()
    assert c.two_track and c.track_l_min == 7 and c.track_l_repairs == 4
