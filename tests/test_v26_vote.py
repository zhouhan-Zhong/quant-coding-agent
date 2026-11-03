"""Unit tests for agent 2.6: majority vote over independent solutions (AGENT_VOTE)."""
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


def _script(v):
    return (F + "python\nimport json, os\njson.dump({'v': " + str(v) + "}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))\n" + F, "stop")


def _vote_solver(tmp_path, replies, **flags):
    s = _solver(tmp_path, n_deliv=1, vote=True, **flags)
    s.house = FakeHouse(replies, s.cfg)
    s.task_context = "task"
    return s


def _primary(s, v):
    code = agent.extract_code(_script(v)[0])
    val, _, _ = s.run_attempt(code, "a1")
    assert val.ok


def _value(s):
    return json.loads((s.out_dir / "f0.json").read_text())["v"]


def test_vote_stops_when_an_independent_solution_agrees(tmp_path):
    s = _vote_solver(tmp_path, [_script(1)])
    _primary(s, 1)
    s.vote(True)
    assert _value(s) == 1 and len(s.house.calls) == 1


def test_vote_replaces_an_outvoted_answer(tmp_path):
    s = _vote_solver(tmp_path, [_script(2), _script(2), _script(3)])
    _primary(s, 1)
    s.vote(True)
    assert _value(s) == 2 and len(s.house.calls) == 3 and s.best_ok


def test_vote_keeps_the_current_answer_on_a_tie(tmp_path):
    s = _vote_solver(tmp_path, [_script(2), _script(3), _script(4)])
    _primary(s, 1)
    s.vote(True)
    assert _value(s) == 1


def test_vote_repairs_a_failing_candidate_once(tmp_path):
    bad = (F + "python\nraise ValueError('boom')\n" + F, "stop")
    s = _vote_solver(tmp_path, [bad, _script(1)])
    _primary(s, 1)
    s.vote(True)
    assert s.house.calls[1][0] == "vote1-fix" and "boom" in s.house.calls[1][2] and _value(s) == 1


def test_vote_rescues_a_failed_unit(tmp_path):
    s = _vote_solver(tmp_path, [_script(5), _script(6), _script(7)])
    s.vote(False)
    assert _value(s) == 5 and s.best_ok
