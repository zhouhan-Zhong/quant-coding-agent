"""Unit tests for agent 2.4: error signatures, same-error strategy change, continuation rollback, restart, memory knob."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))
sys.path.insert(0, str(HERE))

import agent  # noqa: E402
from test_v22_features import FakeHouse, _solver  # noqa: E402

TB = ('Traceback (most recent call last):\n  File "/tmp/w/solution_a1.py", line 55, in <module>\n'
      '    T, N, L = Z_lag.shape\n    ^^^^^^^\nValueError: not enough values to unpack (expected 3, got 2)')


def test_error_signature():
    assert agent.error_signature(TB) == "ValueError@T, N, L = Z_lag.shape"
    assert agent.error_signature(TB.replace("solution_a1", "solution_a2")) == agent.error_signature(TB)
    assert agent.error_signature("") == "" and agent.error_signature("all good\n") == ""
    assert agent.error_signature("KeyError: 'x'").startswith("KeyError@")


def test_same_error_twice_changes_the_repair_prompt(tmp_path):
    s = _solver(tmp_path, n_deliv=1)
    s.house = FakeHouse([("```python\nimport json\nprint(2)\n```", "stop")], s.cfg)
    s.messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "task"},
                  {"role": "assistant", "content": "```python\nprint(1)\n```"}]
    s.last_err_sig = "ValueError@T, N, L = Z_lag.shape"
    s.err_seen = {s.last_err_sig: 2}
    s.repair("STDERR (tail):\n" + TB)
    sent = s.house.calls[-1][2]
    assert agent.SAME_ERROR_NOTE in sent and "DIFFERENT approach" in sent


def test_first_occurrence_keeps_the_plain_prompt(tmp_path):
    s = _solver(tmp_path, n_deliv=1)
    s.house = FakeHouse([("```python\nimport json\nprint(2)\n```", "stop")], s.cfg)
    s.messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "task"},
                  {"role": "assistant", "content": "```python\nprint(1)\n```"}]
    s.last_err_sig = "ValueError@x"
    s.err_seen = {s.last_err_sig: 1}
    s.repair("STDERR (tail):\n" + TB)
    assert agent.SAME_ERROR_NOTE not in s.house.calls[-1][2]


def test_continuation_rolls_back_to_a_complete_statement(tmp_path):
    s = _solver(tmp_path, n_deliv=1)
    cut = "```python\nimport json\nx = 1\ny = compute(x,\n          2"
    s.house = FakeHouse([("```python\ny = compute(x, 2)\nprint(y)\n```", "stop")], s.cfg)
    s.messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "task"}]
    code = s._complete_code(cut, "length")
    # the continuation request shows the rolled-back tail, ending at 'x = 1'
    assert s.house.calls[0][2].rstrip().endswith("x = 1\n```")
    assert code == "import json\nx = 1\ny = compute(x, 2)\nprint(y)"


def test_run_attempt_records_error_signatures(tmp_path):
    s = _solver(tmp_path, n_deliv=1)
    s.deliverables = [agent.Deliverable("f0.json", "json", [], [], "test")]
    s.run_attempt("a, b, c = (1, 2)\n", "a1")
    s.run_attempt("a, b, c = (1, 2)\n\n", "a2")
    assert s.last_err_sig.startswith("ValueError@") and s.err_seen[s.last_err_sig] == 2


def test_restart_runs_an_independent_script(tmp_path):
    s = _solver(tmp_path, n_deliv=1, restart=True)
    s.deliverables = [agent.Deliverable("f0.json", "json", [], [], "test")]
    s.task_context = "TASK CONTEXT"
    out = str(s.out_dir / "f0.json")
    s.house = FakeHouse([(f"```python\nimport json\njson.dump({{'a': 1}}, open({out!r}, 'w'))\n```", "stop")], s.cfg)
    s.messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "task"}]
    code, val, res, fb = s._restart("STDERR (tail):\nValueError: boom")
    assert val is not None and val.ok
    label, roles, prompt, _ = s.house.calls[0]
    assert label == "restart" and roles == ["system", "user"]
    assert "TASK CONTEXT" in prompt and "fresh, complete implementation" in prompt and "ValueError: boom" in prompt
    assert len(s.messages) == 2      # the main conversation is untouched


def test_memory_solutions_knob_reaches_the_plan_prompt(tmp_path):
    s = _solver(tmp_path, n_deliv=1, mem_solutions=3)
    s.house = FakeHouse([("plan text\n```json\n{\"deliverables\": []}\n```", "stop")], s.cfg)
    s.plan("CTX", with_memory=True)
    assert "at most 3 solutions" in s.house.calls[0][2]
