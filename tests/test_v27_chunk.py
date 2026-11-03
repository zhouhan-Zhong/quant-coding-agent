"""Unit tests for agent 2.7: multi-part first script, multi-continuation, part repair, review edits (AGENT_CHUNK)
and the deep-thinking design call (AGENT_DEEP_THINK)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))
sys.path.insert(0, str(HERE))

import agent  # noqa: E402
from test_v22_features import FakeHouse, _solver  # noqa: E402

F = "```"


class KwHouse(FakeHouse):
    def chat(self, messages, **kw):
        self.kw = getattr(self, "kw", []) + [kw]
        return super().chat(messages, **{k: v for k, v in kw.items() if k != "low_effort"})


def _s(tmp_path, replies, n_deliv=1, **flags):
    s = _solver(tmp_path, n_deliv=n_deliv, **flags)
    s.house = KwHouse(replies, s.cfg)
    s.house.no_thinking = False
    return s


def test_chunk_count_uses_deliverables_estimate_and_plan_groups():
    assert agent.chunk_count(None, 1) == (1, [])
    assert agent.chunk_count({"est_lines": 250}, 2) == (2, [])
    assert agent.chunk_count(None, 5)[0] == 2 and agent.chunk_count(None, 7)[0] == 3
    k, groups = agent.chunk_count({"parts": [["load"], ["calc"], ["write", "main"]]}, 2)
    assert k == 3 and groups[2] == ["write", "main"]
    assert agent.chunk_count({"parts": [["a"], ["b"]]}, 7) == (3, [])   # groups that do not match the count are dropped


def test_three_continuations_in_chunk_mode(tmp_path):
    s = _s(tmp_path, [(F + "python\na = 1\nb = 2\n", "length"), (F + "python\nb = 2\nc = 3\n" + F, "stop")], chunk=True)
    s.messages = [{"role": "user", "content": "x"}]
    code = s._complete_code(F + "python\nimport os\na = 1\n", "length")
    assert code.splitlines() == ["import os", "a = 1", "b = 2", "c = 3"]
    assert [c[0] for c in s.house.calls] == ["continue", "continue"] and s.house.kw[0]["thinking"] is False
    assert not s.last_truncated


def test_single_continuation_without_chunk(tmp_path):
    s = _s(tmp_path, [(F + "python\nb = 2\n", "length")])
    s.messages = [{"role": "user", "content": "x"}]
    s._complete_code(F + "python\nimport os\na = 1\n", "length")
    assert len(s.house.calls) == 1


def test_first_script_in_two_parts_is_joined_and_history_collapsed(tmp_path):
    p1 = F + "python\nimport json, os\nOUT = os.environ['OUTPUT_DIR']\ndef payload():\n    return {'v': 1}\n" + F
    p2 = F + "python\ndef main():\n    json.dump(payload(), open(os.path.join(OUT, 'f0.json'), 'w'))\nif __name__ == '__main__':\n    main()\n" + F
    s = _s(tmp_path, [(p1, "stop"), (p2, "stop")], chunk=True)
    s.n_parts = 2
    s.messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "task"}, {"role": "assistant", "content": "plan"}]
    code = s.first_code("task", have_plan=True)
    assert code.startswith("# --- part 1 of 2 ---") and "# --- part 2 of 2 ---" in code and agent.syntax_check(code) is None
    assert "PART 1 of 2" in s.house.calls[0][2] and "PART 2 of 2" in s.house.calls[1][2] and "LAST part" in s.house.calls[1][2]
    assert [m["role"] for m in s.messages] == ["system", "user", "assistant", "user", "assistant"] and code in s.messages[-1]["content"]
    val, _, _ = s.run_attempt(code, "a1")
    assert val.ok


def test_deep_think_notes_reach_the_code_request(tmp_path):
    deep = ("the rate is continuously compounded, so use exp(-r t)</think>DESIGN:\n1. price(r, t) -> float", "stop")
    code = (F + "python\nimport os\nprint(1)\n" + F, "stop")
    s = _s(tmp_path, [deep, code], deep_think=True, low_effort=True)
    s.messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "task"}, {"role": "assistant", "content": "plan"}]
    s.first_code("task", have_plan=True)
    assert s.house.calls[0][0] == "deep-think" and s.house.kw[0]["low_effort"] is False and s.house.kw[0]["thinking"] is True
    req = s.house.calls[1][2]
    assert "DESIGN NOTES" in req and "price(r, t)" in req and "continuously compounded" in req


def test_deep_think_cut_reasoning_becomes_notes(tmp_path):
    s = _s(tmp_path, [("step " * 20 + "key insight: ddof=1", "length")], deep_think=True)
    s.messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "task"}]
    notes = s._deep_think("task", have_plan=False)
    assert "cut off" in notes and "ddof=1" in notes


def test_part_repair_replaces_only_the_failing_part(tmp_path):
    code = "# --- part 1 of 2 ---\nimport os\nx = 1\n# --- part 2 of 2 ---\ny = undefined_name\nprint(y)"
    s = _s(tmp_path, [(F + "python\ny = x + 1\nprint(y)\n" + F, "stop")], chunk=True)
    s.last_code = code
    s.messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": code}]
    pre = agent.make_preamble(s.unit, s.out_dir, s.cfg.seed).count("\n")
    fb = f'Traceback:\n  File "/w/solution_a1.py", line {pre + 5}, in <module>\nNameError: undefined_name'
    new = s._part_repair(fb)
    assert new.splitlines() == ["# --- part 1 of 2 ---", "import os", "x = 1", "# --- part 2 of 2 ---", "y = x + 1", "print(y)"]
    assert "PART 2" in s.house.calls[0][2]


def test_review_applies_edit_blocks(tmp_path):
    s = _s(tmp_path, [("Units are percent.\n<<<<<<< SEARCH\nv = 0.05\n=======\nv = 5.0\n>>>>>>> REPLACE", "stop")], chunk=True)
    s.last_code = "import os\nv = 0.05\n"
    s.messages = [{"role": "user", "content": "x"}]
    assert s.review() == "import os\nv = 5.0\n"


def test_deep_think_tagged_form_still_parses(tmp_path):
    s = _s(tmp_path, [("<think>why</think>DESIGN: f()", "stop")], deep_think=True)
    s.messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "task"}]
    notes = s._deep_think("task", have_plan=False)
    assert notes.startswith("DESIGN: f()") and "why" in notes
