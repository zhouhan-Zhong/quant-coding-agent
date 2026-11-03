"""Unit tests for the agent 2.2 features: data facts probe, two-part code generation, split repair, template kwargs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))

import agent  # noqa: E402


def _unit(tmp_path: Path, n_deliv: int = 6) -> agent.Unit:
    data = tmp_path / "input" / "environment" / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "prices.csv").write_text("date,close,volume\n2020-01-01,1.5,10\n2020-01-02,1.7,12\n")
    (data / "params.json").write_text(json.dumps({"rate": 0.02, "tenors": [1, 2], "name": "x"}))
    try:
        import pandas as pd

        pd.DataFrame([["Fund: SPY", None], ["Ticker", "Weight"], ["AAPL", 0.07]]).to_excel(data / "hold.xlsx", header=False, index=False)
    except Exception:
        pass
    files = [agent.DataFile(p, p.name, p.stat().st_size) for p in sorted(data.iterdir())]
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    deliv = " ".join(f"`f{i}.json`" for i in range(n_deliv))
    u = agent.Unit(task_dir=tmp_path / "input", out_dir=out, unit_id="t1-test", instruction="Write " + deliv + " to /app/output/.",
                   data_root=data, data_files=files)
    return u


def _solver(tmp_path: Path, n_deliv: int = 6, **flags) -> agent.Solver:
    cfg = agent.Config(endpoint="http://127.0.0.1:1", memory=False, selfcheck=False, do_plan=False, **flags)
    u = _unit(tmp_path, n_deliv)
    budget = agent.Budget(600, 700)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    s = agent.Solver(u, cfg, budget, u.out_dir, work)
    s.deliverables = [agent.Deliverable(f"f{i}.json", "json", [], [], "test") for i in range(n_deliv)]
    return s


class FakeHouse:
    """Scripted replies: each call pops the next (content, finish) pair; records the labels and messages."""

    def __init__(self, replies, cfg):
        self.replies = list(replies)
        self.calls = []
        self.requests_made = 0
        self.cfg = cfg
        self.latencies = []

    def chat(self, messages, *, thinking=None, max_tokens=None, label="call", temperature=None):
        self.calls.append((label, [m["role"] for m in messages], messages[-1]["content"], thinking))
        self.requests_made += 1
        return self.replies.pop(0)

    def requests_left(self):
        return 20 - self.requests_made

    def avg_latency(self):
        return 1.0

    @property
    def available(self):
        return True


def test_data_facts_lists_columns_keys_and_sheets(tmp_path):
    u = _unit(tmp_path)
    facts = agent.data_facts(u)
    assert "prices.csv" in facts and "close:float64" in facts and "2 rows" in facts
    assert "params.json" in facts and "rate:float" in facts and "tenors:list" in facts
    if any(df.path.suffix == ".xlsx" for df in u.data_files):
        assert "hold.xlsx" in facts and "raw rows (header=None)" in facts and "Fund: SPY" in facts and "Ticker | Weight" in facts
    assert len(facts) <= 3600


def test_data_err_regex():
    assert agent.DATA_ERR_RE.search("KeyError: 'closing_price'")
    assert agent.DATA_ERR_RE.search("ValueError: operands could not be broadcast together with shapes (1260,5) (1259,5)")
    assert agent.DATA_ERR_RE.search("ValueError: Invalid frequency: M")
    assert not agent.DATA_ERR_RE.search("ZeroDivisionError: float division by zero")


def test_combine_and_split_parts_roundtrip():
    code, marker = agent.combine_parts("import x\na = 1\n# === PART 2 ===\n", "b = a + 1\n")
    assert marker == 3 and code.splitlines()[2] == agent.SPLIT_MARKER
    p1, p2, m = agent.split_parts(code)
    assert p1 == "import x\na = 1" and p2 == "b = a + 1" and m == 3
    assert agent.split_parts("no marker here") is None


def test_error_line_takes_last_frame_in_solution():
    fb = ('File "/tmp/w/solution_a1.py", line 12, in <module>\n    main()\n  File "/tmp/w/solution_a1.py", line 40, in main\n'
          '    x = d["k"]\n  File "/usr/lib/python3.13/site-packages/pandas/core/frame.py", line 3648, in __getitem__')
    assert agent.error_line(fb) == 40
    assert agent.error_line("Deliverable validation problems:\n- missing deliverable: a.json") is None


def test_template_kwargs_low_effort():
    cfg = agent.Config(endpoint="x", low_effort=True)
    h = agent.House(cfg, agent.Budget(10, 20))
    assert h.template_kwargs(True) == {"enable_thinking": True, "low_effort": True}
    assert h.template_kwargs(False) == {"enable_thinking": False}
    assert agent.House(agent.Config(endpoint="x"), agent.Budget(10, 20)).template_kwargs(True) == {"enable_thinking": True}


def test_env_notes_only_when_enabled(tmp_path):
    assert agent.ENV_NOTES not in _solver(tmp_path).system_prompt
    assert agent.ENV_NOTES in _solver(tmp_path, envnotes=True).system_prompt
    assert "'ME'" in agent.ENV_NOTES and "np.trapezoid" in agent.ENV_NOTES


def test_two_part_first_code_combines_and_records_one_assistant_turn(tmp_path):
    s = _solver(tmp_path, n_deliv=6, split=True)
    s.house = FakeHouse([("```python\nimport json\nx = 1\n# === PART 2 ===\n```", "stop"),
                         ("```python\njson.dump({'x': x}, open('f0.json', 'w'))\n```", "stop")], s.cfg)
    code = s.first_code("CONTEXT", have_plan=False)
    assert code == "import json\nx = 1\n# === PART 2 ===\njson.dump({'x': x}, open('f0.json', 'w'))"
    assert s.split_line == 3
    assert [c[0] for c in s.house.calls] == ["code-p1", "code-p2"]
    assert agent.SPLIT_PART1 in s.house.calls[0][2] and s.house.calls[1][2] == agent.SPLIT_PART2
    # history: system, user(code request), assistant(combined) -- the part-2 exchange is not kept
    assert [m["role"] for m in s.messages] == ["system", "user", "assistant"] and "PART 2 ===" in s.messages[-1]["content"]


def test_single_script_when_few_deliverables(tmp_path):
    s = _solver(tmp_path, n_deliv=3, split=True)
    s.house = FakeHouse([("```python\nimport json\nprint(1)\n```", "stop")], s.cfg)
    s.first_code("CONTEXT", have_plan=False)
    assert [c[0] for c in s.house.calls] == ["code"] and s.split_line is None


def test_split_repair_replaces_only_the_failing_part(tmp_path):
    s = _solver(tmp_path, n_deliv=6, split=True)
    s.house = FakeHouse([("```python\nimport json\nx = 1\n# === PART 2 ===\n```", "stop"),
                         ("```python\ny = x / 0\njson.dump({'y': y}, open('f0.json', 'w'))\n```", "stop"),
                         ("```python\ny = x / 2\njson.dump({'y': y}, open('f0.json', 'w'))\n```", "stop")], s.cfg)
    code = s.first_code("CONTEXT", have_plan=False)
    s.last_code = code
    s.seen_code.add(agent.code_fingerprint(code))
    pre = agent.make_preamble(s.unit, s.out_dir, s.cfg.seed).count("\n")
    feedback = f'The script exited with code 1.\n\nSTDERR (tail):\nFile "/tmp/w/solution_a1.py", line {pre + 4}, in <module>\n    y = x / 0\nZeroDivisionError: division by zero'
    new = s.repair(feedback)
    assert s.house.calls[-1][0] == "repair0-p2"
    assert "PART 2" in s.house.calls[-1][2] and "The problem is in PART 2" in s.house.calls[-1][2]
    assert new == "import json\nx = 1\n# === PART 2 ===\ny = x / 2\njson.dump({'y': y}, open('f0.json', 'w'))"
    assert s.split_line == 3 and s.messages[-1]["role"] == "assistant" and "y = x / 2" in s.messages[-1]["content"]


def test_split_repair_part1_when_error_is_before_marker(tmp_path):
    s = _solver(tmp_path, n_deliv=6, split=True)
    s.house = FakeHouse([("```python\nimport json\nx = 1 / 0\n# === PART 2 ===\n```", "stop"),
                         ("```python\njson.dump({'x': x}, open('f0.json', 'w'))\n```", "stop"),
                         ("```python\nimport json\nx = 1 / 2\n```", "stop")], s.cfg)
    code = s.first_code("CONTEXT", have_plan=False)
    s.last_code = code
    pre = agent.make_preamble(s.unit, s.out_dir, s.cfg.seed).count("\n")
    new = s.repair(f'STDERR (tail):\nFile "/tmp/w/solution_a1.py", line {pre + 2}, in <module>\nZeroDivisionError')
    assert s.house.calls[-1][0] == "repair0-p1"
    assert new == "import json\nx = 1 / 2\n# === PART 2 ===\njson.dump({'x': x}, open('f0.json', 'w'))"


def test_split_repair_accepts_a_full_script_answer(tmp_path):
    s = _solver(tmp_path, n_deliv=6, split=True)
    s.house = FakeHouse([("```python\nimport json\nx = 1\n# === PART 2 ===\n```", "stop"),
                         ("```python\njson.dump({'x': x}, open('f0.json', 'w'))\n```", "stop"),
                         ("```python\nimport json\nx = 2\n# === PART 2 ===\njson.dump({'x': x}, open('f0.json', 'w'))\n```", "stop")], s.cfg)
    code = s.first_code("CONTEXT", have_plan=False)
    s.last_code = code
    new = s.repair("Deliverable validation problems:\n- missing deliverable: f1.json")
    assert new.startswith("import json\nx = 2") and s.split_line == 3


def test_probe_feedback_added_once_per_data_error(tmp_path):
    s = _solver(tmp_path, n_deliv=1, probe=True)
    s.deliverables = [agent.Deliverable("f0.json", "json", [], [], "test")]
    bad = "import pandas as pd\ndf = pd.read_csv(" + repr(str(s.unit.data_files[1].path)) + ")\nprint(df['closing_price'])\n"
    val, res, fb = s.run_attempt(bad, "a1")
    assert res is not None and res.returncode != 0
    assert "DATA FACTS" in fb and "close:float64" in fb and s.probes_sent == 1
    val, res, fb = s.run_attempt(bad + "\n", "a2")
    assert s.probes_sent == 2
    val, res, fb = s.run_attempt(bad + "\n\n", "a3")
    assert s.probes_sent == 2 and "DATA FACTS" not in fb


def test_probe_not_added_for_non_data_errors(tmp_path):
    s = _solver(tmp_path, n_deliv=1, probe=True)
    s.deliverables = [agent.Deliverable("f0.json", "json", [], [], "test")]
    val, res, fb = s.run_attempt("x = 1 / 0\n", "a1")
    assert "DATA FACTS" not in fb and s.probes_sent == 0


def test_thinking_fallback_after_repeated_failures(tmp_path):
    s = _solver(tmp_path, n_deliv=1, low_effort=True)
    cut = ("<think>\nlong reasoning that never ends", "length")
    good = ("```python\nimport json\nprint(1)\n```", "stop")
    s.house = FakeHouse([cut, good, cut, good], s.cfg)
    real = agent.House(s.cfg, s.budget)
    s.house.no_thinking = False
    s.house.template_kwargs = real.template_kwargs
    s.messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "task"}]
    s._ask_code("code")
    assert s.think_failures == 1 and not s.house.no_thinking
    s.messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "task"}]
    s._ask_code("code")
    assert s.think_failures == 2 and s.house.no_thinking
    # the retry after a failure is always sent with thinking off
    assert [c[3] for c in s.house.calls if c[0].endswith("-retry")] == [False, False]


def test_real_house_respects_no_thinking_and_scope():
    cfg = agent.Config(endpoint="x", low_effort=True)
    h = agent.House(cfg, agent.Budget(10, 20))
    assert h.template_kwargs(True)["low_effort"] is True
    cfg_plan = agent.Config(endpoint="x", low_effort=True, think_scope="plan")
    assert agent.Config.from_env().think_scope in ("all", "plan")
    # scope=plan: default calls do not think; the House switch overrides everything
    import inspect
    src = inspect.getsource(agent.House.chat)
    assert 'think_scope != "plan"' in src and "self.no_thinking" in src
    assert cfg_plan.think_scope == "plan"
