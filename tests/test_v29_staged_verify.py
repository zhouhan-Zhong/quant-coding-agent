"""Unit tests for agent 2.9: staged first script (AGENT_STAGED: inputs.py / compute.py / main.py with execution in
between, flattened into one script), independent verification of key quantities (AGENT_VERIFY) and the budget report."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))
sys.path.insert(0, str(HERE))

import agent  # noqa: E402
from test_v22_features import FakeHouse, _solver  # noqa: E402

F = "```"


def _py(code: str):
    return (F + "python\n" + code + "\n" + F, "stop")


def _writer(v):
    return "import json, os\njson.dump({'v': " + str(v) + "}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))"


def _value(s):
    return json.loads((s.out_dir / "f0.json").read_text())["v"]


# ---- flattening and helpers -----------------------------------------------------------------------------------
def test_flatten_modules_removes_imports_and_main_guards(tmp_path):
    inputs = "import json\n\ndef load_inputs():\n    return {'a': 1}\n\nif __name__ == '__main__':\n    d = load_inputs()\n    print(d)\n"
    compute = ("from inputs import load_inputs\nimport numpy as np\n\ndef calc(d):\n    return d['a'] * 2\n\n"
               "if __name__ == \"__main__\":\n    print(calc(load_inputs()))\n")
    main = ("import inputs\nfrom compute import calc\nimport json, os\n\ndef main():\n    d = inputs.load_inputs()\n"
            "    json.dump({'v': calc(d)}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))\n\nif __name__ == '__main__':\n    main()\n")
    flat = agent.flatten_modules([("inputs", inputs), ("compute", compute), ("main", main)])
    assert agent.syntax_check(flat) is None
    assert "from inputs import" not in flat and "import inputs" not in flat and "from compute import" not in flat
    assert flat.count("__main__") == 1 and "print(d)" not in flat and "d = load_inputs()" in flat and "import numpy as np" in flat
    script = tmp_path / "flat.py"
    script.write_text(flat)
    env = dict(os.environ, OUTPUT_DIR=str(tmp_path))
    subprocess.run([sys.executable, str(script)], check=True, env=env, cwd=str(tmp_path), timeout=60)
    assert json.loads((tmp_path / "f0.json").read_text()) == {"v": 2}


def test_staged_applicable_rule():
    assert not agent.staged_applicable(1, 1, None)
    assert agent.staged_applicable(3, 0, None) and agent.staged_applicable(1, 2, None)
    assert agent.staged_applicable(1, 1, {"est_lines": 200}) and not agent.staged_applicable(1, 1, {"est_lines": "abc"})


def test_parse_check_lines():
    vals = agent.parse_check_lines("CHECK q1 1.5\nCHECK q2 NA\nnoise\nCHECK q3 1,234.5\nCHECK q4 inf\nCHECKS DONE")
    assert vals == {"q1": 1.5, "q2": None, "q3": 1234.5, "q4": None}


def test_read_quantity_and_auto_quantities(tmp_path):
    (tmp_path / "f.json").write_text(json.dumps({"a": {"b": [1.5, 2.5]}, "c": 3.25, "n": 10, "flag": True}))
    (tmp_path / "t.csv").write_text("x,y\n1,1\n2,1\n3,1\n")
    rq = lambda **q: agent.read_quantity(tmp_path, q)  # noqa: E731
    assert rq(file="f.json", path="a.b[1]") == 2.5 and rq(file="f.json", path="c") == 3.25
    assert rq(file="f.json", path="a.b", agg="mean") == 2.0 and rq(file="f.json", path="flag") is None
    assert rq(file="t.csv", path="x", agg="mean") == 2.0 and rq(file="t.csv", path="x", agg="last") == 3.0
    assert rq(file="t.csv", path="zz", agg="mean") is None and rq(file="missing.json", path="c") is None
    deliv = [agent.Deliverable("f.json", "json", [], [], "t"), agent.Deliverable("t.csv", "csv", [], [], "t")]
    qs = agent.auto_quantities(tmp_path, deliv)
    paths = [(q["file"], q["path"]) for q in qs]
    assert paths[0] == ("f.json", "c") and ("t.csv", "x") in paths and ("t.csv", "y") not in paths and ("f.json", "flag") not in paths


# ---- staged first script --------------------------------------------------------------------------------------
def _staged_replies(s):
    px = str(s.unit.data_files[0].path) if s.unit.data_files[0].path.name == "hold.xlsx" else None
    prices = next(df.path for df in s.unit.data_files if df.path.name == "prices.csv")
    inputs = ("import pandas as pd\n\ndef load_inputs():\n    return {'px': pd.read_csv(r'" + str(prices) + "')}\n\n"
              "if __name__ == '__main__':\n    d = load_inputs()\n    print('px', d['px'].shape, list(d['px'].columns))\n")
    compute = ("from inputs import load_inputs\n\ndef total(d):\n    return float(d['px']['close'].sum())\n\n"
               "if __name__ == '__main__':\n    print('total =', total(load_inputs()))\n")
    main = ("from inputs import load_inputs\nfrom compute import total\nimport json, os\n\ndef main():\n"
            "    json.dump({'v': total(load_inputs())}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))\n\n"
            "if __name__ == '__main__':\n    main()\n")
    return _py(inputs), _py(compute), _py(main)


def test_staged_first_code_runs_modules_and_flattens(tmp_path):
    s = _solver(tmp_path, n_deliv=1, staged=True)
    s.house = FakeHouse(list(_staged_replies(s)), s.cfg)
    code = s.first_code("TASK CONTEXT", have_plan=False, reference="")
    labels = [c[0] for c in s.house.calls]
    assert labels == ["stage-inputs", "stage-compute", "stage-main"]
    assert "(2, 3)" in s.house.calls[1][2] and "total = 3.2" in s.house.calls[2][2]   # observed output fed forward
    assert "from inputs import" not in code and code.count("__main__") == 1
    assert len(s.messages) == 3 and s.head_len == 2 and "OBSERVED WHEN RUNNING THE MODULES" in s.messages[1]["content"]
    assert s.messages[1]["content"].startswith("TASK CONTEXT")
    val, _, _ = s.run_attempt(code, "a1")
    assert val.ok and abs(_value(s) - 3.2) < 1e-9


def test_staged_falls_back_to_a_single_script_when_a_stage_keeps_failing(tmp_path):
    s = _solver(tmp_path, n_deliv=1, staged=True)
    bad = _py("raise ValueError('boom')")
    s.house = FakeHouse([bad, bad, _py(_writer(1))], s.cfg)
    code = s.first_code("TASK CONTEXT", have_plan=False, reference="")
    labels = [c[0] for c in s.house.calls]
    assert labels == ["stage-inputs", "stage-inputs-fix", "code"]
    assert "boom" in s.house.calls[1][2]
    assert "json.dump" in code and len(s.messages) == 3 and s.head_len == 2
    assert "STAGE" not in s.messages[1]["content"] and "inputs.py" not in s.messages[1]["content"]


def test_staged_skipped_for_small_tasks(tmp_path):
    s = _solver(tmp_path, n_deliv=1, staged=True)
    s.unit.data_files = s.unit.data_files[:1]
    s.house = FakeHouse([_py(_writer(1))], s.cfg)
    s.first_code("TASK CONTEXT", have_plan=False, reference="")
    assert [c[0] for c in s.house.calls] == ["code"]


# ---- independent verification -----------------------------------------------------------------------------------
def _verify_solver(tmp_path, replies, v=2.5):
    s = _solver(tmp_path, n_deliv=1, verify=True)
    s.house = FakeHouse(list(replies), s.cfg)
    s.task_context = "task"
    s.plan_json = {"key_quantities": [{"file": "f0.json", "path": "v", "agg": "value", "meaning": "the value v"}]}
    val, _, _ = s.run_attempt(agent.extract_code(_py(_writer(v))[0]), "a1")
    assert val.ok
    return s


def _check(v):
    return _py("print('CHECK q1 " + str(v) + "')\nprint('CHECKS DONE')")


def test_verify_agreeing_check_costs_one_request(tmp_path):
    s = _verify_solver(tmp_path, [_check(2.5)])
    s.verify()
    assert [c[0] for c in s.house.calls] == ["verify"] and _value(s) == 2.5
    assert "CHECK <name> <number>" in s.house.calls[0][2] and "q1: the value v" in s.house.calls[0][2]
    assert "2.5" not in s.house.calls[0][2].split("q1: the value v")[1][:80]   # the check never sees our value


def test_verify_disagreement_fixed_by_judge_edits(tmp_path):
    judge = ("The task defines v as 5.0.\n<<<<<<< SEARCH\njson.dump({'v': 2.5}\n=======\njson.dump({'v': 5.0}\n>>>>>>> REPLACE", "stop")
    s = _verify_solver(tmp_path, [_check(5.0), judge])
    s.verify()
    assert [c[0] for c in s.house.calls] == ["verify", "verify-judge1"]
    assert "yours=2.5 independent=5" in s.house.calls[1][2] and _value(s) == 5.0 and "5.0" in s.last_code


def test_verify_judge_keeps_the_solution(tmp_path):
    s = _verify_solver(tmp_path, [_check(5.0), ("VERDICT: KEEP (the check ignored the task's scaling)", "stop")])
    s.verify()
    assert len(s.house.calls) == 2 and _value(s) == 2.5


def test_verify_restores_outputs_when_the_fix_does_not_help(tmp_path):
    judge = ("<<<<<<< SEARCH\njson.dump({'v': 2.5}\n=======\njson.dump({'v': 2.6}\n>>>>>>> REPLACE", "stop")
    s = _verify_solver(tmp_path, [_check(5.0), judge, ("VERDICT: KEEP", "stop")])
    prev = s.last_code
    s.verify()
    assert _value(s) == 2.5 and s.last_code == prev and len(s.house.calls) == 2


def test_verify_repairs_a_broken_check_script_once(tmp_path):
    s = _verify_solver(tmp_path, [_py("raise RuntimeError('nope')"), _check(2.5)])
    s.verify()
    assert [c[0] for c in s.house.calls] == ["verify", "verify-fix"] and "nope" in s.house.calls[1][2] and _value(s) == 2.5


def test_verify_check_script_cannot_damage_outputs(tmp_path):
    s = _verify_solver(tmp_path, [_py(_writer(9.0) + "\nprint('CHECK q1 2.5')\nprint('CHECKS DONE')")])
    s.verify()
    assert _value(s) == 2.5


def test_verify_without_plan_uses_automatic_quantities(tmp_path):
    s = _verify_solver(tmp_path, [_check(2.5)])
    s.plan_json = None
    qs = s._key_quantities()
    assert qs and qs[0]["path"] == "v" and qs[0]["name"] == "q1"


# ---- budget report and exec timeout cap ---------------------------------------------------------------------------
def test_budget_report_groups_labels(tmp_path):
    s = _solver(tmp_path, n_deliv=1)
    s.house = FakeHouse([], s.cfg)
    s.house.call_log = [("plan", 1.0), ("stage-inputs", 2.0), ("stage-inputs-fix", 1.0), ("patch3", 1.0), ("verify-judge1", 2.0), ("vote2-fix", 1.0), ("code-compact", 3.0)]
    rep = s.budget_report()
    for part in ("plan 1/1s", "stage-inputs 2/3s", "patch 1/1s", "verify 1/2s", "vote 1/1s", "code 1/3s", "soft budget 600s"):
        assert part in rep, (part, rep)


def test_exec_timeout_is_capped_under_a_relaxed_budget(tmp_path):
    s = _solver(tmp_path, n_deliv=1)
    s.budget = agent.Budget(3000, 3100)
    assert s._exec_timeout() <= 450.0
    s.budget = agent.Budget(600, 700)
    assert s._exec_timeout() >= 150.0


# ---- 2.9.1: tolerant CHECK-name matching, truncated judge replies ----------------------------------------------------
def test_match_check_values_by_name_leaf_and_position(tmp_path):
    s = _solver(tmp_path, n_deliv=1)
    qs = [{"name": "q1", "path": "AAPL.total_return"}, {"name": "q2", "path": "sharpe"}]
    assert s._match_check_values(qs, {"q1": 1.0, "Q2": 2.0}) == {"q1": 1.0, "q2": 2.0}
    assert s._match_check_values(qs, {"total_return": 1.0, "sharpe": 2.0}) == {"q1": 1.0, "q2": 2.0}
    assert s._match_check_values(qs, {"a": 1.0, "b": 2.0}) == {"q1": 1.0, "q2": 2.0}      # by position
    assert s._match_check_values(qs, {"a": 1.0}) == {}


def test_verify_applies_complete_edit_blocks_from_a_cut_judge_reply(tmp_path):
    judge = ("<<<<<<< SEARCH\njson.dump({'v': 2.5}\n=======\njson.dump({'v': 5.0}\n>>>>>>> REPLACE\nAnd further explanation that was cut", "length")
    s = _verify_solver(tmp_path, [_check(5.0), judge])
    s.verify()
    assert _value(s) == 5.0


def test_verify_positional_match_when_names_differ(tmp_path):
    s = _verify_solver(tmp_path, [_py("print('CHECK total_value 2.5')\nprint('CHECKS DONE')")])
    s.verify()
    assert len(s.house.calls) == 1   # matched by position -> agreement -> no judge
