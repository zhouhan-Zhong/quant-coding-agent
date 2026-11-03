"""Unit tests for agent 2.9.13: the template-debug route, typed time allocation and the /tmp guard of the vote."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))
sys.path.insert(0, str(HERE))

import agent  # noqa: E402
from test_v22_features import FakeHouse, _solver  # noqa: E402


def _unit_with(tmp_path, instruction, files):
    data = tmp_path / "d"
    data.mkdir(exist_ok=True)
    dfs = []
    for name, text in files.items():
        p = data / name
        p.write_text(text)
        dfs.append(agent.DataFile(p, name, p.stat().st_size))
    return agent.Unit(task_dir=tmp_path, out_dir=tmp_path / "out", unit_id="t1-x", instruction=instruction, data_root=data, data_files=dfs)


def test_template_route_file_detection(tmp_path):
    u = _unit_with(tmp_path, "A colleague wrote `template.py`. Debug and fix all computational bugs. Write `/app/output/a.json`.",
                   {"template.py": "print(1)\n", "helpers.py": "x = 1\n", "q.csv": "a\n1\n"})
    assert agent.template_route_file(u).path.name == "template.py"
    u2 = _unit_with(tmp_path, "Use the helper `html_to_text.py` to parse the filings and write `/app/output/a.json`.",
                    {"html_to_text.py": "print(1)\n"})
    assert agent.template_route_file(u2) is None            # a helper, not a debug task
    u3 = _unit_with(tmp_path, "The script is buggy; fix it.", {"solve.py": "print(1)\n"})
    assert agent.template_route_file(u3).path.name == "solve.py"   # the only .py when none is named


def test_typed_time_factor(tmp_path):
    many = " ".join(f"`/app/output/f{i}.json`" for i in range(7))
    assert agent.typed_time_factor(_unit_with(tmp_path, "Write " + many, {"q.csv": "a\n1\n"})) == (0.75, "7 deliverables")
    f, why = agent.typed_time_factor(_unit_with(tmp_path, "Write `/app/output/a.json`.", {"filing.html": "<html></html>"}))
    assert f == 0.75 and why == "document inputs"
    f, why = agent.typed_time_factor(_unit_with(tmp_path, "Write `/app/output/a.json` and `/app/output/b.csv`.", {"q.csv": "a\n1\n"}))
    assert f == 1.25 and why == "2 deliverables"


def test_template_start_then_forced_fix(tmp_path, capsys):
    s = _solver(tmp_path, n_deliv=1, template_route=True, patch_repair=True)
    tpl = "import json, os\njson.dump({'v': 1.0}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))\n"
    s.template_code, s.template_name = tpl.strip(), "template.py"
    edit = "<<<<<<< SEARCH\njson.dump({'v': 1.0}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))\n=======\njson.dump({'v': 2.0}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))\n>>>>>>> REPLACE"
    s.house = FakeHouse([(edit, "stop")], s.cfg)
    code = s._template_start("CONTEXT", have_plan=False)
    assert code == s.template_code and s.template_pending_fix and s.messages[-1]["role"] == "assistant"
    assert "SCRIPT TO DEBUG" in s._template_section() and s.template_code in s._template_section()
    val, res, fb = s.run_attempt(code, "a1")
    assert val.ok                                             # the template validates by construction ...
    fixed = s.repair(agent.TEMPLATE_FIX_FEEDBACK + "\n\n" + agent.summarize_outputs(s.out_dir, s.initial_files))
    assert "2.0" in fixed and s.house.calls[-1][0] == "patch1"   # ... and the fix came back as an edit of it
    val2, _, _ = s.run_attempt(fixed, "a2", prefer_new_on_tie=True)   # as the solve loop does after the forced fix
    assert val2.ok and json.loads((s.out_dir / "f0.json").read_text()) == {"v": 2.0}
    s.run_attempt(fixed, "a3")                                          # a plain tie keeps the best outputs unchanged
    assert json.loads((s.out_dir / "f0.json").read_text()) == {"v": 2.0}


def test_side_chat_edits_retry_then_fresh(tmp_path):
    s = _solver(tmp_path, n_deliv=1, template_route=True)
    s.template_code = "a = 1\nb = 2\nprint(a + b)"
    s.task_context = "CTX"
    fresh = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(16)) + "\n```"
    s.house = FakeHouse([("<<<<<<< SEARCH\nb = 2\n=======\nb = 3\n>>>>>>> REPLACE", "stop"),
                         ("no edits here", "stop"), ("<<<<<<< SEARCH\nb = 2\n=======\nb = 4\n>>>>>>> REPLACE", "stop"),
                         ("still nothing", "stop"), ("nope", "stop"), (fresh, "stop")], s.cfg)
    assert s._side_chat_edits("p", "vote1") == "a = 1\nb = 3\nprint(a + b)"
    assert s._side_chat_edits("p", "vote2") == "a = 1\nb = 4\nprint(a + b)"        # strict retry succeeded
    assert s.house.calls[-1][0] == "vote2-strict"
    assert s._side_chat_edits("p", "vote3").startswith("x0 = 0")                    # retry failed -> fresh compact script
    assert s.house.calls[-1][0] == "vote3-fresh"
    assert agent.TEMPLATE_LONG_LINES == 120 and "do NOT rewrite" in agent.TEMPLATE_CANDIDATE_LONG


def test_vote_skipped_when_outputs_too_large(tmp_path, capsys, monkeypatch):
    s = _solver(tmp_path, n_deliv=1, vote=True, vote_max=3)
    s.cfg.vote_max_bytes = 10
    s.house = FakeHouse([], s.cfg)
    (s.out_dir / "f0.json").write_text(json.dumps({"v": [1.0] * 50}))
    s.last_code = "print(1)"
    monkeypatch.setattr(s, "_vote_batch", lambda ks, prompt: (_ for _ in ()).throw(AssertionError("no candidates expected")))
    s.vote(True)
    assert "vote skipped" in capsys.readouterr().out


def test_config_parses_2913_flags(monkeypatch):
    monkeypatch.setenv("AGENT_TEMPLATE_ROUTE", "1")
    monkeypatch.setenv("AGENT_TIME_TYPED", "1")
    monkeypatch.setenv("AGENT_VOTE_MAX_MB", "4")
    c = agent.Config.from_env()
    assert c.template_route and c.time_typed and c.vote_max_bytes == 4 * 1024 * 1024
    assert agent.Config(endpoint="x").vote_max_bytes == 8 * 1024 * 1024 and not agent.Config(endpoint="x").template_route


def test_assemble_script_hoists_future_imports(tmp_path):
    pre = "import os as _os\n_os.environ.setdefault('X', '1')\n"
    code = '"""doc"""\nfrom __future__ import annotations\nimport math\nprint(math.pi)'
    out = agent.assemble_script(pre, code)
    assert out.startswith("from __future__ import annotations\n" + pre) and out.count("from __future__") == 1
    assert agent.syntax_check(out) is None and agent.syntax_check(pre + code) is not None   # the plain concatenation is invalid
    assert out.splitlines()[-1] == "print(math.pi)" and agent.assemble_script(pre, "x = 1") == pre + "x = 1\n"
    # the template route's real case: a shipped script that starts with a future import runs under the preamble
    s = _solver(tmp_path, n_deliv=1)
    tpl = "from __future__ import annotations\nimport json, os\njson.dump({'v': 1}, open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))"
    val, res, _ = s.run_attempt(tpl, "a1")
    assert res.returncode == 0 and val.ok
