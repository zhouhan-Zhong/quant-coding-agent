"""Unit tests for agent 2.5: the workspace loop (AGENT_LOOP) - action parsing, cut files, edits, runs, review."""
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


def _loop_solver(tmp_path, replies, n_deliv=1, **flags):
    s = _solver(tmp_path, n_deliv=n_deliv, loop=True, **flags)
    s.house = FakeHouse(replies, s.cfg)
    s.house.no_thinking = False
    return s


def _main_writing(names, extra=""):
    lines = ["import json, os", "out = os.environ['OUTPUT_DIR']"]
    lines += [f"json.dump({{'v': 1}}, open(os.path.join(out, '{n}'), 'w'))" for n in names]
    return "\n".join(lines) + "\n" + extra


def test_parse_actions_kinds_order_and_done():
    text = ("<think>plan it</think>Look first.\n" + F + "python run\nprint(1)\n" + F + "\n" +
            F + "python file=data.py\ndef load():\n    return 1\n" + F + "\n" +
            F + "edit file=main.py\n<<<<<<< SEARCH\na = 1\n=======\na = 2\n>>>>>>> REPLACE\n" + F + "\n" +
            F + "python append=model.py\nx = 3\n" + F + "\n" + F + "json\n{}\n" + F + "\nDONE\n")
    acts, done = agent.parse_actions(text)
    assert [(a.kind, a.name) for a in acts] == [("run", ""), ("write", "data.py"), ("edit", "main.py"), ("append", "model.py")]
    assert acts[1].body == "def load():\n    return 1\n" and all(a.complete for a in acts) and done
    acts, done = agent.parse_actions("text\n" + F + "python file=../evil.sh\nrm -rf /\n" + F + "\nnot DONE yet")
    assert acts[0].kind == "invalid" and not done


def test_parse_actions_marks_a_cut_block():
    acts, _ = agent.parse_actions(F + "python file=main.py\nimport json\nx = compute(1,\n")
    assert len(acts) == 1 and acts[0].kind == "write" and not acts[0].complete


def test_rollback_and_stitch():
    cut = "import json\nx = 1\ny = f(x,\n   2"
    assert agent.rollback_to_statement(cut) == "import json\nx = 1\n"
    assert agent.stitch_code("import json\nx = 1\n", "x = 1\ny = 2\n") == "import json\nx = 1\ny = 2\n"


def test_loop_writes_runs_and_finishes_after_review(tmp_path):
    main = _main_writing(["f0.json"])
    replies = [
        (F + "python run\nimport pandas as pd\nprint(pd.read_csv('prices.csv').shape)\n" + F + "\n" +
         F + "python file=main.py\n" + main + F, "stop"),
        ("All good.\nDONE", "stop"),
    ]
    s = _loop_solver(tmp_path, replies)
    s.messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "task"}]
    s.agent_loop("task", have_plan=False)
    assert json.loads((s.out_dir / "f0.json").read_text()) == {"v": 1}
    obs = s.house.calls[1][2]
    assert "RUN 1 (exit 0" in obs and "(2, 3)" in obs        # the inspection ran with the real data path
    assert "RUN main.py: OK" in obs and agent.LOOP_REVIEW in obs
    assert len(s.house.calls) == 2 and s.best_ok


def test_loop_cut_file_is_kept_and_continued_with_append(tmp_path):
    full = _main_writing(["f0.json"])
    head = "\n".join(full.splitlines()[:2])
    replies = [
        (F + "python file=main.py\n" + head + "\njson.dump({'v': 1}, open(os.path.join(out, 'f0.js", "length"),
        (F + "python append=main.py\n" + full.splitlines()[2] + "\n" + F, "stop"),
        ("DONE", "stop"),
    ]
    s = _loop_solver(tmp_path, replies)
    s.agent_loop("task", have_plan=False)
    obs1 = s.house.calls[1][2]
    assert "cut off by the token limit inside main.py" in obs1 and "append=main.py" in obs1
    assert s.cut_files == 1 and (s.out_dir / "f0.json").exists()


def test_loop_edit_fixes_an_error_and_failed_edit_shows_the_file(tmp_path):
    bad = _main_writing(["f0.json"]).replace("{'v': 1}", "{'v': undefined_name}")
    replies = [
        (F + "python file=main.py\n" + bad + F, "stop"),
        (F + "edit file=main.py\n<<<<<<< SEARCH\nnot in the file\n=======\nx\n>>>>>>> REPLACE\n" + F, "stop"),
        (F + "edit file=main.py\n<<<<<<< SEARCH\n{'v': undefined_name}\n=======\n{'v': 1}\n>>>>>>> REPLACE\n" + F, "stop"),
        ("DONE", "stop"),
    ]
    s = _loop_solver(tmp_path, replies)
    s.agent_loop("task", have_plan=False)
    assert "NameError" in s.house.calls[1][2]
    assert "edit main.py FAILED" in s.house.calls[2][2] and "undefined_name" in s.house.calls[2][2]
    assert "RUN main.py: OK" in s.house.calls[3][2] and s.edit_failures == 1


def test_loop_modules_import_and_bare_block_becomes_main(tmp_path):
    lib = "def payload():\n    return {'v': 2}\n"
    main = "import json, os\nfrom lib import payload\n" + "\n".join(["# pad"] * 20) + "\njson.dump(payload(), open(os.path.join(os.environ['OUTPUT_DIR'], 'f0.json'), 'w'))\n"
    replies = [
        (F + "python file=lib.py\n" + lib + F, "stop"),
        ("Here is the script:\n" + F + "python\n" + main + F, "stop"),
        ("DONE", "stop"),
    ]
    s = _loop_solver(tmp_path, replies)
    s.agent_loop("task", have_plan=False)
    assert "main.py does not exist yet" in s.house.calls[1][2]
    assert json.loads((s.out_dir / "f0.json").read_text()) == {"v": 2}


def test_loop_done_without_valid_outputs_is_ignored_and_budget_stops(tmp_path):
    replies = [("DONE", "stop")] * 30
    s = _loop_solver(tmp_path, replies)
    s.agent_loop("task", have_plan=False)
    assert "DONE ignored" in s.house.calls[1][2]
    assert len(s.house.calls) <= 20


def test_loop_compaction_keeps_head_and_workspace(tmp_path):
    s = _loop_solver(tmp_path, [])
    s.messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "task"}]
    s.head_len = 2
    s.files = {"main.py": "print(1)\n"}
    for i in range(10):
        s.messages += [{"role": "assistant", "content": "a" * 10000}, {"role": "user", "content": f"obs{i}"}]
    assert s._loop_compact()
    assert [m["role"] for m in s.messages] == ["system", "user", "assistant", "user"]
    assert s.messages[-1]["content"].startswith("obs9") and "### main.py" in s.messages[-1]["content"]


def test_loop_reasoning_cut_counts_as_thinking_failure(tmp_path):
    replies = [("<think>" + "long " * 50, "length")] * 2 + [(F + "python file=main.py\n" + _main_writing(["f0.json"]) + F, "stop"), ("DONE", "stop")]
    s = _loop_solver(tmp_path, replies, low_effort=True)
    s.house.no_thinking = False
    s.agent_loop("task", have_plan=False)
    assert s.think_failures >= 2 and s.house.no_thinking and (s.out_dir / "f0.json").exists()


def test_loop_identical_rewrite_is_not_rerun(tmp_path):
    main = _main_writing(["f0.json"])
    replies = [(F + "python file=main.py\n" + main + F, "stop"), (F + "python file=main.py\n" + main + F, "stop")]
    s = _loop_solver(tmp_path, replies)
    s.agent_loop("task", have_plan=False)
    assert s.attempt == 1 and len(s.house.calls) == 2


def test_wrapper_frames_are_dropped_and_signature_uses_the_workspace_line():
    tb = ('Traceback (most recent call last):\n  File "/tmp/w/solution_w3.py", line 14, in <module>\n'
          "    _runpy.run_path('/tmp/w/ws/main.py', run_name='__main__')\n"
          '  File "<frozen runpy>", line 287, in run_path\n  File "<frozen runpy>", line 98, in _run_module_code\n'
          '  File "/tmp/w/ws/main.py", line 7, in <module>\n    x = df["close"]\n        ~~^^^^^^^^^\nKeyError: \'close\'')
    clean = agent.drop_wrapper_frames(tb)
    assert "runpy" not in clean and "solution_w3" not in clean and 'ws/main.py", line 7' in clean and "KeyError" in clean
    assert agent.error_signature(tb) == 'KeyError@x = df["close"]'


def test_bare_block_named_by_comment_or_preceding_text():
    acts, _ = agent.parse_actions("Module **data.py**:\n" + F + "python\ndef f():\n    return 1\n" + F + "\n" +
                                  F + "python\n# file: main.py\nimport data\n" + F)
    assert [(a.kind, a.name) for a in acts] == [("write", "data.py"), ("write", "main.py")]


def test_shell_block_runs_and_heredoc_files_are_picked_up(tmp_path):
    main = _main_writing(["f0.json"])
    replies = [(F + "bash\nls " + "\ncat > main.py <<'EOF'\n" + main + "EOF\n" + F, "stop"), ("DONE", "stop")]
    s = _loop_solver(tmp_path, replies)
    s.agent_loop("task", have_plan=False)
    obs = s.house.calls[1][2]
    assert "SHELL 1 (exit 0" in obs and "created/changed by the shell: main.py" in obs and "RUN main.py: OK" in obs
