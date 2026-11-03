"""Unit tests for small agent.py helpers (run: python -m pytest tests/ -q)."""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))

import agent  # noqa: E402

FINAL = "```python\nimport json\nprint('final answer')\n```"
SKETCH = "```python\nimport os\n# a much longer sketch written while reasoning\nx = 1\ny = 2\nz = 3\nprint(x + y + z)\n```"


def test_strip_reasoning_complete_block():
    text = f"<think>let me sketch\n{SKETCH}\n</think>\nHere is the script:\n{FINAL}"
    assert agent.extract_code(text) == "import json\nprint('final answer')"


def test_strip_reasoning_closing_tag_only():
    # Some servers emit the reasoning inline without the opening tag; the sketch is longer than the answer.
    text = f"Okay, the task wants...\n{SKETCH}\nso the final plan is clear.\n</think>\n{FINAL}"
    assert agent.extract_code(text) == "import json\nprint('final answer')"


def test_strip_reasoning_unclosed_block_is_dropped():
    # Reasoning cut off by the token limit: no final answer exists, so nothing is extracted from it.
    text = f"<think>first idea\n{SKETCH}\nstill thinking"
    assert agent.extract_code(text) == ""


def test_strip_reasoning_leaves_plain_answers_alone():
    assert agent.extract_code(FINAL) == "import json\nprint('final answer')"
    assert agent.strip_reasoning("no tags here") == "no tags here"


def test_plan_json_after_inline_reasoning():
    text = 'thinking about {"deliverables": ["wrong.json"]}\n</think>\n```json\n{"deliverables": ["results.json"]}\n```'
    assert agent.extract_json_block(text) == {"deliverables": ["results.json"]}


def test_make_outputs_readable(tmp_path):
    out = tmp_path / "out"
    (out / "sub").mkdir(parents=True)
    f1 = out / "results.json"
    f2 = out / "sub" / "table.csv"
    f1.write_text("{}")
    f2.write_text("a\n1\n")
    os.chmod(f1, 0o600)
    os.chmod(f2, 0o600)
    os.chmod(out / "sub", 0o700)
    agent.make_outputs_readable(out, set())
    for p in (f1, f2):
        assert stat.S_IMODE(p.stat().st_mode) & 0o444 == 0o444, oct(p.stat().st_mode)
    assert stat.S_IMODE((out / "sub").stat().st_mode) & 0o555 == 0o555
    # files that existed before the run are not touched
    pre = out / "preexisting.txt"
    pre.write_text("x")
    os.chmod(pre, 0o600)
    agent.make_outputs_readable(out, {"preexisting.txt"})
    assert stat.S_IMODE(pre.stat().st_mode) == 0o600
