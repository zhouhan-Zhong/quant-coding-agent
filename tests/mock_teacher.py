#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A deterministic stand-in for devloop's teacher model, for tests (in-process via respond(), or as an
OpenAI-compatible server: MOCK_TEACHER_PORT=8767 python tests/mock_teacher.py).

It recognises the four devloop prompts by their first line ([devloop:attribute|lesson|fold|solve]):
  attribute -> JSON that keeps the rule-based label
  lesson    -> a lesson card with every section
  fold      -> updates the first CURRENT FILE (one general bullet under "Conventions and pitfalls") and, with
               MOCK_TEACHER_CREATE=1 (default), creates a small new playbook whose snippet runs
  solve     -> the canned solution from tests/solutions/<unit>.py when there is one
MOCK_TEACHER_BAD=1 makes the first fold answer carry a GUID, so that the revision path is exercised.
"""
from __future__ import annotations

import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SOL_DIR = Path(os.environ.get("MOCK_SOLUTIONS_DIR", Path(__file__).parent / "solutions"))
STATE = {"fold_calls": 0}

NEW_PLAYBOOK = """---
title: Stating conventions before computing
category: numerics
tags: conventions, annualisation, ddof, day count, reference implementation
summary: A checklist for the conventions that decide whether numbers match a reference implementation.
---
## When this applies
Any task whose outputs are compared with a reference implementation within a tolerance.

## Method and formulas
- Annualised volatility = std(daily returns, ddof=1) * sqrt(periods per year).
- Simple vs log returns: r = P_t / P_(t-1) - 1 or ln(P_t / P_(t-1)); keep one choice throughout.

## Conventions and pitfalls (what graders check)
- Default to ddof=1 for sample statistics unless the task specifies otherwise.
- State the periods-per-year factor once and reuse it.

## Tested snippet
```python
import numpy as np
r = np.array([0.01, -0.02, 0.015, 0.003])
vol = r.std(ddof=1) * np.sqrt(252)
print(round(float(vol), 6))
```
"""


def _user(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")


def respond(messages: list[dict]) -> str:
    system = str(messages[0].get("content", "")) if messages else ""
    user = _user(messages)
    last = str(messages[-1].get("content", "")) if messages else ""
    tag = re.match(r"\[devloop:(\w+)\]", system)
    kind = tag.group(1) if tag else ""
    if kind == "attribute":
        m = re.search(r"RULE-BASED LABEL: ([\w-]+)", user)
        return json.dumps({"category": m.group(1) if m else "conventions",
                           "root_cause": "mock analysis of the checker failure",
                           "general_lesson": "When a task compares numbers with a reference, state each convention first.",
                           "tags": ["conventions", "mock"], "confidence": 0.5})
    if kind == "lesson":
        unit = (re.search(r"UNIT: (t1-[\w-]+)", user) or re.search(r"(t1-[\w-]+)", user)).group(1)
        return (f"# {unit}\ncategory: numerics   tags: mock\n## what the checker really tests\n- every deliverable\n"
                "## method that passed\n- the reference method\n## why the agent failed\n- a convention differed\n"
                "## pitfalls & data quirks\n- none recorded\n## general lesson\n- state conventions before computing\n"
                "## reusable snippets\n```python\nprint('ok')\n```\n")
    if kind == "fold":
        STATE["fold_calls"] += 1
        units = re.findall(r"#### LESSON (t1-[\w-]+)", user) or ["t1-unknown"]
        if last.startswith("These files were rejected"):
            wanted = re.findall(r"^- (\S+):", last, re.M)
            blocks = [_fold_update(user, units, path=w) for w in wanted]
            return "\n\n".join(b for b in blocks if b)
        blocks = [_fold_update(user, units)]
        if os.environ.get("MOCK_TEACHER_BAD") == "1" and STATE["fold_calls"] == 1:
            blocks = [b.replace("explicitly", "explicitly 01234567-89ab-4cde-8f01-23456789abcd") for b in blocks if b]
        if os.environ.get("MOCK_TEACHER_CREATE", "1") == "1":
            blocks.append(f"### FILE: playbooks/devloop-mock-conventions.md (create) sources: {units[0]}\n````markdown\n"
                          + NEW_PLAYBOOK + "````")
        return "\n\n".join(b for b in blocks if b) or "NO CHANGES"
    if kind == "solve":
        unit = re.search(r"UNIT: (t1-[\w-]+)", user).group(1)
        p = SOL_DIR / f"{unit}.py"
        code = p.read_text() if p.is_file() else "import os\nprint('no canned solution')\n"
        return "```python\n" + code + "\n```"
    return "I do not recognise this request."


def _fold_update(user: str, units: list[str], path: str | None = None) -> str:
    for m in re.finditer(r"### CURRENT FILE: (\S+)\n````markdown\n(.*?)\n````", user, re.S):
        if path and m.group(1) != path:
            continue
        text = m.group(2)
        bullet = ("- Reference implementations state their conventions explicitly; default to the textbook "
                  "convention unless the task specifies otherwise.")
        new, n = re.subn(r"(^## Conventions and pitfalls[^\n]*\n)", r"\1" + bullet + "\n", text, count=1, flags=re.M)
        if not n:
            continue
        return f"### FILE: {m.group(1)} (update) sources: {', '.join(units)}\n````markdown\n{new.rstrip()}\n````"
    return ""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[teacher] " + (fmt % args) + "\n")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        content = respond(body.get("messages") or [])
        out = json.dumps({"id": "mock-teacher", "object": "chat.completion",
                          "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                       "finish_reason": "stop"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


if __name__ == "__main__":
    port = int(os.environ.get("MOCK_TEACHER_PORT", "8767"))
    sys.stderr.write(f"[teacher] serving on http://127.0.0.1:{port}\n")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
