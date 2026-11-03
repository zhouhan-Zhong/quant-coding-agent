#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A tiny OpenAI-compatible mock of the Agenthon House route, for plumbing tests only.

It mimics the route's contract:
  * only POST /v1/chat/completions is admitted (anything else -> 403)
  * a request without `Authorization: Bearer ...` -> 401
  * n > 1 -> 400 ; max_tokens > 4000 is clamped to 4000
  * at most 25 admitted requests per "unit" (per server lifetime here)

Canned answers come from a solutions directory: `<unit-id>.py` files whose first lines may carry
`# DELIVERABLES: a.json, b.csv`.  The unit is recognised by its id appearing in the prompt text
(pass --task-dir units/<id> when testing) or by MOCK_UNIT.

Modes (MOCK_MODE): "ok" (default), "buggy" (first script raises, repair returns the good one),
  "specfail" (first script breaks JSON types; the agent's spec self-check must trigger a repair),
  "memnone" (the plan explicitly selects no reference material),
"truncate" (first script is cut, continuation completes it), "nocode" (first answer has no code).
"""
from __future__ import annotations

import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SOL_DIR = Path(os.environ.get("MOCK_SOLUTIONS_DIR", Path(__file__).parent / "solutions"))
MODE = os.environ.get("MOCK_MODE", "ok")
STATE = {"requests": 0, "code_calls": 0, "cont_calls": 0, "ref_seen": False}


def _unit_titles() -> dict[str, str]:
    """With MOCK_KIT=<kit dir>, recognise a unit by the first real line of its instruction.md, for runs where
    the prompt does not contain the unit id (e.g. eval/local_eval.sh mounts the unit at /input)."""
    kit = os.environ.get("MOCK_KIT")
    titles: dict[str, str] = {}
    if not kit:
        return titles
    guid = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
    for p in SOL_DIR.glob("*.py"):
        ins = Path(kit, "units", p.stem, "instruction.md")
        if not ins.is_file():
            continue
        for line in ins.read_text(errors="replace").splitlines():
            line = line.strip()
            low = line.lower()
            if len(line) < 15 or "benchmark data" in low or "canary" in low or line.startswith("<!--") or guid.search(line):
                continue
            titles[p.stem] = line
            break
    return titles


TITLES = _unit_titles()


def find_solution(prompt: str) -> tuple[str, str] | None:
    forced = os.environ.get("MOCK_UNIT")
    if forced:
        p = SOL_DIR / f"{forced}.py"
        if p.is_file():
            return forced, p.read_text()
    for p in sorted(SOL_DIR.glob("*.py")):
        if p.stem in prompt:
            return p.stem, p.read_text()
    for stem, title in TITLES.items():
        if title in prompt:
            return stem, (SOL_DIR / f"{stem}.py").read_text()
    return None


def deliverables_block(code: str) -> str:
    m = re.search(r"#\s*DELIVERABLES:\s*(.+)", code, re.I)
    items = []
    if m:
        for name in [x.strip() for x in m.group(1).split(",") if x.strip()]:
            fmt = name.rsplit(".", 1)[-1]
            items.append({"file": name, "format": fmt})
    return json.dumps({"deliverables": items})


def plan_answer(code: str, with_memory: bool) -> str:
    block = json.loads(deliverables_block(code))
    if with_memory:
        # v2: the plan picks reference material from the REFERENCE LIBRARY list (labels S1.., P1..)
        block["memory"] = {"solutions": [], "playbooks": []} if MODE == "memnone" else {"solutions": ["S1"], "playbooks": ["P1", "P2"]}
    return (
        "1. DELIVERABLES: see block below.\n2. INPUTS: load as described.\n3. METHOD: follow the task formulas.\n"
        "4. PITFALLS: json float conversion.\n```json\n" + json.dumps(block) + "\n```"
    )


SPECFAIL_SNIPPET = """
# --- injected for testing: break the JSON types (bools -> strings, ints -> strings) in every JSON deliverable
import glob as _g, json as _j, os as _o
for _p in _g.glob(_o.path.join(_o.environ.get('OUTPUT_DIR', '/app/output'), '*.json')):
    try:
        _d = _j.load(open(_p))
    except Exception:
        continue
    if isinstance(_d, dict):
        for _k, _v in list(_d.items()):
            if isinstance(_v, bool):
                _d[_k] = 'true' if _v else 'false'
            elif isinstance(_v, int):
                _d[_k] = str(_v)
        _j.dump(_d, open(_p, 'w'))
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter
        sys.stderr.write("[mock] " + (fmt % args) + "\n")

    def _send(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._send(403, {"error": "only POST /v1/chat/completions is admitted"})
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or len(auth) < 8:
            return self._send(401, {"error": "missing bearer"})
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send(400, {"error": "bad json"})
        if body.get("n", 1) != 1:
            return self._send(400, {"error": "multiple alternatives are refused"})
        if MODE == "ctxlong":
            total = sum(len(str(m.get("content", ""))) for m in (body.get("messages") or []))
            if total > int(os.environ.get("MOCK_CTX_LIMIT", "9000")):
                return self._send(400, {"error": {"message": f"This model's maximum context length is 2048 tokens; you requested {total // 4} tokens"}})
        if STATE["requests"] >= int(os.environ.get("MOCK_REQUEST_CAP", "25")):  # raise it when one server serves many units
            return self._send(429, {"error": "request allowance exhausted for this unit"})
        STATE["requests"] += 1
        delay = float(os.environ.get("MOCK_DELAY", "0"))
        if delay > 0:
            import time
            time.sleep(delay)
        max_tokens = min(int(body.get("max_tokens") or 4000), 4000)
        messages = body.get("messages") or []
        text = "\n".join(str(m.get("content", "")) for m in messages)
        last = str(messages[-1].get("content", "")) if messages else ""
        sol = find_solution(text)
        finish = "stop"
        if sol is None:
            content = "I cannot find a canned solution for this unit.\n```python\nprint('no canned solution')\n```"
        else:
            unit, code = sol
            if "implementation plan" in last and "Do not write the code yet" in last:
                content = plan_answer(code, "REFERENCE LIBRARY" in text)
            elif "Continue the script EXACTLY" in last:
                STATE["cont_calls"] += 1
                cut = len(code) // 2
                rest = code[cut:]
                # start from the beginning of the line that was cut
                content = "```python\n" + rest + "\n```"
            elif last.startswith("The script ran successfully"):
                content = "VERDICT: OK"
            else:
                STATE["code_calls"] += 1
                first = STATE["code_calls"] == 1
                if "REFERENCE MATERIAL" in last:
                    n_sol = last.count("SOLUTION TO A DIFFERENT TASK")
                    n_pb = last.count("### PLAYBOOK:") + last.count("### NOTE:")
                    sys.stderr.write(f"[mock] reference material in code prompt: {len(last)} chars, {n_sol} solution(s), {n_pb} note(s)\n")
                    STATE["ref_seen"] = True
                if MODE == "specfail" and first:
                    content = "```python\n" + code + "\n" + SPECFAIL_SNIPPET + "\n```"
                elif MODE == "buggy" and first:
                    content = "```python\n" + code.replace("import json", "import json\nraise RuntimeError('injected bug for testing')", 1) + "\n```"
                elif MODE == "slowscript" and first:
                    content = "```python\nimport time\nwhile True:\n    time.sleep(1)\n```"
                elif MODE == "nanjson" and first:
                    content = "```python\nimport json, os\nos.makedirs('/app/output', exist_ok=True)\njson.dump({'zero_rates': {'1': float('nan')}}, open('/app/output/results.json', 'w'))\n```"
                elif MODE == "nocode" and first:
                    content = "Here is my thinking about the task but no code yet."
                elif MODE == "truncate" and first:
                    cut = len(code) // 2
                    content = "```python\n" + code[:cut]
                    finish = "length"
                else:
                    content = "```python\n" + code + "\n```"
        approx_tokens = max(1, len(content) // 4)
        if approx_tokens > max_tokens:
            content = content[: max_tokens * 4]
            finish = "length"
        self._send(200, {
            "id": "mock", "object": "chat.completion", "model": body.get("model", "house"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": approx_tokens},
        })


def main() -> None:
    port = int(os.environ.get("MOCK_PORT", "8765"))
    host = os.environ.get("MOCK_HOST", "127.0.0.1")
    srv = HTTPServer((host, port), Handler)
    sys.stderr.write(f"[mock] serving on http://127.0.0.1:{port} mode={MODE} solutions={SOL_DIR}\n")
    srv.serve_forever()


if __name__ == "__main__":
    main()
