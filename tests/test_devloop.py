"""Tests for devloop/ (the outer development loop): parsers, rules, guards, statistics, folding, a full round.

Everything runs offline with a fake teacher (tests/mock_teacher.py) and synthetic runs.  The end-to-end test at the
bottom runs a real round with the host runner, the mock House server and the mock teacher; it needs root and a
task kit:  DEVLOOP_E2E_KIT=<kit> python -m pytest tests/test_devloop.py -q -k e2e
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from devloop import attribution, compare, fold, guards, lessons, llm, loop, runner, util  # noqa: E402
from devloop.runs import RESULTS_HEADER, Run, expand_runs, failures, pass_rates  # noqa: E402
import mock_teacher  # noqa: E402

GUID = "01234567-89ab-4cde-8f01-23456789abcd"


class FakeClient:
    """ChatClient stand-in that answers with tests/mock_teacher.respond()."""

    def __init__(self, script=None):
        self.calls = []
        self.script = list(script or [])

    def chat(self, messages, **kw):
        self.calls.append([dict(m) for m in messages])
        if self.script:
            return self.script.pop(0)
        return mock_teacher.respond(messages)

    def chat_json(self, messages, **kw):
        obj = llm.extract_json(self.chat(messages, **kw))
        if obj is None:
            raise llm.LLMError("no json")
        return obj


# --------------------------------------------------------------------------------------------- fixtures

AGENT_LOG_TRUNC = """\
[agent +    0.0s] agent 2.9.16: unit=t1-x card_timeout=1800s soft_budget=480s model=house endpoint=set out=/app/output
[agent +    0.7s] deliverables (with plan): a.json, b.csv, c.csv
[agent +    0.9s] [plan] ok in 3.0s, finish=stop, usage={}, requests=1/25
[agent +    9.9s] [code] ok in 9.0s, finish=length, usage={}, requests=2/25
[agent +    9.9s] code output was truncated at the token limit; asking for a continuation
[agent +   19.9s] continuation also truncated; will rely on the repair loop
[agent +   20.0s] attempt 1: running solution_a1.py (timeout 216s)
[agent +   21.0s] attempt 1: rc=1 timed_out=False in 1.0s; deliverables 0/3; issues=3
[agent +   21.0s] attempt 1: SyntaxError: '(' was never closed (line 184)
[agent +   90.0s] done in 90s; requests used 9; output files: []
"""
AGENT_LOG_CRASH = """\
[agent +    0.0s] agent 2.9.16: unit=t1-y card_timeout=1800s soft_budget=480s model=house endpoint=set out=/app/output
[agent +    1.0s] [code] ok in 1.0s, finish=stop, usage={}, requests=2/25
[agent +    1.6s] attempt 1: rc=1 timed_out=False in 0.6s; deliverables 0/2; issues=2
[agent +    1.6s] repair returned a script identical to an earlier attempt; asking for a different fix
[agent +    2.2s] attempt 2: rc=1 timed_out=False in 0.6s; deliverables 0/2; issues=2
[agent +    2.3s] same error as an earlier attempt (KeyError: 'close'); asking for a different approach
[agent +    2.9s] attempt 3: rc=1 timed_out=False in 0.6s; deliverables 0/2; issues=2
[agent +    3.0s] wrote placeholder results.json
[agent +    3.0s] done in 3s; requests used 25; output files: ['results.json']
"""
AGENT_LOG_OK = """\
[agent +    0.0s] agent 2.9.16: unit=t1-z card_timeout=1800s soft_budget=480s model=house endpoint=set out=/app/output
[agent +    0.9s] [plan] ok in 0.0s, finish=stop, usage={}, requests=1/25
[agent +    1.0s] [code] ok in 0.0s, finish=stop, usage={}, requests=2/25
[agent +    3.6s] attempt 1: rc=0 timed_out=False in 2.7s; deliverables 2/2; issues=0
[agent +    3.6s] done in 4s; requests used 5; output files: ['a.json', 'b.csv']
"""
CHECK_VERIFIER = """\
F                                                                        [100%]
E   AssertionError: Verification failed: WRONG
Classification: WRONG
FAILED test_outputs.py::test_verification - AssertionError: Verification fail...
1 failed in 0.04s
"""
CHECK_APPROX = """\
E   assert 0.0326 == approx(0.0331 +- 3.3e-06)
FAILED test_outputs.py::test_var - assert 0.0326 == approx(0.0331 +- 3.3e-06)
1 failed, 20 passed in 0.5s
"""
CHECK_INVARIANT = """\
E   AssertionError: geometric price must be <= arithmetic price
FAILED test_outputs.py::test_geometric_below_arithmetic - AssertionError: geometric price must be <= arit...
1 failed, 9 passed in 0.3s
"""
CHECK_MISSING = """\
E   FileNotFoundError: [Errno 2] No such file or directory: '/app/output/b.csv'
FAILED test_outputs.py::test_b_columns - FileNotFoundError: [Errno 2]...
1 failed, 3 passed in 0.2s
"""


def write_run(path: Path, rows: list[dict], logs: dict[str, tuple[str, str]] | None = None, outs: dict | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(RESULTS_HEADER)
        for r in rows:
            w.writerow([r["unit"], r.get("pass", 0), r.get("agent_exit", 0), r.get("seconds", 5), r.get("requests", 5),
                        r.get("summary", ""), r.get("files", "")])
    (path / "logs").mkdir(exist_ok=True)
    for unit, (alog, clog) in (logs or {}).items():
        (path / "logs" / f"{unit}.agent.log").write_text(alog)
        (path / "logs" / f"{unit}.check.log").write_text(clog)
    for unit, files in (outs or {}).items():
        d = path / "out" / unit
        d.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (d / name).write_text(text)
    return path


@pytest.fixture()
def kit(tmp_path):
    """A two-unit kit: instruction, card (with a canary), data, checker."""
    root = tmp_path / "kit"
    for unit, cat in (("t1-fake-vol", "risk"), ("t1-fake-bond", "fixed-income")):
        u = root / "units" / unit
        (u / "environment" / "data").mkdir(parents=True)
        (u / "checks").mkdir()
        (u / "instruction.md").write_text(
            f"<!-- canary GUID {GUID} -->\n# {unit}\nCompute the 10-day VaR from `fake_prices_{unit[-3:]}.csv` and save "
            "`/app/output/results.json` with keys var_99 and es_99. Use 1234.5 as notional.\n")
        (u / "card.toml").write_text(f'[task]\nid = "{unit}"\n[metadata]\ncategory = "{cat}"\ntags = ["var", "garch"]\n'
                                     f'[contamination]\ncanary_guid = "{GUID}"\n')
        (u / "environment" / "data" / f"fake_prices_{unit[-3:]}.csv").write_text("date,close\n2024-01-02,100.5\n")
        (u / "checks" / "test_outputs.py").write_text("def test_var():\n    assert abs(var - 0.0331427) < 1e-6\n")
    return util.Kit(root)


@pytest.fixture()
def library(tmp_path):
    """A small memory library: two playbooks, a pinned note, solutions for both kit units, fingerprints."""
    src = REPO / "agent" / "memory"
    lib = tmp_path / "memory"
    (lib / "playbooks").mkdir(parents=True)
    (lib / "notes").mkdir()
    (lib / "solutions").mkdir()
    for name in ("garch-family", "var-es-methods"):
        shutil.copy(src / "playbooks" / f"{name}.md", lib / "playbooks")
    shutil.copy(src / "notes" / "pinned-essentials.md", lib / "notes")
    for unit in ("t1-fake-vol", "t1-fake-bond"):
        (lib / "solutions" / f"{unit}.py").write_text(
            f"# memory-solution\n# unit: {unit} | category: risk | tags: var\n# deliverables: results.json\n"
            "import json, os\nOUT = os.environ.get('OUTPUT_DIR', '/app/output')\n"
            "json.dump({'var_99': 1.0, 'es_99': 1.2}, open(os.path.join(OUT, 'results.json'), 'w'))\n")
    (lib / "fingerprints.json").write_text(json.dumps({"t1-fake-vol": [1, 2], "t1-fake-bond": [3, 4]}))
    (lib / "pb_exclude.json").write_text(json.dumps({"t1-fake-vol": ["garch-family"]}))
    return lib


# --------------------------------------------------------------------------------------------- util / llm

def test_scrub_removes_canaries_and_guids():
    text = f"keep this\n<!-- canary {GUID} -->\nBENCHMARK DATA SHOULD NEVER APPEAR\nid {GUID}\n"
    out = util.scrub(text)
    assert "canary" not in out.lower() and GUID not in out and "BENCHMARK" not in out
    assert "keep this" in out and "id <guid>" in out


def test_dockerfile_env_matches_the_image_configuration():
    env = util.dockerfile_env()
    assert env["AGENT_VOTE_MAX"] == "8" and env["AGENT_TIME_BUDGET"] == "480" and env["HOME"] == "/tmp"
    assert util.parse_env_pairs(["AGENT_VOTE=0 AGENT_CHUNK=1"]) == {"AGENT_VOTE": "0", "AGENT_CHUNK": "1"}
    with pytest.raises(ValueError):
        util.parse_env_pairs(["AGENT_VOTE"])


def test_card_reader_without_tomllib(kit, monkeypatch):
    """Python 3.10 without tomli: the minimal reader still gets the metadata devloop uses."""
    expected = kit.card("t1-fake-vol")
    monkeypatch.setattr(util, "tomllib", None)
    assert kit.card("t1-fake-vol") == expected
    assert kit.category("t1-fake-vol") == "risk" and kit.tags("t1-fake-vol") == ["var", "garch"]
    assert util.mini_toml('[a.b]\nx = "s # not a comment"  # comment\ny = [1, "two"]\nz = true\n') == \
        {"a": {"b": {"x": "s # not a comment", "y": [1, "two"], "z": True}}}


def test_answer_parsers():
    assert llm.extract_json('noise ```json\n{"a": 1}\n``` more') == {"a": 1}
    assert llm.extract_json('the answer: {"a": {"b": "}"}, "c": 2} done') == {"a": {"b": "}"}, "c": 2}
    assert llm.extract_json("no json here") is None
    assert llm.extract_code("```python\nx = 1\n```\ntext\n```python\nimport os\nprint(os.sep)\n```").startswith("import os")
    assert llm.strip_reasoning("<think>hmm</think>answer") == "answer"
    assert llm.chat_url("http://h:8000") == "http://h:8000/v1/chat/completions"
    assert llm.chat_url("http://h/v1") == "http://h/v1/chat/completions"
    text = ("### FILE: playbooks/a-b.md (update) sources: t1-x, t1-y\n````markdown\n---\ntitle: T\n---\n"
            "## Tested snippet\n```python\nprint(1)\n```\n````\n\n### FILE: notes/c.md (create)\n```markdown\nbody\n```\n")
    files = llm.extract_files(text)
    assert [f["path"] for f in files] == ["playbooks/a-b.md", "notes/c.md"]
    assert files[0]["sources"] == ["t1-x", "t1-y"] and "```python\nprint(1)\n```" in files[0]["content"]
    assert files[1]["action"] == "create" and files[1]["sources"] == []


def test_chat_client_retries_on_429(monkeypatch):
    hits = {"n": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            hits["n"] += 1
            if hits["n"] == 1:
                self.send_response(429)
                self.end_headers()
                return
            body = json.dumps({"choices": [{"message": {"content": "<think>x</think>{\"ok\": true}"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    client = llm.ChatClient(llm.LLMConfig(endpoint=f"http://127.0.0.1:{srv.server_port}"))
    assert client.chat_json([{"role": "user", "content": "hi"}]) == {"ok": True}
    assert hits["n"] == 2
    srv.shutdown()


# --------------------------------------------------------------------------------------------- attribution

def test_agent_log_parser():
    s = attribution.parse_agent_log(AGENT_LOG_TRUNC)
    assert s.version == "2.9.16" and s.truncations == 2 and s.requests == 9
    assert s.attempts[0]["present"] == 0 and s.attempts[0]["expected"] == 3
    assert s.errors and "SyntaxError" in s.errors[0] and s.deliverables == ["a.json", "b.csv", "c.csv"]
    c = attribution.parse_agent_log(AGENT_LOG_CRASH)
    assert c.same_error == 1 and c.identical_repairs == 1 and c.placeholders == ["results.json"] and len(c.attempts) == 3


def test_check_log_parser():
    c = attribution.parse_check_log(CHECK_VERIFIER)
    assert c.verifier == "WRONG" and c.failed == 1 and c.failed_tests[0].startswith("test_verification")
    a = attribution.parse_check_log(CHECK_APPROX)
    assert a.failed == 1 and a.passed == 20 and "approx" in a.messages[0]


@pytest.mark.parametrize("alog,clog,files,expected", [
    (AGENT_LOG_TRUNC, CHECK_MISSING, "", "truncation"),
    (AGENT_LOG_CRASH, CHECK_MISSING, "results.json", "repeated-crash"),
    (AGENT_LOG_OK.replace("deliverables 2/2", "deliverables 0/2"), CHECK_MISSING, "", "empty-accepted"),
    (AGENT_LOG_OK, CHECK_VERIFIER, "a.json b.csv", "checker-specific"),
    (AGENT_LOG_OK, CHECK_MISSING, "a.json b.csv", "structure"),
    (AGENT_LOG_OK, CHECK_INVARIANT, "a.json b.csv", "invariants"),
    (AGENT_LOG_OK, CHECK_APPROX, "a.json b.csv", "conventions"),
    (AGENT_LOG_TRUNC.replace("[agent +   90.0s]",                     # a later attempt recovered, a vote candidate failed
                             "[agent +   30.0s] attempt 2: rc=0 timed_out=False in 1.0s; deliverables 3/3; issues=0\n"
                             "[agent +   40.0s] attempt 3: rc=1 timed_out=False in 1.0s; deliverables 0/3; issues=3\n"
                             "[agent +   90.0s]"), CHECK_APPROX, "a.json b.csv c.csv", "conventions"),
    ("[agent +    0.0s] MODEL_ENDPOINT is not set: no model access\n", CHECK_MISSING, "", "infra"),
])
def test_rules(alog, clog, files, expected):
    ag, ck = attribution.parse_agent_log(alog), attribution.parse_check_log(clog)
    cat, why = attribution.classify({"agent_exit": "0"}, ag, ck, files.split())
    assert cat == expected, why
    assert why


def test_attribute_run_with_teacher_and_reports(tmp_path, kit):
    run = write_run(tmp_path / "runs" / "g" / "r1", [
        {"unit": "t1-fake-vol", "pass": 0, "files": "results.json"},
        {"unit": "t1-fake-bond", "pass": 1, "files": "results.json"},
    ], logs={"t1-fake-vol": (AGENT_LOG_OK.replace("a.json, b.csv", "results.json"), CHECK_APPROX),
             "t1-fake-bond": (AGENT_LOG_OK, "1 passed in 0.1s")},
        outs={"t1-fake-vol": {"results.json": json.dumps({"var_99": 0.0326, "note": GUID})}})
    runs = expand_runs([tmp_path / "runs" / "g"])
    client = FakeClient()
    recs = attribution.attribute(runs, kit, client)
    assert [r.unit for r in recs] == ["t1-fake-vol"]
    assert recs[0].category == "conventions" and recs[0].source == "teacher" and recs[0].root_cause
    sent = json.dumps(client.calls)
    assert GUID not in sent and "canary" not in sent.lower()           # nothing canary-like reaches the teacher
    attribution.write_reports(recs, runs, tmp_path / "att", min_units=1)
    md = (tmp_path / "att" / "attribution.md").read_text()
    assert "conventions" in md and "t1-fake-vol" in md
    mech = (tmp_path / "att" / "mechanisms.md").read_text()
    assert "ablate --variant AGENT_" in mech and "AGENT_VOTE=1" not in mech     # AGENT_VOTE is already on
    back = attribution.load_attributions(tmp_path / "att" / "attribution.jsonl")
    assert back[0].unit == "t1-fake-vol" and back[0].category == "conventions"
    assert pass_rates(runs) == {"t1-fake-bond": 1.0, "t1-fake-vol": 0.0} and list(failures(runs)) == ["t1-fake-vol"]
    assert run.name == "r1"


# --------------------------------------------------------------------------------------------- guards

VALID_PB = """---
title: Test card
category: risk
tags: var, es
summary: one line
---
## When this applies
VaR tasks.

## Method and formulas
VaR = -quantile.

## Conventions and pitfalls (what graders check)
- losses positive.

## Tested snippet
```python
print(1 + 1)
```
"""


def test_doc_format_and_hygiene():
    cats = {"risk", "meta"}
    assert guards.doc_format("playbooks/test-card.md", VALID_PB, cats) == []
    bad = guards.doc_format("playbooks/Test Card.md", VALID_PB, cats)
    assert bad and "path" in bad[0].detail
    no_sec = VALID_PB.replace("## Method and formulas", "## Theory")
    assert any("method and formulas" in f.detail.lower() for f in guards.doc_format("playbooks/x.md", no_sec, cats))
    assert any("category" in f.detail for f in guards.doc_format("playbooks/x.md", VALID_PB.replace("risk", "astrology"), cats))
    pinned = VALID_PB.replace("summary: one line", "summary: one line\npinned: true")
    assert any("pinned" in f.detail for f in guards.doc_format("notes/x.md", pinned, cats))
    assert [f.kind for f in guards.hygiene(f"x {GUID} canary")] == ["hygiene", "hygiene"]


def test_numbers_and_leakage(kit):
    assert guards._sig_digits("0.0326") == 3 and guards._sig_digits("1000") == 1 and guards._sig_digits("252") == 3
    assert guards.numbers("x = 0.0331427 and 3.31427e-2 and 0.05") == {"0.0331427"}
    text = VALID_PB + "- with 0.0331427 as the answer, see t1-fake-vol and fake_prices_vol.csv, notional 1234.5\n"
    found = [str(f) for f in guards.leakage(text, ["t1-fake-vol"], kit, original=VALID_PB)]
    assert any("t1-fake-vol" in f for f in found)
    assert any("fake_prices_vol.csv" in f for f in found)
    assert any("0.0331427" in f and "1234.5" in f for f in found)
    assert guards.leakage(VALID_PB, ["t1-fake-vol"], kit, original=VALID_PB) == []   # only new content is judged


def test_snippets_run_only_when_new():
    ok = guards.snippet_findings(VALID_PB, "", runner="host")
    assert ok == []
    broken = VALID_PB.replace("print(1 + 1)", "raise SystemExit(3)")
    found = guards.snippet_findings(broken, "", runner="host")
    assert found and "exit 3" in found[0].detail
    assert guards.snippet_findings(broken, broken, runner="host") == []           # unchanged snippet: not re-run


def test_solution_guard(kit):
    code = "# memory-solution\nimport json\nVAR = 0.0331427\nopen('checks/test_outputs.py')\n"
    found = [str(f) for f in guards.check_solution(code, "t1-fake-vol", kit)]
    assert any("checker" in f and "refers" in f for f in found) and any("0.0331427" in f for f in found)
    assert guards.check_solution("# memory-solution\nN = 1234.5\n", "t1-fake-vol", kit) == []   # task parameter: fine


# --------------------------------------------------------------------------------------------- compare

def test_sign_test_and_decision(tmp_path):
    assert compare.sign_test_p(0, 0) == 1.0
    assert compare.sign_test_p(5, 5) == pytest.approx(0.0625)
    assert compare.sign_test_p(1, 2) == 1.0
    units = [f"t1-u{i}" for i in range(10)]
    a = write_run(tmp_path / "a", [{"unit": u, "pass": int(i < 4)} for i, u in enumerate(units)])
    b = write_run(tmp_path / "b", [{"unit": u, "pass": int(i < 7)} for i, u in enumerate(units)])
    c = compare.compare([Run.load(a)], [Run.load(b)], min_gain=2, sources=["t1-u4"])
    assert c.delta == 3 and c.keep and c.gained == ["t1-u4", "t1-u5", "t1-u6"] and not c.lost
    assert c.held_out_delta == 2
    assert "keep" in compare.render(c)
    c2 = compare.compare([Run.load(a)], [Run.load(b)], min_gain=4)
    assert not c2.keep and "discard" in compare.render(c2)


# --------------------------------------------------------------------------------------------- runner

def test_runner_helpers(tmp_path):
    env = runner.docker_agent_env({"AGENT_VOTE": "0"}, tmp_path)
    assert env == f"-e AGENT_VOTE=0 -e AGENT_MEMORY_DIR=/devloop/memory -v {tmp_path.resolve()}:/devloop/memory:ro"
    with pytest.raises(ValueError):
        runner.docker_agent_env({"AGENT_X": "a b"}, None)
    assert "pytest test_outputs.py" in runner.grade_script()
    t0 = time.time()
    rc = runner.run_logged([sys.executable, "-c", "import time; time.sleep(30)"], env=dict(os.environ), cwd=tmp_path,
                           timeout=1, log_path=tmp_path / "x.log")
    assert rc == 124 and time.time() - t0 < 10


def test_docker_runner_commands(tmp_path, kit, monkeypatch):
    """The docker runner delegates to eval/local_eval.sh with the arm's switches and the read-only library mount."""
    seen = {}
    real_run = subprocess.run

    def fake_run(cmd, env=None, **kw):
        if cmd[:1] != ["bash"]:
            return real_run(cmd, env=env, **kw)
        seen["cmd"], seen["env"] = cmd, env
        out = Path(cmd[-1])
        write_run(out, [{"unit": "t1-fake-vol", "pass": 1}])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    spec = runner.EvalSpec(kit=kit, runner="docker", image="agent:dev", units=["t1-fake-vol"], jobs=2,
                           env={"AGENT_PB_EXCLUDE": "1"}, memory_dir=tmp_path, model_endpoint="http://m:8000")
    run = runner.run_eval(spec, tmp_path / "run", label="x")
    assert seen["cmd"][:2] == ["bash", str(REPO / "eval" / "local_eval.sh")] and seen["cmd"][2:4] == ["agent:dev", str(kit.root)]
    assert seen["env"]["UNITS"] == "t1-fake-vol" and seen["env"]["JOBS"] == "2" and seen["env"]["MODEL_ENDPOINT"] == "http://m:8000"
    assert seen["env"]["AGENT_ENV"].endswith(f"-v {tmp_path.resolve()}:/devloop/memory:ro")
    assert run.meta["passed"] == 1 and run.meta["env"] == {"AGENT_PB_EXCLUDE": "1"}
    with pytest.raises(ValueError):
        runner.run_eval(runner.EvalSpec(kit=kit, runner="docker"), tmp_path / "run2")


def test_docker_solution_commands(tmp_path, kit, monkeypatch):
    calls = []
    real_run = subprocess.run
    monkeypatch.setattr(runner, "run_logged", lambda cmd, **kw: calls.append(cmd) or 0)
    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **kw: (calls.append(cmd) or subprocess.CompletedProcess(
        cmd, 0, "1 passed", "")) if cmd[:1] == ["docker"] else real_run(cmd, **kw))
    script = tmp_path / "sol.py"
    script.write_text("print(1)\n")
    res = runner.run_solution(script, kit, "t1-fake-vol", runner="docker", image="agent:dev", keep_dir=tmp_path / "w")
    run_cmd, grade_cmd = calls
    assert "--network" in run_cmd and "none" in run_cmd and "65534:65534" in run_cmd and run_cmd[-2:] == ["agent:dev", "/devloop/solution.py"]
    assert grade_cmd[-3:-1] == ["agent:dev", "-c"] and "pytest test_outputs.py" in grade_cmd[-1]
    assert not any("checks" in part for part in run_cmd if ":/input:ro" in part) and res.passed
    assert not (tmp_path / "w" / "in" / "checks").exists()          # the solution never sees the checker


@pytest.mark.skipif(not os.environ.get("DEVLOOP_E2E_KIT") or os.geteuid() != 0,
                    reason="set DEVLOOP_E2E_KIT=<kit> and run as root (host runner)")
def test_e2e_host_solution_and_verify():
    k = util.Kit(os.environ["DEVLOOP_E2E_KIT"])
    res = runner.run_solution(REPO / "agent" / "memory" / "solutions" / "t1-merton-jump-diffusion.py", k,
                              "t1-merton-jump-diffusion", runner="host")
    assert res.passed, res.log + res.checker_log
    from devloop.verify import verify_memory
    rep = verify_memory(REPO / "agent" / "memory", kit=k, solutions=True, selfcheck=True, runner="host",
                        units=["t1-merton-jump-diffusion", "t1-cta-basel-capital"])
    assert rep["ok"], rep["problems"]
    assert rep["solutions"]["t1-cta-basel-capital"]["partial"] and rep["solutions"]["t1-merton-jump-diffusion"]["passed"]


# --------------------------------------------------------------------------------------------- fold

def test_fold_build_and_apply(tmp_path, kit, library):
    card = "# t1-fake-vol\ncategory: risk   tags: garch, var\n## general lesson\n- GARCH VaR conventions\n"
    accepted, rejected = fold.fold({"t1-fake-vol": card}, library, FakeClient(), kit, runner="host")
    paths = sorted(e.path for e in accepted)
    assert "playbooks/devloop-mock-conventions.md" in paths and len(paths) == 2 and not rejected
    upd = next(e for e in accepted if e.action == "update")
    assert "state their conventions explicitly" in upd.content and upd.sources == ["t1-fake-vol"]
    cand = fold.build_candidate(library, tmp_path / "cand", accepted, {}, kit)
    excl = json.loads((cand / "pb_exclude.json").read_text())
    assert set(excl["t1-fake-vol"]) >= {upd.doc_id, "devloop-mock-conventions", "garch-family"}
    import memory as agent_memory
    idx = agent_memory.MemoryIndex(cand)
    assert {"devloop-mock-conventions", upd.doc_id} <= {d.id for d in idx.docs} and not idx.dropped
    changed = fold.changed_files(cand, library)
    assert set(changed) == {"pb_exclude.json", "playbooks/devloop-mock-conventions.md", upd.path}
    target = tmp_path / "target"
    shutil.copytree(library, target)
    assert sorted(fold.apply_candidate(cand, target, library)) == sorted(changed)
    same = fold.build_candidate(library, tmp_path / "same", [], {}, kit)
    assert fold.changed_files(same, library) == []                       # no edit, no diff (not even pb_exclude)
    assert (target / "playbooks" / "devloop-mock-conventions.md").is_file()


def test_fold_revision_round(tmp_path, kit, library, monkeypatch):
    monkeypatch.setenv("MOCK_TEACHER_BAD", "1")
    monkeypatch.setenv("MOCK_TEACHER_CREATE", "0")
    mock_teacher.STATE["fold_calls"] = 0
    client = FakeClient()
    card = "# t1-fake-vol\ncategory: risk   tags: garch, var\n## general lesson\n- GARCH VaR conventions\n"
    accepted, rejected = fold.fold({"t1-fake-vol": card}, library, client, kit, runner=None)
    assert len(client.calls) == 2 and "These files were rejected" in client.calls[1][-1]["content"]
    assert len(accepted) == 1 and not rejected and GUID not in accepted[0].content


def test_fold_rejects_leaks(tmp_path, kit, library):
    leak = ("### FILE: playbooks/var-es-methods.md (update) sources: t1-fake-vol\n````markdown\n"
            + (library / "playbooks" / "var-es-methods.md").read_text().rstrip()
            + "\n- the answer is 0.0331427 for fake_prices_vol.csv\n````")
    client = FakeClient(script=[leak, leak])
    card = "# t1-fake-vol\ncategory: risk   tags: var, expected shortfall, historical var\n## general lesson\n- x\n"
    accepted, rejected = fold.fold({"t1-fake-vol": card}, library, client, kit, runner=None)
    assert not accepted and rejected and any(f.kind == "leakage" for f in rejected[0].findings)


# --------------------------------------------------------------------------------------------- lessons

def test_ensure_header_and_solve_public(tmp_path, kit, monkeypatch):
    code = "import json\nprint(1)\n"
    head = lessons.ensure_header(code, "t1-fake-vol", kit, ["results.json"])
    assert head.startswith("# memory-solution\n# unit: t1-fake-vol | category: risk | tags: var, garch\n# deliverables: results.json")
    assert lessons.ensure_header(head, "t1-fake-vol", kit, []) == head
    results = iter([runner.ScriptResult(rc=1, log="boom", checker_rc=1, checker_log="1 failed", out_dir=tmp_path),
                    runner.ScriptResult(rc=0, log="", checker_rc=0, checker_log="1 passed", out_dir=tmp_path)])
    monkeypatch.setattr(lessons, "run_solution", lambda *a, **k: next(results))
    client = FakeClient(script=["```python\nprint('v1')\n```", "```python\nimport os\nprint('v2')\n```"])
    out, history = lessons.solve_public(client, kit, "t1-fake-vol", runner="host", image="", work_dir=tmp_path / "w")
    assert out and out.startswith("# memory-solution") and "print('v2')" in out
    assert [h.get("checker_rc") for h in history] == [1, 0]
    assert "CHECKER REPORT" in client.calls[1][-1]["content"]


# --------------------------------------------------------------------------------------------- a whole round

def _fake_evaluate(results_by_label):
    def fake(cfg, group_dir, memory_dir, label, extra_env=None):
        kind = "candidate" if "candidate" in label else "baseline"
        rows, logs, outs = results_by_label[kind]
        assert memory_dir is not None and Path(memory_dir).is_dir()          # both arms use an explicit library
        return [Run.load(write_run(Path(group_dir) / "r1", rows, logs, outs))]
    return fake


def test_round_keeps_and_applies(tmp_path, kit, library, monkeypatch):
    fail_logs = (AGENT_LOG_OK.replace("a.json, b.csv", "results.json"), CHECK_APPROX)
    base = ([{"unit": "t1-fake-vol", "pass": 0, "files": "results.json"},
             {"unit": "t1-fake-bond", "pass": 0, "files": "results.json"}],
            {"t1-fake-vol": fail_logs, "t1-fake-bond": fail_logs},
            {"t1-fake-vol": {"results.json": "{}"}, "t1-fake-bond": {"results.json": "{}"}})
    cand = ([{"unit": "t1-fake-vol", "pass": 1, "files": "results.json"},
             {"unit": "t1-fake-bond", "pass": 1, "files": "results.json"}], {}, {})
    monkeypatch.setattr(loop, "evaluate", _fake_evaluate({"baseline": base, "candidate": cand}))
    target = tmp_path / "target"
    shutil.copytree(library, target)
    cfg = loop.LoopConfig(kit=kit, work=tmp_path / "work", runner="host", teacher=FakeClient(), apply=True,
                          target_memory=target, min_gain=2)
    results = loop.run_loop(cfg, rounds=2)
    assert results[0].keep and results[0].comparison.delta == 2
    assert (target / "playbooks" / "devloop-mock-conventions.md").is_file()
    rdir = tmp_path / "work" / "rounds" / "01"
    for name in ("attribution.md", "mechanisms.md", "proposals.json", "verify.json", "compare.md", "report.md"):
        assert (rdir / name).is_file(), name
    assert sorted(p.name for p in (rdir / "lessons").glob("t1-*.md")) == ["t1-fake-bond.md", "t1-fake-vol.md"]
    for p in rdir.rglob("*"):
        if p.is_file() and p.suffix in (".md", ".json", ".jsonl"):
            assert GUID not in p.read_text(errors="replace"), p                   # no canary anywhere in the work dir
    assert "kept" in (rdir / "report.md").read_text()
    assert len(results) == 2 and (tmp_path / "work" / "summary.md").is_file()


def test_round_discards_without_gain_and_stops(tmp_path, kit, library, monkeypatch):
    fail_logs = (AGENT_LOG_OK.replace("a.json, b.csv", "results.json"), CHECK_APPROX)
    same = ([{"unit": "t1-fake-vol", "pass": 0, "files": "results.json"},
             {"unit": "t1-fake-bond", "pass": 1, "files": "results.json"}],
            {"t1-fake-vol": fail_logs, "t1-fake-bond": (AGENT_LOG_OK, "1 passed")}, {})
    monkeypatch.setattr(loop, "evaluate", _fake_evaluate({"baseline": same, "candidate": same}))
    target = tmp_path / "target"
    shutil.copytree(library, target)
    before = sorted(p.name for p in (target / "playbooks").iterdir())
    cfg = loop.LoopConfig(kit=kit, work=tmp_path / "work", runner="host", teacher=FakeClient(), apply=True,
                          target_memory=target)
    results = loop.run_loop(cfg, rounds=3)
    assert len(results) == 1 and not results[0].keep and "discarded" in results[0].stopped
    assert sorted(p.name for p in (target / "playbooks").iterdir()) == before          # nothing applied


def test_round_without_teacher_stops_after_attribution(tmp_path, kit, library, monkeypatch):
    fail_logs = (AGENT_LOG_OK.replace("a.json, b.csv", "results.json"), CHECK_APPROX)
    base = ([{"unit": "t1-fake-vol", "pass": 0, "files": "results.json"}], {"t1-fake-vol": fail_logs}, {})
    monkeypatch.setattr(loop, "evaluate", _fake_evaluate({"baseline": base, "candidate": base}))
    cfg = loop.LoopConfig(kit=kit, work=tmp_path / "work", runner="host", teacher=None, target_memory=library)
    res = loop.run_loop(cfg)[0]
    assert not res.keep and "attribution only" in res.stopped
    assert (tmp_path / "work" / "rounds" / "01" / "attribution.md").is_file()


def test_verify_flags_problems(tmp_path, library):
    from devloop.verify import verify_memory
    assert verify_memory(library)["ok"]
    bad = tmp_path / "bad"
    shutil.copytree(library, bad)
    (bad / "playbooks" / "garch-family.md").unlink()
    (bad / "notes" / "broken.md").write_text(f"---\ntitle: x\ncategory: astrology\ntags: a\nsummary: b\n---\nbody {GUID}\n")
    rep = verify_memory(bad, base_memory=library)
    assert not rep["ok"]
    joined = " ".join(rep["problems"])
    assert "GUID" in joined and "garch-family" in joined


# --------------------------------------------------------------------------------------------- end to end (opt-in)

E2E_UNITS = "t1-merton-jump-diffusion t1-zero-coupon-bootstrapping t1-digital-barrier-options t1-momentum-backtest t1-cta-basel-capital"


@pytest.mark.skipif(not os.environ.get("DEVLOOP_E2E_KIT") or os.geteuid() != 0,
                    reason="set DEVLOOP_E2E_KIT=<kit> and run as root (host runner)")
def test_e2e_round_with_mock_models(tmp_path):
    """A real round: host runner, mock House (canned scripts; one of them with a wrong coupon convention, so that a
    knowledge failure exists next to the mechanics failure of t1-cta-basel-capital), mock teacher over HTTP."""
    kit_dir = os.environ["DEVLOOP_E2E_KIT"]
    sols = tmp_path / "house-solutions"
    shutil.copytree(HERE / "solutions", sols)
    zc = sols / "t1-zero-coupon-bootstrapping.py"
    zc.write_text(zc.read_text().replace("coupon = c / freq", "coupon = c / freq * 1.02"))
    house = subprocess.Popen([sys.executable, str(HERE / "mock_model_server.py")], stderr=subprocess.DEVNULL,
                             env={**os.environ, "MOCK_KIT": kit_dir, "MOCK_REQUEST_CAP": "100000", "MOCK_PORT": "8781",
                                  "MOCK_SOLUTIONS_DIR": str(sols)})
    teacher = subprocess.Popen([sys.executable, str(HERE / "mock_teacher.py")], stderr=subprocess.DEVNULL,
                               env={**os.environ, "MOCK_TEACHER_PORT": "8782"})
    try:
        time.sleep(1.0)
        proc = subprocess.run([sys.executable, "-m", "devloop", "round", "--runner", "host", "--kit", kit_dir,
                               "--work", str(tmp_path / "work"), "--units", E2E_UNITS,
                               "--model-endpoint", "http://127.0.0.1:8781", "--teacher-endpoint", "http://127.0.0.1:8782",
                               "--memory", str(REPO / "agent" / "memory")],
                              cwd=REPO, capture_output=True, text=True, timeout=3000)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        rdir = tmp_path / "work" / "rounds" / "01"
        base = Run.load(rdir / "baseline" / "r1")
        assert base.passed() == set(E2E_UNITS.split()) - {"t1-cta-basel-capital", "t1-zero-coupon-bootstrapping"}
        att = {r["unit"]: r["category"] for r in util.read_jsonl(rdir / "attribution.jsonl")}
        assert att == {"t1-cta-basel-capital": "repeated-crash", "t1-zero-coupon-bootstrapping": "conventions"}
        assert sorted(p.name for p in (rdir / "lessons").glob("t1-*.md")) == ["t1-zero-coupon-bootstrapping.md"]
        assert json.loads((rdir / "proposals.json").read_text())["accepted"]
        assert (rdir / "memory" / "playbooks" / "devloop-mock-conventions.md").is_file()
        assert json.loads((rdir / "verify.json").read_text())["ok"]
        cand = Run.load(rdir / "candidate" / "r1")
        assert cand.meta["memory_dir"].endswith("rounds/01/memory") and cand.meta["env"]["AGENT_PB_EXCLUDE"] == "1"
        report = (rdir / "report.md").read_text()
        assert "discarded" in report                       # the mock House ignores the library: no gain to keep
    finally:
        house.terminate()
        teacher.terminate()
