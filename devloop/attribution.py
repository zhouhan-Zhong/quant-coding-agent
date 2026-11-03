"""Failure attribution: why did each failed unit fail?

Rules over the agent log and the checker log come first (cheap, deterministic, reproducible); the teacher model,
when configured, then makes the judgement calls the rules cannot (conventions vs invariants, the root cause in one
sentence, a lesson phrased for *other* tasks).  The categories are the ones used to analyse the 62 failures of
agent 2.0 (truncation 13, empty outputs accepted 2, repeated crash 6, structure 4, and 37 numerical failures split
into invariants 13, conventions 16 and checker-specific 8), plus `infra` for runs that never reached the model.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .llm import ChatClient, LLMError, prompt
from .runs import Run, failures, pass_rates
from .util import Kit, clip, dockerfile_env, log, preview_file, scrub, write_jsonl, write_text

CATEGORIES: dict[str, dict] = {
    "infra": {
        "what": "the run never reached a modelling decision: no model answer, an agent timeout or crash",
        "mechanisms": "check the model server, the request budget and the per-unit time limits before reading anything else",
        "flags": []},
    "truncation": {
        "what": "the program was cut off at the 4,000-token output cap and never became complete",
        "mechanisms": "exact continuation from the cut, two-part generation, SEARCH/REPLACE repairs instead of full rewrites",
        "flags": ["AGENT_CHUNK", "AGENT_SPLIT", "AGENT_TWO_TRACK", "AGENT_STAGED"]},
    "empty-accepted": {
        "what": "the script ran but no real deliverable was written (nothing, or only a placeholder), and the run stopped",
        "mechanisms": "never accept a clean exit whose deliverables are missing or degenerate; feed that back as a failure",
        "flags": ["AGENT_DEGEN"]},
    "repeated-crash": {
        "what": "every attempt crashed, and repairs repeated an earlier script or the same error",
        "mechanisms": "patch repairs, a different fix when the error signature repeats, a fresh rewrite, loop breaking",
        "flags": ["AGENT_RESTART", "AGENT_PROBE"]},
    "structure": {
        "what": "files, keys, columns, row counts or types do not match the task's output contract",
        "mechanisms": "spec self-checks and identifier-column checks fed back as repair feedback; exact input structure",
        "flags": ["AGENT_DEGEN", "AGENT_PROBE2", "AGENT_KEY_CHECK"]},
    "checker-specific": {
        "what": "a checker with its own protocol (e.g. a params verifier recomputing every intermediate) rejected it",
        "mechanisms": "memory notes for the checker family, trigger-pinned notes, template routes for shipped scripts",
        "flags": ["AGENT_TEMPLATE_ROUTE", "AGENT_MEM_TRIGGERS", "AGENT_TEXT_ROUTE", "AGENT_PARAM_ECHO"]},
    "invariants": {
        "what": "a relation the task itself states (parity, bounds, MC vs analytic, totals) does not hold",
        "mechanisms": "executable self-tests of the stated invariants; an independent second solution",
        "flags": ["AGENT_INVARIANTS", "AGENT_SELFTEST", "AGENT_NVERSION", "AGENT_VERIFY"]},
    "conventions": {
        "what": "formats are right but numbers differ: annualisation, ddof, calendars, units, windows, seeds",
        "mechanisms": "convention reads, independent-solution voting, playbooks that state the reference conventions",
        "flags": ["AGENT_VOTE", "AGENT_CONVENTIONS", "AGENT_CONV_VOTE", "AGENT_CLAUSE_AUDIT", "AGENT_MC_RULES"]},
}
KNOWLEDGE = ("conventions", "invariants", "checker-specific", "structure")   # what better memory can fix

_LINE = re.compile(r"^\[agent \+\s*([\d.]+)s\] ?(.*)$")
_ATTEMPT = re.compile(r"attempt (\d+): rc=(-?\d+) timed_out=(True|False) in ([\d.]+)s; deliverables (\d+)/(\d+); issues=(\d+)")
_TRUNC = ("code output was truncated at the token limit", "cut at the token limit", "continuation also truncated",
          "script still truncated after continuation")
_INVARIANT_WORDS = re.compile(
    r"parity|monoton|bound|non-?negative|positive|negative|between|sum(?:s)? to|add(?:s)? up|consisten|identity|"
    r"should not exceed|must be (?:below|above|less|greater)|less than|greater than|<=|>=|\bflag\b|_check|_ok\b|"
    r"is False|is not True|within \d|standard error|\bstd err|\bse\b", re.I)
_CONVENTION_WORDS = re.compile(r"isclose|approx|rtol|atol|toleran|expected|mismatch|differ|relative error|abs\(|!=|==", re.I)
_STRUCTURE_WORDS = re.compile(
    r"FileNotFoundError|No such file|does not exist|not found|\bmissing\b|KeyError|not in (?:\w+\.)?columns|"
    r"columns? (?:mismatch|missing|differ)|set\([^)]*columns|\blen\(|\.shape\b|row count|number of rows|n_rows|"
    r"dtype|isinstance|JSONDecodeError|ParserError|EmptyDataError|isnull|isna\(|\bNaN\b|\bnull\b", re.I)


@dataclass
class AgentSignals:
    version: str = ""
    requests: int | None = None
    truncations: int = 0
    same_error: int = 0
    identical_repairs: int = 0
    attempts: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)
    spec_issues: list[str] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    http_errors: list[str] = field(default_factory=list)
    ok_requests: int = 0
    no_budget: bool = False
    hard_deadline: bool = False
    model_missing: bool = False

    @property
    def last(self) -> dict | None:
        return self.attempts[-1] if self.attempts else None

    @property
    def best(self) -> dict | None:
        """The attempt closest to complete (the agent restores its best outputs, so this is what was submitted)."""
        return max(self.attempts, key=lambda a: (a["rc"] == 0, a["present"] - a["expected"], -a["issues"]), default=None)

    def delivered_all(self) -> bool:
        return any(a["expected"] > 0 and a["present"] >= a["expected"] and a["rc"] == 0 for a in self.attempts)


@dataclass
class CheckSignals:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    failed_tests: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    verifier: str = ""

    @property
    def text(self) -> str:
        return "\n".join(self.failed_tests + self.messages)


def parse_agent_log(text: str) -> AgentSignals:
    s = AgentSignals()
    for raw in text.splitlines():
        m = _LINE.match(raw)
        if not m:
            continue
        msg = m.group(2)
        body = msg.strip()
        if (v := re.match(r"agent ([\w.\-]+): unit=", body)):
            s.version = v.group(1)
        if any(t in body for t in _TRUNC):
            s.truncations += 1
        if body.startswith("same error as an earlier attempt"):
            s.same_error += 1
        if body.startswith("repair returned a script identical"):
            s.identical_repairs += 1
        if (a := _ATTEMPT.search(body)):
            s.attempts.append({"n": int(a.group(1)), "rc": int(a.group(2)), "timed_out": a.group(3) == "True",
                               "seconds": float(a.group(4)), "present": int(a.group(5)), "expected": int(a.group(6)),
                               "issues": int(a.group(7))})
        elif (e := re.match(r"attempt (\d+): (?!rc=|running )(.+)", body)):
            s.errors.append(e.group(2)[:240])
        if (p := re.match(r"wrote placeholder (\S+)", body)):
            s.placeholders.append(p.group(1))
        if body.startswith("spec: ") or body.startswith("check: "):
            s.spec_issues.append(body.split(": ", 1)[1][:240])
        if (d := re.match(r"deliverables \((?:with plan|static)\): (.*)$", body)):
            s.deliverables = [x.strip() for x in d.group(1).split(",") if x.strip()]
        if (h := re.match(r"\[([\w\-]+)\] HTTP (\d{3})", body)):
            s.http_errors.append(f"{h.group(1)}:{h.group(2)}")
        if re.match(r"\[[\w\-]+\] ok in ", body):
            s.ok_requests += 1
        if (r := re.search(r"requests used (\d+)", body)):
            s.requests = int(r.group(1))
        if body.startswith(("no budget left", "stopping the repair loop: no time", "budget exhausted")):
            s.no_budget = True
        if body.startswith("hard deadline reached"):
            s.hard_deadline = True
        if body.startswith("MODEL_ENDPOINT is not set"):
            s.model_missing = True
    return s


def parse_check_log(text: str) -> CheckSignals:
    c = CheckSignals()
    for ln in text.splitlines():
        if (m := re.match(r"^FAILED (\S+)(?: - (.*))?$", ln)):
            c.failed_tests.append(m.group(1).split("::")[-1] + (f" - {m.group(2)}" if m.group(2) else ""))
        elif (m := re.match(r"^ERROR (\S+)", ln)):
            c.failed_tests.append("ERROR " + m.group(1).split("::")[-1])
        elif (m := re.match(r"^E\s+(.+)$", ln)):
            msg = m.group(1).strip()
            if msg and msg not in c.messages and len(c.messages) < 40:
                c.messages.append(msg[:240])
        if (m := re.search(r"Classification: (\w+)|Verification failed: (\w+)", ln)):
            c.verifier = m.group(1) or m.group(2)
    tail = "\n".join(text.splitlines()[-3:])
    for key in ("passed", "failed", "errors"):
        m = re.search(rf"(\d+) {key[:-1] if key == 'errors' else key}", tail)
        if m:
            setattr(c, key, int(m.group(1)))
    return c


def classify(row: dict, ag: AgentSignals, ck: CheckSignals, out_files: list[str]) -> tuple[str, list[str]]:
    """First matching rule wins; the order mirrors how the 2.0 failures were triaged."""
    why: list[str] = []
    rc = int(row.get("agent_exit") or 0) if str(row.get("agent_exit", "")).lstrip("-").isdigit() else 0
    real = [f for f in out_files if f not in set(ag.placeholders) and not f.startswith(".")]
    if ag.model_missing or (ag.ok_requests == 0 and (ag.http_errors or ag.requests == 0)) or rc in (124, 137):
        why.append("no model answers" if ag.ok_requests == 0 else f"agent exit {rc}")
        return "infra", why
    if ag.truncations and not ag.delivered_all():
        why.append(f"{ag.truncations} truncation event(s); the best attempt "
                   + (f"delivered {ag.best['present']}/{ag.best['expected']} (rc={ag.best['rc']})" if ag.best else "never ran"))
        return "truncation", why
    crashed = [a for a in ag.attempts if a["rc"] != 0 or a["timed_out"]]
    if len(ag.attempts) >= 2 and len(crashed) == len(ag.attempts) and (ag.same_error or ag.identical_repairs
                                                                         or len(set(ag.errors[-3:])) == 1):
        why.append(f"all {len(ag.attempts)} attempts failed; same error x{ag.same_error}, identical repairs x{ag.identical_repairs}")
        return "repeated-crash", why
    if not real and (not ag.best or ag.best["rc"] == 0):
        why.append("no deliverable written" + (f" (placeholder {', '.join(ag.placeholders)})" if ag.placeholders else ""))
        return "empty-accepted", why
    if ck.verifier and ck.verifier.upper() != "PERFECT":
        why.append(f"verifier classification {ck.verifier}")
        return "checker-specific", why
    text = ck.text
    if _STRUCTURE_WORDS.search(text) or (ag.best and ag.best["present"] < ag.best["expected"]):
        hit = _STRUCTURE_WORDS.search(text)
        why.append(f"checker: {hit.group(0)!r}" if hit else f"delivered {ag.best['present']}/{ag.best['expected']}")
        return "structure", why
    if _INVARIANT_WORDS.search(text) and not _CONVENTION_WORDS.search(text):
        why.append(f"checker asserts a relation: {_INVARIANT_WORDS.search(text).group(0)!r}")
        return "invariants", why
    if _INVARIANT_WORDS.search(text) and _CONVENTION_WORDS.search(text):
        inv = len(_INVARIANT_WORDS.findall(text))
        conv = len(_CONVENTION_WORDS.findall(text))
        if inv > conv:
            why.append(f"relation words ({inv}) outnumber tolerance words ({conv})")
            return "invariants", why
    why.append("numbers differ from the reference" + (f": {ck.messages[0][:120]}" if ck.messages else ""))
    return "conventions", why


@dataclass
class Attribution:
    unit: str
    category: str
    rule_category: str
    reasons: list[str]
    source: str = "rules"
    root_cause: str = ""
    general_lesson: str = ""
    tags: list[str] = field(default_factory=list)
    confidence: float | None = None
    run: str = ""
    signals: dict = field(default_factory=dict)
    check: dict = field(default_factory=dict)


def evidence_text(kit: Kit | None, run: Run, unit: str, ag: AgentSignals, ck: CheckSignals, limit: int = 14000) -> str:
    """What a model needs to judge a failure: the task, the checker's complaints, what the agent did and wrote."""
    parts = [f"UNIT: {unit}"]
    if kit is not None:
        parts.append("TASK CATEGORY: " + (kit.category(unit) or "?"))
        parts.append("TASK INSTRUCTION (excerpt):\n" + clip(scrub(kit.instruction(unit)), 5000))
    parts.append("CHECKER RESULT: " + f"{ck.passed} passed, {ck.failed} failed" + (f", verifier={ck.verifier}" if ck.verifier else ""))
    if ck.failed_tests or ck.messages:
        parts.append("CHECKER FAILURES:\n" + clip(scrub("\n".join(ck.failed_tests[:15] + ck.messages[:25])), 3500))
    sig = {k: v for k, v in asdict(ag).items() if v not in (None, [], 0, False, "")}
    sig.pop("attempts", None)
    parts.append("AGENT SIGNALS: " + clip(scrub(str(sig)), 1500))
    if ag.attempts:
        parts.append("ATTEMPTS: " + "; ".join(f"#{a['n']} rc={a['rc']} files {a['present']}/{a['expected']} issues={a['issues']}"
                                             for a in ag.attempts[-8:]))
    out = run.out_dir(unit)
    if out.is_dir():
        files = sorted(p for p in out.iterdir() if p.is_file())
        parts.append("AGENT OUTPUTS:\n" + "\n".join(preview_file(p, 900) for p in files[:6]) if files else "AGENT OUTPUTS: none")
    return clip("\n\n".join(parts), limit)


def attribute(runs: list[Run], kit: Kit | None = None, client: ChatClient | None = None) -> list[Attribution]:
    out: list[Attribution] = []
    for unit, run in failures(runs).items():
        row = run.rows[unit]
        ag, ck = parse_agent_log(run.agent_log(unit)), parse_check_log(run.check_log(unit))
        files = (row.get("output_files") or "").split()
        cat, why = classify(row, ag, ck, files)
        rec = Attribution(unit=unit, category=cat, rule_category=cat, reasons=why, run=str(run.path),
                          signals={"truncations": ag.truncations, "same_error": ag.same_error,
                                   "identical_repairs": ag.identical_repairs, "attempts": len(ag.attempts),
                                   "requests": ag.requests, "placeholders": ag.placeholders,
                                   "spec_issues": ag.spec_issues[:5], "errors": [scrub(e) for e in ag.errors[-3:]]},
                          check={"passed": ck.passed, "failed": ck.failed, "verifier": ck.verifier,
                                 "failed_tests": [scrub(t) for t in ck.failed_tests[:10]],
                                 "messages": [scrub(m) for m in ck.messages[:10]]})
        if client is not None and cat != "infra":
            try:
                _refine(rec, client, evidence_text(kit, run, unit, ag, ck))
            except LLMError as exc:
                log(f"attribution of {unit}: teacher unavailable ({exc}); keeping the rule label")
        out.append(rec)
    return out


def _refine(rec: Attribution, client: ChatClient, evidence: str) -> None:
    cats = "\n".join(f"- {k}: {v['what']}" for k, v in CATEGORIES.items())
    msgs = [{"role": "system", "content": prompt("attribute").replace("{categories}", cats)},
            {"role": "user", "content": f"{evidence}\n\nRULE-BASED LABEL: {rec.rule_category} ({'; '.join(rec.reasons)})"}]
    ans = client.chat_json(msgs, max_tokens=1200)
    cat = str(ans.get("category", "")).strip().lower()
    if cat in CATEGORIES and cat != "infra":
        rec.category = cat
    rec.source = "teacher"
    rec.root_cause = scrub(str(ans.get("root_cause", "")))[:600]
    rec.general_lesson = scrub(str(ans.get("general_lesson", "")))[:900]
    rec.tags = [scrub(str(t)).lower()[:40] for t in (ans.get("tags") or [])][:12]
    try:
        rec.confidence = float(ans.get("confidence"))
    except (TypeError, ValueError):
        rec.confidence = None


# ----------------------------------------------------------------------------------------------------------------
# reports
# ----------------------------------------------------------------------------------------------------------------

def signal_table(runs: list[Run]) -> list[tuple[str, int, int]]:
    """How often each log signal occurs across *all* units, and how many of those units still passed."""
    run = runs[0]
    counts: Counter = Counter()
    passed: Counter = Counter()
    for unit in run.units():
        ag = parse_agent_log(run.agent_log(unit))
        ok = unit in run.passed()
        flags = {"truncation": ag.truncations > 0, "repeated error": ag.same_error > 0 or ag.identical_repairs > 0,
                 "placeholder written": bool(ag.placeholders), "spec issues": bool(ag.spec_issues),
                 "budget ran out": ag.no_budget or ag.hard_deadline, "HTTP errors": bool(ag.http_errors)}
        for k, v in flags.items():
            if v:
                counts[k] += 1
                passed[k] += ok
    return [(k, counts[k], passed[k]) for k in counts]


def render_attribution(recs: list[Attribution], runs: list[Run]) -> str:
    rates = pass_rates(runs)
    n = len(rates)
    p = sum(rates.values())
    lines = ["# Failure attribution", "",
             f"Runs: {', '.join(r.label for r in runs)}. Passed {p:g}/{n} on average; {len(recs)} unit(s) failed in at least half of the runs.", "",
             "| category | units | share | what it means |", "|---|---|---|---|"]
    by = Counter(r.category for r in recs)
    for cat in CATEGORIES:
        if by.get(cat):
            lines.append(f"| {cat} | {by[cat]} | {by[cat] / max(len(recs), 1):.0%} | {CATEGORIES[cat]['what']} |")
    lines += ["", "## Per unit", "", "| unit | category | evidence | root cause |", "|---|---|---|---|"]
    for r in sorted(recs, key=lambda r: (list(CATEGORIES).index(r.category), r.unit)):
        label = r.category + ("" if r.category == r.rule_category else f" (rules: {r.rule_category})")
        cause = (r.root_cause or "").replace("|", "/").replace("\n", " ")[:220]
        lines.append(f"| {r.unit} | {label} | {'; '.join(r.reasons).replace('|', '/')[:200]} | {cause} |")
    return "\n".join(lines) + "\n"


def render_mechanisms(recs: list[Attribution], runs: list[Run], min_units: int = 3, image_env: dict | None = None) -> str:
    """Recurring failure classes are mechanism candidates (agent code), reported with ready-to-run flag ablations."""
    image_env = dockerfile_env() if image_env is None else image_env
    by: dict[str, list[str]] = {}
    for r in recs:
        by.setdefault(r.category, []).append(r.unit)
    lines = ["# Mechanism candidates", "",
             "Failure classes that recur across units point at the agent's mechanics rather than at missing knowledge.",
             "devloop never edits agent code; for each class below it lists the existing switches that are off in the image",
             "and the command that A/B-tests one of them on the affected units.", ""]
    found = False
    for cat, units in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(units) < min_units or cat == "infra":
            continue
        found = True
        info = CATEGORIES[cat]
        off = [f for f in info["flags"] if str(image_env.get(f, "0")).strip().lower() in ("0", "", "false", "no")]
        lines += [f"## {cat}: {len(units)} units", "", f"{info['what'].capitalize()}.", "",
                  f"- Mechanism ideas: {info['mechanisms']}.",
                  f"- Units: {', '.join(sorted(units))}"]
        if off:
            unit_arg = " ".join(sorted(units))
            lines.append(f"- Switches off in the image: {', '.join(off)}. To test one on these units:")
            lines += ["", "```bash", f'python -m devloop ablate --variant {off[0]}=1 --units "{unit_arg}" --repeats 2 \\',
                      "    --kit <kit> --image <agent image>", "```"]
        lines.append("")
    if not found:
        lines += [f"No failure class reaches {min_units} units.", ""]
    table = signal_table(runs) if runs else []
    if table:
        lines += ["## Signals across all units (first run)", "", "| signal | units | of which passed |", "|---|---|---|"]
        lines += [f"| {k} | {n} | {p} |" for k, n, p in sorted(table, key=lambda t: -t[1])]
    return "\n".join(lines) + "\n"


def write_reports(recs: list[Attribution], runs: list[Run], out_dir: Path, min_units: int = 3) -> None:
    out_dir = Path(out_dir)
    write_jsonl(out_dir / "attribution.jsonl", [asdict(r) for r in recs])
    write_text(out_dir / "attribution.md", render_attribution(recs, runs))
    write_text(out_dir / "mechanisms.md", render_mechanisms(recs, runs, min_units=min_units))


def load_attributions(path: Path) -> list[Attribution]:
    from .util import read_jsonl
    return [Attribution(**{k: v for k, v in row.items() if k in Attribution.__dataclass_fields__})
            for row in read_jsonl(Path(path))]
