"""Lesson cards for failed public units.

The teacher sees what the agent could not: the unit's checker and a verified reference solution.  When the library
has no solution for the unit, `solve_public` first writes one with the checker in the loop (the workflow of
devloop/guides/SUBAGENT_GUIDE.md).  Cards stay in the round's work directory because they may quote the checker;
only their general part reaches the library, through fold.py and the guards.
"""
from __future__ import annotations

import re
from pathlib import Path

from .attribution import KNOWLEDGE, Attribution
from .guards import check_solution
from .llm import ChatClient, LLMError, extract_code, prompt
from .runner import run_solution
from .runs import Run
from .util import Kit, clip, log, preview_file, read_text, scrub, write_json, write_text


def _task_block(kit: Kit, unit: str, with_checker: bool = True) -> str:
    parts = [f"UNIT: {unit}", f"TASK CATEGORY: {kit.category(unit) or '?'}",
             "TASK INSTRUCTION:\n" + clip(scrub(kit.instruction(unit)), 9000)]
    files = kit.data_files(unit)
    if files:
        parts.append("INPUT FILES (read them from DATA_DIR):\n" + "\n".join(preview_file(p, 700) for p in files[:8]))
    if with_checker:
        parts.append("CHECKER SOURCE (checks/test_outputs.py):\n```python\n" + clip(scrub(kit.checker(unit)), 11000) + "\n```")
    return "\n\n".join(parts)


def _agent_block(run: Run, unit: str, rec: Attribution) -> str:
    parts = [f"THE AGENT'S RESULT: category {rec.category} ({'; '.join(rec.reasons)})"]
    if rec.root_cause:
        parts.append("ROOT CAUSE (first analysis): " + rec.root_cause)
    chk = rec.check or {}
    if chk.get("failed_tests") or chk.get("messages"):
        parts.append("CHECKER FAILURES:\n" + "\n".join((chk.get("failed_tests") or [])[:10] + (chk.get("messages") or [])[:15]))
    out = run.out_dir(unit)
    files = sorted(p for p in out.iterdir() if p.is_file()) if out.is_dir() else []
    parts.append("THE AGENT'S OUTPUTS:\n" + ("\n".join(preview_file(p, 900) for p in files[:6]) if files else "none"))
    return clip(scrub("\n\n".join(parts)), 7000)


def write_lesson(client: ChatClient, kit: Kit, run: Run, rec: Attribution, solution_code: str) -> str:
    user = (_task_block(kit, rec.unit) + "\n\nVERIFIED REFERENCE SOLUTION (passes the checker):\n```python\n"
            + clip(scrub(solution_code), 9000) + "\n```\n\n" + _agent_block(run, rec.unit, rec))
    card = scrub(client.chat([{"role": "system", "content": prompt("lesson")}, {"role": "user", "content": user}],
                             max_tokens=3000))
    card = re.sub(r"^```(?:markdown|md)?\s*\n|\n```\s*$", "", card.strip())
    if not card.lstrip().startswith("#"):
        card = f"# {rec.unit}\n" + card
    return card.strip() + "\n"


def ensure_header(code: str, unit: str, kit: Kit, outputs: list[str]) -> str:
    """Give a passing script the header the memory index needs (`# memory-solution`, unit/category/tags, deliverables)."""
    import memory as agent_memory  # noqa: PLC0415
    lines = code.lstrip("\n").splitlines()
    cats = agent_memory.normalize_category(kit.category(unit))
    cat = cats[0] if cats else (kit.category(unit) or "numerics")
    tags = ", ".join(t.lower() for t in kit.tags(unit)[:10]) or cat
    head = []
    if not lines or lines[0].strip() != "# memory-solution":
        head.append("# memory-solution")
    body = "\n".join(lines[:12])
    if not re.search(r"^#\s*unit:\s*[^|]+\|\s*category:", body, re.M):
        head.append(f"# unit: {unit} | category: {cat} | tags: {tags}")
    if not re.search(r"^#\s*deliverables:", body, re.M):
        head.append("# deliverables: " + ", ".join(sorted(outputs)))
    if not head:
        return code
    if lines and lines[0].strip() == "# memory-solution":
        return "\n".join([lines[0], *head, *lines[1:]]) + "\n"
    return "\n".join([*head, *lines]) + "\n"


def solve_public(client: ChatClient, kit: Kit, unit: str, *, runner: str, image: str, work_dir: Path,
                 attempts: int = 6) -> tuple[str | None, list[dict]]:
    """Write a verified solution with the checker in the loop.  Returns (code or None, attempt history)."""
    msgs = [{"role": "system", "content": prompt("solve")}, {"role": "user", "content": _task_block(kit, unit)}]
    history: list[dict] = []
    for k in range(1, attempts + 1):
        answer = client.chat(msgs, max_tokens=4000)
        code = extract_code(answer)
        msgs.append({"role": "assistant", "content": answer})
        if not code:
            history.append({"attempt": k, "result": "no code"})
            msgs.append({"role": "user", "content": "Reply with the complete script as ONE ```python block."})
            continue
        code = scrub(code) + ("\n" if not code.endswith("\n") else "")
        # hygiene and checker access are judged before running; pasted checker numbers once the checker passes
        problems = [str(f) for f in check_solution(ensure_header(code, unit, kit, []), unit, kit)
                    if f.kind == "hygiene" or "refers to the checker" in f.detail]
        if problems:
            history.append({"attempt": k, "result": "rejected", "problems": problems})
            msgs.append({"role": "user", "content": "The script was not run: " + "; ".join(problems) + ". Fix that."})
            continue
        path = write_text(work_dir / f"{unit}.attempt{k}.py", code)
        res = run_solution(path, kit, unit, runner=runner, image=image, keep_dir=work_dir / f"run{k}")
        outputs = sorted(p.name for p in res.out_dir.iterdir()) if res.out_dir and res.out_dir.is_dir() else []
        history.append({"attempt": k, "rc": res.rc, "checker_rc": res.checker_rc, "outputs": outputs})
        if res.passed:
            final = ensure_header(code, unit, kit, outputs)
            pasted = [str(f) for f in check_solution(final, unit, kit) if "numbers taken from the checker" in f.detail]
            if not pasted:
                log(f"solve {unit}: checker passes after {k} attempt(s)")
                return final, history
            msgs.append({"role": "user", "content": f"The checker passes, but {pasted[0]}. Compute these values."})
            continue
        feedback = (f"The script exited with code {res.rc}.\nLAST OUTPUT LINES:\n{scrub(res.log)}\n\n"
                    f"CHECKER REPORT (exit {res.checker_rc}):\n{clip(scrub(res.checker_log), 3500)}\n\n"
                    "Reply with the complete corrected script as ONE ```python block.")
        msgs.append({"role": "user", "content": feedback})
        if len(msgs) > 8:                                # keep the context bounded
            msgs = msgs[:2] + msgs[-6:]
    log(f"solve {unit}: no passing script in {attempts} attempt(s)")
    return None, history


def write_lessons(recs: list[Attribution], runs_by_unit: dict[str, Run], kit: Kit, client: ChatClient,
                  memory_dir: Path, out_dir: Path, *, runner: str, image: str = "", max_units: int = 12,
                  solve_attempts: int = 6) -> tuple[dict[str, Path], dict[str, str]]:
    """Cards for the failures that knowledge can fix.  Returns ({unit: card path}, {unit: new verified solution})."""
    out_dir = Path(out_dir)
    order = {c: i for i, c in enumerate(KNOWLEDGE)}
    todo = sorted((r for r in recs if r.category in KNOWLEDGE), key=lambda r: (order[r.category], r.unit))[:max_units]
    skipped = [r.unit for r in recs if r.category not in KNOWLEDGE]
    if skipped:
        log(f"lessons: {len(skipped)} failure(s) are mechanics, not knowledge ({', '.join(sorted(skipped)[:8])}...)"
            if len(skipped) > 8 else f"lessons: mechanics, not knowledge: {', '.join(sorted(skipped))}")
    cards: dict[str, Path] = {}
    new_solutions: dict[str, str] = {}
    for rec in todo:
        unit = rec.unit
        if not kit.unit_dir(unit).is_dir():
            log(f"lessons: {unit} is not in the kit; skipped")
            continue
        sol_path = Path(memory_dir) / "solutions" / f"{unit}.py"
        code = read_text(sol_path)
        try:
            if not code and solve_attempts > 0:
                code, history = solve_public(client, kit, unit, runner=runner, image=image,
                                             work_dir=out_dir / "solve" / unit, attempts=solve_attempts)
                write_json(out_dir / "solve" / unit / "history.json", history)
                if code:
                    new_solutions[unit] = code
            if not code:
                log(f"lessons: {unit} has no verified solution; skipped")
                continue
            card = write_lesson(client, kit, runs_by_unit[unit], rec, code)
        except LLMError as exc:
            log(f"lessons: {unit}: teacher failed ({exc})")
            continue
        cards[unit] = write_text(out_dir / f"{unit}.md", card)
        log(f"lesson card {unit} ({rec.category}): {len(card.splitlines())} lines")
    return cards, new_solutions

