"""Verify a memory library before it is evaluated (and before it can ever reach the image).

Always: every document loads (none dropped for a GUID), front matter and categories are valid, no canary text,
solution headers name their unit, pb_exclude.json points at existing documents, fingerprints exist, and nothing was
lost relative to the base library.  Optionally (slow): every solution is re-run against its checker (the former
verify_all.sh) and the spec self-check is calibrated on the passing outputs, where it must raise no hard issue (the
former selfcheck_calib.sh).
"""
from __future__ import annotations

import re
from pathlib import Path

from .guards import front_matter, hygiene
from .runner import run_solution
from .util import Kit, log, read_json


def verify_memory(memory_dir: Path, *, base_memory: Path | None = None, kit: Kit | None = None,
                  solutions: bool = False, selfcheck: bool = False, units: list[str] | None = None,
                  runner: str = "docker", image: str = "") -> dict:
    import memory as agent_memory  # noqa: PLC0415
    import selfcheck as agent_selfcheck  # noqa: PLC0415
    memory_dir = Path(memory_dir)
    idx = agent_memory.MemoryIndex(memory_dir)
    problems: list[str] = []
    warnings: list[str] = []
    kinds: dict[str, int] = {}
    for d in idx.docs:
        kinds[d.kind] = kinds.get(d.kind, 0) + 1
    if idx.dropped:
        problems += [f"dropped (contains a GUID): {p}" for p in idx.dropped]
    cats = set(agent_memory.LIBRARY_CATEGORIES) | {"meta"}
    if base_memory is not None:
        cats |= {d.category for d in agent_memory.MemoryIndex(Path(base_memory)).docs if d.category}
    ids = {d.id for d in idx.docs}
    for d in idx.docs:
        for f in hygiene(d.text):
            problems.append(f"{d.kind} {d.id}: {f.detail}")
        if d.kind == "solution":
            if d.unit_id and d.unit_id != d.id:
                problems.append(f"solution {d.id}: header names unit {d.unit_id}")
            if not d.unit_id:
                warnings.append(f"solution {d.id}: header has no 'unit: ... | category: ... | tags: ...' line")
            continue
        fm = front_matter(d.text)
        if fm is None:
            problems.append(f"{d.kind} {d.id}: no front matter")
        elif fm.get("category") and fm["category"] not in cats:
            problems.append(f"{d.kind} {d.id}: unknown category {fm['category']!r}")
    excl = read_json(memory_dir / "pb_exclude.json", {}) or {}
    dangling = sorted({doc for docs in excl.values() for doc in docs if doc not in ids})
    if dangling:
        warnings.append(f"pb_exclude.json names documents that do not exist: {dangling[:8]}")
    if not idx.fingerprints:
        problems.append("fingerprints.json is missing or empty (same-unit exclusion by content would be off)")
    if base_memory is not None:
        base_ids = {d.id for d in agent_memory.MemoryIndex(Path(base_memory)).docs}
        lost = sorted(base_ids - ids)
        if lost:
            problems.append(f"documents lost relative to the base library: {lost[:8]}")
    report = {"memory": str(memory_dir), "docs": kinds, "problems": problems, "warnings": warnings}

    if solutions or selfcheck:
        if kit is None:
            raise ValueError("--solutions/--selfcheck need --kit")
        sol_report: dict[str, dict] = {}
        for d in sorted((d for d in idx.docs if d.kind == "solution"), key=lambda d: d.id):
            if units and d.id not in units:
                continue
            if not kit.unit_dir(d.id).is_dir():
                continue
            partial = bool(re.search(r"^#\s*verification:\s*PARTIAL", d.header, re.M | re.I))
            res = run_solution(d.path, kit, d.id, runner=runner, image=image, timeout=600)
            entry = {"rc": res.rc, "checker_rc": res.checker_rc, "passed": res.passed, "partial": partial}
            if not res.passed and not partial:
                problems.append(f"solution {d.id} no longer passes its checker (exit {res.rc}, checker {res.checker_rc})")
            if selfcheck and res.passed and res.out_dir is not None:
                contract = agent_selfcheck.parse_contract(kit.instruction(d.id))
                hard, _soft = agent_selfcheck.check_outputs(res.out_dir, contract)
                entry["selfcheck_hard"] = hard
                if hard:
                    problems.append(f"self-check false positive on {d.id}: {hard[0][:160]}")
            sol_report[d.id] = entry
            log(f"verify {d.id}: {'pass' if res.passed else ('partial' if partial else 'FAIL')}")
        report["solutions"] = sol_report
    report["ok"] = not problems
    return report
