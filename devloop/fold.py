"""Fold the general part of lesson cards into the memory library (the workflow of devloop/guides/FOLD_GUIDE.md).

Lessons are grouped by the playbooks they retrieve, each group goes to the teacher with the current text of those
playbooks, and every proposed file passes the guards (format, hygiene, leakage against the source units, new
snippets run); a file that fails gets one revision round.  The candidate library is a copy of the base library with
the accepted files, any new verified solutions, refreshed fingerprints, and pb_exclude.json extended so that a
unit never sees a document derived from its own lesson (with AGENT_PB_EXCLUDE=1 that keeps the A/B held out).
"""
from __future__ import annotations

import filecmp
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .guards import Finding, check_doc
from .llm import ChatClient, LLMError, extract_files, prompt
from .util import Kit, clip, log, read_json, read_text, scrub, write_text


@dataclass
class Edit:
    path: str                       # relative to the memory root: playbooks/<slug>.md or notes/<slug>.md
    action: str                     # create | update
    content: str
    sources: list[str]
    original: str = ""
    findings: list[Finding] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return Path(self.path).stem


def allowed_categories(index) -> set[str]:
    import memory as agent_memory  # noqa: PLC0415
    return set(agent_memory.LIBRARY_CATEGORIES) | {"meta"} | {d.category for d in index.docs if d.category}


def group_lessons(lessons: dict[str, str], index, k: int = 3, max_per_group: int = 4) -> list[tuple[list[str], list[str]]]:
    """[(units, target doc ids)]: lessons that retrieve the same leading playbook share one fold request."""
    targets: dict[str, list[str]] = {}
    for unit, card in lessons.items():
        _, plays = index.retrieve(card, unit_id="", task_dir_name="", n_solutions=0, n_playbooks=k)
        targets[unit] = [d.id for d, _ in plays if not d.pinned][:k]
    groups: dict[str, list[str]] = {}
    for unit, docs in targets.items():
        groups.setdefault(docs[0] if docs else "", []).append(unit)
    out = []
    for lead, units in sorted(groups.items()):
        for i in range(0, len(units), max_per_group):
            chunk = units[i:i + max_per_group]
            docs: list[str] = []
            for u in chunk:
                docs += [d for d in targets[u] if d not in docs]
            out.append((chunk, docs[:4]))
    return out


def _fold_request(lessons: dict[str, str], units: list[str], docs: list[str], index, cats: set[str],
                  overlay: dict[str, "Edit"]) -> list[dict]:
    by_id = {d.id: d for d in index.docs}
    parts = ["LESSON CARDS:"]
    for u in units:
        parts.append(f"#### LESSON {u}\n{clip(scrub(lessons[u]), 6000)}")
    parts.append("CURRENT FILES (the most relevant playbooks for these lessons; update them, or create new ones):")
    for d in docs:
        doc = by_id.get(d)
        if doc is None:
            continue
        rel = f"{'notes' if doc.kind == 'note' else 'playbooks'}/{d}.md"
        text = overlay[rel].content if rel in overlay else doc.text
        parts.append(f"### CURRENT FILE: {rel}\n````markdown\n{text.rstrip()}\n````")
    catalog = [f"- {d.id}: {d.title}" for d in index.docs if d.kind == "playbook"]
    catalog += [f"- {Path(rel).stem}: (new in this round)" for rel, e in overlay.items() if e.action == "create"]
    parts.append("OTHER PLAYBOOKS IN THE LIBRARY (slug: title):\n" + "\n".join(catalog))
    system = prompt("fold").replace("{categories}", ", ".join(sorted(cats)))
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]


def _normalise(block: dict, units: list[str], memory_dir: Path, overlay: dict[str, "Edit"]) -> Edit:
    rel = block["path"].replace("\\", "/")
    for prefix in ("agent/memory/", "memory/", "./"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
    if rel in overlay:                       # already changed by an earlier group: judge only what is new now
        original, action = overlay[rel].content, overlay[rel].action
    else:
        current = memory_dir / rel
        original = read_text(current) if current.is_file() else ""
        action = "update" if original else "create"
    sources = [s for s in block.get("sources", []) if s in units] or list(units)
    content = block["content"] if block["content"].endswith("\n") else block["content"] + "\n"
    return Edit(path=rel, action=action, content=content, sources=sources, original=original)


def fold(lessons: dict[str, str], memory_dir: Path, client: ChatClient, kit: Kit | None, *, runner: str | None,
         image: str = "", revisions: int = 1, log_dir: Path | None = None) -> tuple[list[Edit], list[Edit]]:
    """Returns (accepted edits, rejected edits with their findings).  Groups run one after another and each sees the
    files accepted for the groups before it."""
    import memory as agent_memory  # noqa: PLC0415
    memory_dir = Path(memory_dir)
    index = agent_memory.MemoryIndex(memory_dir)
    cats = allowed_categories(index)
    accepted: dict[str, Edit] = {}
    rejected: list[Edit] = []
    for gi, (units, docs) in enumerate(group_lessons(lessons, index), 1):
        msgs = _fold_request(lessons, units, docs, index, cats, accepted)
        try:
            answer = client.chat(msgs, max_tokens=4000)
        except LLMError as exc:
            log(f"fold group {gi}: teacher failed ({exc})")
            continue
        if log_dir:
            write_text(Path(log_dir) / f"fold-{gi}.md", scrub(answer))
        pending = [_normalise(b, units, memory_dir, accepted) for b in extract_files(answer)]
        n_ok = 0
        for attempt in range(revisions + 1):
            bad: list[Edit] = []
            for e in pending:
                e.findings = check_doc(e.path, e.content, original=e.original, sources=e.sources, kit=kit,
                                       allowed_categories=cats, runner=runner, image=image)
                if e.findings:
                    bad.append(e)
                    continue
                if e.path in accepted:       # provenance accumulates: every source unit is excluded from this doc
                    e.sources = sorted(set(accepted[e.path].sources) | set(e.sources))
                    e.original = accepted[e.path].original if accepted[e.path].action == "update" else ""
                accepted[e.path] = e
                n_ok += 1
            if not bad or attempt == revisions:
                rejected += bad
                break
            notes = "\n".join(f"- {e.path}: " + "; ".join(str(f) for f in e.findings) for e in bad)
            msgs = msgs + [{"role": "assistant", "content": answer},
                           {"role": "user", "content": "These files were rejected:\n" + notes +
                            "\nReply with corrected FILE blocks for these files only (same format)."}]
            try:
                answer = client.chat(msgs, max_tokens=4000)
            except LLMError as exc:
                log(f"fold group {gi}: revision failed ({exc})")
                rejected += bad
                break
            wanted = {e.path for e in bad}
            pending = [e for e in (_normalise(b, units, memory_dir, accepted) for b in extract_files(answer))
                       if e.path in wanted]
            rejected += [e for e in bad if e.path not in {x.path for x in pending}]
        log(f"fold group {gi}: lessons {', '.join(units)} -> {n_ok} file(s) accepted; {len(rejected)} rejected so far")
    return list(accepted.values()), rejected


def build_candidate(base_memory: Path, out_memory: Path, edits: list[Edit], new_solutions: dict[str, str],
                    kit: Kit | None) -> Path:
    """Copy the base library, apply the accepted files and new solutions, extend pb_exclude.json."""
    import memory as agent_memory  # noqa: PLC0415
    base_memory, out_memory = Path(base_memory), Path(out_memory)
    if out_memory.exists():
        shutil.rmtree(out_memory)
    shutil.copytree(base_memory, out_memory)
    for e in edits:
        write_text(out_memory / e.path, e.content)
    for unit, code in new_solutions.items():
        write_text(out_memory / "solutions" / f"{unit}.py", code)
    if new_solutions and kit is not None:
        n = agent_memory.build_fingerprints(kit.root / "units", out_memory / "fingerprints.json")
        log(f"fingerprints rebuilt for {n} units")
    excl = read_json(out_memory / "pb_exclude.json", {}) or {}
    changed = False
    for e in edits:
        for unit in e.sources:
            ids = excl.setdefault(unit, [])
            if e.doc_id not in ids:
                ids.append(e.doc_id)
                changed = True
    if changed:   # same layout as the shipped file (indent 1, no trailing newline): the diff shows only real changes
        (out_memory / "pb_exclude.json").write_text(json.dumps(excl, indent=1, ensure_ascii=False), encoding="utf-8")
    return out_memory


def changed_files(candidate: Path, base: Path) -> list[str]:
    """Relative paths that differ between two library copies (new or modified)."""
    out = []
    for p in sorted(Path(candidate).rglob("*")):
        if p.is_file():
            rel = p.relative_to(candidate)
            q = Path(base) / rel
            if not q.is_file() or not filecmp.cmp(p, q, shallow=False):
                out.append(str(rel).replace("\\", "/"))
    return out


def apply_candidate(candidate: Path, target: Path, base: Path) -> list[str]:
    """Copy into `target` (normally agent/memory) every file that the candidate changed relative to `base`."""
    files = changed_files(candidate, base)
    for rel in files:
        dst = Path(target) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(candidate) / rel, dst)
    return files
