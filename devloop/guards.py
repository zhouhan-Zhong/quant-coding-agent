"""Guards for everything that may enter the memory library.

A document folded from a public unit's lesson must teach a *method*, not that unit's answer: it may not name the
unit, use file names that identify it, or repeat numbers from its checker or task text; it may never carry a
canary.  It must keep the library's format (front matter, the four playbook sections, length limits) and its new
snippets must run.  A solution must compute the checker's numbers, not paste them, and must not read the checker.
Only *new* content is judged, so an update is not blamed for what the document already said.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .runner import run_snippet
from .util import GUID_RE, Kit

PLAYBOOK_SECTIONS = ("when this applies", "method and formulas", "conventions and pitfalls", "tested snippet")
MAX_LINES = 140
MAX_SNIPPET_LINES = 45
GENERIC_FILES = {"results.json", "solution.json", "summary.json", "output.json", "outputs.json", "params.json",
                 "metrics.json", "report.json", "glossary.json", "config.json", "data.json", "results.csv",
                 "output.csv", "data.csv", "prices.csv", "returns.csv", "trades.csv", "positions.csv", "signals.csv",
                 "report.html", "instruction.md", "card.toml", "main.py", "solution.py", "script.py"}
COMMON_NUMBERS = {"252", "365", "360", "365.25", "0.975", "0.995", "0.999", "1.96", "2.326", "2.33", "1.645", "1.65",
                  "2.576", "3.14159", "3.1416", "2.71828", "0.5772", "126", "504", "756", "1260", "3600", "86400",
                  "1440", "8760", "0.6745", "1.4826", "0.3989", "0.7979", "1.253"}
_NUM_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w.])")
_FILE_RE = re.compile(r"[\w\-]+\.(?:json|csv|tsv|parquet|pqt|txt|html|xml|xlsx|xls|png|pdf|npy|npz|pkl|zip|md)\b", re.I)


@dataclass
class Finding:
    kind: str          # hygiene | format | leakage | snippet | solution
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


def hygiene(text: str) -> list[Finding]:
    out = []
    if GUID_RE.search(text):
        out.append(Finding("hygiene", "contains a GUID"))
    if re.search(r"canary", text, re.I):
        out.append(Finding("hygiene", "contains the word 'canary'"))
    if re.search(r"BENCHMARK DATA", text, re.I):
        out.append(Finding("hygiene", "contains a benchmark banner line"))
    return out


def front_matter(text: str) -> dict | None:
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end < 0:
        return None
    fm = {}
    for ln in text[3:end].splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            fm[k.strip().lower()] = v.strip().strip("\"'")
    return fm


def python_blocks(text: str) -> list[str]:
    return [b for b in re.findall(r"```(?:python|py)[ \t]*\n(.*?)```", text, re.S) if b.strip()]


def section(text: str, name: str) -> str:
    """Body of the first `## <name...>` section."""
    m = re.search(r"^##\s+" + re.escape(name) + r".*$", text, re.M | re.I)
    if not m:
        return ""
    nxt = re.search(r"^##\s+", text[m.end():], re.M)
    return text[m.end(): m.end() + nxt.start()] if nxt else text[m.end():]


def doc_format(rel_path: str, text: str, allowed_categories: set[str], original_lines: int = 0) -> list[Finding]:
    rel = rel_path.replace("\\", "/")
    m = re.fullmatch(r"(playbooks|notes)/([a-z0-9][a-z0-9\-]*)\.md", rel)
    if not m:
        return [Finding("format", f"path must be playbooks/<slug>.md or notes/<slug>.md (lowercase, hyphens): {rel}")]
    kind = m.group(1)
    out: list[Finding] = []
    fm = front_matter(text)
    if fm is None:
        out.append(Finding("format", "missing front matter (--- title/category/tags/summary ---)"))
    else:
        for key in ("title", "category", "tags", "summary"):
            if not fm.get(key):
                out.append(Finding("format", f"front matter lacks '{key}'"))
        cat = fm.get("category", "")
        if cat and cat not in allowed_categories:
            out.append(Finding("format", f"category '{cat}' is not in the library vocabulary"))
        if fm.get("pinned", "").lower() in ("1", "true", "yes"):
            out.append(Finding("format", "pinned notes are edited by hand only: they reach every task, the source tasks included"))
    if kind == "playbooks":
        heads = [h.strip().lower() for h in re.findall(r"^##\s+(.+)$", text, re.M)]
        for sec in PLAYBOOK_SECTIONS:
            if not any(h.startswith(sec) for h in heads):
                out.append(Finding("format", f"missing section '## {sec.capitalize()}'"))
        if any(h.startswith("tested snippet") for h in heads) and not python_blocks(section(text, "Tested snippet")):
            out.append(Finding("format", "the Tested snippet section has no ```python block"))
    n = len(text.splitlines())
    if n > max(MAX_LINES, original_lines):
        out.append(Finding("format", f"{n} lines (limit {max(MAX_LINES, original_lines)})"))
    for code in python_blocks(text):
        if len(code.splitlines()) > MAX_SNIPPET_LINES:
            out.append(Finding("format", f"a snippet has {len(code.splitlines())} lines (limit {MAX_SNIPPET_LINES})"))
    return out


def _sig_digits(tok: str) -> int:
    t = tok.lstrip("+-").lower().split("e")[0]
    if "." in t:
        digits = t.replace(".", "").lstrip("0")
    else:
        digits = t.lstrip("0").rstrip("0")
    return len(digits)


def numbers(text: str, min_sig: int = 3) -> set[str]:
    """Numeric literals with at least `min_sig` significant digits, normalised (0.0326 and 3.26e-2 are equal)."""
    out = set()
    for tok in _NUM_RE.findall(text):
        if _sig_digits(tok) >= min_sig:
            try:
                out.add(f"{abs(float(tok)):.10g}")
            except ValueError:
                pass
    return out


_COMMON = {f"{float(x):.10g}" for x in COMMON_NUMBERS}


def unit_file_names(kit: Kit, unit: str) -> set[str]:
    names = {p.name for p in kit.data_files(unit)}
    names |= {m.group(0) for m in _FILE_RE.finditer(kit.instruction(unit))}
    names |= {m.group(0) for m in _FILE_RE.finditer(kit.checker(unit))}
    return {n for n in names if n.lower() not in GENERIC_FILES}


def leakage(text: str, sources: list[str], kit: Kit | None, original: str = "") -> list[Finding]:
    out: list[Finding] = []
    new_ids = set(re.findall(r"\bt1-[a-z0-9][a-z0-9\-]*", text, re.I)) - set(re.findall(r"\bt1-[a-z0-9][a-z0-9\-]*", original, re.I))
    if new_ids:
        out.append(Finding("leakage", f"names public unit(s) {sorted(new_ids)[:5]}"))
    if kit is None:
        return out
    new_numbers = numbers(text) - numbers(original) - _COMMON
    for unit in sources:
        if not kit.unit_dir(unit).is_dir():
            continue
        hits = sorted(n for n in unit_file_names(kit, unit)
                      if re.search(r"(?<![\w.\-])" + re.escape(n) + r"(?![\w\-])", text)
                      and not re.search(r"(?<![\w.\-])" + re.escape(n) + r"(?![\w\-])", original))
        if hits:
            out.append(Finding("leakage", f"file names that identify the source task: {hits[:6]}"))
        task_numbers = numbers(kit.checker(unit)) | numbers(kit.instruction(unit))
        shared = sorted(new_numbers & task_numbers)
        if shared:
            out.append(Finding("leakage", f"numbers that also appear in a source task's checker or text: {shared[:8]}"))
    return out


def snippet_findings(text: str, original: str, *, runner: str, image: str = "") -> list[Finding]:
    """Run every python block that is new or changed compared with the original document."""
    old = {b.strip() for b in python_blocks(original)}
    out = []
    for code in python_blocks(text):
        if code.strip() in old:
            continue
        res = run_snippet(code, runner=runner, image=image)
        if res.rc != 0:
            tail = " | ".join(res.log.splitlines()[-4:])[:400]
            out.append(Finding("snippet", f"a new snippet does not run (exit {res.rc}): {tail}"))
    return out


def check_doc(rel_path: str, text: str, *, original: str, sources: list[str], kit: Kit | None,
              allowed_categories: set[str], runner: str | None = None, image: str = "") -> list[Finding]:
    found = hygiene(text) + doc_format(rel_path, text, allowed_categories, len(original.splitlines()))
    if original and (front_matter(original) or {}).get("pinned", "").lower() in ("1", "true", "yes"):
        found.append(Finding("format", "the original is a pinned note; pinned notes are edited by hand only"))
    found += leakage(text, sources, kit, original)
    if runner and not any(f.kind in ("hygiene", "format") for f in found):
        found += snippet_findings(text, original, runner=runner, image=image)
    return found


def check_solution(code: str, unit: str, kit: Kit) -> list[Finding]:
    found = hygiene(code)
    if not code.lstrip().startswith("# memory-solution"):
        found.append(Finding("solution", "the first line must be '# memory-solution'"))
    if re.search(r"(?<![\w])checks[/\\]|test_outputs|reference_data", code):
        found.append(Finding("solution", "the script refers to the checker or its reference data"))
    pasted = sorted((numbers(code) & numbers(kit.checker(unit))) - numbers(kit.instruction(unit)) - _COMMON)
    if pasted:
        found.append(Finding("solution", f"numbers taken from the checker instead of computed: {pasted[:8]}"))
    return found
