"""Memory library: verified solutions to *other* public units + general playbooks/notes, with retrieval.

Layout (shipped inside the image under /agent/memory):
    solutions/<unit-id>.py      verified reference solutions (header: `# memory-solution`, unit/category/tags/deliverables)
    playbooks/<topic>.md        general method cards: formulas, conventions, pitfalls, tested snippets
    notes/<topic>.md            general notes (grader taste, output hygiene, data quirks)
    fingerprints.json           {unit-id: [sampled shingle hashes of that public unit's instruction]} for same-unit detection

Compliance (competition rule 8): a stored solution may be shown to the model only when solving a *different*
unit.  Exclusion is by id (card id / task directory name) AND by content (Jaccard similarity of the current
instruction's word-shingles against the stored fingerprint of every solution's unit).  Playbooks and notes are
general-purpose material and are always allowed.  Nothing in the library may contain a canary GUID: documents
containing a UUID are dropped at load time.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[_\-][a-z0-9]+)*")
_STOP = set("""
the a an and or of to in for with on by from as at is are be this that these those it its into over under per
each all any only not no than then when where which who while will would should must may can could use using used
you your we our if else also both same other such via etc i e g ie eg file files output outputs input inputs
step steps task tasks compute computed calculate calculated report reported save saved write written create created
column columns row rows value values result results data given following above below section json csv directory
""".split())
_SHINGLE_N = 5
_SHINGLE_SAMPLE_MOD = 4          # keep shingles whose hash % MOD == 0 (consistent sampling, ~25 %)
SAME_UNIT_JACCARD = 0.75         # >= this => treated as the same unit (content match) and excluded; distinct public units score < 0.3, near-duplicate siblings ~0.67

# Library category vocabulary (solution headers / playbook front matter) and how task-card categories map onto it.
# Hidden units set `[metadata] category` to one of the organizers' ten ids (kit/docs/CATEGORIES.md); public (migrated)
# cards use free-form labels, so the map also carries the common synonyms.  "cross-domain" maps to nothing.
LIBRARY_CATEGORIES = ("derivatives-pricing", "volatility", "rates-curves", "credit", "backtest-factors", "portfolio", "risk",
                      "microstructure", "fx-commodities-crypto", "text-filings", "data-engineering", "numerics")
_CATEGORY_MAP: dict[str, tuple[str, ...]] = {
    "derivatives-pricing": ("derivatives-pricing", "volatility"), "derivatives": ("derivatives-pricing", "volatility"),
    "pricing": ("derivatives-pricing",), "option": ("derivatives-pricing",), "volatility": ("volatility", "derivatives-pricing"),
    "stochastic-processes": ("derivatives-pricing",), "interest-rate-derivatives": ("rates-curves", "derivatives-pricing"),
    "fixed-income": ("rates-curves",), "rates": ("rates-curves",), "rates-curves": ("rates-curves",), "bond": ("rates-curves",),
    "cross-currency-rates": ("rates-curves", "fx-commodities-crypto"),
    "credit": ("credit",), "credit-risk": ("credit",), "credit-analysis": ("credit",),
    "factor-research": ("backtest-factors", "portfolio"), "factor-models": ("backtest-factors", "portfolio"),
    "backtesting": ("backtest-factors",), "backtest-factors": ("backtest-factors",), "strategy": ("backtest-factors",),
    "cross-sectional-strategies": ("backtest-factors",), "predictive-alpha-modeling": ("backtest-factors",),
    "performance-attribution": ("backtest-factors", "portfolio"), "portfolio": ("portfolio",), "portfolio-analysis": ("portfolio",),
    "risk-management": ("risk", "portfolio"), "risk": ("risk",), "risk-modeling": ("risk",), "extreme-value-theory": ("risk",),
    "dependence-modeling": ("risk",), "statistical-analysis": ("risk", "volatility"),
    "microstructure": ("microstructure",), "execution": ("microstructure",), "market-microstructure": ("microstructure",),
    "fx": ("fx-commodities-crypto",), "fx-pricing": ("fx-commodities-crypto", "derivatives-pricing"), "fx-strategy": ("fx-commodities-crypto", "backtest-factors"),
    "crypto": ("fx-commodities-crypto",), "commodities": ("fx-commodities-crypto",),
    "nlp-on-finance": ("text-filings",), "nlp": ("text-filings",), "text-filings": ("text-filings",), "fixed-income-nlp": ("text-filings", "rates-curves"),
    "event-driven-analysis": ("text-filings", "backtest-factors"), "data-processing": ("data-engineering",), "data-engineering": ("data-engineering",),
    "tool-using": ("data-engineering",),
}


def normalize_category(raw) -> tuple[str, ...]:
    """Map a task card's category label(s) to library categories (empty for cross-domain/unknown)."""
    if isinstance(raw, (list, tuple)):
        raw = ",".join(str(x) for x in raw)
    out: list[str] = []
    for part in re.split(r"[,;/|]+", str(raw or "").lower()):
        key = part.strip().replace("_", "-").replace(" ", "-")
        if not key or key == "cross-domain":
            continue
        for cat in _CATEGORY_MAP.get(key, ()):
            if cat not in out:
                out.append(cat)
    return tuple(out)



def tokenize(text: str) -> list[str]:
    toks = []
    for t in _TOKEN_RE.findall(text.lower()):
        if len(t) < 2 or t in _STOP or t.isdigit():
            continue
        # split snake/kebab identifiers into parts as well as keeping the whole identifier
        parts = re.split(r"[_\-]", t)
        if len(parts) > 1:
            toks.append(t)
            toks.extend(p for p in parts if len(p) >= 2 and p not in _STOP and not p.isdigit())
        else:
            toks.append(t)
    return [_stem(t) for t in toks]


def _stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def shingles(text: str) -> set[int]:
    """Sampled hashed word 5-gram shingles of a text (canary lines removed first)."""
    lines = [ln for ln in text.splitlines() if not re.search(r"canary|BENCHMARK DATA", ln, re.I)]
    words = [w for w in _TOKEN_RE.findall(" ".join(lines).lower()) if w not in _STOP]
    out: set[int] = set()
    for i in range(max(0, len(words) - _SHINGLE_N + 1)):
        h = int.from_bytes(hashlib.blake2b(" ".join(words[i:i + _SHINGLE_N]).encode(), digest_size=8).digest(), "big")
        if h % _SHINGLE_SAMPLE_MOD == 0:
            out.add(h)
    return out


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / float(len(a) + len(b) - inter)


@dataclass
class MemoryDoc:
    id: str
    kind: str                     # "solution" | "playbook" | "note"
    path: Path
    title: str = ""
    category: str = ""
    tags: list[str] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    unit_id: str = ""             # solutions only
    header: str = ""              # solutions: the leading comment block; playbooks: the summary line(s)
    text: str = ""                # full text
    tf: Counter = field(default_factory=Counter)
    length: int = 0
    pinned: bool = False          # always shown (short essentials), never listed in the TOC
    triggers: list = field(default_factory=list)   # 'file:<glob>' / 'text:<phrase>' patterns that force this doc in (AGENT_MEM_TRIGGERS)

    def toc_line(self) -> str:
        bits = [self.title or self.id]
        if self.category:
            bits.append(self.category)
        if self.tags:
            bits.append("tags: " + ", ".join(self.tags[:9]))
        if self.deliverables:
            bits.append("outputs: " + ", ".join(self.deliverables[:6]))
        return " | ".join(bits)


def _parse_solution(path: Path, text: str) -> MemoryDoc:
    doc = MemoryDoc(id=path.stem, kind="solution", path=path, text=text)
    header_lines: list[str] = []
    for ln in text.splitlines():
        if ln.startswith("#"):
            header_lines.append(ln)
        elif ln.strip() == "":
            continue
        else:
            break
    doc.header = "\n".join(header_lines)
    for ln in header_lines:
        body = ln.lstrip("# ").strip()
        m = re.match(r"unit:\s*([^|]+)\|\s*category:\s*([^|]+)\|\s*tags:\s*(.*)$", body)
        if m:
            doc.unit_id = m.group(1).strip()
            doc.category = m.group(2).strip()
            doc.tags = [t.strip().lower() for t in m.group(3).split(",") if t.strip()]
            continue
        m = re.match(r"deliverables:\s*(.*)$", body)
        if m:
            doc.deliverables = [d.strip() for d in m.group(1).split(",") if d.strip()]
    doc.title = doc.unit_id or path.stem
    return doc


def _parse_markdown(path: Path, text: str, kind: str) -> MemoryDoc:
    doc = MemoryDoc(id=path.stem, kind=kind, path=path, text=text)
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            fm = text[3:end]
            body = text[end + 4:]
            for ln in fm.splitlines():
                if ":" not in ln:
                    continue
                k, v = ln.split(":", 1)
                k, v = k.strip().lower(), v.strip().strip("[]")
                if k == "title":
                    doc.title = v.strip("\"'")
                elif k == "tags":
                    doc.tags = [t.strip().strip("\"'").lower() for t in v.split(",") if t.strip()]
                elif k == "category":
                    doc.category = v.strip("\"'")
                elif k in ("summary", "description"):
                    doc.header = v.strip("\"'")
                elif k == "pinned":
                    doc.pinned = v.strip().lower() in ("1", "true", "yes")
                elif k == "triggers":
                    doc.triggers = [t.strip().strip("\"'").lower() for t in v.split(",") if t.strip()]
    if not doc.title:
        m = re.search(r"^#\s+(.+)$", body, re.M)
        doc.title = m.group(1).strip() if m else path.stem
    if not doc.header:
        # first non-heading paragraph line as the summary
        for ln in body.splitlines():
            s = ln.strip()
            if s and not s.startswith("#"):
                doc.header = s[:200]
                break
    return doc


class MemoryIndex:
    def __init__(self, root: Path, log=None) -> None:
        self.root = Path(root)
        self.docs: list[MemoryDoc] = []
        self.fingerprints: dict[str, set[int]] = {}
        self.df: Counter = Counter()
        self.avg_len = 1.0
        self.log = log or (lambda m: None)
        self.dropped: list[str] = []
        # AGENT_MEMORY_V2=0 hides the second-generation documents listed in v2_docs.json (A/B control arm).
        self.hidden_v2: set[str] = set()
        if os.environ.get("AGENT_MEMORY_V2", "1").strip().lower() in ("0", "false", "no"):
            try:
                self.hidden_v2 = set(json.loads((Path(root) / "v2_docs.json").read_text(encoding="utf-8")))
            except Exception:
                self.hidden_v2 = set()
        # AGENT_MEMORY_V3=0 hides the third-generation convention playbooks listed in v3_docs.json.
        if os.environ.get("AGENT_MEMORY_V3", "1").strip().lower() in ("0", "false", "no"):
            try:
                self.hidden_v2 |= set(json.loads((Path(root) / "v3_docs.json").read_text(encoding="utf-8")))
            except Exception:
                pass
        # AGENT_MEMORY_V4=0 hides the fourth-generation textbook playbooks listed in v4_docs.json (topics with no public unit).
        if os.environ.get("AGENT_MEMORY_V4", "1").strip().lower() in ("0", "false", "no"):
            try:
                self.hidden_v2 |= set(json.loads((Path(root) / "v4_docs.json").read_text(encoding="utf-8")))
            except Exception:
                pass
        self._load()
        # AGENT_PB_EXCLUDE=1: playbooks/notes derived mainly from a unit's own lessons are hidden when solving it
        # (pb_exclude.json: {unit_id: [doc ids]}), the same rule that solutions already follow.
        self.pb_exclude: dict[str, list[str]] = {}
        if os.environ.get("AGENT_PB_EXCLUDE", "0").strip().lower() in ("1", "true", "yes"):
            try:
                self.pb_exclude = json.loads((Path(root) / "pb_exclude.json").read_text(encoding="utf-8"))
            except Exception as exc:
                if log:
                    log(f"memory: pb_exclude.json unreadable ({exc})")

    # -- loading -----------------------------------------------------------------------------
    def _load(self) -> None:
        if not self.root.is_dir():
            return
        for sub, kind, pattern in (("solutions", "solution", "*.py"), ("playbooks", "playbook", "*.md"), ("notes", "note", "*.md")):
            d = self.root / sub
            if not d.is_dir():
                continue
            for p in sorted(d.glob(pattern)):
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if _UUID_RE.search(text):
                    # never let a GUID travel from the library into a prompt (and from there into an output)
                    self.dropped.append(str(p))
                    continue
                if kind == "solution" and "# memory-solution" not in text[:200]:
                    continue
                if p.stem in self.hidden_v2:
                    continue
                doc = _parse_solution(p, text) if kind == "solution" else _parse_markdown(p, text, kind)
                self._index_doc(doc)
                self.docs.append(doc)
        fp = self.root / "fingerprints.json"
        if fp.is_file():
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
                self.fingerprints = {k: set(int(x) for x in v) for k, v in raw.items()}
            except Exception:
                self.fingerprints = {}
        n = len(self.docs)
        if n:
            self.avg_len = sum(d.length for d in self.docs) / n
        if self.dropped:
            self.log(f"memory: dropped {len(self.dropped)} document(s) containing a GUID")
        if self.hidden_v2:
            self.log(f"memory: AGENT_MEMORY_V2/V3/V4=0 hides {len(self.hidden_v2)} document(s)")

    def _index_doc(self, doc: MemoryDoc) -> None:
        if doc.kind == "solution":
            meta = " ".join([doc.unit_id.replace("t1-", " "), doc.category, " ".join(doc.tags), " ".join(doc.deliverables), doc.header])
            toks = tokenize(meta) * 3 + tokenize(doc.text)
        else:
            meta = " ".join([doc.title, doc.category, " ".join(doc.tags), doc.header])
            toks = tokenize(meta) * 3 + tokenize(doc.text)
        tf = Counter(toks)
        # cap term frequency so long scripts do not dominate by sheer size
        doc.tf = Counter({t: min(c, 4) for t, c in tf.items()})
        doc.length = max(1, sum(doc.tf.values()))
        for t in doc.tf:
            self.df[t] += 1

    # -- retrieval ---------------------------------------------------------------------------
    def _bm25(self, query: Counter, doc: MemoryDoc, k1: float = 1.5, b: float = 0.75) -> float:
        n = len(self.docs)
        score = 0.0
        for term, qw in query.items():
            f = doc.tf.get(term)
            if not f:
                continue
            idf = math.log(1.0 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
            score += qw * idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * doc.length / self.avg_len))
        return score

    @staticmethod
    def build_query(instruction: str, deliverables: Iterable[str] = (), data_names: Iterable[str] = (), file_types: bool = False) -> Counter:
        q: Counter = Counter()
        body = Counter(tokenize(instruction))
        for t, c in body.items():
            q[t] += min(3, c)
        headings = " ".join(m.group(1) for m in re.finditer(r"^#{1,6}\s+(.+)$", instruction, re.M))
        for t in tokenize(headings):
            q[t] += 3
        bold = " ".join(re.findall(r"\*\*([^*]{2,60})\*\*", instruction))
        for t in tokenize(bold):
            q[t] += 1
        for name in deliverables:
            for t in tokenize(Path(name).stem.replace("_", " ") + " " + Path(name).stem):
                q[t] += 3
        for name in data_names:
            for t in tokenize(Path(name).stem.replace("_", " ")):
                q[t] += 2
        if file_types:  # formats of the shipped inputs (xml/html/tsv/zip ...) hint at the parsing recipe needed
            exts = Counter(Path(n).suffix.lower().lstrip(".") for n in data_names)
            for ext, c in exts.items():
                if ext and ext not in ("csv", "json") and len(ext) <= 7:
                    q[_stem(ext)] += 2
        return q

    def same_unit(self, doc: MemoryDoc, unit_id: str, task_dir_name: str, instr_shingles: set[int]) -> tuple[bool, str]:
        if doc.kind != "solution":
            return False, ""
        ids = {unit_id.strip().lower(), task_dir_name.strip().lower()}
        if doc.unit_id.strip().lower() in ids or doc.id.strip().lower() in ids:
            return True, "id"
        fp = self.fingerprints.get(doc.unit_id) or self.fingerprints.get(doc.id)
        if fp and instr_shingles:
            j = jaccard(fp, instr_shingles)
            if j >= SAME_UNIT_JACCARD:
                return True, f"content (jaccard={j:.2f})"
        return False, ""

    def retrieve(self, instruction: str, *, unit_id: str, task_dir_name: str, deliverables: Iterable[str] = (),
                 data_names: Iterable[str] = (), n_solutions: int = 6, n_playbooks: int = 6,
                 category_hint: Iterable[str] = (), reserve: int = 2) -> tuple[list[tuple[MemoryDoc, float]], list[tuple[MemoryDoc, float]]]:
        """BM25 candidates; with `category_hint` (library categories from the task card, AGENT_RAG2) up to `reserve`
        slots of each list are guaranteed to documents of those categories when any score at least 0.1 x the leader."""
        if not self.docs:
            return [], []
        hint = tuple(category_hint or ())
        q = self.build_query(instruction, deliverables, data_names, file_types=bool(hint))
        sh = shingles(instruction)
        sols: list[tuple[MemoryDoc, float]] = []
        plays: list[tuple[MemoryDoc, float]] = []
        for doc in self.docs:
            if doc.pinned:
                continue
            same, why = self.same_unit(doc, unit_id, task_dir_name, sh)
            if same:
                self.log(f"memory: excluding solution {doc.id} (same unit by {why})")
                continue
            if doc.kind != "solution" and (doc.id in self.pb_exclude.get(unit_id, ()) or doc.id in self.pb_exclude.get(task_dir_name, ())):
                self.log(f"memory: excluding {doc.kind} {doc.id} (derived from this unit)")
                continue
            s = self._bm25(q, doc)
            if s <= 0:
                continue
            (sols if doc.kind == "solution" else plays).append((doc, s))
        sols.sort(key=lambda x: -x[1])
        plays.sort(key=lambda x: -x[1])
        if hint:
            sols = self._reserve_category(sols, n_solutions, hint, reserve)
            plays = self._reserve_category(plays, n_playbooks, hint, reserve)
        return sols[:n_solutions], plays[:n_playbooks]

    def _reserve_category(self, ranked: list[tuple[MemoryDoc, float]], n: int, hint: tuple[str, ...], reserve: int) -> list[tuple[MemoryDoc, float]]:
        """Re-rank so that at least `reserve` of the first n entries belong to the hinted categories (if available)."""
        if not ranked or reserve <= 0:
            return ranked
        same = lambda d: (d.category.split("/")[0].strip() in hint)  # noqa: E731
        top, rest = ranked[:n], ranked[n:]
        have = sum(1 for d, _ in top if same(d))
        floor = 0.1 * ranked[0][1]   # mild guard against promoting near-zero matches
        pool = [(d, sc) for d, sc in rest if same(d) and sc >= floor]
        while have < reserve and pool:
            victim = next((k for k in range(len(top) - 1, -1, -1) if not same(top[k][0])), None)
            if victim is None:
                break
            d, sc = pool.pop(0)
            self.log(f"memory: category slot {d.id} ({d.category}) replaces {top[victim][0].id} ({top[victim][0].category or '-'})")
            rest.insert(0, top[victim])
            top.pop(victim)
            top.append((d, sc))
            have += 1
        return top + rest

    @property
    def pinned(self) -> list[MemoryDoc]:
        return [d for d in self.docs if d.pinned]

    def triggered(self, instruction: str, data_names: Iterable[str], *, unit_id: str, task_dir_name: str) -> list[MemoryDoc]:
        """Documents whose `triggers:` match this task (file:<glob> against the input file names, text:<phrase> against the
        task text, a bare pattern against either); the same-unit and pb_exclude rules still apply."""
        import fnmatch
        names = [str(n).rsplit("/", 1)[-1].lower() for n in data_names]
        text = (instruction or "").lower()
        out: list[MemoryDoc] = []
        sh = shingles(instruction)
        for doc in self.docs:
            if not doc.triggers or doc.pinned:
                continue
            if doc.kind != "solution" and (doc.id in self.pb_exclude.get(unit_id, ()) or doc.id in self.pb_exclude.get(task_dir_name, ())):
                continue
            if self.same_unit(doc, unit_id, task_dir_name, sh)[0]:
                continue
            hit = None
            for t in doc.triggers:
                if t.startswith("file:"):
                    pat = t[5:]
                    if any(fnmatch.fnmatch(n, pat) for n in names):
                        hit = t
                elif t.startswith("text:"):
                    if t[5:] in text:
                        hit = t
                elif any(fnmatch.fnmatch(n, t) for n in names) or t in text:
                    hit = t
                if hit:
                    break
            if hit:
                self.log(f"memory: trigger {hit!r} -> {doc.id}")
                out.append(doc)
        return out

    # -- rendering ---------------------------------------------------------------------------
    @staticmethod
    def toc(sols: list[tuple[MemoryDoc, float]], plays: list[tuple[MemoryDoc, float]]) -> tuple[str, dict[str, MemoryDoc]]:
        """Compact table of contents for the planning prompt; returns (text, label->doc)."""
        lines: list[str] = []
        labels: dict[str, MemoryDoc] = {}
        if sols:
            lines.append("Verified solutions to OTHER tasks (analogues; the method/outputs may transfer):")
            for i, (d, _) in enumerate(sols, 1):
                lab = f"S{i}"
                labels[lab] = d
                lines.append(f"- {lab}: {d.toc_line()}")
        if plays:
            lines.append("Method notes / playbooks:")
            for i, (d, _) in enumerate(plays, 1):
                lab = f"P{i}"
                labels[lab] = d
                summary = d.header[:140] if d.header else ""
                lines.append(f"- {lab}: {d.toc_line()}" + (f" — {summary}" if summary else ""))
        return "\n".join(lines), labels

    @staticmethod
    def render(docs: list[MemoryDoc], char_budget: int) -> str:
        """Render the chosen documents within a character budget.

        Pinned notes are always complete and do not count against the budget.  The budget is split between
        method cards (playbooks/notes: section-aware excerpts that keep "Conventions and pitfalls" and
        "Method and formulas" before the snippet and the "When this applies" blurb) and solutions
        (head + tail excerpt: formulas/loading at the top, output writing at the bottom).
        """
        if not docs:
            return ""
        sols = [d for d in docs if d.kind == "solution"]
        pinned = [d for d in docs if d.kind != "solution" and d.pinned]
        others = [d for d in docs if d.kind != "solution" and not d.pinned]
        parts: list[str] = []
        for d in pinned:  # always complete (they are short by construction)
            txt = _strip_front_matter(d.text).strip()
            parts.append(f"### {d.title}\n{txt}")
        budget = max(0, char_budget)
        if others and sols:
            share_other = int(budget * 0.55)
        elif others:
            share_other = budget
        else:
            share_other = 0
        share_sol = budget - share_other
        if others:
            per_other = max(900, share_other // len(others))
            for d in others:
                txt = excerpt_markdown(d.text, per_other)
                parts.append(f"### {d.kind.upper()}: {d.title}\n{txt}")
        if sols:
            per_sol = max(1500, share_sol // len(sols))
            for d in sols:
                code = excerpt_code(d.text.strip(), per_sol)
                parts.append(f"### SOLUTION TO A DIFFERENT TASK ({d.unit_id}) — adapt the approach, do not copy inputs/outputs/parameters blindly\n```python\n{code}\n```")
        return "\n\n".join(parts)


_SECTION_PRIORITY = (("convention", 0), ("pitfall", 0), ("method", 1), ("formula", 1), ("snippet", 2), ("when this applies", 3))


def _close_fences(text: str) -> str:
    return text + "\n```" if text.count("```") % 2 == 1 else text


def excerpt_markdown(text: str, limit: int) -> str:
    """Section-aware excerpt of a playbook/note: most valuable sections first, original order kept."""
    body = _strip_front_matter(text).strip()
    if len(body) <= limit:
        return body
    chunks = re.split(r"(?m)^(?=## )", body)
    pre = "" if chunks[0].startswith("## ") else chunks[0]
    secs = chunks if not pre else chunks[1:]
    if not secs:
        return _close_fences(_truncate(body, limit))

    def prio(sec: str) -> int:
        head = sec.splitlines()[0].lower() if sec else ""
        for key, p in _SECTION_PRIORITY:
            if key in head:
                return p
        return 2

    budget = limit - len(pre)
    keep: dict[int, str] = {}
    for i in sorted(range(len(secs)), key=lambda k: (prio(secs[k]), k)):
        if budget < 300:
            break
        sec = secs[i]
        if len(sec) <= budget:
            keep[i] = sec
            budget -= len(sec)
        else:
            keep[i] = _close_fences(_truncate(sec, budget)) + "\n"
            budget = 0
    return (pre + "".join(keep[i] for i in range(len(secs)) if i in keep)).strip()


def excerpt_code(code: str, limit: int) -> str:
    """Head + tail excerpt of a solution script (header/formulas/loading first, output writing last)."""
    if len(code) <= limit:
        return code
    head_n = int(limit * 0.68)
    tail_n = max(0, limit - head_n - 60)
    head = code[:head_n]
    nl = head.rfind("\n")
    if nl > head_n // 2:
        head = head[:nl]
    tail = code[-tail_n:] if tail_n else ""
    nl = tail.find("\n")
    if 0 <= nl < len(tail) // 2:
        tail = tail[nl + 1:]
    return head + "\n# ... (middle of the script omitted) ...\n" + tail


def _strip_front_matter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            return text[end + 4:]
    return text


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - 60)]
    nl = cut.rfind("\n")
    if nl > limit // 2:
        cut = cut[:nl]
    return cut + "\n# ... (truncated)"


def parse_plan_memory(plan_json: dict | None, labels: dict[str, MemoryDoc], max_solutions: int = 2, max_playbooks: int = 2) -> list[MemoryDoc]:
    """Read the model's picks (`"memory": {"solutions": ["S1"], "playbooks": ["P2"]}`) from the plan JSON."""
    if not plan_json or not isinstance(plan_json, dict):
        return []
    mem = plan_json.get("memory")
    picks: list[str] = []
    if isinstance(mem, dict):
        for key in ("solutions", "playbooks", "notes"):
            v = mem.get(key)
            if isinstance(v, list):
                picks.extend(str(x) for x in v)
            elif isinstance(v, str):
                picks.append(v)
    elif isinstance(mem, list):
        picks.extend(str(x) for x in mem)
    elif isinstance(mem, str):
        picks.extend(re.findall(r"[SP]\d+", mem))
    chosen: list[MemoryDoc] = []
    n_s = n_p = 0
    for raw in picks:
        lab = raw.strip().upper()
        m = re.search(r"([SP])\s*(\d+)", lab)
        if not m:
            # maybe an id was given instead of a label
            for d in labels.values():
                if d.id.lower() == raw.strip().lower() and d not in chosen:
                    chosen.append(d)
            continue
        lab = m.group(1) + m.group(2)
        d = labels.get(lab)
        if d is None or d in chosen:
            continue
        if d.kind == "solution":
            if n_s >= max_solutions:
                continue
            n_s += 1
        else:
            if n_p >= max_playbooks:
                continue
            n_p += 1
        chosen.append(d)
    return chosen


def default_picks(sols: list[tuple[MemoryDoc, float]], plays: list[tuple[MemoryDoc, float]], n_solutions: int = 1, n_playbooks: int = 2) -> list[MemoryDoc]:
    """Deterministic fallback when the model made no choice: the BM25 leaders (if they score meaningfully)."""
    out: list[MemoryDoc] = []
    if sols:
        top = sols[0][1]
        for d, s in sols[:n_solutions]:
            if s >= 0.5 * top:
                out.append(d)
    if plays:
        top = plays[0][1]
        for d, s in plays[:n_playbooks]:
            if s >= 0.4 * top:
                out.append(d)
    return out


def build_fingerprints(units_dir: Path, out_path: Path) -> int:
    """Build-time helper: fingerprint every public unit's instruction.md for same-unit detection."""
    fps: dict[str, list[int]] = {}
    for unit in sorted(Path(units_dir).iterdir()):
        instr = unit / "instruction.md"
        if not instr.is_file():
            continue
        fps[unit.name] = sorted(shingles(instr.read_text(encoding="utf-8", errors="replace")))
    out_path.write_text(json.dumps(fps, separators=(",", ":")), encoding="utf-8")
    return len(fps)


if __name__ == "__main__":  # small CLI for build/debug: python memory.py fingerprints <units_dir> | query <task_dir>
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "fingerprints":
        n = build_fingerprints(Path(sys.argv[2]), Path(__file__).parent / "memory" / "fingerprints.json")
        print(f"fingerprinted {n} units")
    elif len(sys.argv) >= 3 and sys.argv[1] == "query":
        task = Path(sys.argv[2])
        idx = MemoryIndex(Path(__file__).parent / "memory", log=print)
        instr = (task / "instruction.md").read_text(encoding="utf-8", errors="replace")
        sols, plays = idx.retrieve(instr, unit_id=task.name, task_dir_name=task.name)
        text, labels = idx.toc(sols, plays)
        print(text)
        for d, s in sols:
            print(f"{s:7.2f}  {d.id}")
        for d, s in plays:
            print(f"{s:7.2f}  {d.id}")
    else:
        print(__doc__)
