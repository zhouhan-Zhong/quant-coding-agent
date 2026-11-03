#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Agenthon 2026 - Track 1 (QFBench 2.0) coding agent.

CLI contract (fixed by the competition):

    solve --task-dir /input --out /app/output

The agent reads the unit (instruction.md, card.toml, environment/data/*, and checks/*.py when
they are mounted), asks the House model (OpenAI-compatible, $MODEL_ENDPOINT/v1) for a plan and a
self-contained Python script, runs the script locally with a timeout, validates the deliverables,
feeds errors back for repair, and finally (optionally) asks the model to review its own outputs.

Design constraints honoured here:
  * House budget: <= 25 admitted requests per unit, <= 4000 output tokens per request, n == 1.
  * Wall clock: the Development stage clock allows ~8 minutes per unit on average, so the agent
    self-caps (AGENT_TIME_BUDGET, default 400 s) and never exceeds the card's [agent].timeout_sec.
  * Sandbox: non-root user, read-only root FS, 64 MiB noexec /tmp, 256 PIDs.  Only /tmp and the
    output directory are written to.
  * Never crash: every failure path still exits 0.
  * Canary hygiene: instruction/checker headers are stripped from prompts and outputs are scrubbed.

v2 additions (see MEMORY.md):
  * Memory library (/agent/memory): verified solutions to OTHER public units + general playbooks/notes.
    BM25 pre-selection -> the model picks the analogues it wants in the plan step -> shown before coding.
    Same-unit exclusion by id and by instruction-content fingerprint (competition rule 8).
  * Specification self-checks: JSON templates / CSV column lists / row counts parsed from the task text
    are verified after every clean run and fed back as repair feedback (no checker access needed).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Python >= 3.11
    import tomllib
except ImportError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:  # v2 modules (memory library + specification self-checks); the agent degrades gracefully without them
    from memory import MemoryIndex, default_picks, parse_plan_memory, normalize_category
except Exception:  # pragma: no cover
    MemoryIndex = None  # type: ignore[assignment,misc]
    normalize_category = None  # type: ignore[assignment]
try:
    from selfcheck import check_outputs, parse_contract
except Exception:  # pragma: no cover
    parse_contract = None  # type: ignore[assignment]

AGENT_VERSION = "2.9.16"

# --------------------------------------------------------------------------------------
# Configuration (environment variables; all optional)
# --------------------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass
class Config:
    endpoint: str = ""
    model: str = ""
    token: str = ""
    time_budget: float = 360.0          # seconds the agent allows itself per unit (soft, scaled by the card)
    max_requests: int = 14              # <= 25 admitted requests per unit (House rule; failed requests count too)
    max_output_tokens: int = 4000       # House cap per request
    thinking: bool = False              # chat_template_kwargs.enable_thinking for code calls
    thinking_plan: bool = False         # ... for the planning call
    send_template_kwargs: bool = True   # send chat_template_kwargs at all
    do_plan: bool = True
    do_review: bool = True
    exec_timeout: float = 150.0         # per attempt, seconds (also bounded by remaining time)
    ignore_checks: bool = False         # do not read /input/checks even when mounted
    temperature: float = 0.0
    seed: int = 0
    workdir: str = "/tmp/agent_work"
    request_timeout: float = 240.0
    max_attempts: int = 8
    verbose: bool = True
    memory: bool = True                 # use the memory library (AGENT_MEMORY)
    memory_chars: int = 22000           # character budget for reference material in the code prompt
    memory_dir: str = ""                # defaults to <agent dir>/memory
    selfcheck: bool = True              # specification-driven self-checks of the deliverables
    retry_temperature: float = 0.7      # sampling temperature when a repair repeats an earlier script
    # A/B switches (off = the 2.1.0 base behaviour)
    patch_repair: bool = False          # AGENT_PATCH_REPAIR: repairs as SEARCH/REPLACE edits, not full scripts
    selftest: bool = False              # AGENT_SELFTEST: model-written invariant checks of the outputs
    nversion: bool = False              # AGENT_NVERSION: independent second solution, compare, adjudicate
    strict_review: bool = False         # AGENT_STRICT_REVIEW: clause-by-clause review instead of the generic one
    low_effort: bool = False            # AGENT_LOW_EFFORT: thinking on with {reasoning effort: low} for code/repair/review
    envnotes: bool = False              # AGENT_ENVNOTES: library-version notes (pandas 3 / numpy 2 gotchas) in the system prompt
    probe: bool = False                 # AGENT_PROBE: exact columns/keys of the input files added to data-error feedback
    split: bool = False                 # AGENT_SPLIT: two-part code generation for long scripts (many deliverables)
    think_scope: str = "all"            # AGENT_THINK_SCOPE: with low_effort, think on "all" calls or only on the "plan"
    think_fail_max: int = 2             # AGENT_THINK_FAIL_MAX: thinking failures in a unit before thinking is switched off
    restart: bool = False               # AGENT_RESTART: after a failed first script, one fresh independent rewrite
    mem_solutions: int = 2              # AGENT_MEMORY_SOLUTIONS: reference solutions the plan may pick
    loop: bool = False                  # AGENT_LOOP: multi-step workspace loop (run/write/edit/append actions) after the plan
    loop_reviews: int = 2               # AGENT_LOOP_REVIEWS: review turns after the outputs first validate
    vote: bool = False                  # AGENT_VOTE: spend the leftover budget on independent solutions, keep the majority answer
    vote_max: int = 3                   # AGENT_VOTE_MAX: independent solutions at most
    chunk: bool = False                 # AGENT_CHUNK: long scripts written in 1-3 parts (one request each), up to 3
                                        #   continuations, no compact rewrite, part-level rewrites, review as edits
    deep_think: bool = False            # AGENT_DEEP_THINK: one full-thinking design call before the code; its
                                        #   reasoning and design are handed to the code requests
    rag2: bool = False                  # AGENT_RAG2: task-card category routes the memory retrieval (reserved slots, file-type tokens)
    verify: bool = False                # AGENT_VERIFY: independent re-derivation of key quantities, adjudicated against the task text
    staged: bool = False                # AGENT_STAGED: first script built as inputs.py / compute.py / main.py with execution in between
    vote_parallel: int = 1              # AGENT_VOTE_PARALLEL: independent solutions requested concurrently (1 = one after another)
    verify_fix2: bool = False           # AGENT_VERIFY_FIX2: a second independent check adjudicates a disagreement and the fix is
                                        #   written in a fresh conversation; verify runs after the vote and is skipped when a vote agreed
    degen: bool = False                 # AGENT_DEGEN: degenerate outputs (all-zero/empty columns or values, files named in the task
                                        #   but never written) fail validation and are fed back
    conventions: bool = False           # AGENT_CONVENTIONS: the plan decides every convention (windows, counts, units, compounding,
                                        #   seeds); the list is carried into the code request, the review and the vote candidates
    invariants: bool = False            # AGENT_INVARIANTS: the script prints INVARIANT <name> OK|FAIL lines for the relations the
                                        #   task implies; a FAIL is repair feedback (twice the same -> soft warning for the review)
    probe2: bool = False                # AGENT_PROBE2: exact input structure added to the feedback after degenerate outputs
    two_track: bool = False             # AGENT_TWO_TRACK: units with many deliverables take the L track (core script + one short
                                        #   script per deliverable group, self-checked; no vote), small units keep the S track
    track_l_min: int = 5                # AGENT_TRACK_L_MIN: deliverables from which the L track applies
    track_l_repairs: int = 2            # AGENT_TRACK_L_REPAIRS: repair requests per core / group script
    conv_vote: int = 0                  # AGENT_CONV_VOTE: extra independent convention reads (JSON only); majority per item
    clause_audit: bool = False          # AGENT_CLAUSE_AUDIT: review = sentence-by-sentence clause mapping with edit blocks
    mc_rules: bool = False              # AGENT_MC_RULES: Monte-Carlo reproducibility rules in the code request; 5 % vote tolerance
    plan_think: bool = False            # AGENT_PLAN_THINK: the plan call reasons at full effort (code stays low-effort)
    probe_plan: bool = False            # AGENT_PROBE_PLAN: exact input structure in the plan context for text / multi-table inputs
    mem_triggers: bool = False          # AGENT_MEM_TRIGGERS: memory docs whose `triggers:` match the inputs / task text are always included
    param_echo: bool = False            # AGENT_PARAM_ECHO: the script prints PARAM <name> <value>; values the task never states are flagged
    text_route: bool = False            # AGENT_TEXT_ROUTE: units with HTML / XML documents get a document probe, EXTRACT checks,
                                        #   forced degenerate checks and triggered text playbooks
    vote_dump: str = ""                 # AGENT_VOTE_DUMP: directory that receives every vote candidate's outputs and a meta.json
                                        #   (local selection studies only; empty = off, never set in the image)
    key_check: bool = False             # AGENT_KEY_CHECK: identifier columns of the outputs (same name as an input column) must
                                        #   take values of that input column
    template_route: bool = False        # AGENT_TEMPLATE_ROUTE: "debug the provided script" tasks start from the full script
                                        #   (run unchanged, then fixed with edit blocks); candidates / review also edit it
    time_typed: bool = False            # AGENT_TIME_TYPED: soft budget x0.75 for units with >= 7 deliverables or documents
                                        #   (they never pass late), x1.25 for the others
    vote_max_bytes: int = 8 * 1024 * 1024   # AGENT_VOTE_MAX_MB: no candidate copies in /tmp (64 MiB on the platform) above this

    @classmethod
    def from_env(cls) -> "Config":
        endpoint = (os.environ.get("MODEL_ENDPOINT") or os.environ.get("OPENAI_BASE_URL") or "").strip()
        model = (os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL") or "").strip()
        token = (os.environ.get("MODEL_TOKEN") or os.environ.get("OPENAI_API_KEY") or "").strip()
        return cls(
            endpoint=endpoint,
            model=model,
            token=token,
            time_budget=_env_float("AGENT_TIME_BUDGET", 360.0),
            max_requests=max(1, min(25, _env_int("AGENT_MAX_REQUESTS", 14))),
            max_output_tokens=max(256, min(4000, _env_int("AGENT_MAX_OUTPUT_TOKENS", 4000))),
            thinking=_env_flag("AGENT_THINKING", False),
            thinking_plan=_env_flag("AGENT_THINKING_PLAN", False),
            send_template_kwargs=_env_flag("AGENT_SEND_TEMPLATE_KWARGS", True),
            do_plan=_env_flag("AGENT_PLAN", True),
            do_review=_env_flag("AGENT_REVIEW", True),
            exec_timeout=_env_float("AGENT_EXEC_TIMEOUT", 150.0),
            ignore_checks=_env_flag("AGENT_IGNORE_CHECKS", False),
            temperature=_env_float("AGENT_TEMPERATURE", 0.0),
            seed=_env_int("QFBENCH_SEED", 0),
            workdir=os.environ.get("AGENT_WORKDIR", "/tmp/agent_work"),
            request_timeout=_env_float("AGENT_REQUEST_TIMEOUT", 240.0),
            max_attempts=max(1, _env_int("AGENT_MAX_ATTEMPTS", 8)),
            verbose=_env_flag("AGENT_VERBOSE", True),
            memory=_env_flag("AGENT_MEMORY", True),
            memory_chars=max(0, _env_int("AGENT_MEMORY_CHARS", 22000)),
            memory_dir=os.environ.get("AGENT_MEMORY_DIR", "").strip(),
            selfcheck=_env_flag("AGENT_SELFCHECK", True),
            retry_temperature=_env_float("AGENT_RETRY_TEMPERATURE", 0.7),
            patch_repair=_env_flag("AGENT_PATCH_REPAIR", False),
            selftest=_env_flag("AGENT_SELFTEST", False),
            nversion=_env_flag("AGENT_NVERSION", False),
            strict_review=_env_flag("AGENT_STRICT_REVIEW", False),
            low_effort=_env_flag("AGENT_LOW_EFFORT", False),
            envnotes=_env_flag("AGENT_ENVNOTES", False),
            probe=_env_flag("AGENT_PROBE", False),
            split=_env_flag("AGENT_SPLIT", False),
            think_scope=(os.environ.get("AGENT_THINK_SCOPE", "all").strip().lower() or "all"),
            think_fail_max=max(1, _env_int("AGENT_THINK_FAIL_MAX", 2)),
            restart=_env_flag("AGENT_RESTART", False),
            mem_solutions=max(1, min(4, _env_int("AGENT_MEMORY_SOLUTIONS", 2))),
            loop=_env_flag("AGENT_LOOP", False),
            loop_reviews=max(0, min(4, _env_int("AGENT_LOOP_REVIEWS", 2))),
            vote=_env_flag("AGENT_VOTE", False),
            vote_max=max(1, min(10, _env_int("AGENT_VOTE_MAX", 3))),
            chunk=_env_flag("AGENT_CHUNK", False),
            deep_think=_env_flag("AGENT_DEEP_THINK", False),
            rag2=_env_flag("AGENT_RAG2", False),
            verify=_env_flag("AGENT_VERIFY", False),
            staged=_env_flag("AGENT_STAGED", False),
            vote_parallel=max(1, min(3, _env_int("AGENT_VOTE_PARALLEL", 1))),
            verify_fix2=_env_flag("AGENT_VERIFY_FIX2", False),
            degen=_env_flag("AGENT_DEGEN", False),
            conventions=_env_flag("AGENT_CONVENTIONS", False),
            invariants=_env_flag("AGENT_INVARIANTS", False),
            probe2=_env_flag("AGENT_PROBE2", False),
            two_track=_env_flag("AGENT_TWO_TRACK", False),
            track_l_min=max(2, min(12, _env_int("AGENT_TRACK_L_MIN", 5))),
            track_l_repairs=max(0, min(4, _env_int("AGENT_TRACK_L_REPAIRS", 2))),
            conv_vote=max(0, min(3, _env_int("AGENT_CONV_VOTE", 0))),
            clause_audit=_env_flag("AGENT_CLAUSE_AUDIT", False),
            mc_rules=_env_flag("AGENT_MC_RULES", False),
            plan_think=_env_flag("AGENT_PLAN_THINK", False),
            probe_plan=_env_flag("AGENT_PROBE_PLAN", False),
            mem_triggers=_env_flag("AGENT_MEM_TRIGGERS", False),
            param_echo=_env_flag("AGENT_PARAM_ECHO", False),
            text_route=_env_flag("AGENT_TEXT_ROUTE", False),
            key_check=_env_flag("AGENT_KEY_CHECK", False),
            vote_dump=(os.environ.get("AGENT_VOTE_DUMP") or "").strip(),
            template_route=_env_flag("AGENT_TEMPLATE_ROUTE", False),
            time_typed=_env_flag("AGENT_TIME_TYPED", False),
            vote_max_bytes=int(max(1, _env_int("AGENT_VOTE_MAX_MB", 8)) * 1024 * 1024),
        )


# --------------------------------------------------------------------------------------
# Logging (stdout only; nothing is ever written into the output directory by the logger)
# --------------------------------------------------------------------------------------

_T0 = time.time()


def log(msg: str) -> None:
    try:
        sys.stdout.write(f"[agent +{time.time() - _T0:7.1f}s] {msg}\n")
        sys.stdout.flush()
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Time budget
# --------------------------------------------------------------------------------------


class Budget:
    def __init__(self, soft_seconds: float, hard_seconds: float) -> None:
        self.start = time.time()
        self.soft = float(soft_seconds)
        self.hard = float(hard_seconds)

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> float:
        return self.soft - self.elapsed()

    def hard_remaining(self) -> float:
        return self.hard - self.elapsed()


# --------------------------------------------------------------------------------------
# Canary / sanitisation helpers
# --------------------------------------------------------------------------------------

_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_CANARY_LINE_RE = re.compile(r"canary|BENCHMARK DATA SHOULD NEVER APPEAR", re.IGNORECASE)


def sanitize_text(text: str) -> str:
    """Drop lines that carry a canary GUID / benchmark header so they never reach a prompt."""
    out = []
    for line in text.splitlines():
        if _CANARY_LINE_RE.search(line) or _UUID_RE.search(line):
            continue
        out.append(line)
    return "\n".join(out).strip("\n")


def truncate(text: str, limit: int, tail: int = 0) -> str:
    if len(text) <= limit:
        return text
    if tail > 0 and limit > tail + 40:
        return text[: limit - tail - 30] + "\n...[truncated]...\n" + text[-tail:]
    return text[:limit] + "\n...[truncated]..."


# --------------------------------------------------------------------------------------
# Unit reading
# --------------------------------------------------------------------------------------

_TEXT_EXT = {".md", ".txt", ".py", ".json", ".jsonl", ".csv", ".tsv", ".xml", ".html", ".htm", ".toml", ".yaml", ".yml"}
_SKIP_TOP = {"checks", "tests", "solution", "dev", "card.toml", "manifest.json", "instruction.md", ".git"}


@dataclass
class DataFile:
    path: Path        # absolute real path
    rel: str          # path relative to the data root (e.g. "filings/filing_01.html")
    size: int


@dataclass
class Unit:
    task_dir: Path
    out_dir: Path
    unit_id: str = ""
    instruction: str = ""
    card: dict = field(default_factory=dict)
    agent_timeout: float = 1800.0
    canary: str = ""
    data_root: Path | None = None
    data_files: list[DataFile] = field(default_factory=list)
    checks_text: str = ""
    checks_present: bool = False
    aliases: dict[str, Path] = field(default_factory=dict)   # Dockerfile COPY targets -> real files (renamed data)


def read_card(task_dir: Path) -> dict:
    card_path = task_dir / "card.toml"
    if not card_path.is_file() or tomllib is None:
        return {}
    try:
        return tomllib.loads(card_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # pragma: no cover
        log(f"card.toml unreadable: {exc}")
        return {}


def find_data_root(task_dir: Path) -> Path | None:
    for cand in ("environment/data", "data", "environment", "input", "app/data"):
        p = task_dir / cand
        if p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file() and f.name != "Dockerfile"]
            if files:
                return p
    return None


def collect_data_files(task_dir: Path, data_root: Path | None) -> list[DataFile]:
    files: list[DataFile] = []
    seen: set[Path] = set()

    def add(root: Path, f: Path) -> None:
        if f in seen or not f.is_file():
            return
        if f.name == "Dockerfile" or f.name.startswith("."):
            return
        seen.add(f)
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        files.append(DataFile(path=f.resolve(), rel=str(f.relative_to(root)).replace(os.sep, "/"), size=size))

    if data_root is not None:
        for f in sorted(data_root.rglob("*")):
            add(data_root, f)
    # Anything else at the top level that is not part of the harness (rare layouts).
    for entry in sorted(task_dir.iterdir()):
        if entry.name in _SKIP_TOP:
            continue
        if data_root is not None and (entry == data_root or data_root.is_relative_to(entry)):
            continue
        if entry.is_file():
            add(task_dir, entry)
        elif entry.is_dir():
            for f in sorted(entry.rglob("*")):
                add(task_dir, f)
    return files


def parse_dockerfile_aliases(task_dir: Path) -> dict[str, Path]:
    """The unit's environment Dockerfile places (and sometimes renames) the data inside the checker image,
    e.g. `COPY data/stock_chars.pqt /app/data/stock_data.parquet`.  The task text then refers to the
    renamed path, which does not exist under /input.  Return {absolute alias path: real file}."""
    env_dir = task_dir / "environment"
    dockerfile = env_dir / "Dockerfile"
    aliases: dict[str, Path] = {}
    if not dockerfile.is_file():
        return aliases
    try:
        lines = dockerfile.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return aliases
    for ln in lines:
        m = re.match(r"^\s*(?:COPY|ADD)\s+(?:--\S+\s+)*(\S+)\s+(\S+)\s*$", ln)
        if not m:
            continue
        src, dst = m.group(1).strip("\"'"), m.group(2).strip("\"'")
        if not dst.startswith("/"):
            dst = "/app/" + dst.lstrip("./")
        src_path = (env_dir / src).resolve()
        if src_path.is_file():
            alias = dst + src_path.name if dst.endswith("/") else dst
            aliases[alias] = src_path
        elif src_path.is_dir():
            for f in sorted(src_path.rglob("*")):
                if f.is_file() and f.name != "Dockerfile":
                    rel = str(f.relative_to(src_path)).replace(os.sep, "/")
                    aliases[dst.rstrip("/") + "/" + rel] = f.resolve()
    return aliases


def read_checks(task_dir: Path, limit: int = 16000) -> tuple[str, bool]:
    """Return sanitized checker source when the unit mounts it (local dev); '' otherwise."""
    checks = task_dir / "checks"
    if not checks.is_dir():
        return "", False
    parts: list[str] = []
    for name in ("test_outputs.py", "conftest.py"):
        p = checks / name
        if p.is_file():
            try:
                txt = sanitize_text(p.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            parts.append(f"### checks/{name}\n{txt}")
    for p in sorted(checks.glob("test_*.py")):
        if p.name == "test_outputs.py":
            continue
        try:
            txt = sanitize_text(p.read_text(encoding="utf-8", errors="replace"))
            parts.append(f"### checks/{p.name}\n{txt}")
        except Exception:
            continue
    if (checks / "verifier.py").is_file():
        parts.append(
            "### note\nThe checker delegates to a generic verifier that compares every value in "
            "results.json (deliverables) and every `intermediates` entry in solution.json against "
            "hidden reference values with tight tolerances. Exact key names matter."
        )
    text = "\n\n".join(parts)
    if not text:
        return "", True
    if len(text) > limit:
        text = truncate(text, limit, tail=limit // 4)
    return text, True


def load_unit(task_dir: Path, out_dir: Path, cfg: Config) -> Unit:
    unit = Unit(task_dir=task_dir, out_dir=out_dir)
    unit.card = read_card(task_dir)
    unit.unit_id = str(unit.card.get("task", {}).get("id") or task_dir.name)
    try:
        unit.agent_timeout = float(unit.card.get("agent", {}).get("timeout_sec") or 1800.0)
    except (TypeError, ValueError):
        unit.agent_timeout = 1800.0
    unit.canary = str(unit.card.get("contamination", {}).get("canary_guid") or "").lower()

    instr = task_dir / "instruction.md"
    if not instr.is_file():
        cands = sorted(task_dir.glob("*.md"))
        instr = cands[0] if cands else instr
    if instr.is_file():
        raw = instr.read_text(encoding="utf-8", errors="replace")
        if not unit.canary:
            m = _UUID_RE.search(raw)
            if m and _CANARY_LINE_RE.search(raw):
                unit.canary = m.group(0).lower()
        unit.instruction = sanitize_text(raw)
    else:
        unit.instruction = ""

    unit.data_root = find_data_root(task_dir)
    unit.data_files = collect_data_files(task_dir, unit.data_root)
    try:
        unit.aliases = parse_dockerfile_aliases(task_dir)
    except Exception:
        unit.aliases = {}
    if not cfg.ignore_checks:
        unit.checks_text, unit.checks_present = read_checks(task_dir)
    return unit


# --------------------------------------------------------------------------------------
# Data previews
# --------------------------------------------------------------------------------------


def _count_lines(path: Path, max_bytes: int = 80_000_000) -> int | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        n = 0
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                n += chunk.count(b"\n")
        return n
    except Exception:
        return None


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    return re.sub(r"\s+", " ", text).strip()


def preview_file(df: DataFile, budget_chars: int) -> str:
    """Best-effort, cheap preview of one input file for the prompt."""
    p = df.path
    ext = p.suffix.lower()
    head = f"- `{p}`  ({df.size:,} bytes)"
    if budget_chars < 200:
        return head
    try:
        if ext in (".csv", ".tsv") or (ext == ".txt" and df.size < 2_000_000):
            sep = "\t" if ext == ".tsv" else ","
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                lines = [next(fh, "") for _ in range(6)]
            raw_head = "".join(lines).strip()
            n_lines = _count_lines(p)
            info = f"  rows(excl. header)≈{n_lines - 1 if n_lines else '?'}"
            dtypes = ""
            try:
                import pandas as pd

                sample = pd.read_csv(p, sep=sep, nrows=200, engine="python", on_bad_lines="skip")
                dtypes = "  dtypes: " + ", ".join(f"{c}:{t}" for c, t in sample.dtypes.astype(str).items())
                if len(dtypes) > 1200:
                    dtypes = dtypes[:1200] + "..."
            except Exception:
                pass
            body = raw_head
            if len(body) > budget_chars - 300:
                body = body[: budget_chars - 300] + "\n...[truncated]"
            return f"{head}{info}\n{dtypes}\n  first lines:\n" + "\n".join("    " + ln for ln in body.splitlines())
        if ext == ".json":
            txt = p.read_text(encoding="utf-8", errors="replace")
            if len(txt) <= min(budget_chars - 100, 3500):
                return f"{head}\n  full content:\n" + "\n".join("    " + ln for ln in txt.strip().splitlines())
            try:
                obj = json.loads(txt)
            except Exception:
                return f"{head}\n  head:\n" + "\n".join("    " + ln for ln in txt[: budget_chars - 200].splitlines())
            return f"{head}\n  structure:\n" + _describe_json(obj, budget_chars - 200)
        if ext == ".jsonl":
            n_lines = _count_lines(p)
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                lines = [next(fh, "").strip() for _ in range(3)]
            lines = [ln[:400] + ("..." if len(ln) > 400 else "") for ln in lines if ln]
            return f"{head}  lines≈{n_lines}\n  first records:\n" + "\n".join("    " + ln for ln in lines)
        if ext in (".parquet", ".pqt", ".pq"):
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(p)
            schema = ", ".join(f"{f.name}:{f.type}" for f in pf.schema_arrow)
            rows = pf.metadata.num_rows
            batch = next(pf.iter_batches(batch_size=5))
            import pandas as pd

            sample = batch.to_pandas()
            with pd.option_context("display.max_columns", 30, "display.width", 200):
                sample_txt = sample.head(5).to_string()
            if len(schema) > 1500:
                schema = schema[:1500] + "..."
            return f"{head}  rows={rows}\n  schema: {schema}\n  head:\n" + "\n".join("    " + ln for ln in sample_txt.splitlines()[:8])
        if ext in (".xlsx", ".xlsm", ".xls"):
            import openpyxl

            wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
            parts = [head, f"  sheets: {wb.sheetnames}"]
            for ws in wb.worksheets[:4]:
                try:
                    dims = f"max_row={ws.max_row}, max_col={ws.max_column}"
                except Exception:
                    dims = "?"
                parts.append(f"  sheet '{ws.title}' ({dims}) first rows:")
                for i, row in enumerate(ws.iter_rows(values_only=True)):
                    if i >= 8:
                        break
                    cells = [str(c)[:30] if c is not None else "" for c in row[:15]]
                    parts.append("    " + " | ".join(cells))
            wb.close()
            return "\n".join(parts)[:budget_chars]
        if ext in (".html", ".htm"):
            txt = _strip_html(p.read_text(encoding="utf-8", errors="replace"))
            return f"{head}\n  text excerpt: {txt[: min(1500, budget_chars - 100)]}"
        if ext == ".xml":
            txt = p.read_text(encoding="utf-8", errors="replace")
            return f"{head}\n  head:\n" + "\n".join("    " + ln for ln in txt[: min(1500, budget_chars - 100)].splitlines())
        if ext in (".py", ".md", ".txt", ".toml", ".yaml", ".yml"):
            txt = sanitize_text(p.read_text(encoding="utf-8", errors="replace"))
            limit = min(6000, budget_chars - 100)
            return f"{head}\n  content:\n" + "\n".join("    " + ln for ln in truncate(txt, limit).splitlines())
        if ext == ".zip":
            import zipfile

            with zipfile.ZipFile(p) as zf:
                names = zf.namelist()
            shown = ", ".join(names[:40]) + (" ..." if len(names) > 40 else "")
            return f"{head}\n  zip members ({len(names)}): {shown}"
    except Exception as exc:
        return f"{head}\n  (preview failed: {type(exc).__name__}: {str(exc)[:120]})"
    return head


def _describe_json(obj: Any, limit: int, depth: int = 0) -> str:
    pad = "    " + "  " * depth
    lines: list[str] = []
    if isinstance(obj, dict):
        for i, (k, v) in enumerate(obj.items()):
            if i >= 40:
                lines.append(f"{pad}... ({len(obj)} keys total)")
                break
            if isinstance(v, (dict, list)) and depth < 3:
                kind = f"dict({len(v)})" if isinstance(v, dict) else f"list({len(v)})"
                lines.append(f"{pad}{k!r}: {kind}")
                lines.append(_describe_json(v, limit, depth + 1))
            else:
                lines.append(f"{pad}{k!r}: {json.dumps(v)[:120]}")
    elif isinstance(obj, list):
        if obj:
            lines.append(f"{pad}[0] = {json.dumps(obj[0])[:300]}")
            if len(obj) > 1:
                lines.append(f"{pad}[-1] = {json.dumps(obj[-1])[:200]}")
    else:
        lines.append(f"{pad}{json.dumps(obj)[:200]}")
    text = "\n".join(lines)
    return text[:limit]


def build_previews(unit: Unit, total_chars: int = 14000) -> str:
    if not unit.data_files:
        return "(this unit ships no input data files; everything needed is in the task text)"
    n = len(unit.data_files)
    per_file = max(400, min(5000, total_chars // max(1, n)))
    parts: list[str] = []
    used = 0
    for df in unit.data_files:
        if used > total_chars:
            parts.append(f"- `{df.path}`  ({df.size:,} bytes)")
            continue
        txt = preview_file(df, min(per_file, total_chars - used + 200))
        parts.append(txt)
        used += len(txt)
    return "\n".join(parts)


# --------------------------------------------------------------------------------------
# House model client
# --------------------------------------------------------------------------------------


class ModelError(Exception):
    pass


class BudgetExhausted(ModelError):
    pass


class ContextTooLong(ModelError):
    pass


class House:
    def __init__(self, cfg: Config, budget: Budget) -> None:
        self.cfg = cfg
        self.budget = budget
        self.requests_made = 0
        self.failures = 0
        self.latencies: list[float] = []
        self.call_log: list[tuple[str, float]] = []   # (label, seconds) of every answered request
        ep = cfg.endpoint.rstrip("/")
        if ep.endswith("/chat/completions"):
            self.url = ep
        elif ep.endswith("/v1"):
            self.url = ep + "/chat/completions"
        else:
            self.url = ep + "/v1/chat/completions"
        self.send_template_kwargs = cfg.send_template_kwargs
        self.no_thinking = False      # set by the solver after repeated thinking failures (fallback for this unit)
        self._lock = threading.Lock()  # admission / refund bookkeeping is shared by concurrent vote candidates
        self.rate_limited = False     # a 429 that was not the request allowance: concurrent calls fall back to one at a time

    @property
    def available(self) -> bool:
        return bool(self.cfg.endpoint)

    def template_kwargs(self, thinking: bool, low_effort: bool | None = None) -> dict:
        """chat_template_kwargs for one call: thinking on/off, plus the template's low-effort reasoning switch."""
        kw: dict[str, Any] = {"enable_thinking": bool(thinking)}
        if thinking and (self.cfg.low_effort if low_effort is None else low_effort):
            kw["low_effort"] = True
        return kw

    def requests_left(self) -> int:
        return self.cfg.max_requests - self.requests_made

    def _refund(self) -> None:
        with self._lock:
            self.requests_made -= 1

    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 45.0

    def chat(self, messages: list[dict], *, thinking: bool | None = None, max_tokens: int | None = None,
             label: str = "call", temperature: float | None = None, low_effort: bool | None = None) -> tuple[str, str]:
        """Return (content, finish_reason). Raises ModelError / BudgetExhausted."""
        if not self.available:
            raise ModelError("no MODEL_ENDPOINT configured")
        import requests  # baked into the sandbox base image

        if thinking is None:
            thinking = self.cfg.thinking or (self.cfg.low_effort and self.cfg.think_scope != "plan")
        if thinking and self.no_thinking:
            thinking = False
        max_tokens = min(self.cfg.max_output_tokens, max_tokens or self.cfg.max_output_tokens)
        body: dict[str, Any] = {
            "model": self.cfg.model or "house",
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(self.cfg.temperature if temperature is None else temperature),
            "seed": int(self.cfg.seed),
            "stream": False,
        }
        if self.send_template_kwargs:
            body["chat_template_kwargs"] = self.template_kwargs(thinking, low_effort)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.cfg.token:
            headers["Authorization"] = f"Bearer {self.cfg.token}"

        attempts = 0
        while True:
            attempts += 1
            remaining = self.budget.remaining()
            if remaining < 20:
                raise ModelError("no time left for a model call")
            timeout = max(15.0, min(self.cfg.request_timeout, remaining - 5.0))
            with self._lock:   # the counter is shared by concurrent candidates: check and charge atomically
                if self.requests_left() <= 0:
                    raise BudgetExhausted("request allowance for this unit is used up")
                self.requests_made += 1  # charged on admission, before forwarding
            t0 = time.time()
            try:
                resp = requests.post(self.url, json=body, headers=headers, timeout=(15.0, timeout))
            except requests.exceptions.RequestException as exc:
                self.failures += 1
                log(f"[{label}] request error: {type(exc).__name__}: {str(exc)[:200]}")
                proxy_refused = isinstance(exc, requests.exceptions.ProxyError)
                if proxy_refused:
                    self._refund()   # the egress proxy refused the tunnel (e.g. 407 on CONNECT): nothing reached the route
                if attempts < (4 if proxy_refused else 3) and self.budget.remaining() > 45:
                    time.sleep(2.0 * attempts if proxy_refused else 3.0)
                    continue
                raise ModelError(f"request failed: {exc}") from exc
            dt = time.time() - t0
            status = resp.status_code
            if status in (401, 403):
                # Refused before admission -> not charged.
                self._refund()
                self.failures += 1
                log(f"[{label}] HTTP {status}: {resp.text[:300]}")
                if attempts < 3 and self.budget.remaining() > 45:
                    time.sleep(4.0)
                    continue
                raise ModelError(f"refused: HTTP {status}")
            if status == 407:
                # Proxy Authentication Required is answered by the egress proxy, not by the House route: the request
                # was never admitted, so the slot is given back and the call is retried after a short back-off.
                self._refund()
                self.failures += 1
                log(f"[{label}] HTTP 407 from the proxy: {resp.text[:200]}")
                if attempts < 4 and self.budget.remaining() > 45:
                    time.sleep(2.0 * attempts)
                    continue
                raise ModelError("proxy refused the request: HTTP 407")
            if status == 400:
                self._refund()  # invalid request, refused before admission
                txt = resp.text[:400]
                low = txt.lower()
                log(f"[{label}] HTTP 400: {txt}")
                if ("context" in low and ("length" in low or "window" in low)) or ("maximum" in low and "token" in low) or "too long" in low:
                    raise ContextTooLong(f"prompt too long for the model: {txt}")
                # Strip optional parameters one at a time; the House route documents only the core fields.
                if "chat_template_kwargs" in body:
                    self.send_template_kwargs = False
                    body.pop("chat_template_kwargs", None)
                    continue
                if "seed" in body:
                    body.pop("seed", None)
                    continue
                if "temperature" in body:
                    body.pop("temperature", None)
                    continue
                raise ModelError(f"bad request: {txt}")
            if status == 429:
                self.failures += 1
                log(f"[{label}] HTTP 429: {resp.text[:300]}")
                txt = resp.text.lower()
                if "request" in txt and ("allowance" in txt or "quota" in txt or "budget" in txt or "limit" in txt):
                    raise BudgetExhausted("House route reports the request allowance is exhausted")
                self.rate_limited = True
                if attempts < 3 and self.budget.remaining() > 60:
                    time.sleep(8.0)
                    continue
                raise ModelError("rate limited")
            if status >= 500 or status in (408,):
                self.failures += 1
                log(f"[{label}] HTTP {status}: {resp.text[:300]}")
                if attempts < 3 and self.budget.remaining() > 60:
                    time.sleep(4.0)
                    continue
                raise ModelError(f"server error HTTP {status}")
            if status != 200:
                self.failures += 1
                raise ModelError(f"unexpected HTTP {status}: {resp.text[:300]}")
            try:
                data = resp.json()
            except Exception as exc:
                self.failures += 1
                raise ModelError(f"non-JSON response: {exc}") from exc
            with self._lock:
                self.latencies.append(dt)
                self.call_log.append((label, dt))
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            if isinstance(content, list):  # some servers return content parts
                content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
            if not isinstance(content, str):
                content = str(content)
            finish = str(choice.get("finish_reason") or "")
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip() and "<think>" not in content:
                # a server with a reasoning parser returns the thinking separately: keep it inline, like ours
                cut = finish == "length" and not content.strip()
                content = "<think>" + reasoning + ("" if cut else "</think>") + content
            usage = data.get("usage") or {}
            log(f"[{label}] ok in {dt:.1f}s, finish={finish}, usage={usage}, requests={self.requests_made}/{self.cfg.max_requests}")
            return content, finish


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert quantitative-finance engineer and Python developer. You write correct, deterministic, self-contained Python scripts that solve a task EXACTLY as specified and write the required deliverable files.

Runtime environment of the script: Python 3.13; installed packages: numpy, pandas, scipy, statsmodels, scikit-learn, arch, polars, pyarrow, numba, matplotlib (Agg backend), seaborn, plotly, openpyxl, lxml, beautifulsoup4, TA-Lib (talib), tqdm. No internet, no pip install, no GPU required.

Hard rules:
1. Read inputs ONLY from the absolute paths listed under "ACTUAL FILE LAYOUT" (the task text may name other locations such as /app/data/...; the listed absolute paths are the real ones).
2. Write EVERY deliverable file into the output directory given, using EXACTLY the file names, JSON keys, CSV column names, orderings, units and rounding that the task specifies. Never skip a deliverable; never rename keys; do not write extra files.
3. Follow the task's formulas and conventions literally (annualization factor, ddof, log vs simple returns, sign conventions, day-count, compounding, calendar rules). If something is ambiguous, use the most standard textbook convention and note it in a short comment.
4. JSON: convert numpy/pandas scalars to Python float/int/str before json.dump; never emit NaN/Infinity unless the task explicitly asks for it (use None only if allowed). CSV: write with index=False unless the task wants an index column.
5. Deterministic: seed numpy and random with the given seed; no time-of-day dependent logic; no randomness beyond what the task requires.
6. Efficient: vectorize with numpy/pandas; the whole script must finish within ~60 seconds on 16 CPUs. No plotting windows or interactive calls (matplotlib must use Agg; plotly figures are written with write_html/write_image only if required).
7. Never copy the task text, headers or any GUID into an output file; keep outputs to the deliverables only.
8. The script must be a single file that runs top to bottom with `python script.py` and exits 0. Do not read from stdin or command-line arguments. The harness saves and runs your script itself: do NOT write the script to disk (ignore instructions such as "save it as /app/solve.py"), and write temporary files only under /tmp (small) or the output directory — the rest of the filesystem is read-only.
9. The grader is a hidden pytest suite: it checks file names/schemas, financial invariants (no-arbitrage bounds, monotonicity, parity, consistency between files) and usually compares numbers with a reference implementation of the SAME method under tolerances (often 1e-3 to a few percent). Implement the exact textbook method the task names, with full precision; do not approximate, subsample, or change the method for speed unless the task allows it.
10. Defaults ONLY when the task does not specify: 252 trading days per year, sample std with ddof=1, simple returns = pct_change, risk-free rate 0, max drawdown reported as a positive magnitude, dates written as YYYY-MM-DD, rows kept in input order, float values unrounded unless a rounding rule is given.
11. Reference material (verified solutions to OTHER tasks, method notes) may be shown to you. Reuse their formulas, conventions and output hygiene where the tasks match, but THIS task's text always wins when they differ; never copy their constants, file names, keys or parameters blindly.
"""

ENV_NOTES = """
Library versions in the sandbox: Python 3.13, numpy 2.5, pandas 3.0, scipy 1.18, statsmodels 0.15, arch 8.0, scikit-learn 1.9, polars 1.44, pyarrow 25, numba 0.67, matplotlib 3.11, openpyxl 3.1. Consequences:
- pandas 3: the offset aliases 'M', 'Q', 'Y', 'A', 'H', 'T', 'S' raise errors; use 'ME', 'QE', 'YE', 'h', 'min', 's' (e.g. '5min'). String columns have dtype 'str'. Copy-on-write is on: assign with df.loc[mask, col] = value, never through a chained selection. DataFrame.append, fillna(method=...) and Series.drop_duplicates(subset=...) no longer exist (use pd.concat, .ffill()/.bfill()). pct_change(fill_method=None). For Excel inputs look at the raw rows first (header=None): workbooks often have title rows above the real header.
- numpy 2: np.trapz is gone (np.trapezoid); numpy scalars/bools are not JSON serialisable (cast with float()/int()/bool()); an array in an if-statement needs .any()/.all().
- scipy: minimize(fun, x0, tol=...) (ftol belongs in options=); brentq/bisect need a sign change, so bracket adaptively.
- Never wrap the main computation in a try/except that swallows the error or exits 0: let exceptions propagate so they can be fixed.
"""

SPLIT_PART1 = """The script will be long, so write it in TWO parts. Now output PART 1 of 2 only: imports, constants and paths, loading and cleaning of every input, and the core computations, keeping every result that is needed later in plain variables. Do NOT write any output file in part 1. End part 1 with the exact comment line `# === PART 2 ===`. Keep part 1 under about 2500 tokens. Output ONLY one ```python block."""

SPLIT_PART2 = """Now output PART 2 of 2 only: it continues directly after part 1 in the same file (same variables in scope; no re-imports, no re-loading), computes whatever remains and writes EVERY deliverable with the exact names, keys and columns. Keep it under about 2500 tokens. Output ONLY one ```python block containing just part 2."""

SPLIT_REWRITE = """Your script was cut off by the output-token limit, so it will be rewritten in TWO parts. Output PART 1 of 2 only: imports, constants and paths, loading and cleaning of inputs, and the core computations, keeping every result that is needed later in plain variables; do NOT write any output file in part 1. End part 1 with the exact comment line `# === PART 2 ===`. Keep part 1 under about 2500 tokens. Output ONLY one ```python block."""

SPLIT_REPAIR = """The script consists of PART 1 (lines 1-{s1}) and PART 2 (lines {s2}-{end}, after the `# === PART 2 ===` marker). The problem is in PART {k}. Fix it and output ONLY the complete corrected PART {k} as ONE ```python block; the other part is kept unchanged, so keep the same variable names at the boundary. Keep every deliverable, exact names/keys/columns."""

PLAN_INSTRUCTIONS = """Before coding, write a concise implementation plan (at most 600 words). Cover:
1. DELIVERABLES: every output file with its exact name, format, and the exact keys/columns/order it must contain (quote the task).
2. INPUTS: how each input file is loaded (columns, date parsing, units, missing values).
3. METHOD: the computation step by step with the exact formulas and conventions the task requires (annualization, ddof, compounding, sign, rounding, edge cases).
4. PITFALLS: anything that commonly goes wrong for this kind of task, and how the code avoids it.
Finish with a machine-readable block listing the deliverables, exactly in this form:
```json
{"deliverables": [{"file": "results.json", "format": "json", "required_keys": ["..."]}, {"file": "table.csv", "format": "csv", "required_columns": ["..."]}]}
```
Do not write the code yet."""

PLAN_MEMORY_INSTRUCTIONS = """Before coding, write a concise implementation plan (at most 600 words). Cover:
1. DELIVERABLES: every output file with its exact name, format, and the exact keys/columns/order it must contain (quote the task).
2. INPUTS: how each input file is loaded (columns, date parsing, units, missing values).
3. METHOD: the computation step by step with the exact formulas and conventions the task requires (annualization, ddof, compounding, sign, rounding, edge cases).
4. PITFALLS: anything that commonly goes wrong for this kind of task, and how the code avoids it.
5. REFERENCE MATERIAL: from the "REFERENCE LIBRARY" list, pick the entries whose method or output structure is closest to THIS task (at most {n_sol} solutions and 2 playbooks/notes; pick none if nothing fits). They will be shown to you in full before you write the code.
Finish with a machine-readable block, exactly in this form:
```json
{"deliverables": [{"file": "results.json", "format": "json", "required_keys": ["..."]}, {"file": "table.csv", "format": "csv", "required_columns": ["..."]}], "memory": {"solutions": ["S1"], "playbooks": ["P1", "P2"]}}
```
Do not write the code yet."""

CHUNK_PLAN_NOTE = """In the JSON block also include "est_lines" (your estimate of the script length in lines) and "parts": the script's function names split into 1-3 consecutive parts of at most about 150 lines each, e.g. "parts": [["load_inputs", "clean"], ["compute_x", "compute_y"], ["write_outputs", "main"]]. Use one part when the script is short."""

DEEP_THINK = """Before any code is written, think this task through carefully: the exact formulas and conventions (annualisation, ddof, compounding, signs, day counts, rounding), how each input file must be parsed, edge cases, and how every key/column of every deliverable is computed. Then output a short DESIGN (at most 60 lines, no full code):
1. the script's functions in order, each with its signature and a one-line description;
2. for every deliverable: the function that produces it and its exact keys/columns;
3. the key formulas, one line each."""

CHUNK_PART = """The script is written in {n} consecutive parts, one per answer; the parts are concatenated in order into ONE script. Output ONLY PART {k} of {n} as one ```python block{scope}. {rule}Keep this part under about 150 lines so it fits in one answer, with minimal comments. Do not repeat code from earlier parts; use the names they define."""

PART_REPAIR = """The script consists of {n} parts, each starting with a `# --- part k of {n} ---` line. The failure is in PART {k} (lines {a}-{b}). Output ONLY the complete corrected PART {k} as ONE ```python block (without its marker line); the other parts are kept unchanged, so keep every name they use from this part. Keep every deliverable and exact names/keys/columns."""

REVIEW_EDITS = """The script ran successfully. Here is a summary of the files it wrote:

{summary}

Carefully compare these outputs with the task specification: exact file names, keys/columns and their order, units (percent vs decimal, annualized vs daily), sign conventions, rounding, row counts, date formats, and whether the magnitudes are financially plausible. Also re-check the formulas in the code against the task text.
If everything is correct, reply with exactly: VERDICT: OK
Otherwise list the concrete problems briefly and then fix them with edit blocks of the current script, each exactly in this form:
<<<<<<< SEARCH
(lines copied VERBATIM from the current script)
=======
(the replacement lines)
>>>>>>> REPLACE"""

MEMORY_PREFACE = """## REFERENCE MATERIAL
The following are verified solutions to DIFFERENT tasks and general method notes. Use them for formulas, conventions and output hygiene; this task's own text decides every name, key, column, parameter and method.

"""

STAGED_PLAN_NOTE = """In the JSON block also include "est_lines": your estimate of the complete script's length in lines."""

CONVENTIONS_PLAN_NOTE = """After that JSON block, add a SECOND, separate ```json block {"conventions": [...]} (keep the first block exactly as specified; valid JSON, double quotes, no trailing commas): the list of every convention this task leaves to be decided and the value you will implement, each as {"item": "<what>", "task_text": "<the words of the task that decide it, or 'not stated'>", "choice": "<exactly what the code will do>"}. Cover at least: the sample window and warm-up / lookback start (first and last date or observation actually used, inclusive or exclusive ends), every observation or row count the outputs report, units and scaling of every reported number (percent vs decimal, bps, per-1% vs per-unit sensitivities, currency/notional scale), compounding and day-count conventions, annualisation factors, ddof, sign conventions, random seeds and generator calls, rounding, and tie-breaking / de-duplication rules. Be literal: when the task states a value, quote it; when it does not, say so and pick the standard convention."""

TRACK_L_PLAN_NOTE = """This task has many deliverables, so the code will be written in PIECES: first a CORE script that loads the inputs and computes every quantity that two or more deliverables share, then one SHORT script per group of deliverables.  After the other JSON block(s), add a separate ```json block {"groups": [{"name": "<short>", "files": ["<deliverable file>", ...], "needs": ["<core quantity>", ...]}, ...]} that assigns EVERY deliverable file to exactly one group (2 to 4 groups, 1-3 files each - never all files in one group, and no group named core - files that depend on each other together, in the order they should be produced) and names the core quantities each group needs."""

TEMPLATE_PLAN_NOTE = """This task is to DEBUG the provided script (shown in full under SCRIPT TO DEBUG), not to design a solution from scratch: in the plan, go through the script function by function and list every suspected bug (function, what it does, what the task text / conventions say it must do instead).  Keep the JSON deliverables block."""

TEMPLATE_START = """The task is to fix the colleague's script shown in full above.  The harness runs that script UNCHANGED first and shows you the result; you then fix the bugs with SEARCH/REPLACE edit blocks against it (change only what is wrong; keep its I/O structure, file names, keys and columns).  The script as it stands:"""

TEMPLATE_FIX_FEEDBACK = """The script ran end-to-end and wrote the deliverables, but it is the colleague's buggy template: the task says it contains computational bugs (conventions, curve usage, ordering, valuation semantics, formulas).  Go through it against the task text and the convention file(s), find EVERY bug and fix them all; do not rewrite working parts.  Summary of what it wrote:"""

TEMPLATE_CANDIDATE = """The task is to fix the script shown under SCRIPT TO DEBUG.  Independently find and fix EVERY bug in it: output SEARCH/REPLACE edit blocks against that script, each exactly in this form:
<<<<<<< SEARCH
(lines copied VERBATIM from the script)
=======
(the replacement lines)
>>>>>>> REPLACE
or, if the changes are extensive, the complete corrected script in ONE ```python block."""

TEMPLATE_CANDIDATE_LONG = """The task is to fix the script shown under SCRIPT TO DEBUG.  Independently find and fix EVERY bug in it.  The script is long, so do NOT rewrite it: output ONLY SEARCH/REPLACE edit blocks (no explanation, no full script), each exactly in this form:
<<<<<<< SEARCH
(at most 8 consecutive lines copied CHARACTER FOR CHARACTER from the script, enough to be unique)
=======
(the replacement lines)
>>>>>>> REPLACE"""

TEMPLATE_EDIT_STRICT = """NOTE: your previous edit blocks did not match the script and were discarded.  Output ONLY edit blocks (no prose), each SEARCH part copied character for character from the script as shown (same indentation, no abbreviations, no '...'), at most 8 lines per block, one block per change."""

TEMPLATE_LONG_LINES = 120   # above this length a full rewrite of the shipped script does not fit in one answer

CORE_INSTRUCTIONS = """Write the CORE script now: load every input, compute every shared quantity ({needs}) exactly as the task defines it, and store ALL of them in one pickle file `os.path.join(os.environ["CORE_DIR"], "core.pkl")` (a dict: name -> DataFrame / Series / ndarray / scalar / dict; keep raw cleaned inputs too).  Write NO deliverable files in this script.  At the end print a SUMMARY: one line per stored key as `CORE <key> <type> <shape or value>` (scalars with their value, tables with shape and the first column names), then `CORE DONE`.  Output ONLY one ```python block (target under 200 lines, minimal comments)."""

GROUP_INSTRUCTIONS = """The core quantities were computed by the script below and stored in the pickle `os.path.join(os.environ["CORE_DIR"], "core.pkl")`; its printed summary follows.

Core script:
```python
{core}
```
Core summary:
{summary}

Now write ONE SHORT script that writes ONLY these deliverables, exactly as the task specifies them: {files}.  Load the pickle and use its quantities ({needs}); do not recompute them, do not read other deliverables, do not write any other file.  {invariants}  Output ONLY one ```python block (target under 150 lines, minimal comments)."""

GROUP_FIX = """The script did not produce correct deliverables. Details:

{feedback}

Fix the problem(s) and output the COMPLETE corrected script as ONE ```python block (still only these files: {files})."""

CONV_VOTE_PROMPT = """Read the task above once more, independently, and output ONLY one ```json block of the form {"conventions": [{"item": "<what>", "task_text": "<the words of the task that decide it, or 'not stated'>", "choice": "<exactly what a correct implementation does>"}, ...]} - no plan, no code.  Cover at least: the sample window and warm-up / lookback start (first and last date or observation actually used, inclusive or exclusive ends), every observation or row count the outputs report, units and scaling of every reported number (percent vs decimal, bps, per-1% vs per-unit sensitivities, currency/notional scale), compounding and day-count conventions, annualisation factors, ddof, sign conventions, random seeds and generator calls, rounding, tie-breaking / de-duplication rules.  Be literal: when the task states a value, quote it; when it does not, say so and name the standard convention."""

CLAUSE_AUDIT = """The script ran successfully. Here is a summary of the files it wrote:

{summary}

Audit the script against the task text clause by clause.  For EVERY sentence of the task that fixes a number, date, window, unit, formula, rounding, ordering, filter, threshold or file/key/column name, write one line exactly in the form `CLAUSE: <short quote of the task> -> <function or line of the script that implements it> -> OK` or `... -> MISMATCH: <what the code does instead>`.  Then, if there is any MISMATCH, fix each one with an edit block of the current script, exactly in this form:
<<<<<<< SEARCH
(lines copied VERBATIM from the current script, enough to be unique)
=======
(the replacement lines)
>>>>>>> REPLACE
If every clause is OK, end with exactly: VERDICT: OK"""

MC_NOTE = """MONTE CARLO / SIMULATION RULES (this task involves simulation): use exactly the number of paths, steps, repetitions and the seed the task states (`np.random.default_rng(seed)` unless the task names another generator call), draw in the order the task describes, use antithetic or control variates only if the task asks, report standard errors where asked, and never reduce the sizes to save time - vectorise instead.  Where the task compares simulation with a closed form, the closed form must be implemented exactly, and the simulated value must land within the stated tolerance."""

EXTRACT_NOTE = """This task reads values out of documents.  For EVERY value the script reads from a document (before any computation), print one line `EXTRACT <field> <value> | <the exact short source text it was read from, at most 100 characters>`.  Never write placeholder values (empty strings, 0, None) into the deliverables: if a value cannot be found, print `EXTRACT <field> MISSING | <what was searched for>` and fix the parsing.  The DOCUMENT PROBE above shows where the values are; parse the documents in the script, do not paste numbers from the probe into the code."""

PARAM_ECHO_NOTE = """Also print, before anything else is written, one line `PARAM <name> <value>` for EVERY numeric parameter the task fixes and the script uses (window lengths, lookbacks, counts, seeds, numbers of paths / steps / simulations, confidence levels, tolerances, rates, fees, thresholds, start / end dates as YYYY-MM-DD), with the value actually used in the computation (plain number or ISO date)."""

INVARIANTS_NOTE = """At the END of the script (after every deliverable is written), recompute from the written values the relations the task text implies a correct solution must satisfy - e.g. parity or no-arbitrage identities, components adding up to totals, repricing of inputs, monotonicity in a parameter, positivity / bounds, methods that must agree within the task's tolerance, row counts and date ranges implied by the task - and print exactly one line per relation: `INVARIANT <short_name> OK` or `INVARIANT <short_name> FAIL <the numbers>`. Wrap each in try/except; never let these checks change the outputs or the exit code."""

VERIFY_PLAN_NOTE = """In the JSON block also include "key_quantities": 3 to 6 scalar results that best test the correctness of the whole computation (prices, rates, risk numbers, statistics that depend on the core formulas; not counts or echoes of inputs), each as {"file": "<deliverable file>", "path": "<JSON key path such as summary.var_99 or a CSV column name>", "agg": "value" for a JSON scalar or one of first/last/mean/sum/min/max for a CSV column, "meaning": "<what it is and how the task defines it, in one line>"}."""

STAGE_INPUTS = """The script is built in THREE modules, one per answer, that live in the same directory: inputs.py (loading and cleaning), compute.py (the computations) and main.py (writing the deliverables). They are joined into one script at the end, so define everything at module level with plain names and import between them only with `from inputs import ...` / `from compute import ...`.
Write ONLY inputs.py now, as ONE ```python block: a function `load_inputs()` that reads every input file from its absolute path, applies the parsing/cleaning the plan requires (dtypes, dates, sorting, filters, parameters from JSON) and returns a dict of named objects; then an `if __name__ == "__main__":` block that calls it and prints, for every object: its name, type, shape or length, column names with dtypes (or the keys/values of parameters), the first 3 rows, the number of missing values, and the min/max of any date column. Do not write any output file. Under 120 lines."""

STAGE_COMPUTE = """inputs.py ran successfully. Its printed output:
{observed}

Now write ONLY compute.py, as ONE ```python block: start with `from inputs import load_inputs`; implement the computations of the plan as functions that take the loaded objects/parameters and return results (keep every quantity the deliverables need); then an `if __name__ == "__main__":` block that loads the inputs, runs every computation, prints each key result as `name = value` and asserts the invariants the task implies (parity, bounds, monotonicity, components summing to totals). Do not write any output file. Under 150 lines; import only from inputs and the standard libraries."""

STAGE_MAIN = """compute.py ran successfully. Its printed output:
{observed}

Now write ONLY main.py, as ONE ```python block: `from inputs import load_inputs` and `from compute import <the functions you need>`; call them and write EVERY deliverable into the output directory with the exact file names, keys, columns, order, units and rounding the task specifies (cast numpy/pandas scalars to Python float/int/str/bool; no NaN in JSON). The module runs top to bottom when executed (call a main() at the bottom). Under 120 lines."""

STAGE_FIX = """Running {name}.py failed:
{feedback}

Output the COMPLETE corrected {name}.py as ONE ```python block (keep the same functions and names; keep it short)."""

STAGE_COMBINE = """Combine the three modules you wrote (inputs.py, compute.py, main.py) into ONE self-contained Python script that runs top to bottom and writes every deliverable: drop the inter-module imports and the diagnostic __main__ blocks of inputs.py and compute.py, keep every function and the main() call. Output ONLY one ```python block, under 250 lines."""

VERIFY_PROMPT = """A solution to the task above has been written and run separately; you have NOT seen it and must not guess what it did. Independently recompute the following quantities directly from the input files, using the task text's own definitions and conventions (formula variants, ddof, annualisation, units, day counts, seeds exactly as stated):
{quantities}

Write ONE short Python script (at most ~90 lines, no comments) that only READS the input files (never writes anything anywhere), computes each quantity with the most direct reliable method the task allows (closed forms and direct formulas where possible; Monte Carlo only where the task requires it, with the task's sizes and seed), and prints exactly one line per quantity in the form `CHECK <name> <number>` using EXACTLY the names given above (q1, q2, ...; plain decimal, no units or commas), or `CHECK <name> NA <short reason, e.g. the exception message>` if it cannot be computed, followed by a final line `CHECKS DONE`. Load every input inside try/except too, so a loading problem still lets the other quantities print. Wrap each quantity in try/except so one failure does not stop the others. Output ONLY one ```python block."""

VERIFY_FIX = """The check script failed:
{feedback}

Output the COMPLETE corrected check script as ONE ```python block (same CHECK lines, nothing written to disk)."""

VERIFY_JUDGE = """For cross-checking, the following quantities were recomputed INDEPENDENTLY from the input files by a separate script that followed the task text literally, and they DIFFER from your outputs:
{diffs}

The independent script:
```python
{other}
```
Decide, from the task text, which value follows the task's definitions for each quantity (formula variant, convention, parameters, units, input handling). If your script is right and the independent computation misread the task, reply with exactly: VERDICT: KEEP (one line of reason).
Otherwise fix your script by EDITING it with one or more edit blocks and NOTHING else (no explanation, no full script - the answer must stay short), each exactly in this form:
<<<<<<< SEARCH
(lines copied VERBATIM from the current script, enough to be unique)
=======
(the replacement lines)
>>>>>>> REPLACE
Keep every deliverable, name, key and column."""

VERIFY_SECOND = """

This is a second, separate re-derivation: where the task allows more than one valid implementation route (a library routine versus a direct formula, equivalent estimator forms), take the route you would not pick first, and read every convention literally from the task text."""

VERIFY_FIX2 = """The script below was written for the task above and its outputs were checked: key quantities were recomputed TWICE, independently, by two separate scripts that followed the task text literally and never saw this script. The two independent computations agree with each other and DIFFER from the script's outputs:
{diffs}

Current script:
```python
{code}
```
Find the cause in the script (formula variant, convention, parameter, units, day count, input handling, sample window) by re-reading the task text, and fix it by EDITING the script with one or more edit blocks and NOTHING else (no explanation, no full script), each exactly in this form:
<<<<<<< SEARCH
(lines copied VERBATIM from the current script, enough to be unique)
=======
(the replacement lines)
>>>>>>> REPLACE
Keep every deliverable, name, key and column; change only what the disagreement requires."""

CODE_INSTRUCTIONS = """Now write the complete Python script that implements the plan and produces every deliverable. Output ONLY one ```python code block and nothing else. Keep it compact (target under 250 lines, minimal comments) so it is not cut off. Use the absolute input paths and the output directory exactly as listed."""

COMPACT_REWRITE = """Your script was cut off by the output-token limit and is incomplete. Rewrite the COMPLETE script much more compactly so that it fits in one answer: at most about 180 lines, no comments or docstrings, no print statements except essential ones, short helper functions reused instead of repeated code, no unused branches. Keep every deliverable, every name/key/column and the exact method. Output ONLY one ```python block."""

PATCH_REPAIR = """Fix the problem(s) by EDITING the script of your previous answer (the one that was run) instead of rewriting it. Output one or more edit blocks and nothing else, each exactly in this form:
<<<<<<< SEARCH
(lines copied VERBATIM from the current script, enough to be unique)
=======
(the replacement lines)
>>>>>>> REPLACE
Blocks are applied in order. Keep edits minimal; keep every deliverable, name, key and column."""

SELFTEST_PROMPT = """Your script for the task above has run and written its deliverables into {out_dir}. Current script:
```python
{code}
```
Summary of the outputs:
{summary}

Write a SEPARATE short Python script (at most ~120 lines) that CHECKS these outputs against the task text. It must only READ files (the outputs in {out_dir} and the inputs); never write anything. Test properties that the task text itself implies and that a correct implementation must satisfy, for example: quantities that must reprice/reconstruct given inputs; parity or no-arbitrage relations; components that must add up to totals; consistency between different output files; bounds, signs and monotonicity; methods that the task says should agree (within a stated tolerance or Monte-Carlo error); row counts and date ranges implied by the task; units and magnitudes (percent vs decimal, annualised vs per-period). Do NOT simply recompute everything with the same code. Wrap each check in try/except so one failure does not stop the others. Print exactly one line `FAIL: <what is wrong, with the numbers>` for every violated check, and `CHECKS DONE` at the end. Always exit 0. Output ONLY one ```python block."""

SELFTEST_FEEDBACK = """The outputs were checked against properties implied by the task, and these checks FAILED:
{fails}
A failed check usually means a formula, convention, unit or input-handling error in the script; but the check itself can be wrong, so first decide from the task text whether it really applies. Fix the real cause in the script."""

NV_JUDGE = """For cross-checking, an INDEPENDENT implementation of the same task was written and run. Its outputs DIFFER from yours:
{diffs}

The independent script:
```python
{other}
```
Decide, from the task text, which implementation is correct for each differing quantity (formula, convention, parameters, input handling, units). If your script is correct, reply with exactly: VERDICT: KEEP
Otherwise output the COMPLETE corrected version of YOUR script as ONE ```python block (keep every deliverable, name, key and column)."""

STRICT_REVIEW = """The script ran successfully. Here is a summary of the files it wrote:

{summary}

Audit the script against the task text, clause by clause. First list, one terse line each (at most 25 lines), every explicit requirement of the task: formulas, parameters, conventions (units, percent vs decimal, annualisation, day-count/calendar, ddof, sign, compounding, rounding), input filtering/cleaning rules, and file schemas. For each line write MATCH or MISMATCH and the reason. Then check 2-3 output numbers for plausibility against the summary.
If everything matches, end with exactly: VERDICT: OK
Otherwise end with the COMPLETE corrected script as ONE ```python block."""

SAME_ERROR_NOTE = """NOTE: this exact error (same exception at the same line) already occurred in an earlier attempt, so the previous fix did not address its cause. Re-read the input-data previews and the traceback, question the assumption behind that line (shapes, column names, dtypes, index alignment), and restructure that part instead of patching the symptom."""

DUPLICATE_REPAIR = """That script is identical to one that was already run and failed the same way, so it cannot fix the problem. Look again at the error/validation messages above, find the actual cause (read the traceback line numbers; check column names, dtypes, shapes, index alignment, and that every output file is written), and fix it differently — restructure the failing part if needed. Output the COMPLETE corrected script as ONE ```python block."""

CODE_ONLY_INSTRUCTIONS = """Write the complete Python script that solves the task and produces every deliverable. Output ONLY one ```python code block and nothing else. Keep it compact (target under 250 lines, minimal comments) so it is not cut off. Use the absolute input paths and the output directory exactly as listed."""

LOOP_INSTRUCTIONS = """Now solve the task step by step in a small WORKSPACE, like a developer at a terminal. Each of your answers contains one or more ACTIONS as fenced blocks; they are executed in order and the results come back to you.

ACTIONS
1. Inspect or experiment (prints are returned; about 60 s limit; do not write deliverables here):
```python run
import pandas as pd
df = pd.read_csv("/abs/path/of/input.csv"); print(df.dtypes); print(df.head(3))
```
2. Write a whole workspace file (creates or replaces it):
```python file=data.py
...complete file...
```
3. Edit a workspace file (SEARCH text copied exactly from the current file; one or more blocks):
```edit file=main.py
<<<<<<< SEARCH
exact old lines
=======
new lines
>>>>>>> REPLACE
```
4. Append to a workspace file (to continue a file that was cut off):
```python append=model.py
...the remaining lines...
```
5. Finish: a line containing only `DONE`, once the deliverables are validated and you have checked them.

RULES
- The solution is the workspace. `main.py` is the entry point, run as `python main.py`; it may import your other workspace modules (e.g. `from data import load_prices`). After every answer that changes files, main.py is run automatically and you get its exit status, the stderr/stdout tails, a check of every deliverable and a preview of the output files.
- Each answer is limited to about 4000 tokens INCLUDING your reasoning, and an answer that hits the limit is cut off. Keep the reasoning brief and keep every file under about 150 lines: put a long solution into several modules (e.g. data.py for loading and cleaning, model.py for the computations, main.py for writing the deliverables) written over one or more answers. Write main.py last: it runs as soon as it exists.
- Fix problems with `edit` blocks rather than rewriting whole files, unless the file is short or the approach itself is wrong.
- Look before you code: if the previews leave any doubt about columns, sheets, dtypes, date formats or JSON structure, inspect first, with ALL the prints you need in ONE `run` block (it can be in the same answer as file writes). Never spend an answer on a single print: every answer costs one of your limited answers.
- Use exactly the fence headers shown above (```python run, ```python file=NAME.py, ```edit file=NAME.py, ```python append=NAME.py); no other format (no tool-call syntax, no JSON) is executed.
- All earlier rules still hold: exact file names, keys, columns and methods; inputs from the listed absolute paths; deliverables into the output directory; deterministic and fast."""

LOOP_REVIEW = """All deliverables were written and pass the automatic checks. Before finishing, review them: compare the output preview with the task text (file names, keys/columns and their order, units such as percent vs decimal, annualisation, signs, rounding, row counts, date formats, plausible magnitudes) and re-check the formulas in the workspace files against the task. If everything is right, answer with the single line DONE. Otherwise fix the problems with edit blocks."""

LOOP_NO_ACTION = """No action was found in your answer. Reply with action blocks exactly as described (```python run, ```python file=NAME.py, ```edit file=NAME.py, ```python append=NAME.py) or the line DONE."""


def build_context(unit: Unit, out_dir: Path, previews: str, compact: int = 0, memory_toc: str = "") -> str:
    """compact: 0 = full context, 1 = abridged checker + small previews, 2 = instruction + file list only."""
    instruction = unit.instruction if unit.instruction.strip() else "(no instruction text was provided)"
    if compact >= 2 and len(instruction) > 10000:
        instruction = truncate(instruction, 10000, tail=2500)
    elif compact == 1 and len(instruction) > 14000:
        instruction = truncate(instruction, 14000, tail=3000)
    lines = [
        "## TASK (instruction.md)",
        instruction,
        "",
        "## ACTUAL FILE LAYOUT",
        f"Output directory (write all deliverables here; it exists and is writable): `{out_dir}`",
    ]
    if unit.data_files:
        lines.append("Input files (absolute paths; the task text may refer to them by other names):")
        for df in unit.data_files:
            alias = ""
            if unit.data_root is not None:
                alias = f"  ← referred to as `/app/data/{df.rel}` or `/app/{df.rel}` or `{df.rel}` in the task text"
            renamed = [a for a, real in unit.aliases.items() if real == df.path and Path(a).name != df.path.name]
            if renamed:
                alias += " — the task text calls this file " + " / ".join(f"`{a}`" for a in renamed[:3]) + " (same content, use the real path above)"
            lines.append(f"- `{df.path}`{alias}")
        if unit.data_root is not None:
            lines.append(f"Data root directory: `{unit.data_root}` (helper .py files there can be imported: it is on sys.path)")
    else:
        lines.append("Input files: none (the unit ships no data files).")
    lines += ["", "## INPUT DATA PREVIEWS", previews]
    if unit.checks_text and compact == 0:
        lines += [
            "",
            "## CHECKER SOURCE (the pytest file that will grade the deliverables; satisfy every assertion)",
            unit.checks_text,
        ]
    elif unit.checks_text and compact == 1:
        lines += ["", "## CHECKER SOURCE (abridged)", truncate(unit.checks_text, 3000, tail=800)]
    if memory_toc and compact < 2:
        lines += ["", "## REFERENCE LIBRARY (verified solutions to OTHER tasks and method notes; pick the closest analogues in the plan)", memory_toc]
    if unit.agent_timeout:
        lines += ["", f"(Per-unit wall-clock limit: {int(unit.agent_timeout)} s in total; keep the script fast.)"]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Code handling
# --------------------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:python|py|python3)?[ \t]*\r?\n(.*?)```", re.S)


def strip_reasoning(text: str) -> str:
    """Drop inline reasoning, which never contains the final answer: complete <think>...</think> blocks,
    everything up to a stray closing tag (some servers emit the reasoning without the opening tag), and an
    unclosed reasoning block that was cut off by the token limit."""
    text = re.sub(r"(?s)<think>.*?</think>", "", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text


def extract_code(text: str) -> str:
    text = strip_reasoning(text)
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip("\n")
    # Unterminated fence (truncated output)
    m = re.search(r"```(?:python|py|python3)?[ \t]*\r?\n(.*)$", text, re.S)
    if m:
        return m.group(1).strip("\n")
    if re.search(r"^\s*(import |from \S+ import )", text, re.M):
        return text.strip("\n")
    return ""


def _loads_lenient(raw: str):
    """json.loads, then with the usual model slips repaired: trailing commas, smart quotes, a stray fence line."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    fixed = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    fixed = re.sub(r"^\s*```.*$", "", fixed, flags=re.M)
    try:
        return json.loads(fixed)
    except Exception:
        return None


def extract_json_block(text: str) -> dict | None:
    """The plan's machine-readable part: every ```json block that parses to an object, merged (later blocks add or
    override keys), so a second block (e.g. the conventions list) that fails to parse never costs the first one."""
    text = strip_reasoning(text)
    merged: dict | None = None
    unparsed = 0
    for m in re.finditer(r"```json[ \t]*\r?\n(.*?)```", text, re.S):
        obj = _loads_lenient(m.group(1))
        if isinstance(obj, dict):
            merged = obj if merged is None else {**merged, **obj}
        else:
            unparsed += 1
    if merged is not None:
        if unparsed:
            log(f"plan: {unparsed} JSON block(s) could not be parsed; using the {len(merged)} key(s) that did")
        return merged
    m = re.search(r"\{\s*\"deliverables\"\s*:\s*\[.*?\]\s*\}", text, re.S)
    if m:
        obj = _loads_lenient(m.group(0))
        if isinstance(obj, dict):
            return obj
    if "```json" in text:
        log("plan: the JSON block(s) could not be parsed")
    return None


def rewrite_paths(code: str, unit: Unit, out_dir: Path, protected_names: set[str] | None = None) -> str:
    """Map the conventional paths of the task text to the real mount layout.

    The task texts say `/app/data/x.csv`, `/app/x.json`, `data/x.csv` or just `x.csv`; the harness mounts
    the unit at /input, so the data really lives under e.g. /input/environment/data/.  Literal path strings
    in the generated code are rewritten to the real locations; output aliases (/app/output, /output,
    output/) are mapped onto the real output directory.
    """
    real_root = str(unit.data_root) if unit.data_root is not None else None
    out = str(out_dir)
    protected = protected_names or set()
    boundary_after = r"(?=[/\"'\s)\]]|$)"
    # Dockerfile aliases (possibly renamed files) take precedence over the generic rules below.
    for alias, real in sorted(unit.aliases.items(), key=lambda kv: -len(kv[0])):
        real_s = str(real)
        if alias == real_s:
            continue
        code = re.sub(r"(?<![\w])" + re.escape(alias) + r"(?![\w])", real_s, code)
        aname = Path(alias).name
        if aname != real.name and aname not in protected:
            code = re.sub(r"([\"'])(?:\./)?(?:(?:environment/)?data/)?" + re.escape(aname) + r"\1",
                          lambda m, p=real_s: m.group(1) + p + m.group(1), code)
    if real_root:
        for alias in ("/app/data", "/app/environment/data", "/input/data", "/data", "/app/input", "/workspace/data", "/input/environment/data"):
            if alias == real_root or real_root.startswith(alias + "/"):
                continue
            code = re.sub(r"(?<![\w])" + re.escape(alias) + boundary_after, real_root, code)
        for df in unit.data_files:
            real = str(df.path)
            names = {df.rel, df.path.name}
            for name in names:
                # bare "/app/<file>" (build-time layout of the unit image)
                code = re.sub(r"(?<![\w])/app/" + re.escape(name) + r"(?![\w.])", real, code)
                # "/input/<file>" when the file is actually deeper (e.g. /input/environment/data/<file>)
                if not real.startswith("/input/" + name):
                    code = re.sub(r"(?<![\w])/input/" + re.escape(name) + r"(?![\w.])", real, code)
            if df.path.name in protected:
                continue  # an output deliverable shares this name; do not touch bare relative uses
            for name in names:
                # quoted relative forms: "data/<rel>", "./data/<rel>", "environment/data/<rel>", "<rel>"
                code = re.sub(r"([\"'])(?:\./)?(?:environment/)?data/" + re.escape(name) + r"\1",
                              lambda m, p=real: m.group(1) + p + m.group(1), code)
                code = re.sub(r"([\"'])(?:\./)?" + re.escape(name) + r"\1",
                              lambda m, p=real: m.group(1) + p + m.group(1), code)
    if out != "/app/output":
        code = re.sub(r"(?<![\w])/app/output" + boundary_after, out, code)
    if out != "/output":
        code = re.sub(r"(?<![\w/])/output" + boundary_after, out, code)
    code = re.sub(r"([\"'])(?:\./)?output/", lambda m: m.group(1) + out + "/", code)
    return code


def make_preamble(unit: Unit, out_dir: Path, seed: int) -> str:
    data_root = str(unit.data_root) if unit.data_root is not None else ""
    lines = [
        "import os as _os, sys as _sys, warnings as _warnings, random as _random",
        "_warnings.filterwarnings('ignore')",
        "_os.environ.setdefault('MPLBACKEND', 'Agg')",
        f"_random.seed({int(seed)})",
        "try:",
        f"    import numpy as _np; _np.random.seed({int(seed)})",
        "except Exception:",
        "    pass",
        f"_os.makedirs({str(out_dir)!r}, exist_ok=True)",
    ]
    if data_root:
        lines.append(f"_sys.path.insert(0, {data_root!r})")
    lines.append("# ---- generated script follows ----")
    return "\n".join(lines) + "\n"


_FUTURE_RE = re.compile(r"^\s*from\s+__future__\s+import\s+[^\n]+$", re.M)


def assemble_script(preamble: str, code: str) -> str:
    """Preamble + script.  `from __future__` imports must be the first statements of a module, so they are hoisted in
    front of the preamble (their original lines stay as blanks, keeping every other line number in place)."""
    futures = _FUTURE_RE.findall(code)
    if futures:
        code = _FUTURE_RE.sub("", code)
    return ("".join(f.strip() + "\n" for f in futures) if futures else "") + preamble + code + "\n"


_EDIT_RE = re.compile(r"<{5,7} ?SEARCH[^\n]*\n(.*?)\n={5,7}[^\n]*\n(.*?)\n?>{5,7} ?REPLACE", re.S)


def apply_edit_blocks(code: str, reply: str) -> str | None:
    """Apply SEARCH/REPLACE blocks to code; None when there are none or one does not match."""
    blocks = _EDIT_RE.findall(reply)
    if not blocks:
        return None
    for search, replace in blocks:
        if search and search in code:
            code = code.replace(search, replace, 1)
            continue
        # tolerate trailing-whitespace / indentation-preserving differences line by line
        lines = code.splitlines()
        want = [l.rstrip() for l in search.splitlines()]
        n = len(want)
        hit = -1
        for i in range(len(lines) - n + 1):
            if [l.rstrip() for l in lines[i:i + n]] == want:
                hit = i
                break
        if hit >= 0 and n > 0:
            lines[hit:hit + n] = replace.splitlines()
            code = "\n".join(lines)
            continue
        span = _fuzzy_span(lines, search)   # the judge / review often re-indents or re-spaces the quoted lines
        if span is None:
            return None
        lines[span[0]:span[1] + 1] = replace.splitlines()
        code = "\n".join(lines)
    return code


def _fuzzy_span(lines: list[str], search: str) -> tuple[int, int] | None:
    """Line span of the unique place where the non-blank lines of `search` equal the code's non-blank lines once all
    whitespace is removed; None when there is no such place or more than one."""
    want = ["".join(l.split()) for l in search.splitlines() if l.strip()]
    if not want:
        return None
    idx = [(i, "".join(l.split())) for i, l in enumerate(lines) if l.strip()]
    hits = [k for k in range(len(idx) - len(want) + 1) if [t for _, t in idx[k:k + len(want)]] == want]
    if len(hits) != 1:
        return None
    k = hits[0]
    return idx[k][0], idx[k + len(want) - 1][0]


_FILE_IN_TEXT_RE = re.compile(r"\b([A-Za-z0-9_][A-Za-z0-9_\-]{1,60}\.(?:csv|json|parquet|txt|md|html|png|svg|pdf|xlsx))\b")
_INVARIANT_RE = re.compile(r"^\s*INVARIANT\s+(\S+)\s+(OK|FAIL)\b\s*(.*)$", re.I)


# --------------------------------------------------------------------------------------
# Text route (AGENT_TEXT_ROUTE): a code-only probe of HTML / XML documents
# --------------------------------------------------------------------------------------

TEXT_DOC_EXTS = (".htm", ".html", ".xml", ".xbrl")
_XBRL_LINKBASE_RE = re.compile(r"_(?:cal|def|lab|pre)\.xml$|\.xsd$", re.I)
_TEXT_KW_STOP = set(("the a an of to in for and or is are be by with on as at from that this it its all each per into than then over "
                     "file files output outputs json csv tsv write writes use value values column columns row rows string int float bool "
                     "true false null none list dict number numbers field fields name names data input inputs report reports must should "
                     "only also not one two three when where which with without using used based given following").split())
# Generic vocabulary of current-report / filing concepts: a task keyword in a group activates the whole group for
# sentence anchoring (press releases say "outlook" where the task says "guidance").
FILING_CONCEPTS: dict[str, tuple[str, ...]] = {
    "guidance": ("guidance", "outlook", "expects", "expected", "expect", "forecast", "anticipates", "projected", "range of"),
    "revenue": ("revenue", "revenues", "net sales", "sales"),
    "departure": ("resign", "resigned", "resignation", "step down", "stepped down", "depart", "departure", "retire",
                  "retirement", "successor", "succeed"),
    "debt": ("notes due", "principal amount", "aggregate principal", "senior notes", "debentures", "offering of",
             "credit agreement", "term loan", "borrowings", "indenture"),
    "dividend": ("dividend", "payable on", "record date"),
    "repurchase": ("repurchase", "repurchased", "buyback", "share repurchase"),
    "earnings": ("net income", "earnings per share", "diluted", "operating income"),
    "acquisition": ("acquire", "acquisition", "merger", "purchase price", "consideration"),
    "insider": ("sold", "weighted average", "weighted-average", "price range", "ranging from", "10b5-1", "trust", "indirect"),
}
_HAS_NUMBER_RE = re.compile(r"\$\s?\d|\d[\d,]*\.?\d*\s?(?:%|percent|million|billion|thousand)|\b\d{1,3}(?:,\d{3})+\b|\b(?:19|20)\d{2}\b")


_DEBUG_TASK_RE = re.compile(r"\bdebug|fix (?:all |the )?(?:computational |remaining )?bugs|\bbuggy\b|colleague wrote|bugs are in|contains bugs|has bugs", re.I)


def template_route_file(unit: "Unit", max_chars: int = 60_000):
    """The script a 'debug and fix' task ships (AGENT_TEMPLATE_ROUTE): the task text talks about debugging / fixing bugs and
    names a .py input (or ships exactly one).  None for every other unit."""
    if not _DEBUG_TASK_RE.search(unit.instruction or ""):
        return None
    pys = [df for df in unit.data_files if df.path.suffix.lower() == ".py" and 0 < df.size <= max_chars]
    if not pys:
        return None
    named = [df for df in pys if df.path.name in unit.instruction]
    if len(named) == 1:
        return named[0]
    if not named and len(pys) == 1:
        return pys[0]
    return None


def typed_time_factor(unit: "Unit") -> tuple[float, str]:
    """AGENT_TIME_TYPED: units with >= 7 deliverables or document inputs never pass in their late minutes locally ->
    0.75 x the soft budget; the others get 1.25 x (more vote candidates where agreement is still reachable)."""
    names = {df.path.name for df in unit.data_files}
    try:
        n = len(deliverables_from_instruction(unit.instruction, names))
    except Exception:
        n = 0
    try:
        docs = bool(text_route_docs(unit))
    except Exception:
        docs = False
    if docs:
        return 0.75, "document inputs"
    if n >= 7:
        return 0.75, f"{n} deliverables"
    return 1.25, f"{n} deliverables"


def _dir_bytes(d: Path) -> int:
    try:
        return sum(p.stat().st_size for p in Path(d).rglob("*") if p.is_file())
    except OSError:
        return 0


def text_route_docs(unit: "Unit") -> list[dict]:
    """HTML / XML documents among the inputs, including members of zip inputs (the text route's trigger)."""
    import zipfile
    docs: list[dict] = []
    for df in unit.data_files:
        ext = df.path.suffix.lower()
        if ext in TEXT_DOC_EXTS:
            docs.append({"path": df.path, "member": None, "name": df.rel or df.path.name, "size": df.size, "ext": ext})
        elif ext == ".zip":
            try:
                with zipfile.ZipFile(df.path) as z:
                    for i in z.infolist():
                        e = os.path.splitext(i.filename)[1].lower()
                        if e in TEXT_DOC_EXTS and not i.is_dir():
                            docs.append({"path": df.path, "member": i.filename, "name": f"{df.path.name}:{i.filename}",
                                         "size": i.file_size, "ext": e})
            except Exception:
                continue
    return docs


def read_text_doc(d: dict, max_bytes: int = 30_000_000) -> str:
    import zipfile
    if d.get("size", 0) > max_bytes:
        return ""
    if d.get("member"):
        with zipfile.ZipFile(d["path"]) as z:
            raw = z.read(d["member"])
    else:
        raw = Path(d["path"]).read_bytes()
    return raw.decode("utf-8", errors="replace")


def task_keywords(instruction: str, limit: int = 30) -> list[str]:
    """Salient lowercase words of the task: quoted / back-ticked terms, JSON-template keys and Capitalised phrases."""
    from collections import Counter
    words: Counter = Counter()

    def add(term: str, w: int) -> None:
        term = re.sub(r"([a-z])([A-Z])", r"\1 \2", term)
        for t in re.split(r"[_\W]+", term):
            t = t.lower()
            if len(t) > 3 and t not in _TEXT_KW_STOP and not t.isdigit():
                words[t] += w
    text = sanitize_text(instruction)
    for m in re.finditer(r"[`\"']([A-Za-z][\w .'/-]{2,80})[`\"']", text):
        add(m.group(1), 1)
    for m in re.finditer(r'"([A-Za-z][\w.\'-]{2,80})"\s*:', text):
        add(m.group(1), 1)
    for m in re.finditer(r"\b([A-Z][a-zA-Z]+(?:[_ ][A-Z][a-zA-Z.']+)+)\b", text):
        add(m.group(1), 2)
    return [w for w, _ in words.most_common(limit)]


def small_input_values(unit: "Unit", max_bytes: int = 20000, limit: int = 40) -> list[str]:
    """Identifier-like values (tickers, ids) of small CSV inputs: where they occur in the documents is a strong anchor."""
    import csv as _csv
    vals: list[str] = []
    for df in unit.data_files:
        if df.path.suffix.lower() not in (".csv", ".tsv") or df.size > max_bytes:
            continue
        try:
            with open(df.path, encoding="utf-8", errors="replace", newline="") as fh:
                rows = list(_csv.DictReader(fh, delimiter="\t" if df.path.suffix.lower() == ".tsv" else ","))
        except Exception:
            continue
        for k in (rows[0].keys() if rows else []):
            col = [str(r.get(k) or "").strip() for r in rows]
            col = [c for c in col if c]
            if col and len(col) <= 60 and all(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,7}", c) for c in col):
                vals += col
    out: list[str] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out[:limit]


def _concepts_for(keywords: list[str]) -> list[str]:
    terms: list[str] = []
    for key, group in FILING_CONCEPTS.items():
        if any(k == key or k in group or any(k in g.split() for g in group) for k in keywords):
            terms += list(group)
    return terms


def html_probe(name: str, html: str, keywords: list[str], values: list[str], budget: int = 2500) -> str:
    """Inline-XBRL facts, item codes, where small-input values occur, then the task-keyword sentences and tables.
    Keywords that occur in most tables / sentences of this document (e.g. "short" in every balance sheet) are not
    used for ranking them."""
    import warnings
    from bs4 import BeautifulSoup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    head = [f"== {name} ({len(html):,} characters of HTML)"]
    facts = soup.find_all(re.compile(r"^ix:non(?:fraction|numeric)$"))
    if facts:
        head.append(f"inline XBRL: {len(facts)} tagged facts (<ix:nonFraction> / <ix:nonNumeric> with attributes name, contextRef, scale); "
                    "dei facts and facts whose name matches the task:")
        shown = 0
        for f in facts:
            nm = f.get("name", "")
            low = re.sub(r"([a-z])([A-Z])", r"\1 \2", nm).lower()
            if nm.startswith("dei:") or any(k in low for k in keywords):
                head.append(f"  {nm} contextRef={f.get('contextref', '')} scale={f.get('scale', '')} = {f.get_text(' ', strip=True)[:70]!r}")
                shown += 1
                if shown >= 16:
                    break
    for t in soup.find_all(re.compile(r"^ix:header$")):
        t.decompose()   # hidden XBRL contexts / units: noise for the text below (facts were read above)
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    items = []
    for it in re.findall(r"\bItem\s+(\d{1,2}\.\d{2})", text):
        if it not in items:
            items.append(it)
    if items:
        head.append("Item codes in the text (document order): " + ", ".join(items[:20]))
    for v in values:
        m = re.search(rf"(.{{0,90}})\b{re.escape(v)}\b(.{{0,60}})", text)
        if m:
            head.append(f"input value {v!r} occurs: ...{m.group(1)}[{v}]{m.group(2)}...")
    # ---- sentences with a number, ranked by filing concepts and specific task keywords
    concepts = _concepts_for(keywords)
    cands = []
    pieces = []
    for sent in re.split(r"(?<=[.;:])\s+(?=[A-Z(])", text):
        pieces += [x for x in re.split(r"\s*[\u2022\u25aa\u25cf\u25e6\ufffd\u00b7]\s*", sent) if x]
    for si, sent in enumerate(pieces):
        if 25 <= len(sent) and _HAS_NUMBER_RE.search(sent):
            cands.append((si, sent, sent.lower()))
    kw_s = [k for k in keywords if len(cands) < 30 or sum(1 for _, _, low in cands if k in low) <= 0.2 * len(cands)]
    sents = []
    for si, sent, low in cands:
        sc = 2 * sum(1 for c in concepts if c in low) + sum(1 for k in kw_s if k in low)
        if sc >= 2:
            sents.append((sc, si, sent))
    sents.sort(key=lambda x: (-x[0], x[1]))
    def window(sent: str) -> str:
        if len(sent) <= 240:
            return sent
        low = sent.lower()
        pos = min([low.find(t) for t in concepts + kw_s if t in low] or [0])
        a = max(0, pos - 60)
        return ("..." if a else "") + sent[a:a + 240] + "..."
    sent_lines = [f"sentence: {window(sent)}" for sc, si, sent in sorted(sents[:6], key=lambda x: x[1])]
    # ---- tables ranked by the task keywords that are specific to them
    tables = soup.find_all("table")
    parsed = []
    for ti, t in enumerate(tables):
        rows = []
        for tr in t.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c and c not in ("$", "%", ")")]
            if cells:
                rows.append(cells)
        if rows:
            parsed.append((ti, rows, " ".join(" ".join(r) for r in rows).lower()))
    kw_t = [k for k in keywords if len(parsed) < 5 or sum(1 for _, _, j in parsed if k in j) <= 0.3 * len(parsed)]
    scored = []
    for ti, rows, joined in parsed:
        score = sum(1 for k in kw_t if k in joined)
        if score >= 2:
            scored.append((score, ti, rows))
    scored.sort(key=lambda x: (-x[0], x[1]))
    table_lines = []
    for score, ti, rows in scored[:3]:
        table_lines.append(f"table #{ti} of {len(tables)} (matches {score} task keywords):")
        table_lines += ["  | " + " | ".join(r)[:170] for r in rows[:8]]
    out = "\n".join(head + sent_lines)
    rest = budget - len(out) - 1
    if rest > 120 and table_lines:
        out += "\n" + "\n".join(table_lines)[:rest]
    return out[:budget]


def xml_probe(name: str, xml: str, budget: int = 2000) -> str:
    import xml.etree.ElementTree as ET
    from collections import Counter, defaultdict
    try:
        root = ET.fromstring(xml)
    except Exception as exc:
        return f"== {name}: XML could not be parsed ({type(exc).__name__}); head:\n" + xml[:600]
    paths: Counter = Counter()
    leaf: dict = defaultdict(Counter)
    foot: list[str] = []
    first: dict[str, ET.Element] = {}

    def strip(tag: str) -> str:
        return tag.split("}")[-1]

    def walk(e, path):
        tag = strip(e.tag)
        p = f"{path}/{tag}"
        paths[p] += 1
        first.setdefault(p, e)
        txt = (e.text or "").strip()
        if len(e) == 0 and txt and len(txt) <= 50:
            leaf[p][txt] += 1
        if tag.lower() == "footnote":
            foot.append(f"{e.get('id', '?')}: {' '.join(''.join(e.itertext()).split())[:220]}")
        for c in e:
            walk(c, p)
    walk(root, "")
    out = [f"== {name}: XML root <{strip(root.tag)}>, {sum(paths.values())} elements"]
    rep_el = [(p, c) for p, c in paths.items() if c > 1 and len(first[p]) > 0 and p.count("/") <= 3]
    if rep_el:
        out.append("repeated records: " + ", ".join(f"{p.rsplit('/', 1)[-1]} x{c}" for p, c in rep_el[:8]))
    key = re.compile(r"code|type|ownership|nature|title|acquired|disposed|form|status|class", re.I)
    shown = 0
    for p, cnt in leaf.items():
        tot = sum(cnt.values())
        if tot >= 2 and len(cnt) <= 8 and key.search(p.rsplit("/", 2)[-2] + "/" + p.rsplit("/", 1)[-1]):
            out.append(f"  {'/'.join(p.split('/')[-4:])}: " + ", ".join(f"{v} x{k}" for v, k in cnt.most_common(8)))
            shown += 1
            if shown >= 12:
                break
    if rep_el:
        p0 = max(rep_el, key=lambda x: x[1])[0]
        recs = []

        def leaves(el, parent=""):
            for c in el:
                t = (c.text or "").strip()
                tag = strip(c.tag)
                if len(c) == 0 and t:
                    recs.append(f"{parent if tag == 'value' and parent else tag}={t[:40]}")
                else:
                    leaves(c, tag)
        leaves(first[p0])
        out.append(f"first <{p0.rsplit('/', 1)[-1]}>: " + "; ".join(recs[:24]))
    for f in foot[:12]:
        out.append(f"  footnote {f}")
    return "\n".join(out)[:budget]


def helper_code_notes(unit: "Unit") -> str:
    """Python helpers shipped with the inputs (e.g. an html_to_text.py), with their functions and first docstring line."""
    import ast
    notes = []
    for df in unit.data_files:
        if df.path.suffix.lower() != ".py" or df.path.name.lower().startswith("template") or df.size > 200_000:
            continue
        try:
            tree = ast.parse(df.path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        doc = (ast.get_docstring(tree) or "").strip().splitlines()
        fns = [f"{f.name}({', '.join(a.arg for a in f.args.args)})" for f in tree.body if isinstance(f, ast.FunctionDef)]
        notes.append(f"- {df.path}: " + (doc[0][:120] + "; " if doc else "") + ("functions: " + ", ".join(fns[:8]) if fns else "no functions"))
    if not notes:
        return ""
    return "Helper code shipped with the inputs (prefer using it; import it with sys.path.insert(0, <its directory>)):\n" + "\n".join(notes)


def text_probe(unit: "Unit", docs: list[dict], budget: int = 9000) -> str:
    """The DOCUMENT PROBE section for the text route."""
    keywords = task_keywords(unit.instruction)
    values = small_input_values(unit)
    main = [d for d in docs if d["ext"] in (".htm", ".html") or not _XBRL_LINKBASE_RE.search(d["name"])]
    linkbase = [d for d in docs if d not in main]
    # inside a zip, probe the largest HTML member(s) - the main document - first
    main.sort(key=lambda d: (d["member"] is None, -(d["size"] if d["member"] else 0), d["name"]))
    zipped_html = [d for d in main if d["member"] and d["ext"] in (".htm", ".html")]
    by_zip: dict = {}
    for d in zipped_html:
        by_zip.setdefault(str(d["path"]), d)   # largest html member of each zip
    chosen = list(by_zip.values()) + [d for d in main if not d["member"]]
    chosen = chosen[:8]
    per = max(1200, budget // max(1, len(chosen)) - 100)
    parts = ["DOCUMENT PROBE (read by the agent from the actual input documents, no model involved; it shows WHERE the "
             "values are - parse the documents in your script):",
             f"task keywords used for anchoring: {', '.join(keywords[:15]) or '(none)'}"]
    if values:
        parts.append("identifier values of small inputs: " + ", ".join(values[:20]))
    for d in chosen:
        try:
            raw = read_text_doc(d)
            if not raw:
                parts.append(f"== {d['name']}: too large to probe ({d['size']:,} bytes)")
                continue
            how = f"(inside {Path(d['path']).name}: read with zipfile.ZipFile(path).read({d['member']!r}))" if d["member"] else ""
            if d["ext"] in (".htm", ".html") or (d["ext"] == ".xml" and "<html" in raw[:2000].lower()):
                body = html_probe(d["name"], raw, keywords, values, per)
            else:
                body = xml_probe(d["name"], raw, per)
            parts.append((how + "\n" if how else "") + body)
        except Exception as exc:
            parts.append(f"== {d['name']}: probe failed ({type(exc).__name__}: {str(exc)[:80]})")
    if linkbase:
        parts.append("XBRL linkbase / schema members (labels, calculations; usually not needed): " +
                     ", ".join(d["name"] for d in linkbase[:10]))
    helpers = helper_code_notes(unit)
    if helpers:
        parts.append(helpers)
    return "\n".join(parts)[:budget + 2000]


def doc_plain_text(docs: list[dict], max_chars: int = 6_000_000) -> str:
    """Lower-cased, whitespace-collapsed text of the routed documents (EXTRACT source checks)."""
    import warnings
    out = []
    used = 0
    for d in docs:
        if used >= max_chars or _XBRL_LINKBASE_RE.search(d["name"]):
            continue
        try:
            raw = read_text_doc(d)
            if d["ext"] in (".htm", ".html"):
                from bs4 import BeautifulSoup
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    txt = BeautifulSoup(raw, "lxml").get_text(" ", strip=True)
            else:
                txt = re.sub(r"<[^>]+>", " ", raw)
            txt = re.sub(r"\s+", " ", txt).lower()
            out.append(txt[: max_chars - used])
            used += len(out[-1])
        except Exception:
            continue
    return "\n".join(out)


_EXTRACT_RE = re.compile(r"^\s*EXTRACT\s+(\S+)\s+(.*?)\s*(?:\|\s*(.*))?$")


def parse_extract_lines(stdout: str) -> list[tuple[str, str, str]]:
    out = []
    for ln in (stdout or "").splitlines():
        m = _EXTRACT_RE.match(ln)
        if m:
            out.append((m.group(1), (m.group(2) or "").strip(), (m.group(3) or "").strip()))
    return out


def extract_issues(lines: list[tuple[str, str, str]], doc_text: str) -> tuple[list[str], list[str]]:
    """(hard, soft) findings about EXTRACT lines: missing / empty values, mostly-zero extraction, sources not in the docs."""
    hard: list[str] = []
    soft: list[str] = []
    if not lines:
        return hard, ["the script printed no EXTRACT lines (print one per value read from a document)"]
    _EMPTY = ("", "missing", "none", "null", "nan", "n/a", "''", '""')
    empty = [f for f, v, _ in lines if v.strip().lower() in _EMPTY]
    zero = [f for f, v, _ in lines if re.fullmatch(r"[-+]?0+(?:\.0+)?", v.strip() or "x")]
    if empty:
        hard.append(f"extraction failed for {len(empty)} of {len(lines)} field(s): {', '.join(empty[:8])} - print the text around "
                    "where each should be found (the DOCUMENT PROBE shows the tables / tags) and fix the parsing")
    if len(lines) >= 4 and (len(empty) + len(zero)) * 2 >= len(lines):
        hard.append(f"{len(empty) + len(zero)} of {len(lines)} extracted values are empty or 0: the document parsing does not find the values")
    elif zero:
        soft.append(f"extracted value 0 for {', '.join(zero[:6])} - check that 0 is really what the document says")
    if doc_text:
        missing_src = []
        for f, v, src in lines:
            if v.strip().lower() in _EMPTY:
                continue   # for a missing value the text is what was searched for, not a quote
            s2 = re.sub(r"\s+", " ", src).strip().lower().strip("'\"")
            if len(s2) >= 12 and s2[:60] not in doc_text:
                missing_src.append(f)
        if missing_src:
            soft.append(f"the source text printed for {', '.join(missing_src[:6])} does not occur in the documents - verify those values")
    return hard, soft


def degenerate_issues(out_dir: Path, deliverables: list, instruction: str = "", input_names: set[str] | None = None,
                      initial: set[str] | None = None) -> list[str]:
    """Outputs that exist and parse but cannot be a real answer: numeric columns that are all zero / all missing, text
    columns that are all empty, JSON documents whose numbers are (mostly) all zero or whose strings are empty, and
    output files the task text names (by <name>.<ext>) that were never written.  Pure file inspection, no model."""
    issues: list[str] = []
    for d in deliverables:
        p = out_dir / d.file
        if not p.is_file():
            continue
        ext = d.file.rsplit(".", 1)[-1].lower() if "." in d.file else d.fmt
        try:
            if ext in ("csv", "tsv"):
                import pandas as pd

                df = pd.read_csv(p, nrows=5000, sep="\t" if ext == "tsv" else ",")
                if df.shape[0] < 1:
                    continue
                bad = []
                for c in df.columns:
                    col = df[c]
                    if col.isna().all():
                        bad.append(f"{c} (all missing)")
                    elif pd.api.types.is_numeric_dtype(col):
                        if df.shape[0] >= 3 and (col.fillna(0) == 0).all():
                            bad.append(f"{c} (all zero)")
                    else:
                        if (col.astype(str).str.strip() == "").all():
                            bad.append(f"{c} (all empty strings)")
                if bad:
                    issues.append(f"degenerate: {d.file}: columns with no information: {', '.join(bad[:6])}")
            elif ext == "json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                nums = _numeric_leaves(obj)
                vals = [v for k, v in nums.items() if not k.endswith(("seed", "tier"))]
                if len(vals) >= 3 and sum(1 for v in vals if v == 0.0) * 2 >= len(vals):
                    zeros = [k for k, v in nums.items() if v == 0.0][:6]
                    issues.append(f"degenerate: {d.file}: {sum(1 for v in vals if v == 0.0)} of {len(vals)} numeric values are exactly 0 (e.g. {', '.join(zeros)})")
                empties = _empty_string_leaves(obj)
                if empties:
                    issues.append(f"degenerate: {d.file}: empty strings at {', '.join(empties[:6])}")
                if isinstance(obj, dict):
                    empty_lists = [k for k, v in obj.items() if isinstance(v, (list, dict)) and len(v) == 0]
                    if empty_lists and len(empty_lists) * 2 >= max(1, len(obj)):
                        issues.append(f"degenerate: {d.file}: empty lists/objects at {', '.join(empty_lists[:6])}")
        except Exception:
            continue
    if instruction:
        written = {str(q.relative_to(out_dir)) for q in out_dir.rglob("*") if q.is_file()}
        written_names = {w.rsplit("/", 1)[-1] for w in written} | set(initial or ())
        inputs = {x.rsplit("/", 1)[-1] for x in (input_names or set())}
        deliv_names = {d.file.rsplit("/", 1)[-1] for d in deliverables}
        named = []
        for m in _FILE_IN_TEXT_RE.finditer(instruction):
            name = m.group(1)
            if name in inputs or name in written_names or name in named or name.lower().startswith(("readme", "requirements", "dockerfile", "test_")):
                continue
            if name in deliv_names:
                continue  # already a missing deliverable
            named.append(name)
        if named:
            issues.append("degenerate: files named in the task text but not written: " + ", ".join(named[:6]))
    return issues


def _empty_string_leaves(obj: Any, path: str = "", out: list | None = None) -> list[str]:
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _empty_string_leaves(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:100]):
            _empty_string_leaves(v, f"{path}[{i}]", out)
    elif isinstance(obj, str) and obj.strip() == "":
        out.append(path or "(root)")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Referential integrity (AGENT_KEY_CHECK): an output column / JSON key named like an identifier column of an input table
# (ticker, tenor, rating, scenario, filing_id ...) must take values of that input column.  Typical failures it catches:
# a regex that grabs a neighbouring header word instead of the ticker, file paths written as ids, "c"/"p" for "call"/"put",
# dates written into a label column, invented pillars.  Pure file inspection, no model.
# ---------------------------------------------------------------------------------------------------------------------
_KEY_DATE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?([ T].*)?$|^\d{1,2}/\d{1,2}/\d{2,4}$|^\d{8}$"
                          r"|^\d{1,2}[-/ ][A-Za-z]{3,9}[-/ ]\d{2,4}$|^[A-Za-z]{3,9}\.? \d{1,2},? \d{4}$|^\d{4}/\d{1,2}/\d{1,2}$")
_KEY_NUM_RE = re.compile(r"[-+]?[\d.,eE%]+")
_KEY_EMPTY = ("", "nan", "none", "null")


def input_key_sets(paths: list, max_bytes: int = 30_000_000, max_distinct: int = 2000) -> dict:
    """Identifier-like string columns of the input tables (csv / tsv / parquet): lower-case column name ->
    (source "file:column", set of values, whether the input column itself has empty cells).  A column qualifies when it
    has 2..max_distinct distinct values of at most 40 characters that are mostly neither numbers nor dates; a column
    name shared by several inputs takes the union of their values."""
    import pandas as pd

    keys: dict = {}
    for p in paths:
        p = Path(p)
        ext = p.suffix.lower()
        try:
            if ext not in (".csv", ".tsv", ".parquet") or p.stat().st_size > max_bytes:
                continue
            if ext == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p, sep="\t" if ext == ".tsv" else ",", dtype=str, nrows=200_000)
        except Exception:
            continue
        for c in df.columns:
            raw = df[c]
            if not (raw.dtype == object or pd.api.types.is_string_dtype(raw)):
                continue
            col = raw.dropna().astype(str).str.strip()
            col = col[col != ""]
            if len(col) < 2:
                continue
            vals = set(col.unique())
            if not (2 <= len(vals) <= max_distinct) or any(len(v) > 40 for v in vals):
                continue
            if sum(bool(_KEY_NUM_RE.fullmatch(v)) for v in vals) > len(vals) * 0.2:
                continue
            if sum(bool(_KEY_DATE_RE.match(v)) for v in vals) > len(vals) * 0.2:
                continue
            name = str(c).strip().lower()
            if not re.search(r"[a-z]", name):
                continue
            has_empty = len(col) < len(raw)
            prev = keys.get(name)
            if prev is None:
                keys[name] = (f"{p.name}:{c}", vals, has_empty)
            else:
                src = prev[0] if f"{p.name}:" in prev[0] else f"{prev[0]} / {p.name}:{c}"
                keys[name] = (src, prev[1] | vals, prev[2] or has_empty)
    return keys


def _named_in_task(value: str, instruction: str) -> bool:
    """A literal the task text asks for: quoted / back-ticked, or an upper-case token (TOTAL, EW, PORTFOLIO) as a word."""
    if not value or not instruction:
        return False
    v = re.escape(value)
    if re.search(rf"[`\"'“‘]{v}[`\"'”’]", instruction):
        return True
    return len(value) >= 2 and bool(re.fullmatch(r"[A-Z0-9_]+", value)) and bool(
        re.search(rf"(?<![A-Za-z0-9_]){v}(?![A-Za-z0-9_])", instruction))


def key_issues(out_files: list, keys: dict, instruction: str = "") -> tuple[list[str], list[str]]:
    """(hard, soft) findings for output CSV columns and JSON keys named like an input identifier column: values that are
    not values of that column (values quoted in the task text are allowed), or empty values where the input column is
    never empty.  Differences only in letter case are soft."""
    import pandas as pd

    hard: list[str] = []
    soft: list[str] = []
    if not keys:
        return hard, soft
    instr = instruction or ""

    def judge(where: str, name: str, values: list) -> None:
        src, allowed, input_empty = keys[name]
        bad: list[str] = []
        empty = 0
        for v in values:
            sv = "" if v is None else str(v).strip()
            if sv.lower() in _KEY_EMPTY:
                empty += 1
                continue
            if sv in allowed or _named_in_task(sv, instr):
                continue
            bad.append(sv)
        if empty and not input_empty:
            hard.append(f"key: {where}: {empty} of {len(values)} value(s) are empty, but every {name} in the input {src} is "
                        f"filled; fill it with the matching input value")
        if not bad:
            return
        lower = {a.lower() for a in allowed}
        if all(b.lower() in lower for b in bad):
            soft.append(f"key: {where}: {len(bad)} value(s) differ only in letter case from the input {src} (e.g. {bad[:3]})")
            return
        uniq = list(dict.fromkeys(bad))
        ex = sorted(allowed)
        hard.append(f"key: {where}: {len(bad)} value(s) are not values of the input {src}: {uniq[:4]}; each {name} must be "
                    f"one of the input's values (e.g. {', '.join(ex[:8])}{', ...' if len(ex) > 8 else ''}) - identify it by "
                    f"matching against those values, not by position or a loose regex")

    for p in out_files:
        p = Path(p)
        ext = p.suffix.lower()
        try:
            if ext in (".csv", ".tsv"):
                df = pd.read_csv(p, sep="\t" if ext == ".tsv" else ",", dtype=str, nrows=20000, keep_default_na=False)
                for c in df.columns:
                    n = str(c).strip().lower()
                    if n in keys:
                        judge(f"{p.name} column {c}", n, df[c].tolist())
            elif ext == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                stack: list = [("", obj)]
                seen = 0
                while stack and seen < 20000:
                    path, o = stack.pop()
                    seen += 1
                    if isinstance(o, dict):
                        for k, v in list(o.items())[:2000]:
                            kp = f"{path}.{k}" if path else str(k)
                            toks = {t for t in re.split(r"[^a-z0-9]+", str(k).lower()) if t}
                            hit = [n for n in keys if n == str(k).lower() or n in toks]
                            if hit and (isinstance(v, str) or v is None):
                                judge(f"{p.name}:{kp}", hit[0], [v])
                            elif hit and isinstance(v, list) and v and all(isinstance(x, str) for x in v[:50]):
                                judge(f"{p.name}:{kp}", hit[0], v[:500])
                            else:
                                stack.append((kp, v))
                    elif isinstance(o, list):
                        for i, v in enumerate(o[:500]):
                            stack.append((f"{path}[{i}]", v))
        except Exception:
            continue
    return hard, soft


def conventions_text(plan_json: dict | None, limit: int = 3500) -> str:
    """The plan's conventions list rendered for the code request, the review and the vote candidates."""
    raw = plan_json.get("conventions") if isinstance(plan_json, dict) else None
    if not isinstance(raw, list):
        return ""
    lines = []
    for c in raw[:25]:
        if not isinstance(c, dict):
            continue
        item, choice, src = str(c.get("item", "")).strip(), str(c.get("choice", "")).strip(), str(c.get("task_text", "")).strip()
        if not item or not choice:
            continue
        lines.append(f"- {item}: {choice}" + (f"  [task: {src[:160]}]" if src and src.lower() != "not stated" else "  [not stated in the task; standard convention]"))
    if not lines:
        return ""
    text = "CONVENTIONS decided in the plan - implement EXACTLY these (they are what the task's expected values depend on):\n" + "\n".join(lines)
    return text[:limit]


def _conv_key(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return " ".join(w for w in words if w not in ("the", "of", "a", "an", "for", "and", "to", "in", "used", "value"))[:60]


def parse_conventions_reply(text: str) -> list[dict]:
    """The conventions list from a model reply: a parsable JSON block, else any fenced/raw {"conventions": [...]} object,
    else item/choice pairs picked out by regex from a malformed block."""
    obj = extract_json_block(text)
    lst = obj.get("conventions") if isinstance(obj, dict) else None
    if isinstance(lst, list) and lst:
        return [c for c in lst if isinstance(c, dict)]
    body = strip_reasoning(text)
    m = re.search(r"\{\s*\"conventions\"\s*:\s*\[.*\]\s*\}", body, re.S)
    if m:
        obj = _loads_lenient(m.group(0))
        lst = obj.get("conventions") if isinstance(obj, dict) else None
        if isinstance(lst, list) and lst:
            return [c for c in lst if isinstance(c, dict)]
    out = []
    pat = r"\"item\"\s*:\s*\"((?:[^\"\\\\]|\\\\.)*)\"(.*?)\"choice\"\s*:\s*\"((?:[^\"\\\\]|\\\\.)*)\""
    for im in re.finditer(pat, body, re.S):
        tt = re.search(r"\"task_text\"\s*:\s*\"((?:[^\"\\\\]|\\\\.)*)\"", im.group(2))
        out.append({"item": im.group(1), "task_text": tt.group(1) if tt else "", "choice": im.group(3)})
    return out[:30]


def merge_conventions(reads: list[list]) -> tuple[list[dict], int, int]:
    """Majority per convention item over independent reads (the first read is the plan's own list).  Items are matched
    by their normalised name, choices by normalised text; ties keep the first read.  Returns (merged, items seen in
    >= 2 reads, items whose first-read choice was outvoted)."""
    by_key: dict[str, list[tuple[int, dict]]] = {}
    order: list[str] = []
    for i, lst in enumerate(reads):
        for c in lst or []:
            if not isinstance(c, dict) or not str(c.get("item", "")).strip() or not str(c.get("choice", "")).strip():
                continue
            k = _conv_key(c["item"])
            if not k:
                continue
            if k not in by_key:
                by_key[k] = []
                order.append(k)
            by_key[k].append((i, c))
    merged: list[dict] = []
    multi = changed = 0
    for k in order:
        entries = by_key[k]
        if len({i for i, _ in entries}) >= 2:
            multi += 1
        votes: dict[str, int] = {}
        first: dict[str, dict] = {}
        for i, c in entries:
            ck = _conv_key(c["choice"])
            votes[ck] = votes.get(ck, 0) + 1
            first.setdefault(ck, c)
        win = max(votes, key=lambda ck: (votes[ck], -list(votes).index(ck)))
        chosen = dict(first[win])
        if entries[0][0] == 0 and _conv_key(entries[0][1]["choice"]) != win:
            changed += 1
        merged.append({"item": str(chosen.get("item", "")), "task_text": str(chosen.get("task_text", "")), "choice": str(chosen.get("choice", ""))})
    return merged[:30], multi, changed


_PARAM_LINE_RE = re.compile(r"^\s*PARAM\s+(\S+)\s+(\S+)\s*$")
_NUM_RE = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)(\s*%)?(?![\w])")
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def parse_param_lines(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for ln in (stdout or "").splitlines():
        m = _PARAM_LINE_RE.match(ln)
        if m:
            out[m.group(1)] = m.group(2).strip().strip(",;")
    return out


def stated_values(*texts: str) -> tuple[set[float], set[str]]:
    """Numbers (with percent variants) and ISO dates that appear in the task text / parameter files."""
    nums: set[float] = set()
    dates: set[str] = set()
    for t in texts:
        if not t:
            continue
        for m in _NUM_RE.finditer(t):
            try:
                v = float(m.group(1).replace(",", ""))
            except ValueError:
                continue
            nums.add(v)
            if m.group(2):
                nums.add(v / 100.0)
        dates.update(_DATE_RE.findall(t))
    return nums, dates


def unstated_params(echo: dict[str, str], nums: set[float], dates: set[str]) -> list[str]:
    """PARAM echoes whose value is neither stated in the task nor trivially derived (0/1, a stated number x or / 100, 252 / 365 / 360)."""
    out: list[str] = []
    trivial = {0.0, 1.0, 2.0, 252.0, 250.0, 365.0, 360.0, 12.0, 100.0}
    for name, raw in echo.items():
        if _DATE_RE.fullmatch(raw):
            if dates and raw not in dates:
                out.append(f"{name}={raw} (dates stated in the task: {', '.join(sorted(dates)[:6])})")
            continue
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        if v in trivial or any(abs(v - x) <= 1e-9 * max(1.0, abs(x)) for x in nums) or \
           any(abs(v - x / 100.0) <= 1e-9 or abs(v - x * 100.0) <= 1e-6 for x in nums):
            continue
        out.append(f"{name}={raw} is not a value stated in the task")
    return out[:8]


def parse_invariants(stdout: str) -> tuple[list[str], list[str]]:
    """(failed, ok) INVARIANT lines printed by the script."""
    failed, ok = [], []
    for ln in (stdout or "").splitlines():
        m = _INVARIANT_RE.match(ln)
        if not m:
            continue
        if m.group(2).upper() == "FAIL":
            failed.append(f"{m.group(1)} {m.group(3).strip()}"[:220])
        else:
            ok.append(m.group(1))
    return failed, ok


def _numeric_leaves(obj: Any, path: str = "", out: dict | None = None) -> dict:
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _numeric_leaves(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:200]):
            _numeric_leaves(v, f"{path}[{i}]", out)
    elif isinstance(obj, bool):
        out[path] = float(obj)
    elif isinstance(obj, (int, float)):
        out[path] = float(obj)
    return out


def _differs(a: float, b: float, rtol: float = 2e-3, atol: float = 1e-8) -> bool:
    import math
    if math.isnan(a) and math.isnan(b):
        return False
    return not (abs(a - b) <= atol + rtol * max(abs(a), abs(b)))


def compare_outputs(ours: Path, other: Path, deliverables: list, limit: int = 15, rtol: float = 2e-3) -> list[str]:
    """Human-readable list of numeric disagreements between two output directories (rtol: relative tolerance)."""
    diffs: list[str] = []
    for d in deliverables:
        a, b = ours / d.file, other / d.file
        if not a.is_file():
            continue
        if not b.is_file():
            diffs.append(f"{d.file}: the independent implementation did not write it")
            continue
        ext = d.file.rsplit(".", 1)[-1].lower()
        try:
            if ext == "json":
                la = _numeric_leaves(json.loads(a.read_text(encoding="utf-8")))
                lb = _numeric_leaves(json.loads(b.read_text(encoding="utf-8")))
                for k, va in la.items():
                    if k in lb and _differs(va, lb[k], rtol):
                        diffs.append(f"{d.file} {k}: yours={va:.6g} independent={lb[k]:.6g}")
            elif ext in ("csv", "tsv"):
                import pandas as pd
                sep = "\t" if ext == "tsv" else ","
                da, db = pd.read_csv(a, sep=sep), pd.read_csv(b, sep=sep)
                if da.shape != db.shape:
                    diffs.append(f"{d.file}: shape yours={da.shape} independent={db.shape}")
                    continue
                for c in da.columns:
                    if c not in db.columns:
                        continue
                    xa = pd.to_numeric(da[c], errors="coerce"); xb = pd.to_numeric(db[c], errors="coerce")
                    if xa.notna().sum() == 0:
                        continue
                    bad = [(i, x, y) for i, (x, y) in enumerate(zip(xa, xb)) if not (pd.isna(x) and pd.isna(y))
                           and (pd.isna(x) or pd.isna(y) or _differs(float(x), float(y), rtol))]
                    if bad:
                        i, x, y = bad[0]
                        diffs.append(f"{d.file} column {c}: {len(bad)}/{len(xa)} rows differ, e.g. row {i}: yours={x} independent={y}")
        except Exception as exc:
            diffs.append(f"{d.file}: could not compare ({type(exc).__name__})")
        if len(diffs) >= limit:
            break
    return diffs[:limit]


# --------------------------------------------------------------------------------------
# Staged build (AGENT_STAGED) and independent verification (AGENT_VERIFY) helpers
# --------------------------------------------------------------------------------------

_MAIN_GUARD_RE = re.compile(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:\s*(#.*)?$")
_STAGE_IMPORT_RE = re.compile(r"^\s*(?:from\s+(inputs|compute)\s+import\s+.*|import\s+(inputs|compute)(?:\s+as\s+\w+)?\s*)$")
_CHECK_LINE_RE = re.compile(r"^\s*CHECK\s+(\S+)\s+(.+?)\s*$")


def _strip_main_guard(code: str) -> str:
    """Drop every `if __name__ == '__main__':` block (with its indented body) from a module."""
    lines = code.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _MAIN_GUARD_RE.match(lines[i]):
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i][0] in " \t"):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def flatten_modules(mods: list[tuple[str, str]]) -> str:
    """Join staged modules into one script: imports between the modules are removed, `import inputs` style
    attribute access becomes plain names, and every module but the last loses its __main__ block."""
    parts: list[str] = []
    for k, (name, code) in enumerate(mods):
        body = code if k == len(mods) - 1 else _strip_main_guard(code)
        kept: list[str] = []
        aliases: dict[str, str] = {}
        for ln in body.splitlines():
            m = _STAGE_IMPORT_RE.match(ln)
            if m and (m.group(1) or m.group(2)) != name:
                if m.group(2):
                    am = re.match(r"^\s*import\s+(\w+)\s+as\s+(\w+)", ln)
                    aliases[am.group(2) if am else m.group(2)] = m.group(2)
                continue
            kept.append(ln)
        text = "\n".join(kept)
        for alias in aliases:  # `inputs.load_inputs()` -> `load_inputs()`
            text = re.sub(r"(?<![\w.])" + re.escape(alias) + r"\.(?=[A-Za-z_])", "", text)
        parts.append(f"# ---- {name}.py ----\n" + text.strip("\n"))
    return "\n\n".join(parts) + "\n"


def staged_applicable(n_deliverables: int, n_data_files: int, plan_json: dict | None) -> bool:
    """Staged construction pays off for tasks with several deliverables or inputs or a long script."""
    est = 0
    if isinstance(plan_json, dict):
        try:
            est = int(float(plan_json.get("est_lines") or 0))
        except Exception:
            est = 0
    return n_deliverables >= 3 or n_data_files >= 2 or est >= 150


def _error_line(text: str, limit: int = 160) -> str:
    """The most informative line of a failure report: the last exception / CHECK NA / kill line, else the last line."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    for l in reversed(lines):
        if re.match(r"^[A-Za-z_.]*(Error|Exception|Warning)\b", l) or l.startswith("CHECK ") or l.startswith("The script was killed"):
            return l[:limit]
    return (lines[-1] if lines else "")[:limit]


def _close(a: float, b: float, tol: float) -> bool:
    """Agreement test of the verification: relative difference within tol (at least 2e-3) or absolute 1e-9."""
    rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
    return rel <= max(2e-3, tol) or abs(a - b) <= 1e-9


def parse_check_lines(stdout: str) -> dict[str, float | None]:
    """`CHECK <name> <number|NA>` lines of an independent check script -> {name: value or None}."""
    import math
    vals: dict[str, float | None] = {}
    for ln in stdout.splitlines():
        m = _CHECK_LINE_RE.match(ln)
        if not m:
            continue
        name, raw = m.group(1), m.group(2).strip().strip(",;")
        try:
            v = float(raw.replace(",", ""))
            vals[name] = v if math.isfinite(v) else None
        except ValueError:
            vals[name] = None
    return vals


def _aggregate(nums: list[float], agg: str) -> float | None:
    if not nums:
        return None
    if agg == "last":
        return nums[-1]
    if agg == "mean":
        return sum(nums) / len(nums)
    if agg == "sum":
        return sum(nums)
    if agg == "min":
        return min(nums)
    if agg == "max":
        return max(nums)
    if agg == "count":
        return float(len(nums))
    return nums[0]  # first / value


def read_quantity(out_dir: Path, q: dict) -> float | None:
    """Value of a key quantity from the written outputs: a JSON path (a.b[2].c) or a CSV column aggregate."""
    file, path, agg = str(q.get("file", "")), str(q.get("path", "")), str(q.get("agg", "value") or "value").lower()
    p = out_dir / file
    if not p.is_file():
        return None
    ext = file.rsplit(".", 1)[-1].lower()
    try:
        if ext == "json":
            obj: Any = json.loads(p.read_text(encoding="utf-8"))
            for tok in re.findall(r"\[\d+\]|[^.\[\]]+", path):
                if tok.startswith("["):
                    obj = obj[int(tok[1:-1])]
                elif isinstance(obj, dict):
                    obj = obj[tok] if tok in obj else obj[tok.strip()]
                elif isinstance(obj, list) and tok.isdigit():
                    obj = obj[int(tok)]
                else:
                    return None
            if isinstance(obj, list):
                nums = [float(x) for x in obj if isinstance(x, (int, float)) and not isinstance(x, bool)]
                return _aggregate(nums, agg)
            if isinstance(obj, bool) or not isinstance(obj, (int, float)):
                return None
            return float(obj)
        if ext in ("csv", "tsv"):
            import pandas as pd

            df = pd.read_csv(p, sep="\t" if ext == "tsv" else ",")
            if path not in df.columns:
                return None
            col = pd.to_numeric(df[path], errors="coerce").dropna()
            return _aggregate([float(x) for x in col], agg)
    except Exception:
        return None
    return None


def auto_quantities(out_dir: Path, deliverables: list, limit: int = 6) -> list[dict]:
    """Fallback key quantities when the plan named none: scalar JSON leaves (fractional values first), then means
    of numeric CSV columns with more than two distinct values."""
    out: list[dict] = []

    def fractional(v: float) -> bool:
        try:
            return v != int(v)
        except (ValueError, OverflowError):
            return False

    for d in deliverables:
        p = out_dir / d.file
        ext = d.file.rsplit(".", 1)[-1].lower() if "." in d.file else ""
        if not p.is_file():
            continue
        try:
            if ext == "json":
                leaves = _numeric_leaves(json.loads(p.read_text(encoding="utf-8")))
                scal = [(k, v) for k, v in leaves.items() if "[" not in k]
                picks = [(k, v) for k, v in scal if fractional(v)] or [(k, v) for k, v in scal if v not in (0.0, 1.0)] or scal
                for k, v in picks:
                    if len(out) >= limit:
                        break
                    out.append({"file": d.file, "path": k, "agg": "value", "meaning": k})
            elif ext in ("csv", "tsv"):
                import pandas as pd

                df = pd.read_csv(p, sep="\t" if ext == "tsv" else ",", nrows=5000)
                for c in df.columns:
                    if len(out) >= limit:
                        break
                    col = pd.to_numeric(df[c], errors="coerce")
                    if col.notna().sum() >= 1 and col.nunique() > 2:
                        out.append({"file": d.file, "path": str(c), "agg": "mean", "meaning": f"mean of column {c}"})
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out[:limit]


DATA_ERR_RE = re.compile(r"KeyError|not in index|No columns|has no attribute|IndexError|out of bounds|could not convert|"
                         r"Invalid frequency|time data|unpack|broadcast|not enough values|too many values|shape|Unnamed|"
                         r"UnicodeDecodeError|ParserError|EmptyDataError|FileNotFoundError|No such file|Worksheet|sheet", re.I)
SPLIT_MARKER = "# === PART 2 ==="


def data_facts(unit: Unit, limit: int = 3500, max_files: int = 12) -> str:
    """Exact structure of the input files, read from disk (no model involved): columns with dtypes and row counts of
    tables, raw leading rows of Excel sheets, top-level keys of JSON documents.  Added to the repair feedback after a
    data-shaped exception so the fix is based on facts, not on guessed column names."""
    parts: list[str] = []
    used = 0
    for df in unit.data_files[:max_files]:
        p = df.path
        ext = p.suffix.lower()
        txt = ""
        try:
            if ext in (".csv", ".tsv") or (ext == ".txt" and df.size < 2_000_000):
                import pandas as pd

                sample = pd.read_csv(p, sep="\t" if ext == ".tsv" else ",", nrows=50, engine="python", on_bad_lines="skip")
                n = _count_lines(p)
                cols = ", ".join(f"{c}:{t}" for c, t in sample.dtypes.astype(str).items())
                row0 = sample.iloc[0].to_dict() if len(sample) else {}
                row0_s = ", ".join(f"{k}={str(v)[:24]}" for k, v in list(row0.items())[:14])
                txt = f"{n - 1 if n else '?'} rows; columns: {cols}\n    first row: {row0_s}"
            elif ext in (".xlsx", ".xlsm", ".xls"):
                import pandas as pd

                xl = pd.ExcelFile(p)
                txt = f"sheets: {xl.sheet_names}"
                for sh in xl.sheet_names[:3]:
                    raw = xl.parse(sh, header=None, nrows=10)
                    txt += f"\n    sheet {sh!r}: {raw.shape[1]} columns; raw rows (header=None):"
                    for i, row in raw.iterrows():
                        cells = [("" if (isinstance(v, float) and v != v) else str(v)[:22]) for v in row.tolist()[:12]]
                        txt += f"\n      {i}: " + " | ".join(cells)
            elif ext == ".json":
                obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                if isinstance(obj, dict):
                    keys = list(obj.keys())
                    txt = f"JSON object with {len(keys)} keys: " + ", ".join(f"{k}:{type(obj[k]).__name__}" for k in keys[:30])
                elif isinstance(obj, list):
                    txt = f"JSON array of {len(obj)} items"
                    if obj and isinstance(obj[0], dict):
                        txt += "; item keys: " + ", ".join(f"{k}:{type(v).__name__}" for k, v in list(obj[0].items())[:30])
                else:
                    txt = f"JSON {type(obj).__name__}"
            elif ext in (".parquet", ".pqt", ".pq"):
                import pyarrow.parquet as pq

                pf = pq.ParquetFile(p)
                txt = f"{pf.metadata.num_rows} rows; schema: " + ", ".join(f"{f.name}:{f.type}" for f in pf.schema_arrow)
            elif ext == ".jsonl":
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    first = fh.readline()
                rec = json.loads(first) if first.strip() else {}
                txt = f"JSONL, ~{_count_lines(p)} lines; keys of the first record: " + ", ".join(f"{k}:{type(v).__name__}" for k, v in (rec.items() if isinstance(rec, dict) else []))
            else:
                continue
        except Exception as exc:
            txt = f"(could not be read: {type(exc).__name__}: {str(exc)[:100]})"
        block = f"- {p}: {txt}"
        if used + len(block) > limit:
            parts.append(f"- {p}: ...")
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def combine_parts(part1: str, part2: str) -> tuple[str, int]:
    """Join the two halves of a split script; returns (code, line number of the marker line, 1-based)."""
    p1 = [ln for ln in part1.rstrip().splitlines() if ln.strip() != SPLIT_MARKER]
    p2 = [ln for ln in part2.strip("\n").splitlines() if ln.strip() != SPLIT_MARKER]
    return "\n".join(p1 + [SPLIT_MARKER] + p2), len(p1) + 1


def split_parts(code: str) -> tuple[str, str, int] | None:
    """Inverse of combine_parts when the marker is present."""
    lines = code.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == SPLIT_MARKER:
            return "\n".join(lines[:i]), "\n".join(lines[i + 1:]), i + 1
    return None


_PART_MARK_RE = re.compile(r"^# --- part (\d+) of (\d+) ---\s*$")


def chunk_count(plan_json: Any, n_deliverables: int) -> tuple[int, list[list[str]]]:
    """Number of parts (1-3) for the first script, and the plan's function groups when they match it.
    Truncation of the first script rises with the deliverables: 7 % at 1, ~25 % at 2-4, 40 % at 5, 80 % at 6+."""
    est, groups = None, []
    if isinstance(plan_json, dict):
        try:
            est = float(plan_json.get("est_lines"))
        except (TypeError, ValueError):
            est = None
        raw = plan_json.get("parts")
        if isinstance(raw, list) and 1 <= len(raw) <= 3:
            groups = [[str(x) for x in g] if isinstance(g, list) else [str(g)] for g in raw]
    if n_deliverables >= 6 or (est is not None and est > 320):
        k = 3
    elif n_deliverables >= 4 or (est is not None and est > 180):
        k = 2
    else:
        k = 1
    k = max(k, len(groups))
    return min(3, k), (groups if len(groups) == k else [])


def error_line(feedback: str) -> int | None:
    """Line number (in the executed script file) of the last traceback frame inside the solution script."""
    hits = re.findall(r"solution_\w+\.py\", line (\d+)", feedback)
    return int(hits[-1]) if hits else None


def error_signature(stderr: str) -> str:
    """Exception type + the source line it was raised from (inside the solution script); '' if none."""
    if not stderr or not stderr.strip():
        return ""
    lines = [l for l in stderr.strip().splitlines() if l.strip()]
    exc = ""
    for l in reversed(lines):
        m = re.match(r"\s*([A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt))\b", l)
        if m:
            exc = m.group(1)
            break
    src = ""
    for i, l in enumerate(lines):
        if re.search(r"(?:solution_\w+|/ws/\w+)\.py\", line \d+", l) and "_runpy.run_path(" not in (lines[i + 1] if i + 1 < len(lines) else "") and i + 1 < len(lines):
            src = lines[i + 1].strip()
    return f"{exc}@{src}" if exc else ""


def code_fingerprint(code: str) -> str:
    """Whitespace-insensitive identity of a script (two repairs that differ only in blank lines are the same)."""
    norm = "\n".join(line.rstrip() for line in code.strip().splitlines() if line.strip())
    return hashlib.sha1(norm.encode("utf-8", "replace")).hexdigest()


def syntax_check(code: str) -> str | None:
    try:
        compile(code, "solution.py", "exec")
        return None
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno}): {exc.text.strip() if exc.text else ''}"


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float


_CURRENT_PROCS: set = set()          # running generated programs (several when vote candidates run concurrently)
_PROC_LOCK = threading.Lock()


def kill_all_processes() -> None:
    with _PROC_LOCK:
        procs = list(_CURRENT_PROCS)
    for p in procs:
        kill_process(p)


def kill_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_script(script: Path, out_dir: Path, timeout: float, workdir: Path, extra_env: dict | None = None,
               cwd: Path | None = None) -> RunResult:
    env = dict(os.environ)
    threads = str(max(1, min(8, os.cpu_count() or 4)))
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "NUMBA_NUM_THREADS": threads,
        "PYTHONHASHSEED": "0",
        "OUTPUT_DIR": str(out_dir),  # some task texts say "the output directory ($OUTPUT_DIR)"
    })
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items() if v})
    env.setdefault("HOME", str(workdir))
    env.setdefault("MPLCONFIGDIR", str(workdir / "mpl"))
    env.setdefault("NUMBA_CACHE_DIR", str(workdir / "numba"))
    env.setdefault("XDG_CACHE_HOME", str(workdir / "xdg"))
    for d in (env["MPLCONFIGDIR"], env["NUMBA_CACHE_DIR"], env["XDG_CACHE_HOME"]):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
    # Do not leak the model credential to the generated program.
    for k in ("MODEL_TOKEN", "OPENAI_API_KEY"):
        env.pop(k, None)
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script)], cwd=str(cwd or out_dir), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
    except Exception as exc:
        return RunResult(-1, "", f"failed to start interpreter: {exc}", False, 0.0)
    with _PROC_LOCK:
        _CURRENT_PROCS.add(proc)
    timed_out = False
    try:
        out_b, err_b = proc.communicate(timeout=max(5.0, timeout))
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_process(proc)
        try:
            out_b, err_b = proc.communicate(timeout=10)
        except Exception:
            out_b, err_b = b"", b""
    finally:
        with _PROC_LOCK:
            _CURRENT_PROCS.discard(proc)
    dt = time.time() - t0
    out = out_b.decode("utf-8", errors="replace") if out_b else ""
    err = err_b.decode("utf-8", errors="replace") if err_b else ""
    return RunResult(proc.returncode if not timed_out else -9, out, err, timed_out, dt)


# --------------------------------------------------------------------------------------
# Deliverable contract + validation
# --------------------------------------------------------------------------------------

_OUT_EXT = r"(?:json|csv|parquet|pqt|txt|html|png|jpg|jpeg|svg|pdf|xlsx|md|pkl|npy|npz|tsv|feather)"


@dataclass
class Deliverable:
    file: str
    fmt: str = ""
    required_keys: list[str] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    source: str = ""


def deliverables_from_instruction(instr: str, input_names: set[str] | None = None) -> list[Deliverable]:
    """Output file names named by the task text.

    Explicit `/app/output/<name>` (or `/output/<name>`) references are taken as-is.  Inside an
    output/deliverables section, bare `name.ext` tokens (back-ticked, bold or plain) are taken
    too, except names that are input files of the unit.
    """
    inputs = input_names or set()
    found: dict[str, Deliverable] = {}

    def add(rel: str, source: str) -> None:
        rel = rel.strip("/")
        if not rel or rel.endswith(("reward.json", "pytest_report.json")):
            return
        if source != "instruction" and rel.rsplit("/", 1)[-1] in inputs:
            return
        found.setdefault(rel, Deliverable(file=rel, fmt=rel.rsplit(".", 1)[-1].lower(), source=source))

    for m in re.finditer(r"(?:/app/output/|/output/|(?<![\w/])output/)([\w][\w./-]*\." + _OUT_EXT + r")\b", instr):
        add(m.group(1), "instruction")
    heading = re.compile(
        r"(?im)^\s*(?:"
        r"#{1,6}\s+[^\n]*?(?:output|deliverable)[^\n]*"          # any markdown heading mentioning outputs
        r"|\*\*[^\n]*?(?:output|deliverable)[^\n]*?\*\*[^\n]*"  # a bold label line
        r"|(?:outputs?|deliverables?)(?:\s+files?)?\s*:[^\n]*"     # 'Output files:' / 'Deliverables:'
        r")$"
    )
    heads = list(heading.finditer(instr))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(instr)
        chunk = instr[h.start(): min(end, h.start() + 6000)]  # the heading line itself may name the file
        for m in re.finditer(r"(?<![\w/.-])([A-Za-z0-9_][A-Za-z0-9_.-]*\." + _OUT_EXT + r")(?![\w/-])", chunk):
            name = m.group(1)
            if name.count(".") > 2 or name.lower() in ("e.g.", "i.e.") or name.endswith((".pdf", ".md", ".txt")):
                continue  # e.g. `norm.pdf` is a SciPy call, not a file
            add(name, "instruction-section")
    return list(found.values())


def deliverables_from_checks(task_dir: Path) -> list[Deliverable]:
    """Mirror conformance.sh: quoted output-rooted or bare filenames inside checks/*.py."""
    checks = task_dir / "checks"
    if not checks.is_dir():
        return []
    names: set[str] = set()
    root_re = re.compile(r"(data|input|ref|reference|log)_(dir|path)|input_(json|csv|parquet|text)|/app/data|/input|/tests/reference|/logs|[^a-z0-9_](data|ref) */", re.I)
    lit_re = re.compile(r"['\"]([A-Za-z0-9_./{}-]+\.(?:json|parquet|csv|txt|html|png|xlsx|tsv|pqt))['\"]")
    for p in checks.glob("*.py"):
        try:
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if not lit_re.search(line) or root_re.search(line):
                    continue
                for m in lit_re.finditer(line):
                    lit = m.group(1)
                    if "{" in lit or "}" in lit:
                        continue
                    if "/" not in lit or lit.startswith(("/app/output/", "/output/")):
                        base = lit.rsplit("/", 1)[-1]
                        if base and not base.startswith(("reward", "pytest_report", "diagnostic")):
                            names.add(base)
        except Exception:
            continue
    return [Deliverable(file=n, fmt=n.rsplit(".", 1)[-1].lower(), source="checks") for n in sorted(names)]


def merge_deliverables(*groups: list[Deliverable]) -> list[Deliverable]:
    merged: dict[str, Deliverable] = {}
    for group in groups:
        for d in group:
            key = d.file.strip("/")
            if not key:
                continue
            if key in merged:
                cur = merged[key]
                for k in d.required_keys:
                    if k not in cur.required_keys:
                        cur.required_keys.append(k)
                for c in d.required_columns:
                    if c not in cur.required_columns:
                        cur.required_columns.append(c)
            else:
                merged[key] = Deliverable(file=key, fmt=d.fmt or key.rsplit(".", 1)[-1].lower(),
                                          required_keys=list(d.required_keys), required_columns=list(d.required_columns), source=d.source)
    return list(merged.values())


def deliverables_from_plan(plan_json: dict | None) -> list[Deliverable]:
    out: list[Deliverable] = []
    if not plan_json:
        return out
    for item in plan_json.get("deliverables") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("file") or item.get("name") or item.get("path") or "").strip()
        if not name:
            continue
        name = re.sub(r"^(/app/output/|/output/|output/|\./)", "", name)
        keys = [str(k) for k in (item.get("required_keys") or item.get("keys") or []) if isinstance(k, (str, int))]
        cols = [str(c) for c in (item.get("required_columns") or item.get("columns") or []) if isinstance(c, (str, int))]
        out.append(Deliverable(file=name, fmt=str(item.get("format") or name.rsplit(".", 1)[-1]).lower(),
                               required_keys=keys[:80], required_columns=cols[:80], source="plan"))
    return out


def _walk_nan(obj: Any, path: str = "", found: list[str] | None = None, limit: int = 5) -> list[str]:
    if found is None:
        found = []
    if len(found) >= limit:
        return found
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            found.append(path or "<root>")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk_nan(v, f"{path}.{k}" if path else str(k), found, limit)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5000]):
            _walk_nan(v, f"{path}[{i}]", found, limit)
    return found


@dataclass
class Validation:
    ok: bool
    issues: list[str]
    present: int
    expected: int
    score: float


def validate_outputs(out_dir: Path, deliverables: list[Deliverable], initial: set[str]) -> Validation:
    issues: list[str] = []
    present = 0
    quality = 0.0
    if not deliverables:
        # No contract known: at least one new, non-empty file must exist.
        new_files = [p for p in out_dir.rglob("*") if p.is_file() and str(p.relative_to(out_dir)) not in initial]
        if not new_files:
            return Validation(False, ["no output file was written to the output directory"], 0, 1, 0.0)
        return Validation(True, [], len(new_files), 1, 1.0)
    for d in deliverables:
        p = out_dir / d.file
        if not p.is_file():
            issues.append(f"missing deliverable: {d.file}")
            continue
        try:
            if p.stat().st_size == 0:
                issues.append(f"deliverable is empty: {d.file}")
                continue
        except OSError:
            issues.append(f"deliverable unreadable: {d.file}")
            continue
        present += 1
        ext = d.file.rsplit(".", 1)[-1].lower() if "." in d.file else d.fmt
        try:
            if ext == "json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                nan_paths = _walk_nan(obj)
                if nan_paths:
                    issues.append(f"{d.file} contains NaN/Infinity at: {', '.join(nan_paths[:5])}")
                if d.required_keys:
                    if isinstance(obj, dict):
                        missing = [k for k in d.required_keys if k not in obj]
                        if missing and len(missing) < len(d.required_keys):
                            issues.append(f"{d.file} is missing top-level keys: {missing[:15]}")
                        elif missing:
                            # maybe nested (e.g. results['SPY'][...]); only warn softly
                            flat = json.dumps(obj)
                            really_missing = [k for k in missing if f'"{k}"' not in flat]
                            if really_missing:
                                issues.append(f"{d.file} does not contain keys: {really_missing[:15]}")
                    else:
                        issues.append(f"{d.file} top level is a {type(obj).__name__}, expected an object")
                if isinstance(obj, (dict, list)) and len(obj) == 0:
                    issues.append(f"{d.file} is empty ({{}} / [])")
            elif ext in ("csv", "tsv"):
                import pandas as pd

                df = pd.read_csv(p, nrows=2000, sep="\t" if ext == "tsv" else ",")
                if df.shape[0] == 0:
                    issues.append(f"{d.file} has no rows")
                if d.required_columns:
                    missing = [c for c in d.required_columns if c not in df.columns]
                    if missing:
                        issues.append(f"{d.file} is missing columns: {missing[:15]} (has {list(df.columns)[:20]})")
            elif ext in ("parquet", "pqt"):
                import pyarrow.parquet as pq

                pf = pq.ParquetFile(p)
                if pf.metadata.num_rows == 0:
                    issues.append(f"{d.file} has no rows")
                if d.required_columns:
                    cols = set(pf.schema_arrow.names)
                    missing = [c for c in d.required_columns if c not in cols]
                    if missing:
                        issues.append(f"{d.file} is missing columns: {missing[:15]}")
            quality += 1.0
        except Exception as exc:
            issues.append(f"{d.file} could not be parsed as {ext}: {type(exc).__name__}: {str(exc)[:160]}")
    expected = len(deliverables)
    ok = present == expected and not issues
    score = present + 0.5 * quality / max(1, expected) - 0.01 * len(issues)
    return Validation(ok, issues, present, expected, score)


# --------------------------------------------------------------------------------------
# Output directory bookkeeping (best-attempt backup / restore, scrubbing)
# --------------------------------------------------------------------------------------


def list_new_files(out_dir: Path, initial: set[str]) -> list[Path]:
    files = []
    for p in out_dir.rglob("*"):
        if p.is_file() and str(p.relative_to(out_dir)) not in initial:
            files.append(p)
    return files


def clear_outputs(out_dir: Path, initial: set[str]) -> None:
    for p in list_new_files(out_dir, initial):
        try:
            p.unlink()
        except Exception:
            pass
    # remove now-empty sub directories
    for d in sorted((x for x in out_dir.rglob("*") if x.is_dir()), key=lambda x: -len(str(x))):
        try:
            d.rmdir()
        except Exception:
            pass


def backup_outputs(out_dir: Path, initial: set[str], dest: Path, max_bytes: int = 24 * 1024 * 1024) -> bool:
    files = list_new_files(out_dir, initial)
    total = 0
    for f in files:
        try:
            total += f.stat().st_size
        except OSError:
            pass
    if total > max_bytes:
        log(f"outputs too large to back up ({total} bytes); keeping in place")
        return False
    try:
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            rel = f.relative_to(out_dir)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
        return True
    except Exception as exc:
        log(f"backup failed: {exc}")
        return False


def restore_outputs(out_dir: Path, initial: set[str], src: Path) -> None:
    if not src.is_dir():
        return
    clear_outputs(out_dir, initial)
    for f in src.rglob("*"):
        if f.is_file():
            rel = f.relative_to(src)
            target = out_dir / rel
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
            except Exception as exc:
                log(f"restore failed for {rel}: {exc}")


def make_outputs_readable(out_dir: Path, initial: set[str]) -> None:
    """The scorer may read the output tree as another uid: files created 0600 (tempfile.mkstemp, some
    writers) would be unreadable.  Add read (and directory traversal) permission for everyone."""
    for p in list_new_files(out_dir, initial):
        try:
            for d in [p.parent, *p.parent.parents]:
                if d == out_dir or out_dir not in d.parents:
                    break
                d.chmod(d.stat().st_mode | 0o555)
            p.chmod(p.stat().st_mode | 0o444)
        except Exception:
            pass


def scrub_outputs(out_dir: Path, initial: set[str], canary: str) -> None:
    """Remove canary GUIDs / benchmark headers from text deliverables (g2 gate hygiene)."""
    text_ext = {".py", ".json", ".txt", ".md", ".csv", ".log", ".tsv", ".html", ".xml", ".yaml", ".yml"}
    for p in list_new_files(out_dir, initial):
        if p.suffix.lower() not in text_ext:
            continue
        try:
            if p.stat().st_size > 64 * 1024 * 1024:
                continue
            raw = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        low = raw.lower()
        hit = (canary and canary in low) or ("benchmark data should never appear" in low) or ("canary" in low and _UUID_RE.search(raw))
        if not hit:
            continue
        new = raw
        if canary:
            new = re.sub(re.escape(canary), "", new, flags=re.I)
        new = "\n".join(ln for ln in new.splitlines() if not _CANARY_LINE_RE.search(ln))
        try:
            p.write_text(new, encoding="utf-8")
            log(f"scrubbed canary text from {p.name}")
        except Exception:
            pass


def summarize_outputs(out_dir: Path, initial: set[str], limit: int = 7000) -> str:
    parts: list[str] = []
    used = 0
    for p in sorted(list_new_files(out_dir, initial)):
        if used > limit:
            parts.append(f"- {p.relative_to(out_dir)} ({p.stat().st_size} bytes)")
            continue
        rel = str(p.relative_to(out_dir))
        ext = p.suffix.lower()
        txt = ""
        try:
            if ext == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                txt = json.dumps(obj, indent=1)[:1800]
            elif ext in (".csv", ".tsv"):
                import pandas as pd

                df = pd.read_csv(p, sep="\t" if ext == ".tsv" else ",")
                with pd.option_context("display.max_columns", 25, "display.width", 200):
                    txt = f"shape={df.shape}\n" + df.head(5).to_string()[:1500]
                    try:
                        desc = df.describe().T[["mean", "min", "max"]]
                        txt += "\n" + desc.head(20).to_string()[:1200]
                    except Exception:
                        pass
            elif ext in (".parquet", ".pqt"):
                import pandas as pd

                df = pd.read_parquet(p)
                txt = f"shape={df.shape}\n" + df.head(5).to_string()[:1500]
            else:
                txt = f"({p.stat().st_size} bytes)"
        except Exception as exc:
            txt = f"(unreadable: {exc})"
        block = f"### {rel}\n{txt}"
        parts.append(block)
        used += len(block)
    return "\n".join(parts) if parts else "(no files were written)"


# --------------------------------------------------------------------------------------
# Workspace loop (AGENT_LOOP): action parsing
# --------------------------------------------------------------------------------------


@dataclass
class Action:
    kind: str           # run | write | append | edit | bare | invalid
    name: str           # workspace file name ('' for run/bare)
    body: str
    complete: bool = True


_FENCE_OPEN_RE = re.compile(r"^[ \t]*```[ \t]*([^\n`]*)\n", re.M)
_WS_NAME_RE = re.compile(r"^[A-Za-z_]\w*\.py$")


def _classify_fence(info: str) -> tuple[str, str]:
    m = re.search(r"\bappend\s*=\s*[\"']?([^\s\"'`]+)", info)
    if m:
        return "append", m.group(1)
    m = re.search(r"\bfile\s*=\s*[\"']?([^\s\"'`]+)", info)
    if m:
        return ("edit" if re.match(r"\s*edit\b", info, re.I) else "write"), m.group(1)
    words = info.lower().split()
    if "run" in words:
        return "run", ""
    if not words or words[0] in ("python", "py", "python3"):
        return "bare", ""
    if words[0] in ("bash", "sh", "shell", "console", "terminal", "zsh"):
        return "shell", ""
    return "other", ""


def parse_actions(text: str) -> tuple[list[Action], bool]:
    """Action blocks of a loop answer, in order, and whether the answer says DONE (outside any block).
    A block left open by the token limit is returned with complete=False."""
    text = strip_reasoning(text)
    actions: list[Action] = []
    outside: list[str] = []
    pos = 0
    while True:
        m = _FENCE_OPEN_RE.search(text, pos)
        if not m:
            outside.append(text[pos:])
            break
        outside.append(text[pos:m.start()])
        info = m.group(1).strip()
        close = re.compile(r"^[ \t]*```[ \t]*$", re.M).search(text, m.end())
        if close is None:
            body, complete, pos = text[m.end():], False, len(text)
        else:
            body, complete, pos = text[m.end():close.start()], True, close.end()
        kind, name = _classify_fence(info)
        if kind == "other":
            continue
        if kind == "bare":
            # a plain block named by its first comment line or by the text right before it is a file write
            first = body.lstrip("\n").split("\n", 1)[0]
            hint = re.match(r"\s*#\s*(?:file(?:name)?\s*[:=]\s*)?([A-Za-z_]\w*\.py)\s*$", first)
            before = [ln for ln in outside[-1].splitlines() if ln.strip()][-1:] if outside[-1].strip() else []
            hint2 = re.search(r"[`*\s(]([A-Za-z_]\w*\.py)[`*)]*\s*:?[`*]*\s*$", " " + before[0]) if before else None
            if hint or hint2:
                kind, name = "write", (hint or hint2).group(1)
        if name:
            name = name.rsplit("/", 1)[-1]
            if not _WS_NAME_RE.match(name):
                actions.append(Action("invalid", name, f"`{name}` is not a valid workspace file name (use names like data.py; Python files only)."))
                continue
        actions.append(Action(kind, name, body.rstrip("\n").rstrip() if kind == "edit" else body.rstrip("\n") + "\n", complete))
    done = bool(re.search(r"(?m)^\s*[*_`]*DONE[*_`.!]*\s*$", "".join(outside)))
    return actions, done


def drop_wrapper_frames(text: str) -> str:
    """Remove the runpy / workspace-wrapper frames from a traceback (they are noise to the model)."""
    out, skip = [], False
    for line in text.splitlines():
        if re.match(r'\s*File "(?:<frozen runpy>|[^"]*solution_w\d+\.py)", line \d+', line):
            skip = True
            continue
        if skip and (line.startswith("    ") or line.strip().startswith(("^", "~"))) and not line.lstrip().startswith('File "'):
            continue
        skip = False
        out.append(line)
    return "\n".join(out)


def stitch_code(code: str, more: str) -> str:
    """Join a continuation onto code that was cut off: drop a repeated overlap, prefer a variant that compiles."""
    code_lines = code.rstrip("\n").splitlines()
    more_lines = more.rstrip("\n").splitlines()
    first_more = more_lines[0].strip() if more_lines else ""
    if first_more:
        for i in range(len(code_lines) - 1, max(-1, len(code_lines) - 15), -1):
            if code_lines[i].strip() == first_more:
                code_lines = code_lines[:i]
                break
    variants = ["\n".join(code_lines + more_lines)]
    if code_lines:
        variants.append("\n".join(code_lines[:-1] + more_lines))
        if more_lines:
            variants.append("\n".join(code_lines[:-1] + [code_lines[-1] + more_lines[0]] + more_lines[1:]))
    for v in variants:
        if syntax_check(v) is None:
            return v + "\n"
    return variants[0] + "\n"


def rollback_to_statement(code: str, max_lines: int = 60) -> str:
    """Longest prefix (dropping at most max_lines lines) that is valid Python; the input when none is."""
    if syntax_check(code) is None:
        return code
    lines = code.rstrip("\n").splitlines()
    for k in range(1, min(max_lines, len(lines))):
        prefix = "\n".join(lines[:-k])
        if syntax_check(prefix) is None:
            return prefix + "\n"
    return code


# --------------------------------------------------------------------------------------
# The solve loop
# --------------------------------------------------------------------------------------


class Solver:
    def __init__(self, unit: Unit, cfg: Config, budget: Budget, out_dir: Path, workdir: Path) -> None:
        self.unit = unit
        self.cfg = cfg
        self.budget = budget
        self.out_dir = out_dir
        self.workdir = workdir
        self.house = House(cfg, budget)
        self.initial_files: set[str] = {str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()}
        self.best_dir = workdir / "best_outputs"
        self.best_score = -1.0
        self.best_ok = False
        self.backup_ok = True   # False when the best outputs were too large to copy into /tmp
        self.attempt = 0
        self.messages: list[dict] = []
        self.n_parts, self.part_groups = 1, []  # AGENT_CHUNK: parts of the first script
        self.plan_json: dict | None = None      # the plan's JSON block (key_quantities, est_lines, deliverables)
        self.head_len = 2  # number of leading messages that are always kept (system + task context [+ plan turn])
        self.deliverables: list[Deliverable] = []
        # v2: memory library + specification contract
        self.memory = None
        self.mem_candidates: tuple[list, list] = ([], [])
        self.mem_labels: dict = {}
        self.mem_picks: list = []
        self.contract = None
        self.soft_warnings: list[str] = []
        self.seen_code: set[str] = set()   # fingerprints of every script already executed
        self.last_code = ""                # the script of the latest attempt (before path rewriting)
        self.last_truncated = False        # the latest code answer hit the token limit and was not repaired
        self.exec_timeouts = 0             # runs killed at the exec allowance so far (each halves the next allowance)
        self.vote_agreed = False           # an independent solution agreed with the final outputs (vote)
        self.inv_fails: dict[str, int] = {}   # INVARIANT names that failed, with counts (repeat -> soft warning)
        self.track_l_done = False          # the L track produced the outputs (review / vote skipped)
        self.track = "S"                  # S = single script, L = core + deliverable groups
        self.mc_task = False               # the task text involves simulation (AGENT_MC_RULES)
        self.stated: tuple[set, set] | None = None   # numbers / dates stated in the task (AGENT_PARAM_ECHO)
        self.text_docs: list[dict] = []    # HTML / XML documents of the unit (AGENT_TEXT_ROUTE)
        self.text_routed = False
        self.text_probe_text = ""
        self._doc_text: str | None = None
        self.extract_fails: dict[str, int] = {}
        self.key_sets: dict | None = None  # identifier columns of the input tables (AGENT_KEY_CHECK)
        self.key_fails: dict[str, int] = {}
        self.template_code = ""            # AGENT_TEMPLATE_ROUTE: the shipped script to debug (empty = route not taken)
        self.template_name = ""
        self.template_pending_fix = False  # the template ran; the bug-fix request is still owed
        self.task_context = ""
        self.system_prompt = SYSTEM_PROMPT + (ENV_NOTES if cfg.envnotes else "")
        self.split_line: int | None = None   # split mode: line of the PART 2 marker in the current script
        self.probes_sent = 0
        self.think_failures = 0
        self.err_seen: dict[str, int] = {}
        self.last_err_sig = ""
        self.restarted = False
        # workspace loop state (AGENT_LOOP)
        self.ws = workdir / "ws"
        self.files: dict[str, str] = {}
        self.turn = 0
        self.probes_run = 0
        self.cut_files = 0
        self.edit_failures = 0
        if cfg.memory and MemoryIndex is not None:
            mem_dir = Path(cfg.memory_dir) if cfg.memory_dir else Path(_HERE) / "memory"
            try:
                self.memory = MemoryIndex(mem_dir, log=log)
                log(f"memory: {len(self.memory.docs)} documents loaded from {mem_dir}")
            except Exception as exc:
                log(f"memory: disabled ({exc})")
                self.memory = None
        if cfg.selfcheck and parse_contract is not None:
            try:
                self.contract = parse_contract(unit.instruction)
            except Exception:
                self.contract = None

    # -- helpers -------------------------------------------------------------------------
    def _write_script(self, code: str, tag: str) -> Path:
        script = self.workdir / f"solution_{tag}.py"
        script.write_text(assemble_script(make_preamble(self.unit, self.out_dir, self.cfg.seed), code), encoding="utf-8")
        return script

    def _exec_timeout(self) -> float:
        # Heavier units (larger card timeouts) get a larger soft budget, and with it a longer run allowance.
        allowance = min(max(self.cfg.exec_timeout, 0.45 * self.budget.soft), 450.0)   # capped: a relaxed soft budget must not mean 10-minute runs
        if self.exec_timeouts:   # a run already hit the allowance: halve it each time (floor 60 s) instead of paying it again and again
            allowance = max(min(60.0, allowance), allowance / (2 ** self.exec_timeouts))
        return max(10.0, min(allowance, self.budget.remaining() - 10.0))

    def _can_call(self, min_seconds: float = 40.0) -> bool:
        """Enough requests and time for one more model call (plus a short run) given observed latency."""
        need = max(min_seconds, 1.4 * self.house.avg_latency() + 12.0)
        return self.house.requests_left() > 0 and self.budget.remaining() > need

    def _trim_history(self) -> None:
        """Keep the context bounded: system, task, plan turn(s), the last two code/feedback exchanges."""
        if len(self.messages) <= self.head_len + 5:
            return
        head = self.messages[: self.head_len]
        tail = self.messages[-4:]
        # keep roles alternating: head ends with a user message when it includes the code request
        if head and tail and head[-1]["role"] == tail[0]["role"]:
            tail = tail[1:]
        self.messages = head + tail

    def _complete_code(self, content: str, finish: str, temperature: float | None = None) -> str:
        """Handle a code answer that was cut at the token limit."""
        code = extract_code(content)
        self.last_truncated = finish == "length" and bool(code)
        if finish != "length" or not code:
            return code
        rounds = 3 if self.cfg.chunk else 1
        for rnd in range(rounds):
            log("code output was truncated at the token limit; asking for a continuation" + (f" ({rnd + 1}/{rounds})" if rounds > 1 else ""))
            if not self._can_call(45):
                break
            # Roll back to the longest prefix that is valid Python (the cut lands mid-statement), so the continuation
            # starts at a statement boundary instead of inside an expression.
            lines = code.splitlines()
            for k in range(1, min(60, len(lines))):
                if syntax_check("\n".join(lines[:-k])) is None:
                    log(f"continuation: rolled back {k} line(s) to the last complete statement")
                    code = "\n".join(lines[:-k])
                    break
            content = "```python\n" + code + "\n"
            tail_lines = code.splitlines()[-12:]
            cont_prompt = (
                "Your previous answer was cut off by the output-token limit. Continue the script EXACTLY from "
                "after the last complete line shown below. Output ONLY the remaining code in one ```python block, "
                "without repeating anything that was already written:\n```python\n" + "\n".join(tail_lines) + "\n```"
            )
            try:
                more, fin2 = self.house.chat(self.messages + [{"role": "assistant", "content": content}, {"role": "user", "content": cont_prompt}],
                                             label="continue", temperature=temperature, thinking=False if self.cfg.chunk else None)
            except ModelError as exc:
                log(f"continuation failed: {exc}")
                break
            more_code = extract_code(more)
            if not more_code:
                break
            code = stitch_code(code, more_code).rstrip("\n")
            if fin2 != "length":
                break
            log("continuation also truncated" + ("; continuing again" if rnd + 1 < rounds else "; will rely on the repair loop"))
        self.last_truncated = syntax_check(code) is not None
        return code

    # -- steps ----------------------------------------------------------------------------
    def plan(self, context: str, with_memory: bool = False) -> dict | None:
        instr = PLAN_MEMORY_INSTRUCTIONS.replace("{n_sol}", str(self.cfg.mem_solutions)) if with_memory else PLAN_INSTRUCTIONS
        if self.cfg.chunk:
            instr += "\n" + CHUNK_PLAN_NOTE
        if self.cfg.staged:
            instr += "\n" + STAGED_PLAN_NOTE
        if self.cfg.verify:
            instr += "\n" + VERIFY_PLAN_NOTE
        if self.cfg.conventions:
            instr += "\n" + CONVENTIONS_PLAN_NOTE
        if self.cfg.two_track and len(self.deliverables) >= max(3, self.cfg.track_l_min - 1):
            instr += "\n" + TRACK_L_PLAN_NOTE
        if self.template_code:
            instr += "\n" + TEMPLATE_PLAN_NOTE
        self.messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": context + "\n\n" + instr}]
        try:
            if self.cfg.plan_think:
                content, finish = self.house.chat(self.messages, thinking=True, low_effort=False, label="plan")
            else:
                content, finish = self.house.chat(self.messages, thinking=self.cfg.thinking_plan or (self.cfg.low_effort and self.cfg.think_scope == "plan"), label="plan")
        except ContextTooLong:
            raise
        except ModelError as exc:
            log(f"plan step failed: {exc}")
            self.messages = [{"role": "system", "content": self.system_prompt}]
            return None
        if not content.strip():
            log("plan step returned no visible content (thinking consumed the budget?)")
            self.messages = [{"role": "system", "content": self.system_prompt}]
            return None
        self.messages.append({"role": "assistant", "content": content})
        obj = extract_json_block(content)
        if self.cfg.conventions and not (isinstance(obj, dict) and isinstance(obj.get("conventions"), list) and obj["conventions"]):
            rec = parse_conventions_reply(content)
            if rec:
                obj = dict(obj or {})
                obj["conventions"] = rec
                log(f"plan: conventions recovered from a malformed block ({len(rec)} items)")
        return obj

    def _ask_code(self, label: str, temperature: float | None = None) -> str:
        """Send the current conversation, expect a script back; retry once if no code came back.

        The assistant turn stored in the history is the normalised code block (stitched if the
        answer was cut at the token limit), so later repair turns see exactly what was executed.
        """
        content, finish = self.house.chat(self.messages, label=label, temperature=temperature)
        code = self._complete_code(content, finish, temperature)
        if code and self.last_truncated and not self.cfg.chunk and self._can_call(45):
            # The answer was cut at the token limit and the continuation did not give valid Python: running it is
            # a wasted attempt and a full-script repair would be cut again.  Ask for a shorter script instead.
            self.messages.append({"role": "assistant", "content": "```python\n" + code + "\n```"})
            if self.cfg.split and self.split_line is None:
                log("script still truncated after continuation; rewriting it in two parts")
                code2 = self._two_part_code(SPLIT_REWRITE, label + "-split", temperature)
                self.messages = self.messages[:-1]
                if code2 and (syntax_check(code2) is None or syntax_check(code) is not None):
                    code = code2
                else:
                    self.split_line = None
            else:
                log("script still truncated after continuation; asking for a compact rewrite")
                self.messages.append({"role": "user", "content": COMPACT_REWRITE})
                content2, finish2 = self.house.chat(self.messages, label=label + "-compact", temperature=temperature)
                code2 = self._complete_code(content2, finish2, temperature)
                self.messages = self.messages[:-2]
                if code2 and (syntax_check(code2) is None or syntax_check(code) is not None):
                    code = code2
        if not code or (finish == "length" and "<think>" in content and "</think>" not in content):
            self._think_failure("no code" if not code else "reasoning cut at the token limit")
        if not code and self._can_call(40):
            log("no code in the answer; asking again for code only")
            self.messages.append({"role": "assistant", "content": content or "(empty)"})
            self.messages.append({"role": "user", "content": "You did not output any code. Output ONLY the complete Python script now, as ONE ```python block, nothing else."})
            content, finish = self.house.chat(self.messages, label=label + "-retry", temperature=temperature, thinking=False)
            code = self._complete_code(content, finish, temperature)
        if not code:
            self.messages.append({"role": "assistant", "content": content or "(empty)"})
            raise ModelError("the model returned no code")
        self.messages.append({"role": "assistant", "content": "```python\n" + code + "\n```"})
        return code

    def _restart(self, feedback: str):
        """One fresh, independent first script (new conversation, retry temperature) after the first one failed.
        Correctness is mostly decided by the first script that runs; a second sample beats repairing a broken one."""
        ref = self._reference_material(0)
        err = ""
        for line in reversed(feedback.splitlines()):
            if re.match(r"\s*[A-Za-z_][\w.]*(Error|Exception)\b", line) or line.startswith("- missing deliverable"):
                err = line.strip()[:300]
                break
        prompt = (self.task_context + "\n\n" + ((MEMORY_PREFACE + ref + "\n\n") if ref else "") + CODE_ONLY_INSTRUCTIONS +
                  "\nWrite a fresh, complete implementation." + (f" A previous attempt failed with: {err}" if err else ""))
        try:
            new_code = self._side_chat_code(prompt, "restart", temperature=self.cfg.retry_temperature)
        except ModelError as exc:
            log(f"restart: no code ({exc})")
            return "", None, None, ""
        log("restart: running a fresh independent script")
        val, res, fb = self.run_attempt(new_code, "rs")
        return new_code, val, res, fb

    def _think_failure(self, why: str) -> None:
        """Thinking consumed the answer: after think_fail_max such events, this unit continues with thinking off
        (also the safety net if the House route ignores low_effort and reasons at full length)."""
        if not (self.cfg.thinking or self.cfg.low_effort) or self.house.no_thinking:
            return
        self.think_failures += 1
        log(f"thinking failure {self.think_failures}/{self.cfg.think_fail_max}: {why}")
        if self.think_failures >= self.cfg.think_fail_max:
            self.house.no_thinking = True
            log("thinking switched off for the rest of this unit")

    def _two_part_code(self, part1_prompt: str, label: str, temperature: float | None = None) -> str:
        """Ask for PART 1, then PART 2 of the script and join them (sets self.split_line).  The history is left
        as it was on entry plus nothing: the caller records the combined script.  '' when a part did not come."""
        n0 = len(self.messages)
        self.messages.append({"role": "user", "content": part1_prompt})
        try:
            content, finish = self.house.chat(self.messages, label=label + "-p1", temperature=temperature)
            p1 = self._complete_code(content, finish, temperature)
            if not p1:
                return ""
            self.messages.append({"role": "assistant", "content": "```python\n" + p1 + "\n```"})
            self.messages.append({"role": "user", "content": SPLIT_PART2})
            content, finish = self.house.chat(self.messages, label=label + "-p2", temperature=temperature)
            p2 = self._complete_code(content, finish, temperature)
            if not p2:
                return ""
        finally:
            del self.messages[n0:]
        code, self.split_line = combine_parts(p1, p2)
        log(f"two-part script: part 1 = {self.split_line - 1} lines, part 2 = {len(code.splitlines()) - self.split_line} lines")
        return code

    # -- A/B: deep-thinking design call and multi-part first script (AGENT_DEEP_THINK / AGENT_CHUNK) --------
    def _deep_think(self, context: str, have_plan: bool) -> str:
        """One call with full reasoning (low_effort off) whose answer is a short design.  Returns notes for the code
        requests: the design plus the reasoning (only the reasoning tail when it was cut by the token limit)."""
        if have_plan:
            msgs = self.messages + [{"role": "user", "content": DEEP_THINK}]
        else:
            msgs = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": context + "\n\n" + DEEP_THINK}]
        try:
            content, finish = self.house.chat(msgs, label="deep-think", thinking=True, low_effort=False)
        except ModelError as exc:
            log(f"deep-think failed: {exc}")
            return ""
        # the chat template opens the reasoning itself, so the answer usually arrives as "reasoning</think>design"
        m = re.search(r"(?s)^(?:<think>)?(.*?)(</think>|$)", content) if ("</think>" in content or "<think>" in content or finish == "length") else None
        reasoning = (m.group(1) if m else "").strip()
        closed = bool(m and m.group(2))
        design = strip_reasoning(content).strip() if closed else ""
        log(f"deep-think: finish={finish}, reasoning {len(reasoning)} chars ({'complete' if closed or not m else 'cut'}), design {len(design)} chars")
        if design and (closed or not m):
            notes = design
            if reasoning:
                r = reasoning if len(reasoning) <= 6000 else reasoning[:1500] + "\n[...]\n" + reasoning[-4500:]
                notes += "\n\nYour analysis behind this design:\n" + r
            return notes
        if reasoning:
            return "Your analysis so far (it was cut off by the length limit):\n" + reasoning[-7000:]
        return ""

    def _first_code_parts(self, context: str, have_plan: bool, reference: str = "") -> str:
        ref = (MEMORY_PREFACE + reference + "\n\n") if reference else ""
        notes = self._deep_think(context, have_plan) if self.cfg.deep_think and self._can_call(90) else ""
        if notes:
            ref += "## DESIGN NOTES (your own analysis of this task; follow them unless they contradict the task text)\n" + notes + "\n\n"
        if have_plan:
            head, intro, whole = self.messages[:], ref, ref + CODE_INSTRUCTIONS
            self.head_len = 4
        else:
            head = [{"role": "system", "content": self.system_prompt}]
            intro, whole = context + "\n\n" + ref, context + "\n\n" + ref + CODE_ONLY_INSTRUCTIONS
            self.head_len = 2
        n = self.n_parts if self.cfg.chunk else 1
        if n <= 1:
            self.messages = head + [{"role": "user", "content": whole}]
            return self._ask_code("code")
        log(f"chunk: writing the first script in {n} parts" + (f" {self.part_groups}" if self.part_groups else ""))
        default_scope = {2: ["imports, constants, paths, loading and cleaning of the inputs, and the core computation functions",
                             "the remaining computations and writing every deliverable"],
                         3: ["imports, constants, paths, loading and cleaning of the inputs",
                             "the computation functions", "assembling the results and writing every deliverable"]}[n]
        self.messages = head[:]
        parts: list[str] = []
        for k in range(1, n + 1):
            names = self.part_groups[k - 1] if self.part_groups else []
            scope = (" - it contains: " + ", ".join(names[:20])) if names else (" - it contains " + default_scope[k - 1])
            rule = ("Start with all imports, constants and the input/output paths. " if k == 1 else "") + \
                   ("This is the LAST part: it must run everything and write every deliverable (a main() called under "
                    "`if __name__ == \"__main__\":`). " if k == n else "Do not write any output file in this part. ")
            req = CHUNK_PART.format(n=n, k=k, scope=scope, rule=rule)
            self.messages.append({"role": "user", "content": (intro + req) if k == 1 else req})
            part = ""
            for attempt in range(2):
                content, finish = self.house.chat(self.messages, label=f"code-p{k}" + ("-retry" if attempt else ""),
                                                  thinking=False if attempt else None)
                part = self._complete_code(content, finish)
                if part:
                    break
                self._think_failure("no code")
                if not self._can_call(40):
                    break
            if not part:
                log(f"chunk: no code for part {k}; falling back to a single script")
                self.messages = head + [{"role": "user", "content": whole}]
                return self._ask_code("code")
            part = "\n".join(l for l in part.splitlines() if not _PART_MARK_RE.match(l.strip())).strip("\n")
            parts.append(part)
            self.messages.append({"role": "assistant", "content": "```python\n" + part + "\n```"})
        code = "\n".join(f"# --- part {k} of {n} ---\n{p}" for k, p in enumerate(parts, 1))
        log(f"chunk: {n} parts joined ({', '.join(str(len(p.splitlines())) for p in parts)} lines)" +
            ("" if syntax_check(code) is None else "; the joined script does not compile"))
        self.messages = head + [{"role": "user", "content": whole}, {"role": "assistant", "content": "```python\n" + code + "\n```"}]
        return code

    def _part_repair(self, feedback: str) -> str:
        """Chunk mode, after edits failed: rewrite only the part that holds the failing line; '' when unknown."""
        lines = self.last_code.splitlines()
        marks = [i for i, l in enumerate(lines) if _PART_MARK_RE.match(l.strip())]
        line = error_line(feedback)
        if len(marks) < 2 or line is None:
            return ""
        idx = line - make_preamble(self.unit, self.out_dir, self.cfg.seed).count("\n") - 1
        if idx < marks[0]:
            return ""
        k = max(j for j, m in enumerate(marks) if m <= idx)
        start, end = marks[k], (marks[k + 1] if k + 1 < len(marks) else len(lines))
        prompt = ("The script did not produce correct deliverables. Details:\n\n" + feedback + "\n\n" +
                  PART_REPAIR.format(n=len(marks), k=k + 1, a=start + 1, b=end))
        self.messages.append({"role": "user", "content": prompt})
        self._trim_history()
        part = self._ask_code(f"repair{self.attempt}-part{k + 1}")
        part_lines = [l for l in part.splitlines() if not _PART_MARK_RE.match(l.strip())]
        if sum(bool(_PART_MARK_RE.match(l.strip())) for l in part.splitlines()) >= 2 or len(part_lines) > 0.85 * len(lines):
            full = part  # the model answered with the whole script
        else:
            full = "\n".join(lines[:start + 1] + part_lines + lines[end:])
            log(f"part repair: part {k + 1} replaced ({len(part_lines)} lines)")
        self.messages[-1] = {"role": "assistant", "content": "```python\n" + full + "\n```"}
        return full

    def first_code(self, context: str, have_plan: bool, reference: str = "") -> str:
        if self.cfg.staged:
            if staged_applicable(len(self.deliverables), len(self.unit.data_files), self.plan_json):
                code = self._staged_first_code(context, have_plan, reference)
                if code:
                    return code
            else:
                log("staged: small task (few deliverables and inputs); writing a single script")
        if self.cfg.chunk or self.cfg.deep_think:
            return self._first_code_parts(context, have_plan, reference)
        ref = (MEMORY_PREFACE + reference + "\n\n") if reference else ""
        ref += self._code_notes()
        two_part = self.cfg.split and len(self.deliverables) >= 6
        if have_plan:
            self.messages.append({"role": "user", "content": ref + (SPLIT_PART1 if two_part else CODE_INSTRUCTIONS)})
            self.head_len = 4
        else:
            self.messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": context + "\n\n" + ref + (SPLIT_PART1 if two_part else CODE_ONLY_INSTRUCTIONS)}]
            self.head_len = 2
        if two_part:
            # the PART 1 request is already the last user message: _two_part_code appends its own, so hand it the
            # same text and drop the duplicate afterwards
            req = self.messages.pop()
            code = self._two_part_code(req["content"], "code")
            self.messages.append(req)
            if code:
                self.messages.append({"role": "assistant", "content": "```python\n" + code + "\n```"})
                return code
            log("two-part generation failed; falling back to a single script")
            self.split_line = None
            self.messages[-1] = {"role": "user", "content": req["content"].replace(SPLIT_PART1, CODE_INSTRUCTIONS if have_plan else CODE_ONLY_INSTRUCTIONS)}
        return self._ask_code("code")

    def _conventions_vote(self, context: str) -> None:
        """AGENT_CONV_VOTE: 1-3 extra independent convention reads (JSON only, fresh conversations); the plan's list is
        replaced by the per-item majority."""
        base = self.plan_json.get("conventions") if isinstance(self.plan_json, dict) else None
        reads: list[list] = [base if isinstance(base, list) else []]
        for k in range(self.cfg.conv_vote):
            if not (self._can_call(40) and self.house.requests_left() >= 2):
                break
            msgs = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": context + "\n\n" + CONV_VOTE_PROMPT}]
            try:
                content, _ = self.house.chat(msgs, label=f"conv{k + 1}", temperature=self.cfg.retry_temperature)
            except ModelError as exc:
                log(f"conv-vote: read {k + 1} failed ({exc})")
                break
            lst = parse_conventions_reply(content)
            if lst:
                reads.append(lst)
            else:
                head = re.sub(r"\s+", " ", strip_reasoning(content))[:160]
                log(f"conv-vote: read {k + 1} returned no conventions (reply starts: {head!r})")
        if len(reads) < 2:
            log("conv-vote: not enough reads; the plan's conventions stand")
            return
        merged, multi, changed = merge_conventions(reads)
        if not isinstance(self.plan_json, dict):
            self.plan_json = {}
        self.plan_json["conventions"] = merged
        log(f"conv-vote: {len(reads)} reads -> {len(merged)} items ({multi} seen in >= 2 reads, {changed} plan choices outvoted)")

    def _code_notes(self) -> str:
        """Extra text in front of every code request: the plan's conventions and the invariant-printing rule."""
        parts = []
        if self.cfg.conventions:
            conv = conventions_text(self.plan_json)
            if conv:
                parts.append(conv)
            else:
                log("conventions: the plan listed none")
        if self.cfg.invariants:
            parts.append(INVARIANTS_NOTE)
        if self.cfg.mc_rules and self.mc_task:
            parts.append(MC_NOTE)
        if self.cfg.param_echo:
            parts.append(PARAM_ECHO_NOTE)
        if self.text_routed:
            parts.append(EXTRACT_NOTE)
        return ("\n\n".join(parts) + "\n\n") if parts else ""

    def _extra_checks(self, out_dir: Path, stdout: str) -> list[str]:
        """Hard issues beyond the shape checks: degenerate outputs (AGENT_DEGEN) and failed INVARIANT lines (AGENT_INVARIANTS);
        an invariant that fails a second time becomes a soft warning for the review instead of blocking."""
        issues: list[str] = []
        if self.text_routed:
            try:
                if self._doc_text is None:
                    self._doc_text = doc_plain_text(self.text_docs)
                lines = parse_extract_lines(stdout)
                hard, soft = extract_issues(lines, self._doc_text)
                log(f"text route: {len(lines)} EXTRACT lines, {len(hard)} hard / {len(soft)} soft findings")
                for h in hard:
                    sig = h.split(':')[0][:60]
                    self.extract_fails[sig] = self.extract_fails.get(sig, 0) + 1
                    if self.extract_fails[sig] >= 3:
                        self.soft_warnings.append(h + " (repeated; check the parsing once more)")
                    else:
                        issues.append(h)
                self.soft_warnings.extend(soft)
            except Exception as exc:
                log(f"text route: EXTRACT check error {type(exc).__name__}: {exc}")
        if self.cfg.degen or self.text_routed:
            try:
                issues += degenerate_issues(out_dir, self.deliverables, self.unit.instruction,
                                            {df.path.name for df in self.unit.data_files} | {d["name"].rsplit(":", 1)[-1].rsplit("/", 1)[-1] for d in self.text_docs},
                                            self.initial_files if out_dir == self.out_dir else set())
            except Exception as exc:
                log(f"degenerate check error: {type(exc).__name__}: {exc}")
        if self.cfg.key_check:
            try:
                if self.key_sets is None:
                    self.key_sets = input_key_sets([df.path for df in self.unit.data_files])
                    log(f"key check: {len(self.key_sets)} identifier column(s) in the inputs: {sorted(self.key_sets)[:10]}")
                if self.key_sets:
                    files = [out_dir / d.file for d in self.deliverables if (out_dir / d.file).is_file()]
                    hard, soft = key_issues(files, self.key_sets, self.unit.instruction)
                    if hard or soft:
                        log(f"key check: {len(hard)} hard / {len(soft)} soft findings")
                    for h in hard:
                        sig = h.split(": ", 2)[1][:80] if h.count(": ") >= 2 else h[:80]
                        self.key_fails[sig] = self.key_fails.get(sig, 0) + 1
                        if self.key_fails[sig] >= 3:
                            self.soft_warnings.append(h + " (repeated; check once more whether the value really comes from that input)")
                        else:
                            issues.append(h)
                    self.soft_warnings.extend(soft)
            except Exception as exc:
                log(f"key check error: {type(exc).__name__}: {exc}")
        if self.cfg.param_echo:
            echo = parse_param_lines(stdout)
            if echo:
                if self.stated is None:
                    extra_text = ""
                    for df in self.unit.data_files:
                        if df.path.suffix.lower() in (".json", ".toml", ".yaml", ".yml", ".txt") and df.size < 200_000:
                            try:
                                extra_text += "\n" + df.path.read_text(encoding="utf-8", errors="replace")
                            except OSError:
                                pass
                    self.stated = stated_values(self.unit.instruction, extra_text)
                bad = unstated_params(echo, *self.stated)
                log(f"param echo: {len(echo)} parameters, {len(bad)} not stated in the task" + (f": {bad[0][:120]}" if bad else ""))
                for b in bad:
                    self.soft_warnings.append(f"PARAM {b} - re-read the task: every fixed parameter must take exactly the stated value")
        if self.cfg.invariants:
            failed, ok = parse_invariants(stdout)
            if failed or ok:
                log(f"invariants: {len(ok)} ok, {len(failed)} failed")
            for f in failed[:8]:
                name = f.split()[0]
                self.inv_fails[name] = self.inv_fails.get(name, 0) + 1
                if self.inv_fails[name] >= 3:
                    self.soft_warnings.append(f"INVARIANT {f} (failed repeatedly; check whether the relation really applies)")
                else:
                    issues.append(f"invariant failed: {f} (a relation the task implies; fix the cause in the computation, or drop the check only if the task text says it does not apply)")
        return issues

    def _reference_material(self, compact: int) -> str:
        """Render the chosen memory documents within the budget for the current compaction level."""
        if not self.memory or compact >= 2:
            return ""
        docs = [d for d in self.memory.pinned] + [d for d in self.mem_picks if not d.pinned]
        if not docs:
            return ""
        budget = self.cfg.memory_chars if compact == 0 else min(self.cfg.memory_chars, 8000)
        if budget <= 0:
            return ""
        try:
            text = self.memory.render(docs, budget)
        except Exception as exc:
            log(f"memory: render failed ({exc})")
            return ""
        if _UUID_RE.search(text):  # belt and braces: never forward a GUID
            return ""
        log(f"memory: using {[d.id for d in docs]} ({len(text)} chars)")
        return text

    def repair(self, feedback: str) -> str:
        prompt = (
            "The script did not produce correct deliverables. Details:\n\n" + feedback +
            "\n\nFix the problem(s) and output the COMPLETE corrected script as ONE ```python block (no prose). "
            "Keep every deliverable, keep exact names/keys/columns, keep it compact so it is not cut off."
        )
        if self.last_err_sig and self.err_seen.get(self.last_err_sig, 0) >= 2:
            # the same exception on the same line again: the previous fix did not address the cause
            log(f"same error as an earlier attempt ({self.last_err_sig[:120]}); asking for a different approach")
            feedback += "\n\n" + SAME_ERROR_NOTE
            prompt = prompt.replace("\n\nFix the problem(s)", "\n\n" + SAME_ERROR_NOTE + "\n\nFix the problem(s) with a DIFFERENT approach for the failing part")
            temp = self.cfg.retry_temperature
        else:
            temp = None
        code = ""
        if self.cfg.patch_repair and self.last_code:
            code = self._patch_repair(feedback)
            if not code and self.template_code and self._can_call(60):
                log("template route: edits did not apply; one strict retry")
                code = self._patch_repair(feedback + "\n\n" + TEMPLATE_EDIT_STRICT)
                if not code and len(self.last_code.splitlines()) > TEMPLATE_LONG_LINES and self._can_call(60):
                    log("template route: edits failed twice; a fresh compact script replaces the rewrite of the long template")
                    fresh = (self.task_context + "\n\n" + self._code_notes() + CODE_ONLY_INSTRUCTIONS +
                             "\nThe shipped script above is only a reference for the I/O shape: write your own compact implementation.")
                    code = self._side_chat_code(fresh, f"repair{self.attempt}-fresh")
                    self.messages.append({"role": "user", "content": "The edits could not be applied; a fresh script was written instead."})
                    self.messages.append({"role": "assistant", "content": "```python\n" + code + "\n```"})
                    return code
        if not code and self.split_line is not None and self.last_code:
            code = self._split_repair(feedback)
        if not code and self.cfg.chunk and self.last_code:
            code = self._part_repair(feedback)
        if not code:
            self.messages.append({"role": "user", "content": prompt})
            self._trim_history()
            code = self._ask_code(f"repair{self.attempt}", temperature=temp)
        if code_fingerprint(code) in self.seen_code and self._can_call(45):
            # temperature 0 + an unchanged (trimmed) history gives the same answer again: change the question
            log("repair returned a script identical to an earlier attempt; asking for a different fix")
            self.messages.append({"role": "user", "content": DUPLICATE_REPAIR})
            code = self._ask_code(f"repair{self.attempt}-alt", temperature=self.cfg.retry_temperature)
        return code

    def _split_repair(self, feedback: str) -> str:
        """Split mode: ask only for the half that contains the failure; '' when the result is unusable."""
        parts = split_parts(self.last_code)
        if parts is None:
            self.split_line = None
            return ""
        p1, p2, marker = parts
        total = len(self.last_code.splitlines())
        pre = make_preamble(self.unit, self.out_dir, self.cfg.seed).count("\n")
        line = error_line(feedback)
        k = 2 if line is None or (line - pre) > marker else 1
        prompt = ("The script did not produce correct deliverables. Details:\n\n" + feedback + "\n\n" +
                  SPLIT_REPAIR.format(s1=marker - 1, s2=marker + 1, end=total, k=k))
        self.messages.append({"role": "user", "content": prompt})
        self._trim_history()
        part = self._ask_code(f"repair{self.attempt}-p{k}")
        if split_parts(part) is not None or len(part.splitlines()) > 0.85 * total:
            full = part  # the model answered with the whole script
            sp = split_parts(full)
            self.split_line = sp[2] if sp else None
            if sp is None:
                log("repair returned a single script; leaving split mode")
        else:
            full, self.split_line = combine_parts(part, p2) if k == 1 else combine_parts(p1, part)
            log(f"split repair: part {k} replaced ({len(part.splitlines())} lines)")
        self.messages[-1] = {"role": "assistant", "content": "```python\n" + full + "\n```"}
        return full

    def _patch_repair(self, feedback: str) -> str:
        """A/B: ask for SEARCH/REPLACE edits of the last script; '' when they do not apply (caller falls back)."""
        self.messages.append({"role": "user", "content": "The script did not produce correct deliverables. Details:\n\n" + feedback + "\n\n" + PATCH_REPAIR})
        self._trim_history()
        content, finish = self.house.chat(self.messages, label=f"patch{self.attempt}")
        new = apply_edit_blocks(self.last_code, content) if finish != "length" else None
        how = f"applied {len(_EDIT_RE.findall(content))} edit block(s)"
        if new is None:
            full = extract_code(content)
            if full and len(full.splitlines()) >= 15 and finish != "length" and syntax_check(full) is None:
                new, how = full, "edit request answered with a full script; using it"
        if new is None:
            log("edit blocks did not apply; falling back to a full-script repair")
            self.messages.pop()  # drop the edit request; the caller asks for a full script
            return ""
        log(how)
        self.messages.append({"role": "assistant", "content": "```python\n" + new + "\n```"})
        if self.split_line is not None:
            sp = split_parts(new)
            self.split_line = sp[2] if sp else None
        return new

    # -- A/B: invariant self-test ----------------------------------------------------------
    def _template_section(self, limit: int = 60_000) -> str:
        body = self.template_code if len(self.template_code) <= limit else self.template_code[:limit] + "\n# ... (cut)"
        return (f"\n\n## SCRIPT TO DEBUG (full text of `{self.template_name}`; the task asks you to fix its bugs - start from it and "
                f"change only what is wrong)\n```python\n{body}\n```")

    def _template_start(self, context: str, have_plan: bool, reference: str = "") -> str:
        """Template route: the shipped script is the first attempt.  The conversation records it as the assistant's script so
        that the usual SEARCH/REPLACE repairs apply to it; the loop then owes one bug-fix request even when it validates."""
        ref = (MEMORY_PREFACE + reference + "\n\n") if reference else ""
        ref += self._code_notes()
        if have_plan:
            self.messages.append({"role": "user", "content": ref + TEMPLATE_START})
            self.head_len = 4
        else:
            self.messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": context + "\n\n" + ref + TEMPLATE_START}]
            self.head_len = 2
        self.messages.append({"role": "assistant", "content": "```python\n" + self.template_code + "\n```"})
        self.template_pending_fix = True
        return self.template_code

    def _side_chat_edits(self, prompt: str, label: str, temperature: float | None = None) -> str:
        """Template route candidates: independent SEARCH/REPLACE edits of the shipped script (a full script is accepted too)."""
        saved, saved_head = self.messages, self.head_len
        self.messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        try:
            content, finish = self.house.chat(self.messages, label=label, temperature=temperature)
        finally:
            self.messages, self.head_len = saved, saved_head
        def usable(content: str, finish: str) -> str | None:
            body = strip_reasoning(content)
            new = apply_edit_blocks(self.template_code, body) if finish != "length" else None
            if new is None:
                full = extract_code(content)
                if full and len(full.splitlines()) >= 15 and finish != "length" and syntax_check(full) is None:
                    new = full
            return new
        new = usable(content, finish)
        if not new and self._can_call(60) and self.house.requests_left() >= 3:
            log(f"{label}: edits did not apply; one strict retry")
            saved, saved_head = self.messages, self.head_len
            self.messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt},
                             {"role": "assistant", "content": strip_reasoning(content)[:6000] or "(empty)"}, {"role": "user", "content": TEMPLATE_EDIT_STRICT}]
            try:
                content, finish = self.house.chat(self.messages, label=label + "-strict", temperature=temperature, thinking=False)
            finally:
                self.messages, self.head_len = saved, saved_head
            new = usable(content, finish)
        if not new:
            log(f"{label}: no usable edits; writing a fresh compact script instead")
            fresh = (self.task_context + "\n\n" + self._code_notes() + CODE_ONLY_INSTRUCTIONS +
                     "\nThe shipped script above is only a reference for the I/O shape: write your own compact implementation.")
            return self._side_chat_code(fresh, label + "-fresh", temperature=temperature)
        return new

    def _side_chat_code(self, prompt: str, label: str, temperature: float | None = None) -> str:
        """Ask for a script in a separate conversation (system + task context only); main history untouched."""
        saved, saved_head = self.messages, self.head_len
        self.messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        try:
            return self._ask_code(label, temperature=temperature)
        finally:
            self.messages, self.head_len = saved, saved_head

    def _run_check_script(self, code: str) -> list[str] | None:
        """Run a read-only check script; FAIL lines, or None if the check script itself is unusable."""
        code = rewrite_paths(code, self.unit, self.out_dir, {d.file.rsplit("/", 1)[-1] for d in self.deliverables})
        if syntax_check(code):
            return None
        cwd = self.workdir / "selftest_cwd"
        cwd.mkdir(parents=True, exist_ok=True)
        script = cwd / "check.py"
        script.write_text(code + "\n", encoding="utf-8")
        before = {str(p) for p in self.out_dir.rglob("*") if p.is_file()}
        res = run_script(script, cwd, min(60.0, max(10.0, self.budget.remaining() - 60.0)), self.workdir,
                         extra_env={"OUTPUT_DIR": str(self.out_dir)})
        for p in self.out_dir.rglob("*"):  # a check script must not leave files behind
            if p.is_file() and str(p) not in before:
                try:
                    p.unlink()
                except OSError:
                    pass
        fails = [l.strip()[5:].strip() for l in res.stdout.splitlines() if l.strip().startswith("FAIL:")]
        if res.timed_out or ("CHECKS DONE" not in res.stdout and not fails):
            log(f"selftest: check script unusable (rc={res.returncode}, timed_out={res.timed_out})")
            return None
        return fails[:12]

    def selftest(self) -> None:
        if not (self._can_call(150) and self.house.requests_left() >= 4):
            return
        prompt = self.task_context + "\n\n" + SELFTEST_PROMPT.format(
            out_dir=self.out_dir, code=truncate(self.last_code, 14000), summary=summarize_outputs(self.out_dir, self.initial_files, 5000))
        check = self._side_chat_code(prompt, "selftest")
        fails = self._run_check_script(check)
        if not fails:
            log("selftest: no failed checks" if fails is not None else "selftest: skipped")
            return
        for f in fails:
            log(f"   selftest FAIL: {f[:200]}")
        for rnd in range(2):
            if not (self._can_call(100) and self.house.requests_left() >= 2):
                return
            pre = self.workdir / f"pre_selftest{rnd}"
            prev_code = self.last_code
            if not backup_outputs(self.out_dir, self.initial_files, pre):
                return
            new_code = self.repair(SELFTEST_FEEDBACK.format(fails="\n".join("- " + f for f in fails)))
            val, res, _ = self.run_attempt(new_code, f"st{rnd + 1}", prefer_new_on_tie=True)
            new_fails = self._run_check_script(check) if val.ok else None
            if new_fails is None or len(new_fails) >= len(fails):
                log(f"selftest: revision did not reduce the failed checks ({len(fails)} -> {'?' if new_fails is None else len(new_fails)}); keeping the previous outputs")
                restore_outputs(self.out_dir, self.initial_files, pre)
                backup_outputs(self.out_dir, self.initial_files, self.best_dir)
                self.last_code = prev_code
                return
            log(f"selftest: failed checks {len(fails)} -> {len(new_fails)}; revision kept")
            fails = new_fails
            if not fails:
                return

    # -- A/B: independent second implementation ---------------------------------------------
    def nversion(self) -> None:
        if not (self._can_call(150) and self.house.requests_left() >= 3):
            return
        shadow = self.workdir / "nv_out"
        shutil.rmtree(shadow, ignore_errors=True)
        shadow.mkdir(parents=True, exist_ok=True)
        prompt = self.task_context + "\n\n" + CODE_ONLY_INSTRUCTIONS + "\nWrite your own implementation directly from the task text."
        try:
            other = self._side_chat_code(prompt, "nv-code", temperature=self.cfg.retry_temperature)
        except ModelError as exc:
            log(f"nversion: no second implementation ({exc})")
            return
        code2 = rewrite_paths(other, self.unit, shadow, {d.file.rsplit("/", 1)[-1] for d in self.deliverables})
        if syntax_check(code2):
            log("nversion: second implementation does not compile; skipped")
            return
        script = self.workdir / "solution_nv.py"
        script.write_text(assemble_script(make_preamble(self.unit, shadow, self.cfg.seed), code2), encoding="utf-8")
        res = run_script(script, shadow, self._exec_timeout(), self.workdir,
                         extra_env={"DATA_DIR": str(self.unit.data_root) if self.unit.data_root else ""})
        val2 = validate_outputs(shadow, self.deliverables, set())
        if res.returncode != 0 or res.timed_out or val2.present == 0:
            log(f"nversion: second implementation failed (rc={res.returncode}, files {val2.present}/{val2.expected}); skipped")
            return
        diffs = compare_outputs(self.out_dir, shadow, self.deliverables)
        if not diffs:
            log("nversion: independent implementation agrees")
            return
        log(f"nversion: {len(diffs)} disagreements, e.g. {diffs[0][:160]}")
        if not (self._can_call(90) and self.house.requests_left() >= 1):
            return
        self.messages.append({"role": "user", "content": NV_JUDGE.format(diffs="\n".join("- " + x for x in diffs), other=truncate(other, 12000))})
        self._trim_history()
        content, finish = self.house.chat(self.messages, label="nv-judge")
        self.messages.append({"role": "assistant", "content": content})
        if re.search(r"VERDICT:\s*KEEP", strip_reasoning(content), re.I):
            log("nversion: judge keeps the original")
            return
        code = self._complete_code(content, finish)
        if not code or len(code.splitlines()) < 15:
            return
        val, res, _ = self.run_attempt(code, "nv", prefer_new_on_tie=True)
        log("nversion: revised script " + ("accepted" if val.ok else "failed validation; previous outputs kept"))

    def review(self) -> str | None:
        summary = summarize_outputs(self.out_dir, self.initial_files)
        if self.cfg.conventions:
            conv = conventions_text(self.plan_json)
            if conv:
                summary += "\n\n" + conv.replace("implement EXACTLY these", "verify that the code implements EXACTLY these, one by one")
        if self.soft_warnings:
            summary += "\n\nAutomatic checks flagged (verify against the task; fix only if the task requires it):\n- " + "\n- ".join(self.soft_warnings[:10])
        if self.cfg.clause_audit and self.last_code:
            prompt = CLAUSE_AUDIT.format(summary=summary)
        elif (self.cfg.chunk or self.template_code) and self.last_code:
            prompt = REVIEW_EDITS.format(summary=summary)
        elif self.cfg.strict_review:
            prompt = STRICT_REVIEW.format(summary=summary)
        else:
            prompt = (
                "The script ran successfully. Here is a summary of the files it wrote:\n\n" + summary +
                "\n\nCarefully compare these outputs with the task specification: exact file names, keys/columns and "
                "their order, units (percent vs decimal, annualized vs daily), sign conventions, rounding, row counts, "
                "date formats, and whether the magnitudes are financially plausible. Also re-check the formulas in the code "
                "against the task text.\nIf everything is correct, reply with exactly: VERDICT: OK\n"
                "Otherwise list the concrete problems briefly and then output the COMPLETE corrected script as ONE ```python block."
            )
        self.messages.append({"role": "user", "content": prompt})
        self._trim_history()
        content, finish = self.house.chat(self.messages, label="review")
        self.messages.append({"role": "assistant", "content": content})
        if self.cfg.clause_audit:
            body = strip_reasoning(content)
            n_cl = len(re.findall(r"^\s*CLAUSE:", body, re.M)); n_mm = len(re.findall(r"->\s*MISMATCH", body))
            log(f"clause audit: {n_cl} clauses, {n_mm} mismatches")
        if re.search(r"VERDICT:\s*OK", strip_reasoning(content), re.I):
            return None
        if (self.cfg.chunk or self.cfg.clause_audit or self.template_code) and self.last_code and _EDIT_RE.search(strip_reasoning(content)):
            new = apply_edit_blocks(self.last_code, strip_reasoning(content)) if finish != "length" else None
            log("review: " + ("edit blocks applied" if new else "edit blocks did not apply; outputs kept"))
            return new
        code = self._complete_code(content, finish)
        if code and len(code.splitlines()) < 15:
            return None  # a snippet, not a full replacement script
        return code or None

    # -- A/B: two tracks (AGENT_TWO_TRACK) - L track = core script + one short script per deliverable group -------------
    def _derive_groups(self) -> list[dict]:
        """The plan's groups (files restricted to real deliverables), every uncovered deliverable appended in pairs."""
        files = [d.file for d in self.deliverables]
        groups: list[dict] = []
        raw = self.plan_json.get("groups") if isinstance(self.plan_json, dict) else None
        seen: set[str] = set()
        if isinstance(raw, list):
            for g in raw[:6]:
                if not isinstance(g, dict):
                    continue
                fs = [f for f in (g.get("files") or []) if isinstance(f, str) and f in files and f not in seen]
                if not fs:
                    continue
                seen.update(fs)
                needs = [str(x) for x in (g.get("needs") or []) if isinstance(x, (str, int, float))][:12]
                name = re.sub(r"[^A-Za-z0-9_]+", "_", str(g.get("name") or f"g{len(groups) + 1}"))[:24] or f"g{len(groups) + 1}"
                if name.lower() in ("core", "main", "all"):
                    name = f"g{len(groups) + 1}"
                for j in range(0, len(fs), 3):   # a group is one short script: at most 3 files each
                    groups.append({"name": name if j == 0 else f"{name}_{j // 3 + 1}", "files": fs[j:j + 3], "needs": needs})
        rest = [f for f in files if f not in seen]
        for i in range(0, len(rest), 2):
            groups.append({"name": f"g{len(groups) + 1}", "files": rest[i:i + 2], "needs": []})
        return groups

    def _run_piece(self, code: str, tag: str, core_dir: Path) -> RunResult:
        """Run a core / group script against the real output directory without clearing it (incremental)."""
        self.attempt += 1
        protected = {d.file.rsplit("/", 1)[-1] for d in self.deliverables}
        run_code = rewrite_paths(code, self.unit, self.out_dir, protected)
        err = syntax_check(run_code)
        if err:
            return RunResult(1, "", f"SyntaxError: {err}", False, 0.0)
        script = self._write_script(run_code, tag)
        timeout = self._exec_timeout()
        log(f"attempt {self.attempt}: running {script.name} (timeout {timeout:.0f}s)")
        res = run_script(script, self.out_dir, timeout, self.workdir,
                         extra_env={"DATA_DIR": str(self.unit.data_root) if self.unit.data_root else "", "CORE_DIR": str(core_dir)})
        if res.timed_out:
            self.exec_timeouts += 1
        if (res.returncode != 0 or res.timed_out) and res.stderr.strip():
            for line in res.stderr.strip().splitlines()[-2:]:
                log(f"   stderr: {line[:200]}")
        return res

    def _piece_feedback(self, res: RunResult, issues: list[str]) -> str:
        wrote_none = res.returncode == 0 and not res.timed_out and bool(issues) and all(i.startswith("missing deliverable") for i in issues)
        return "\n\n".join(x for x in (
            (f"The script exited normally but wrote none of its files into {self.out_dir}: make sure the main computation is actually "
             "executed (a defined main() must be called at the bottom) and that every file is written to that directory.") if wrote_none else "",
            "The script was killed (too slow); reduce the computational cost." if res.timed_out else (f"The script exited with code {res.returncode}." if res.returncode else ""),
            ("STDERR (tail):\n" + truncate(res.stderr.strip(), 3000, tail=2200)) if res.stderr.strip() else "",
            ("STDOUT (tail):\n" + truncate(res.stdout.strip(), 1000, tail=800)) if res.stdout.strip() else "",
            ("Deliverable validation problems:\n- " + "\n- ".join(issues[:12])) if issues else "") if x)

    def _piece_request(self, messages: list[dict], label: str) -> str:
        """One code request in its own conversation (the main history is untouched)."""
        saved, saved_head = self.messages, self.head_len
        self.messages = messages
        try:
            return self._ask_code(label)
        finally:
            self.messages, self.head_len = saved, saved_head

    def _track_l(self, reference: str) -> bool:
        """Core script (shared quantities -> $CORE_DIR/core.pkl) + one short script per deliverable group, each run
        incrementally and self-checked (shape, degenerate outputs, INVARIANT lines).  Returns False only when the core
        could not be produced (the caller falls back to the single-script track)."""
        core_dir = self.workdir / "core"
        core_dir.mkdir(parents=True, exist_ok=True)
        groups = self._derive_groups()
        needs = sorted({x for g in groups for x in g["needs"]})
        log("track L: groups " + "; ".join(f"{g['name']}={g['files']}" for g in groups))
        ref = (MEMORY_PREFACE + reference + "\n\n") if reference else ""
        notes = self._code_notes()
        base = self.task_context + "\n\n" + ref + notes
        # ---- core
        core_msgs = [{"role": "system", "content": self.system_prompt},
                     {"role": "user", "content": base + CORE_INSTRUCTIONS.format(needs=", ".join(needs) if needs else "everything the deliverables share")}]
        core_code = ""
        summary = ""
        for rnd in range(1 + self.cfg.track_l_repairs):
            if not (self._can_call(60) and self.house.requests_left() >= 1):
                log("track L: no budget for the core script")
                return False
            try:
                core_code = self._piece_request(core_msgs if rnd == 0 else core_msgs, "core" if rnd == 0 else f"core-fix{rnd}")
            except ModelError as exc:
                log(f"track L: core request failed ({exc})")
                return False
            res = self._run_piece(core_code, f"core{rnd}", core_dir)
            pkl = core_dir / "core.pkl"
            keys = [l.strip() for l in res.stdout.splitlines() if l.strip().startswith("CORE ") and not l.strip().startswith("CORE DONE")]
            ok = res.returncode == 0 and not res.timed_out and pkl.is_file() and pkl.stat().st_size > 0
            log(f"track L: core run {rnd + 1} rc={res.returncode} timed_out={res.timed_out} pickle={'yes' if pkl.is_file() else 'no'} keys={len(keys)}")
            if ok:
                summary = "\n".join(keys[:60]) or truncate(res.stdout.strip(), 1500, tail=1200)
                break
            fb = self._piece_feedback(res, [] if pkl.is_file() else ["core.pkl was not written to $CORE_DIR"])
            core_msgs = core_msgs[:2] + [{"role": "assistant", "content": "```python\n" + core_code + "\n```"},
                                         {"role": "user", "content": GROUP_FIX.format(feedback=fb, files="(no deliverables: the core pickle and its summary)")}]
        else:
            return False
        # ---- groups
        pieces = [("core", core_code)]
        done: list[str] = []
        pending = list(groups)
        for pass_no in range(2):   # a second pass retries the groups that failed, if budget remains
            still = []
            for g in pending:
                if not (self._can_call(60) and self.house.requests_left() >= 1):
                    log("track L: no budget left for the remaining groups")
                    still.extend(pending[pending.index(g):])
                    break
                if self._track_l_group(g, core_code, summary, base, core_dir, pieces, repairs=self.cfg.track_l_repairs if pass_no == 0 else 1):
                    done.append(g["name"])
                else:
                    still.append(g)
            pending = still
            if not pending or not (self._can_call(60) and self.house.requests_left() >= 2):
                break
        val = validate_outputs(self.out_dir, self.deliverables, self.initial_files)
        extra = degenerate_issues(self.out_dir, self.deliverables)   # self-checks are part of the L track
        self.last_code = "\n\n".join(f"# ===== {name} =====\n{code}" for name, code in pieces)
        self.best_score = val.score + (1.0 if val.ok else 0.0)
        self.best_ok = val.ok and not extra
        self.backup_ok = backup_outputs(self.out_dir, self.initial_files, self.best_dir)
        log(f"track L: {len(done)}/{len(groups)} groups done, deliverables {val.present}/{val.expected}, issues={len(val.issues) + len(extra)}; requests used {self.house.requests_made}")
        return True

    def _track_l_group(self, g: dict, core_code: str, summary: str, base: str, core_dir: Path, pieces: list, repairs: int) -> bool:
        files = g["files"]
        delivs = [d for d in self.deliverables if d.file in files]
        req = GROUP_INSTRUCTIONS.format(core=truncate(core_code, 12000), summary=truncate(summary, 3000), files=", ".join(files),
                                        needs=", ".join(g["needs"]) if g["needs"] else "whatever these files need",
                                        invariants="Also, at the end, print `INVARIANT <name> OK` or `INVARIANT <name> FAIL <numbers>` for every relation the task "
                                                   "implies these files must satisfy (totals, parity, monotonicity, bounds, row counts), wrapped in try/except.")
        msgs = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": base + req}]
        code = ""
        for rnd in range(1 + repairs):
            if not (self._can_call(60) and self.house.requests_left() >= 1):
                return False
            try:
                code = self._piece_request(msgs, f"group{g['name']}" if rnd == 0 else f"group{g['name']}-fix{rnd}")
            except ModelError as exc:
                log(f"track L: group {g['name']} request failed ({exc})")
                return False
            before = {str(p.relative_to(self.out_dir)): p.stat().st_mtime_ns for p in self.out_dir.rglob("*") if p.is_file()}
            res = self._run_piece(code, f"group_{g['name']}_{rnd}", core_dir)
            val = validate_outputs(self.out_dir, delivs, self.initial_files)
            issues = list(val.issues)
            touched = [f for f, p in ((str(p.relative_to(self.out_dir)), p) for p in self.out_dir.rglob("*") if p.is_file())
                       if f not in files and f not in self.initial_files and before.get(f) != p.stat().st_mtime_ns]
            if touched:
                issues.append(f"the script wrote files outside its group: {touched[:5]}; write only {files}")
            if val.ok and res.returncode == 0 and not res.timed_out:
                issues += degenerate_issues(self.out_dir, delivs)
                failed, ok_inv = parse_invariants(res.stdout)
                if failed or ok_inv:
                    log(f"   group {g['name']}: invariants {len(ok_inv)} ok, {len(failed)} failed")
                for f in failed[:4]:
                    name = f.split()[0]
                    self.inv_fails[name] = self.inv_fails.get(name, 0) + 1
                    if self.inv_fails[name] < 3:
                        issues.append(f"invariant failed: {f} (fix the cause, or drop the check only if the task text says it does not apply)")
            ok = res.returncode == 0 and not res.timed_out and val.ok and not issues
            log(f"track L: group {g['name']} run {rnd + 1} rc={res.returncode} timed_out={res.timed_out} files {val.present}/{val.expected} issues={len(issues)}")
            for iss in issues[:5]:
                log(f"   - {iss[:200]}")
            if ok:
                pieces.append((g["name"], code))
                backup_outputs(self.out_dir, self.initial_files, self.best_dir)
                return True
            fb = self._piece_feedback(res, issues)
            msgs = msgs[:2] + [{"role": "assistant", "content": "```python\n" + code + "\n```"}, {"role": "user", "content": GROUP_FIX.format(feedback=fb, files=", ".join(files))}]
        if val.present == val.expected:   # the files exist but a check still fails: keep them (partial credit beats nothing)
            pieces.append((g["name"], code))
            backup_outputs(self.out_dir, self.initial_files, self.best_dir)
        return False

    # -- A/B: majority vote over independent solutions (AGENT_VOTE) ---------------------------
    def _vote_candidate(self, k: int, prompt: str) -> tuple[Path, str] | None:
        """One independent solution in its own conversation and output directory, with at most one repair.
        Returns (output dir, code) when it runs cleanly and its outputs validate, else None."""
        shadow = self.workdir / f"vote{k}"
        protected = {d.file.rsplit("/", 1)[-1] for d in self.deliverables}
        if self.template_code:
            code = self._side_chat_edits(prompt, f"vote{k}", temperature=self.cfg.retry_temperature)
        else:
            code = self._side_chat_code(prompt, f"vote{k}", temperature=self.cfg.retry_temperature)
        for rnd in range(2):
            shutil.rmtree(shadow, ignore_errors=True)
            shadow.mkdir(parents=True, exist_ok=True)
            run_code = rewrite_paths(code, self.unit, shadow, protected)
            err = syntax_check(run_code)
            if err:
                fb = f"The script has a syntax error and could not start:\n{err}"
            else:
                script = self.workdir / f"solution_vote{k}_{rnd}.py"
                script.write_text(assemble_script(make_preamble(self.unit, shadow, self.cfg.seed), run_code), encoding="utf-8")
                res = run_script(script, shadow, self._exec_timeout(), self.workdir,
                                 extra_env={"DATA_DIR": str(self.unit.data_root) if self.unit.data_root else ""})
                val = validate_outputs(shadow, self.deliverables, set())
                issues = list(val.issues)
                if val.ok and self.contract is not None and (self.contract.json_specs or self.contract.csv_specs):
                    try:
                        issues += ["spec: " + h for h in check_outputs(shadow, self.contract)[0]]
                    except Exception:
                        pass
                if val.ok and res.returncode == 0 and not res.timed_out:
                    saved_soft = list(self.soft_warnings)
                    issues += self._extra_checks(shadow, res.stdout)
                    self.soft_warnings = saved_soft
                ok = res.returncode == 0 and not res.timed_out and val.ok and not issues
                log(f"vote{k}: run {rnd + 1} rc={res.returncode} timed_out={res.timed_out} deliverables {val.present}/{val.expected} issues={len(issues)}")
                if ok and _dir_bytes(shadow) > self.cfg.vote_max_bytes:
                    log(f"vote{k}: candidate outputs exceed the /tmp allowance; discarded")
                    shutil.rmtree(shadow, ignore_errors=True)
                    return None
                if ok:
                    return shadow, code
                fb = "\n\n".join(x for x in (
                    "The script was killed (too slow)." if res.timed_out else f"The script exited with code {res.returncode}.",
                    ("STDERR (tail):\n" + truncate(res.stderr.strip(), 3000, tail=2200)) if res.stderr.strip() else "",
                    ("Deliverable validation problems:\n- " + "\n- ".join(issues[:12])) if issues else "") if x)
            if rnd == 1 or not (self._can_call(90) and self.house.requests_left() >= 2):
                return None
            saved, saved_head = self.messages, self.head_len
            self.messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt},
                             {"role": "assistant", "content": "```python\n" + code + "\n```"},
                             {"role": "user", "content": "The script did not produce correct deliverables. Details:\n\n" + fb +
                              "\n\nFix the problem(s) and output the COMPLETE corrected script as ONE ```python block (no prose)."}]
            try:
                code = self._ask_code(f"vote{k}-fix")
            finally:
                self.messages, self.head_len = saved, saved_head
        return None

    def _dump_candidate(self, j: int, attempt: int, cand_dir: Path, meta: list) -> None:
        """AGENT_VOTE_DUMP: copy candidate j (attempt number `attempt`, 0 = the current outputs) for offline grading."""
        if not self.cfg.vote_dump:
            return
        try:
            dst = Path(self.cfg.vote_dump) / f"c{j}"
            size = sum(p.stat().st_size for p in cand_dir.rglob("*") if p.is_file())
            if size <= 50 * 1024 * 1024:
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(cand_dir, dst)
            meta.append({"j": j, "attempt": attempt, "elapsed": round(self.budget.elapsed(), 1),
                         "requests": int(getattr(self.house, "requests_made", 0)), "bytes": size, "copied": size <= 50 * 1024 * 1024})
        except Exception as exc:
            log(f"vote dump: candidate {j} not saved ({type(exc).__name__}: {exc})")

    def _dump_meta(self, meta: list, **extra) -> None:
        if not self.cfg.vote_dump:
            return
        try:
            Path(self.cfg.vote_dump).mkdir(parents=True, exist_ok=True)
            (Path(self.cfg.vote_dump) / "meta.json").write_text(json.dumps(dict(extra, candidates=meta, unit=self.unit.unit_id,
                                                                                   vote_max=self.cfg.vote_max), default=str))
        except Exception as exc:
            log(f"vote dump: meta not saved ({type(exc).__name__}: {exc})")

    def vote(self, success: bool) -> None:
        """Self-consistency: while time and requests remain, write independent solutions (fresh conversations,
        retry temperature) and keep the answer most of them agree on.  The current outputs win ties; a failed unit
        takes any independent solution that validates."""
        cands: list[tuple[Path, str]] = []
        dump_meta: list = []
        if success:
            c0 = self.workdir / "vote0"
            if not backup_outputs(self.out_dir, self.initial_files, c0):
                return
            size = _dir_bytes(c0)
            if size > self.cfg.vote_max_bytes:
                log(f"vote: outputs are {size / 1048576:.1f} MiB; candidate copies would not fit in /tmp - vote skipped")
                shutil.rmtree(c0, ignore_errors=True)
                return
            cands.append((c0, self.last_code))
            self._dump_candidate(0, 0, c0, dump_meta)
        ref = self._reference_material(0)
        if self.template_code:
            cand_note = TEMPLATE_CANDIDATE_LONG if len(self.template_code.splitlines()) > TEMPLATE_LONG_LINES else TEMPLATE_CANDIDATE
            prompt = self.task_context + "\n\n" + ((MEMORY_PREFACE + ref + "\n\n") if ref else "") + self._code_notes() + cand_note
        else:
            prompt = (self.task_context + "\n\n" + ((MEMORY_PREFACE + ref + "\n\n") if ref else "") + self._code_notes() + CODE_ONLY_INSTRUCTIONS +
                      "\nWrite your own complete implementation directly from the task text.")
        agree: dict[tuple[int, int], bool] = {}
        par = max(1, min(self.cfg.vote_parallel, self.cfg.vote_max))
        k = 1
        while k <= self.cfg.vote_max:
            if not (self._can_call(150) and self.house.requests_left() >= 3):
                log("vote: no time/requests left for another independent solution")
                break
            n = 1 if (par == 1 or getattr(self.house, "rate_limited", False)) else min(par, self.cfg.vote_max - k + 1)
            while n > 1 and self.house.requests_left() < 2 * n + 1:   # every concurrent candidate needs at least two requests
                n -= 1
            ks = list(range(k, k + n))
            k += n
            stop = False
            for kk, cand in zip(ks, self._vote_batch(ks, prompt)):
                if cand is None:
                    continue
                j = len(cands)
                cands.append(cand)
                self._dump_candidate(j, kk, cand[0], dump_meta)
                for i in range(j):
                    rt = 0.05 if (self.cfg.mc_rules and self.mc_task) else 2e-3
                    diffs = compare_outputs(cands[i][0], cand[0], self.deliverables, rtol=rt) + compare_outputs(cand[0], cands[i][0], self.deliverables, rtol=rt)
                    agree[(i, j)] = agree[(j, i)] = not diffs
                    log(f"vote: candidate {j} vs {i}: " + ("agree" if not diffs else f"{len(diffs)} disagreements, e.g. {diffs[0][:140]}"))
                if success and any(agree.get((0, x)) for x in range(1, len(cands))):
                    stop = True  # the current answer is confirmed by an independent solution
                if not success and len(cands) >= 2 and any(agree.values()):
                    stop = True
            if stop:
                break
        if not cands:
            self._dump_meta(dump_meta, success=success, attempts=k - 1)
            return
        support = [sum(agree.get((i, x), False) for x in range(len(cands)) if x != i) for i in range(len(cands))]
        win = max(range(len(cands)), key=lambda i: (support[i], i == 0, -i))
        log(f"vote: {len(cands)} candidate(s), support {support}; winner {win}")
        self._dump_meta(dump_meta, success=success, attempts=k - 1, support=support, winner=win,
                        agree=sorted([i, j] for (i, j), v in agree.items() if v and i < j))
        self.vote_agreed = support[win] >= 1
        if success and (win == 0 or support[win] <= support[0]):
            return
        restore_outputs(self.out_dir, self.initial_files, cands[win][0])
        self.backup_ok = backup_outputs(self.out_dir, self.initial_files, self.best_dir)
        self.best_ok = True
        self.last_code = cands[win][1]
        log(f"vote: outputs replaced by candidate {win}")

    def _vote_batch(self, ks: list[int], prompt: str) -> list:
        """Candidates ks; concurrently when more than one (AGENT_VOTE_PARALLEL), each on a shallow copy of the solver so
        the side conversations do not share state.  Daemon threads: the hard deadline must never wait for them."""
        def one(kk: int, solver):
            try:
                cand = solver._vote_candidate(kk, prompt)
            except ModelError as exc:
                log(f"vote{kk}: no solution ({exc})")
                return None
            if cand is None:
                log(f"vote{kk}: independent solution did not validate")
            return cand

        if len(ks) == 1:
            return [one(ks[0], self)]
        out: dict[int, Any] = {}

        def run(kk: int) -> None:
            try:
                out[kk] = one(kk, copy.copy(self))
            except Exception as exc:  # a thread must never take the unit down
                log(f"vote{kk}: error {type(exc).__name__}: {exc}")
                out[kk] = None

        threads = [threading.Thread(target=run, args=(kk,), daemon=True, name=f"vote{kk}") for kk in ks]
        log(f"vote: candidates {ks[0]}-{ks[-1]} requested concurrently")
        for t in threads:
            t.start()
        deadline = time.time() + max(30.0, self.budget.hard_remaining() - 15.0)
        for t in threads:
            t.join(timeout=max(1.0, deadline - time.time()))
        if any(t.is_alive() for t in threads):
            log("vote: a concurrent candidate is still running at the deadline; abandoned")
        return [out.get(kk) for kk in ks]

    # -- A/B: staged build with execution between modules (AGENT_STAGED) ---------------------
    def _run_module(self, name: str, code: str) -> RunResult:
        """Execute one staged module from the stage directory (so `from inputs import ...` resolves there)."""
        stage = self.workdir / "stage"
        stage.mkdir(parents=True, exist_ok=True)
        protected = {d.file.rsplit("/", 1)[-1] for d in self.deliverables}
        run_code = rewrite_paths(code, self.unit, self.out_dir, protected)
        err = syntax_check(run_code)
        if err:
            return RunResult(1, "", f"SyntaxError: {err}", False, 0.0)
        script = stage / f"{name}.py"
        script.write_text(assemble_script(make_preamble(self.unit, self.out_dir, self.cfg.seed), run_code), encoding="utf-8")
        timeout = max(10.0, min(90.0, self.budget.remaining() - 30.0))
        return run_script(script, self.out_dir, timeout, self.workdir,
                          extra_env={"DATA_DIR": str(self.unit.data_root) if self.unit.data_root else ""}, cwd=stage)

    def _stage_module(self, name: str, request: str) -> tuple[str, str] | None:
        """Ask for one module, run it, repair it once; (code, printed output) on success, None when it still fails."""
        self.messages.append({"role": "user", "content": request})
        code = self._ask_code(f"stage-{name}")
        for rnd in range(2):
            res = self._run_module(name, code)
            if res.returncode == 0 and not res.timed_out:
                log(f"stage {name}: ok in {res.seconds:.1f}s ({len(code.splitlines())} lines)")
                return code, truncate(res.stdout.strip(), 2500, tail=800) or "(nothing printed)"
            tail = res.stderr.strip().splitlines()[-1][:200] if res.stderr.strip() else ""
            log(f"stage {name}: rc={res.returncode} timed_out={res.timed_out} {tail}")
            if rnd == 1 or not self._can_call(60):
                return None
            fb = "\n\n".join(x for x in (
                "The module was killed (too slow)." if res.timed_out else f"Exit code {res.returncode}.",
                ("STDERR (tail):\n" + truncate(res.stderr.strip(), 3000, tail=2200)) if res.stderr.strip() else "",
                ("STDOUT (tail):\n" + truncate(res.stdout.strip(), 800, tail=600)) if res.stdout.strip() else "") if x)
            self.messages.append({"role": "user", "content": STAGE_FIX.format(name=name, feedback=fb)})
            code = self._ask_code(f"stage-{name}-fix")
        return None

    def _staged_first_code(self, context: str, have_plan: bool, reference: str = "") -> str:
        """inputs.py -> compute.py -> main.py, each executed (and repaired once) before the next is requested, then
        flattened into one script for the normal run/repair flow.  '' (history restored) when a stage fails."""
        saved, saved_head = list(self.messages), self.head_len
        ref = (MEMORY_PREFACE + reference + "\n\n") if reference else ""
        if have_plan:
            self.head_len = 4
            first = ref + STAGE_INPUTS
        else:
            self.messages = [{"role": "system", "content": self.system_prompt}]
            self.head_len = 2
            first = context + "\n\n" + ref + STAGE_INPUTS
        observed: list[str] = []
        mods: list[tuple[str, str]] = []
        try:
            for name in ("inputs", "compute", "main"):
                if not self._can_call(60):
                    raise ModelError("no budget for the next stage")
                if name == "inputs":
                    request = first
                elif name == "compute":
                    request = STAGE_COMPUTE.format(observed=observed[-1])
                else:
                    self.messages.append({"role": "user", "content": STAGE_MAIN.format(observed=observed[-1])})
                    mods.append((name, self._ask_code("stage-main")))
                    break
                got = self._stage_module(name, request)
                if got is None:
                    raise ModelError(f"stage {name} did not run")
                mods.append((name, got[0]))
                observed.append(got[1])
            flat = flatten_modules(mods)
            if syntax_check(flat) is not None:
                log("staged: the flattened script does not compile; asking for a combined script")
                if not self._can_call(60):
                    raise ModelError("no budget to combine the modules")
                self.messages.append({"role": "user", "content": STAGE_COMBINE})
                flat = self._ask_code("stage-combine")
        except ModelError as exc:
            log(f"staged build abandoned ({exc}); falling back to a single script")
            self.messages, self.head_len = saved, saved_head
            return ""
        obs = "\n\n".join(f"### {n}.py printed\n{o}" for n, o in zip(("inputs", "compute"), observed))
        code_req = ref + (CODE_INSTRUCTIONS if have_plan else CODE_ONLY_INSTRUCTIONS)
        if obs:
            code_req += "\n\n## OBSERVED WHEN RUNNING THE MODULES (facts about the inputs and the intermediate results)\n" + obs
        answer = {"role": "assistant", "content": "```python\n" + flat + "\n```"}
        if have_plan:
            self.messages = saved[:3] + [{"role": "user", "content": code_req}, answer]
            self.head_len = 4
        else:
            self.messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": context + "\n\n" + code_req}, answer]
            self.head_len = 2
        log(f"staged: {len(mods)} modules flattened into a {len(flat.splitlines())}-line script")
        return flat

    # -- A/B: independent re-derivation of key quantities (AGENT_VERIFY) ----------------------
    def _key_quantities(self) -> list[dict]:
        """The plan's key quantities that exist in the written outputs (else automatic picks), named q1..qn."""
        qs: list[dict] = []
        raw = self.plan_json.get("key_quantities") if isinstance(self.plan_json, dict) else None
        files = {d.file for d in self.deliverables} | {str(p.relative_to(self.out_dir)) for p in list_new_files(self.out_dir, self.initial_files)}
        if isinstance(raw, list):
            for q in raw[:8]:
                if isinstance(q, dict) and str(q.get("file", "")) in files and str(q.get("path", "")).strip():
                    qs.append({"file": str(q["file"]), "path": str(q["path"]).strip(), "agg": str(q.get("agg", "value") or "value").lower(),
                               "meaning": str(q.get("meaning", "") or q["path"])[:220]})
        qs = [q for q in qs if read_quantity(self.out_dir, q) is not None]
        if len(qs) < 2:
            for q in auto_quantities(self.out_dir, self.deliverables):
                if len(qs) >= 6:
                    break
                if not any(q["file"] == x["file"] and q["path"] == x["path"] for x in qs):
                    qs.append(q)
        for i, q in enumerate(qs, 1):
            q["name"] = f"q{i}"
        return qs[:6]

    def _snapshot_outputs(self) -> dict[str, tuple[int, int]]:
        snap: dict[str, tuple[int, int]] = {}
        for p in list_new_files(self.out_dir, self.initial_files):
            try:
                st = p.stat()
                snap[str(p)] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
        return snap

    def _run_verify_script(self, code: str) -> tuple[dict[str, float | None] | None, str]:
        """Run the independent check script read-only (outputs are protected and restored if touched);
        (values, feedback) - values is None when the script is unusable."""
        protected = {d.file.rsplit("/", 1)[-1] for d in self.deliverables}
        code = rewrite_paths(code, self.unit, self.out_dir, protected)
        err = syntax_check(code)
        if err:
            return None, f"SyntaxError: {err}"
        cwd = self.workdir / "verify_cwd"
        cwd.mkdir(parents=True, exist_ok=True)
        script = cwd / "check.py"
        script.write_text(assemble_script(make_preamble(self.unit, self.out_dir, self.cfg.seed), code), encoding="utf-8")
        guard = self.workdir / "pre_check"
        guarded = backup_outputs(self.out_dir, self.initial_files, guard)
        snap = self._snapshot_outputs()
        res = run_script(script, cwd, max(10.0, min(120.0, self.budget.remaining() - 60.0)), self.workdir,
                         extra_env={"DATA_DIR": str(self.unit.data_root) if self.unit.data_root else ""}, cwd=cwd)
        if guarded and self._snapshot_outputs() != snap:
            log("verify: the check script touched the outputs; restoring them")
            restore_outputs(self.out_dir, self.initial_files, guard)
        vals = parse_check_lines(res.stdout)
        na = [ln.strip()[:200] for ln in res.stdout.splitlines() if re.match(r"^\s*CHECK\s+\S+\s+NA\b", ln)]
        fb = "\n\n".join(x for x in (
            "The script was killed (too slow)." if res.timed_out else (f"Exit code {res.returncode}." if res.returncode else ""),
            ("STDERR (tail):\n" + truncate(res.stderr.strip(), 3000, tail=2200)) if res.stderr.strip() else "",
            ("STDOUT (tail):\n" + truncate(res.stdout.strip(), 800, tail=600)) if res.stdout.strip() else "",
            ("Quantities the check could not compute (fix these):\n" + "\n".join(na[:8])) if na else "") if x)
        if res.timed_out or (not vals and "CHECKS DONE" not in res.stdout):
            return None, fb or "no CHECK lines were printed"
        return vals, fb

    @staticmethod
    def _tolerance(q: dict) -> float:
        return 0.05 if re.search(r"monte.?carlo|simulat|\bmc\b|random|paths", str(q.get("meaning", "")), re.I) else 1e-2

    @staticmethod
    def _match_check_values(qs: list[dict], vals: dict[str, float | None]) -> dict[str, float | None]:
        """CHECK names as given (q1..), else case-insensitively, else by the quantity's path leaf, else by position."""
        out: dict[str, float | None] = {}
        low = {k.lower(): v for k, v in vals.items()}
        for q in qs:
            name = q["name"]
            if name in vals:
                out[name] = vals[name]
            elif name.lower() in low:
                out[name] = low[name.lower()]
            else:
                leaf = re.split(r"[.\[\]]+", str(q.get("path", "")).strip())
                leaf = next((t for t in reversed(leaf) if t), "").lower()
                if leaf and leaf in low:
                    out[name] = low[leaf]
        if not out and len(vals) == len(qs):
            out = {q["name"]: v for q, v in zip(qs, vals.values())}
        return out

    def _compare_quantities(self, qs: list[dict], vals: dict[str, float | None]) -> tuple[list[str], int, int]:
        """(disagreement lines, number agreeing, number unavailable); Monte-Carlo-like quantities get a 5% band."""
        diffs: list[str] = []
        agree = na = 0
        matched = self._match_check_values(qs, vals)
        for q in qs:
            ours = read_quantity(self.out_dir, q)
            theirs = matched.get(q["name"])
            if ours is None or theirs is None:
                na += 1
                continue
            tol = self._tolerance(q)
            rel = abs(ours - theirs) / max(abs(ours), abs(theirs), 1e-12)
            if _close(ours, theirs, tol):
                agree += 1
            else:
                where = f"{q['file']}: {q['path']}" + ("" if q["agg"] in ("value", "") else f" ({q['agg']} over the column)")
                diffs.append(f"{q['name']} = {q['meaning']} [{where}]: yours={ours:.8g} independent={theirs:.8g} (relative difference {rel:.1%})")
        return diffs, agree, na

    def verify(self) -> None:
        """Independent re-derivation of the key quantities in a fresh conversation that never sees the solution;
        disagreements are adjudicated against the task text and fixed by edits when the solution is wrong
        (the previous outputs come back when a fix does not reduce the disagreements)."""
        if self.cfg.verify_fix2 and self.vote_agreed:
            log("verify: skipped (an independent solution agreed with the outputs)")
            return
        if not (self._can_call(150) and self.house.requests_left() >= 3):
            log("verify: no time/requests left")
            return
        qs = self._key_quantities()
        if not qs:
            log("verify: no key quantities to check")
            return
        ref = self._reference_material(0)
        qtext = "\n".join(f"- {q['name']}: {q['meaning']} (written to {q['file']} at {q['path']}" +
                          ("" if q["agg"] in ("value", "") else f", {q['agg']} over the column") + ")" for q in qs)
        prompt = self.task_context + "\n\n" + ((MEMORY_PREFACE + ref + "\n\n") if ref else "") + VERIFY_PROMPT.format(quantities=qtext)
        try:
            check = self._side_chat_code(prompt, "verify", temperature=self.cfg.retry_temperature)
        except ModelError as exc:
            log(f"verify: no check script ({exc})")
            return
        vals, fb = self._run_verify_script(check)
        first_vals = vals
        mostly_na = vals is not None and 2 * sum(v is None for v in vals.values()) >= max(1, len(qs))
        if (vals is None or mostly_na) and self._can_call(90) and self.house.requests_left() >= 2:
            log(("verify: check script failed; one repair" if vals is None else "verify: the check printed NA for most quantities; one repair")
                + (f" [{_error_line(fb)}]" if fb else ""))
            saved, saved_head = self.messages, self.head_len
            self.messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt},
                             {"role": "assistant", "content": "```python\n" + check + "\n```"},
                             {"role": "user", "content": VERIFY_FIX.format(feedback=fb)}]
            try:
                check = self._ask_code("verify-fix")
            except ModelError as exc:
                log(f"verify: check script repair failed ({exc})")
                check = ""
            finally:
                self.messages, self.head_len = saved, saved_head
            if check:
                vals, fb = self._run_verify_script(check)
                if first_vals is not None and (vals is None or sum(v is not None for v in vals.values()) < sum(v is not None for v in first_vals.values())):
                    vals = first_vals   # the repair did not help: keep the first check's values
        if vals is None:
            log("verify: independent check script unusable; skipped" + (f" [{_error_line(fb)}]" if fb else ""))
            return
        diffs, agree, na = self._compare_quantities(qs, vals)
        log("verify: CHECK lines: " + "; ".join(f"{k}={'NA' if v is None else format(v, '.6g')}" for k, v in list(vals.items())[:8]) +
            " | ours: " + "; ".join(f"{q['name']}={format(read_quantity(self.out_dir, q), '.6g') if read_quantity(self.out_dir, q) is not None else 'NA'}" for q in qs))
        log(f"verify: {len(qs)} quantities: {agree} agree, {len(diffs)} disagree, {na} unavailable")
        for d in diffs[:6]:
            log(f"   verify: {d[:240]}")
        if not diffs:
            return
        if self.cfg.verify_fix2:
            self._verify_fix2(qs, vals, check, prompt)
            return
        pre = self.workdir / "pre_verify"
        for rnd in range(2):
            if not (self._can_call(100) and self.house.requests_left() >= 2):
                return
            prev_code = self.last_code
            if not backup_outputs(self.out_dir, self.initial_files, pre):
                return
            self.messages.append({"role": "user", "content": VERIFY_JUDGE.format(diffs="\n".join("- " + d for d in diffs), other=truncate(check, 6000))})
            self._trim_history()
            content, finish = self.house.chat(self.messages, label=f"verify-judge{rnd + 1}")
            self.messages.append({"role": "assistant", "content": content})
            body = strip_reasoning(content)
            if re.search(r"VERDICT:\s*KEEP", body, re.I):
                log("verify: judge keeps the solution (independent computation judged wrong)")
                return
            new = apply_edit_blocks(prev_code, body)   # complete blocks apply even when the reply was cut afterwards
            if new is not None and syntax_check(new) is not None:
                new = None
            if new is None and finish != "length":
                full = extract_code(content)
                if full and len(full.splitlines()) >= 15 and syntax_check(full) is None:
                    new = full
            if new is None or code_fingerprint(new) == code_fingerprint(prev_code):
                log("verify: no applicable fix from the judge" + (" (reply cut at the token limit)" if finish == "length" else "") + "; outputs kept")
                return
            self.messages[-1] = {"role": "assistant", "content": "```python\n" + new + "\n```"}
            val, res, _ = self.run_attempt(new, f"vf{rnd + 1}", prefer_new_on_tie=True)
            if not val.ok:
                log("verify: revised script failed validation; the previous outputs were kept")
                self.last_code = prev_code
                self.messages[-1] = {"role": "assistant", "content": "```python\n" + prev_code + "\n```"}
                return
            vals2, _ = self._run_verify_script(check)  # no request: the same check script again
            new_diffs = self._compare_quantities(qs, vals2)[0] if vals2 is not None else None
            if new_diffs is not None and len(new_diffs) < len(diffs):
                log(f"verify: disagreements {len(diffs)} -> {len(new_diffs)}; revision kept")
                diffs = new_diffs
                if not diffs:
                    return
                continue
            log(f"verify: revision did not reduce the disagreements ({len(diffs)} -> {'?' if new_diffs is None else len(new_diffs)}); restoring the previous outputs")
            restore_outputs(self.out_dir, self.initial_files, pre)
            self.backup_ok = backup_outputs(self.out_dir, self.initial_files, self.best_dir)
            self.best_ok = True
            self.last_code = prev_code
            self.messages[-1] = {"role": "assistant", "content": "```python\n" + prev_code + "\n```"}
            return

    def _verify_fix2(self, qs: list[dict], vals1: dict[str, float | None], check1: str, prompt: str) -> None:
        """Second-check adjudication (AGENT_VERIFY_FIX2).  A second independent check script is written; a disputed
        quantity counts as a confirmed error only when both independent values agree with each other against ours.
        The fix is then asked for in a FRESH conversation that sees the script and the three values (no self-defence),
        and kept only when it validates and brings the outputs closer to the independent values."""
        if not (self._can_call(120) and self.house.requests_left() >= 3):
            log("verify: no time/requests left for the second check")
            return
        try:
            check2 = self._side_chat_code(prompt + VERIFY_SECOND, "verify2", temperature=self.cfg.retry_temperature)
        except ModelError as exc:
            log(f"verify: no second check script ({exc})")
            return
        vals2, fb2 = self._run_verify_script(check2)
        if vals2 is None:
            log("verify: second check script unusable; outputs kept" + (f" [{_error_line(fb2)}]" if fb2 else ""))
            return
        m1, m2 = self._match_check_values(qs, vals1), self._match_check_values(qs, vals2)
        confirmed: list[tuple[dict, float, float, float]] = []
        ours_ok = unresolved = 0
        for q in qs:
            ours, a, b = read_quantity(self.out_dir, q), m1.get(q["name"]), m2.get(q["name"])
            if ours is None or a is None or _close(ours, a, self._tolerance(q)):
                continue   # unavailable, or the first check agreed with us
            if b is None:
                unresolved += 1
            elif _close(a, b, self._tolerance(q)):
                confirmed.append((q, ours, a, b))
            elif _close(ours, b, self._tolerance(q)):
                ours_ok += 1
            else:
                unresolved += 1
        log(f"verify: second check: {len(confirmed)} disagreement(s) confirmed, {ours_ok} resolved in our favour, {unresolved} unresolved")
        if not confirmed:
            return
        pre = self.workdir / "pre_verify"
        n_bad = len(confirmed)
        for rnd in range(2):
            if not (self._can_call(100) and self.house.requests_left() >= 2):
                return
            prev_code = self.last_code
            if not backup_outputs(self.out_dir, self.initial_files, pre):
                return
            dtext = "\n".join(
                f"- {q['name']} = {q['meaning']} [{q['file']}: {q['path']}" + ("" if q["agg"] in ("value", "") else f" ({q['agg']} over the column)") +
                f"]: script={ours:.8g} independent #1={a:.8g} independent #2={b:.8g}" for q, ours, a, b in confirmed)
            msgs = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.task_context + "\n\n" + VERIFY_FIX2.format(diffs=dtext, code=truncate(prev_code, 24000))}]
            content, finish = self.house.chat(msgs, label=f"verify-edit{rnd + 1}")
            body = strip_reasoning(content)
            new = apply_edit_blocks(prev_code, body)
            if new is not None and syntax_check(new) is not None:
                new = None
            if new is None and finish != "length":
                full = extract_code(content)
                if full and len(full.splitlines()) >= 15 and syntax_check(full) is None:
                    new = full
            if new is None or code_fingerprint(new) == code_fingerprint(prev_code):
                log("verify: no applicable fix" + (" (reply cut at the token limit)" if finish == "length" else "") + "; outputs kept")
                return
            val, res, _ = self.run_attempt(new, f"vx{rnd + 1}", prefer_new_on_tie=True)
            if not val.ok:
                log("verify: revised script failed validation; the previous outputs were kept")
                self.last_code = prev_code
                return
            still = []
            for q, _, a, b in confirmed:
                now = read_quantity(self.out_dir, q)
                if now is None or not (_close(now, a, self._tolerance(q)) or _close(now, b, self._tolerance(q))):
                    still.append((q, now if now is not None else float("nan"), a, b))
            if len(still) < n_bad:
                log(f"verify: confirmed disagreements {n_bad} -> {len(still)}; revision kept")
                n_bad = len(still)
                confirmed = still
                if not confirmed:
                    return
                continue
            log(f"verify: revision did not reduce the confirmed disagreements ({n_bad}); restoring the previous outputs")
            restore_outputs(self.out_dir, self.initial_files, pre)
            self.backup_ok = backup_outputs(self.out_dir, self.initial_files, self.best_dir)
            self.best_ok = True
            self.last_code = prev_code
            return

    def budget_report(self) -> str:
        """One log line: requests and model seconds per step family, elapsed time and the soft budget."""
        groups: dict[str, list[float]] = {}
        for label, dt in getattr(self.house, "call_log", []):
            m = re.match(r"[A-Za-z]+(?:-[A-Za-z]+)?", label)
            key = re.sub(r"-(fix|retry|compact|split|alt|judge|code)$", "", (m.group(0) if m else label))
            groups.setdefault(key, []).append(dt)
        parts = [f"{k} {len(v)}/{sum(v):.0f}s" for k, v in groups.items()]
        total_s = sum(sum(v) for v in groups.values())
        return (f"budget report: requests {self.house.requests_made}, model time {total_s:.0f}s, elapsed {self.budget.elapsed():.0f}s, "
                f"soft budget {self.budget.soft:.0f}s | " + (", ".join(parts) if parts else "no answered requests"))

    # -- workspace loop (AGENT_LOOP) --------------------------------------------------------
    def _ws_protected(self) -> set[str]:
        return {d.file.rsplit("/", 1)[-1] for d in self.deliverables}

    def _ws_write(self, name: str, text: str) -> None:
        """Keep the model's text (edits must match it); write the path-rewritten version to disk."""
        self.files[name] = text
        disk = rewrite_paths(text, self.unit, self.out_dir, self._ws_protected())
        (self.ws / name).write_text(disk, encoding="utf-8")

    def _ws_listing(self) -> str:
        if not self.files:
            return "empty"
        return ", ".join(f"{n} ({len(t.splitlines())} lines)" for n, t in self.files.items())

    def _ws_state(self, limit: int = 30000) -> str:
        """Full text of the workspace files (most recently written first) within limit characters."""
        parts, used = [], 0
        for name in reversed(list(self.files)):
            text = self.files[name]
            block = f"### {name}\n```python\n{text.rstrip()}\n```"
            if used + len(block) > limit:
                parts.append(f"### {name}: {len(text.splitlines())} lines (not shown; rewrite it with file={name} if it needs changes)")
                continue
            parts.append(block)
            used += len(block)
        return "\n".join(parts)

    def _loop_run_snippet(self, code: str, k: int) -> str:
        code = rewrite_paths(code, self.unit, self.out_dir, self._ws_protected())
        err = syntax_check(code)
        if err:
            return f"RUN {k}: not started, {err}"
        script = self.workdir / f"probe_{self.turn}_{k}.py"
        script.write_text(assemble_script(make_preamble(self.unit, self.out_dir, self.cfg.seed) +
                                          f"_sys.path.insert(0, {str(self.ws)!r})\n", code), encoding="utf-8")
        timeout = max(5.0, min(60.0, self.budget.remaining() - 30.0))
        res = run_script(script, self.out_dir, timeout, self.workdir,
                         extra_env={"DATA_DIR": str(self.unit.data_root) if self.unit.data_root else ""}, cwd=self.ws)
        self.probes_run += 1
        text = (res.stdout.strip() + ("\n" + res.stderr.strip() if res.stderr.strip() else "")).strip() or "(no output)"
        head = f"RUN {k} (exit {res.returncode}" + (f", killed after {timeout:.0f}s" if res.timed_out else "") + f", {res.seconds:.1f}s):"
        return head + "\n" + truncate(text, 3000, tail=1200)

    def _loop_shell(self, cmd: str, k: int) -> str:
        """A shell block (```bash): run it in the workspace like an inspection; files it creates are picked up."""
        cmd = rewrite_paths(cmd, self.unit, self.out_dir, self._ws_protected())
        script = self.workdir / f"shell_{self.turn}_{k}.py"
        script.write_text("import subprocess, sys\nsys.exit(subprocess.call(" + repr(cmd) + ", shell=True, executable='/bin/bash'))\n",
                          encoding="utf-8")
        timeout = max(5.0, min(60.0, self.budget.remaining() - 30.0))
        res = run_script(script, self.out_dir, timeout, self.workdir, cwd=self.ws)
        self.probes_run += 1
        text = (res.stdout.strip() + ("\n" + res.stderr.strip() if res.stderr.strip() else "")).strip() or "(no output)"
        return f"SHELL {k} (exit {res.returncode}{', killed' if res.timed_out else ''}):\n" + truncate(text, 3000, tail=1200)

    def _ws_sync(self) -> list[str]:
        """Pick up workspace .py files that a shell block created or changed."""
        changed = []
        for f in sorted(self.ws.glob("*.py")):
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:
                continue
            if f.name not in self.files or rewrite_paths(self.files[f.name], self.unit, self.out_dir, self._ws_protected()) != text:
                if _WS_NAME_RE.match(f.name):
                    self.files[f.name] = text
                    changed.append(f.name)
        return changed

    def _loop_exec(self, prefer_new: bool) -> tuple[Validation, RunResult | None, str]:
        """Run the workspace (main.py and its modules) as one attempt: run_attempt keeps the best outputs."""
        errs = []
        for name, text in self.files.items():
            err = syntax_check(rewrite_paths(text, self.unit, self.out_dir, self._ws_protected()))
            if err:
                errs.append(f"{name}: {err}")
        if errs:
            log("loop: syntax errors, main.py not run: " + "; ".join(errs)[:300])
            return (Validation(False, errs, 0, len(self.deliverables), -1.0), None,
                    "main.py was not run because of syntax errors:\n- " + "\n- ".join(errs))
        digest = hashlib.sha1(json.dumps(self.files, sort_keys=True).encode("utf-8", "replace")).hexdigest()[:12]
        wrapper = (f"# workspace {digest}\nimport runpy as _runpy\n_sys.path.insert(0, {str(self.ws)!r})\n"
                   f"_sys.argv = ['main.py']\n_runpy.run_path({str(self.ws / 'main.py')!r}, run_name='__main__')\n")
        val, res, fb = self.run_attempt(wrapper, f"w{self.turn}", prefer_new_on_tie=prefer_new)
        return val, res, drop_wrapper_frames(fb)

    def _loop_compact(self, force: bool = False) -> bool:
        """Bound the prompt: keep the head (system, task, plan, loop instructions) and the last step, and put the
        current workspace files into the last observation.  True when something was dropped."""
        total = sum(len(m["content"]) for m in self.messages)
        if (not force and total < 90000) or len(self.messages) <= self.head_len + 2:
            return False
        tail = [dict(m) for m in self.messages[-2:]]
        dropped = len(self.messages) - self.head_len - 2
        if dropped <= 0:
            return False
        tail[-1]["content"] += ("\n\n(Earlier steps were removed to save space.)\nCURRENT WORKSPACE FILES:\n" +
                                self._ws_state(20000 if force else 30000))
        self.messages = self.messages[: self.head_len] + tail
        log(f"loop: compacted the history ({total} chars, {dropped} messages dropped)")
        return True

    def _loop_apply(self, a: Action, obs: list[str]) -> bool:
        """Apply one write/append/edit action; True when the workspace changed."""
        name = a.name
        if a.kind == "edit":
            if not a.complete:
                obs.append(f"The edit of {name} was cut off by the token limit and was NOT applied; send shorter edits.")
                return False
            if name not in self.files:
                obs.append(f"edit {name} FAILED: there is no such file (workspace: {self._ws_listing()}).")
                return False
            new = apply_edit_blocks(self.files[name], a.body)
            if new is None:
                self.edit_failures += 1
                cur = self.files[name]
                show = f"\nCurrent {name}:\n```python\n{cur.rstrip()}\n```" if len(cur) <= 8000 else ""
                obs.append(f"edit {name} FAILED: a SEARCH text was not found verbatim (or no SEARCH/REPLACE block was given); "
                           f"nothing in {name} was changed. Copy the SEARCH lines exactly from the current file." + show)
                log(f"loop: edit of {name} did not apply")
                return False
            self._ws_write(name, new if new.endswith("\n") else new + "\n")
            obs.append(f"edited {name} ({len(_EDIT_RE.findall(a.body))} block(s); now {len(new.splitlines())} lines)")
            return True
        body = a.body
        if a.kind == "append" and name in self.files:
            body = stitch_code(self.files[name], body) if a.complete else self.files[name].rstrip("\n") + "\n" + body
            what = "appended to"
        else:
            what = "wrote"
        if not a.complete:
            kept = rollback_to_statement(body)
            self.cut_files += 1
            last = next((ln.strip() for ln in reversed(kept.splitlines()) if ln.strip()), "")
            log(f"loop: {name} cut at the token limit; kept {len(kept.splitlines())} lines")
            obs.append(f"Your answer was cut off by the token limit inside {name}: kept lines 1-{len(kept.splitlines())} "
                       f"(the last kept line is `{last[:120]}`). Continue it with an append={name} block containing only the "
                       f"remaining lines, or split the file into smaller modules. Keep the reasoning short.")
            body = kept
        if self.files.get(name) == body:
            obs.append(f"{name} is unchanged (same text as before), so nothing was run again.")
            return False
        self._ws_write(name, body)
        obs.append(f"{what} {name} (now {len(body.splitlines())} lines)")
        return True

    def agent_loop(self, context: str, have_plan: bool, reference: str = "") -> None:
        """Multi-step loop: the model inspects data, writes/edits workspace modules, and sees every run's result."""
        self.ws = self.workdir / "ws"
        self.ws.mkdir(parents=True, exist_ok=True)
        ref = (MEMORY_PREFACE + reference + "\n\n") if reference else ""
        start = (ref + LOOP_INSTRUCTIONS + f"\n\nBudget for this task: {self.house.requests_left()} answers and about "
                 f"{int(self.budget.remaining())} seconds in total. Start now.")
        if have_plan:
            self.messages.append({"role": "user", "content": start})
        else:
            self.messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": context + "\n\n" + start}]
        self.head_len = len(self.messages)
        success = False
        after_success = 0
        idle = 0
        last_clean_issues: list[str] | None = None
        temperature: float | None = None
        inspect_only = 0
        while True:
            if not self._can_call(40):
                log("loop: stopping, no time/requests left")
                break
            self._loop_compact()
            try:
                content, finish = self.house.chat(self.messages, label=f"step{self.turn + 1}", temperature=temperature)
            except ContextTooLong as exc:
                log(f"loop: context too long ({exc})")
                if not self._loop_compact(force=True):
                    break
                continue
            except ModelError as exc:
                log(f"loop: stopping ({exc})")
                break
            self.turn += 1
            temperature = None
            answer = strip_reasoning(content)
            actions, done = parse_actions(content)
            labelled = [a for a in actions if a.kind != "bare"]
            if not labelled:
                # tolerate a plain ```python block: a long one is the main script, a short one an inspection
                bare = [a for a in actions if a.kind == "bare"]
                if bare:
                    big = max(bare, key=lambda a: len(a.body))
                    labelled = [Action("write", "main.py", big.body, big.complete)] if len(big.body.splitlines()) >= 20 \
                        else [Action("run", "", big.body, big.complete)]
            actions = labelled
            if not actions and not done:
                log("loop: no action parsed; answer starts: " + repr(answer.strip()[:400]))
            reasoning_cut = finish == "length" and "<think>" in content and "</think>" not in content
            log(f"loop step {self.turn}: finish={finish} reasoning={len(content) - len(answer)} chars answer={len(answer)} chars "
                f"actions={[a.kind + (':' + a.name if a.name else '') + ('' if a.complete else '(cut)') for a in actions]}"
                + (" DONE" if done else ""))
            if reasoning_cut:
                self._think_failure("reasoning cut at the token limit")
            self.messages.append({"role": "assistant", "content": answer.strip() or "(empty answer)"})
            obs: list[str] = []
            changed = False
            runs = 0
            for a in actions:
                if a.kind == "run":
                    runs += 1
                    if not a.complete:
                        obs.append(f"RUN {runs}: cut off by the token limit, not run.")
                    elif runs <= 3 and self.budget.remaining() > 60:
                        obs.append(self._loop_run_snippet(a.body, runs))
                    else:
                        obs.append(f"RUN {runs}: skipped (at most 3 inspections per answer, and not when time is short).")
                elif a.kind == "shell":
                    runs += 1
                    if a.complete and runs <= 3 and self.budget.remaining() > 60:
                        obs.append(self._loop_shell(a.body, runs))
                        synced = self._ws_sync()
                        if synced:
                            obs.append("workspace files created/changed by the shell: " + ", ".join(synced))
                            changed = True
                    else:
                        obs.append(f"SHELL {runs}: not run (cut off, or too many inspections in one answer).")
                elif a.kind == "invalid":
                    obs.append(a.body)
                else:
                    changed = self._loop_apply(a, obs) or changed
            inspect_only = inspect_only + 1 if actions and all(a.kind in ("run", "shell") for a in actions) and not changed else 0
            if inspect_only >= 3 and not success:
                obs.append("You have spent several answers only inspecting. Stop inspecting now and write the solution files "
                           "(main.py and any modules) in your next answer.")
            if not actions and not done:
                idle += 1
                obs.append("Your answer was cut off by the token limit before any complete action. Keep the reasoning much "
                           "shorter and act now, with small files." if finish == "length" else LOOP_NO_ACTION)
                if idle >= 2:
                    self._think_failure("no action in two answers")
            else:
                idle = 0
            ran_ok = False
            if changed and "main.py" in self.files:
                val, res, fb = self._loop_exec(prefer_new=success)
                clean = res is not None and res.returncode == 0 and not res.timed_out
                if val.ok:
                    ran_ok = True
                    if not success:
                        log(f"loop: deliverables validated at step {self.turn}")
                    success = True
                    obs.append("RUN main.py: OK, every deliverable was written and passed the automatic checks.\n"
                               "OUTPUT PREVIEW:\n" + summarize_outputs(self.out_dir, self.initial_files, limit=3500))
                else:
                    if clean and last_clean_issues is not None and val.issues == last_clean_issues and val.present == val.expected:
                        log("loop: same validation issues twice on a clean run; keeping the outputs as they are")
                        success = True
                        break
                    last_clean_issues = val.issues if clean else None
                    note = ""
                    if self.last_err_sig and self.err_seen.get(self.last_err_sig, 0) >= 2:
                        log(f"same error as an earlier attempt ({self.last_err_sig[:120]}); asking for a different approach")
                        note = "\n\n" + SAME_ERROR_NOTE
                        temperature = self.cfg.retry_temperature
                    preview = ("\n\nOUTPUT PREVIEW:\n" + summarize_outputs(self.out_dir, self.initial_files, limit=2500)) if clean and val.present else ""
                    obs.append("RUN main.py: FAILED.\n" + fb + preview + note)
            elif changed:
                obs.append("main.py does not exist yet, so nothing was run; write it once the modules it needs are in place.")
            if done:
                if success:
                    log(f"loop: DONE at step {self.turn}")
                    break
                obs.append("DONE ignored: " + ("there is no main.py yet." if "main.py" not in self.files else
                                               "the last run did not produce valid deliverables; fix the problems above."))
            if success:
                after_success += 1
                if after_success > self.cfg.loop_reviews or (after_success > 1 and not changed):
                    log("loop: review turns used; finishing")
                    break
                if ran_ok or not changed:
                    review = LOOP_REVIEW
                    if self.soft_warnings:
                        review += "\nAutomatic checks flagged (verify against the task; fix only if the task requires it):\n- " + "\n- ".join(self.soft_warnings[:10])
                    obs.append(review)
            elif self.house.requests_left() <= 2 or self.budget.remaining() < 2.5 * self.house.avg_latency() + 60:
                obs.append("LAST CHANCE: almost no answers/time are left. In your next answer make main.py write EVERY deliverable "
                           "(fix the error above with the smallest possible edit).")
            obs.append(f"[workspace: {self._ws_listing()}; left: {self.house.requests_left()} answers, ~{int(self.budget.remaining())} s]")
            self.messages.append({"role": "user", "content": "\n\n".join(obs)})
        log(f"loop: finished after {self.turn} steps; success={success}; inspections={self.probes_run}; "
            f"cut files={self.cut_files}; failed edits={self.edit_failures}; workspace: {self._ws_listing()}")

    def run_attempt(self, code: str, tag: str, prefer_new_on_tie: bool = False) -> tuple[Validation, RunResult | None, str]:
        """Execute code; returns (validation, run result, feedback text for the model)."""
        self.attempt += 1
        self.seen_code.add(code_fingerprint(code))
        self.last_code = code
        protected = {d.file.rsplit("/", 1)[-1] for d in self.deliverables}
        code = rewrite_paths(code, self.unit, self.out_dir, protected)
        err = syntax_check(code)
        if err:
            log(f"attempt {self.attempt}: {err}")
            return Validation(False, [err], 0, len(self.deliverables), -1.0), None, f"The script has a syntax error and could not start:\n{err}"
        script = self._write_script(code, tag)
        # Preserve the best outputs so far, then start from a clean output directory.  If they were
        # too large to copy into /tmp, leave them in place (the new run simply overwrites its files).
        if self.best_score >= 0:
            self.backup_ok = backup_outputs(self.out_dir, self.initial_files, self.best_dir)
        if self.backup_ok:
            clear_outputs(self.out_dir, self.initial_files)
        timeout = self._exec_timeout()
        log(f"attempt {self.attempt}: running {script.name} (timeout {timeout:.0f}s)")
        res = run_script(script, self.out_dir, timeout, self.workdir,
                         extra_env={"DATA_DIR": str(self.unit.data_root) if self.unit.data_root else ""})
        val = validate_outputs(self.out_dir, self.deliverables, self.initial_files)
        run_clean = (res.returncode == 0) and not res.timed_out
        self.soft_warnings = []
        if run_clean and val.present == val.expected:
            total = sum((p.stat().st_size for p in list_new_files(self.out_dir, self.initial_files)), 0)
            if total > 60 * 1024 * 1024:
                val.issues.append(f"outputs total {total / 1048576:.0f} MiB; the platform limit is 64 MiB (write only the deliverables)")
            if self.contract is not None and (self.contract.json_specs or self.contract.csv_specs):
                try:
                    hard, soft = check_outputs(self.out_dir, self.contract)
                except Exception:
                    hard, soft = [], []
                if hard:
                    val.issues.extend("spec: " + h for h in hard)
                    for h in hard[:6]:
                        log(f"   spec: {h}")
                self.soft_warnings = soft
            extra = self._extra_checks(self.out_dir, res.stdout)
            if extra:
                val.issues.extend(extra)
                for h in extra[:6]:
                    log(f"   check: {h[:200]}")
            if val.issues:
                val = Validation(False, val.issues, val.present, val.expected, val.score - 0.01 * len(val.issues))
        log(f"attempt {self.attempt}: rc={res.returncode} timed_out={res.timed_out} in {res.seconds:.1f}s; "
            f"deliverables {val.present}/{val.expected}; issues={len(val.issues)}")
        for iss in val.issues[:8]:
            log(f"   - {iss}")
        if (res.returncode != 0 or res.timed_out) and res.stderr.strip():
            for line in res.stderr.strip().splitlines()[-4:]:
                log(f"   stderr: {line[:240]}")
        self.last_err_sig = error_signature(res.stderr) if res.returncode != 0 else ""
        if self.last_err_sig:
            self.err_seen[self.last_err_sig] = self.err_seen.get(self.last_err_sig, 0) + 1
        # Feedback for the model
        fb: list[str] = []
        if res.timed_out:
            self.exec_timeouts += 1
            fb.append(f"The script was killed after {timeout:.0f} seconds (too slow); the next run is limited to {self._exec_timeout():.0f} seconds. "
                      f"Reduce the computational cost substantially (vectorize, coarser grids / fewer simulations or iterations while keeping "
                      f"the required accuracy, avoid Python loops over rows).")
        elif res.returncode != 0:
            fb.append(f"The script exited with code {res.returncode}.")
        if res.stderr.strip():
            fb.append("STDERR (tail):\n" + truncate(res.stderr.strip(), 3500, tail=2500))
        if res.stdout.strip():
            fb.append("STDOUT (tail):\n" + truncate(res.stdout.strip(), 1200, tail=900))
        if val.issues:
            fb.append("Deliverable validation problems:\n- " + "\n- ".join(val.issues[:12]))
        if self.cfg.probe2 and self.probes_sent < 2 and any(i.startswith("degenerate:") for i in val.issues):
            facts = data_facts(self.unit)
            if facts:
                self.probes_sent += 1
                log("probe: degenerate outputs; adding the input files' exact structure to the feedback")
                fb.append("The outputs above carry no information, which usually means the inputs were parsed wrongly (wrong column, sheet, "
                          "key, date format or filter). DATA FACTS (read from the actual input files; use these exact names and shapes):\n" + facts)
        if self.cfg.probe and self.probes_sent < 2 and res.returncode != 0 and DATA_ERR_RE.search(res.stderr or ""):
            facts = data_facts(self.unit)
            if facts:
                self.probes_sent += 1
                log("probe: adding the input files' exact structure to the feedback")
                fb.append("DATA FACTS (read from the actual input files; use these exact names, sheets and shapes):\n" + facts)
        if res.returncode == 0 and not res.timed_out and val.present == 0:
            fb.append(f"The script exited normally but wrote none of the deliverables into {self.out_dir}. Make sure the "
                      "main computation is actually executed (e.g. a defined main() is called at the bottom) and that "
                      "every file is written to that directory.")
        run_ok = (res.returncode == 0) and not res.timed_out
        score = val.score + (1.0 if run_ok else 0.0)
        improved = score > self.best_score or (prefer_new_on_tie and score >= self.best_score and run_ok and val.ok)
        if improved or not self.backup_ok:
            self.best_score = max(score, self.best_score) if not improved else score
            self.best_ok = run_ok and val.ok
            self.backup_ok = backup_outputs(self.out_dir, self.initial_files, self.best_dir)
        else:
            log("attempt did not improve on the best result; restoring the best outputs")
            restore_outputs(self.out_dir, self.initial_files, self.best_dir)
        return Validation(val.ok and run_ok, val.issues, val.present, val.expected, score), res, "\n\n".join(fb)

    # -- main -----------------------------------------------------------------------------
    def solve(self) -> None:
        unit = self.unit
        previews = build_previews(unit)
        context = build_context(unit, self.out_dir, previews)
        log(f"context size: {len(context)} chars; data files: {len(unit.data_files)}; checks present: {unit.checks_present}")

        input_names = {df.path.name for df in unit.data_files}
        static_deliverables = merge_deliverables(
            deliverables_from_instruction(unit.instruction, input_names),
            deliverables_from_checks(unit.task_dir) if not self.cfg.ignore_checks else [],
        )
        self.deliverables = static_deliverables
        log("deliverables (static): " + ", ".join(d.file for d in static_deliverables) if static_deliverables else "deliverables (static): none detected")

        if self.contract is not None and (self.contract.json_specs or self.contract.csv_specs):
            log("spec contract: " + truncate(self.contract.summary(), 600))

        # v2: retrieve candidate memory documents (same unit excluded by id and by content)
        memory_toc = ""
        if self.memory is not None and self.memory.docs:
            try:
                hint: tuple = ()
                if self.cfg.rag2 and normalize_category is not None:
                    raw_cat = (unit.card.get("metadata") or {}).get("category", "") if isinstance(unit.card, dict) else ""
                    hint = normalize_category(raw_cat)
                    log(f"memory: card category {raw_cat!r} -> library categories {list(hint)}")
                sols, plays = self.memory.retrieve(
                    unit.instruction, unit_id=unit.unit_id, task_dir_name=unit.task_dir.name,
                    deliverables=[d.file for d in static_deliverables], data_names=[df.path.name for df in unit.data_files],
                    category_hint=hint)
                self.mem_candidates = (sols, plays)
                memory_toc, self.mem_labels = self.memory.toc(sols, plays)
                log(f"memory: candidates {[d.id for d, _ in sols]} + {[d.id for d, _ in plays]}")
            except Exception as exc:
                log(f"memory: retrieval failed ({exc})")
                memory_toc = ""
        context = build_context(unit, self.out_dir, previews, memory_toc=memory_toc)
        self.mc_task = bool(re.search(r"monte.?carlo|simulat|\bpaths\b|bootstrap|random (?:draws|numbers|walk)|\bseed\b", unit.instruction, re.I))
        if self.cfg.text_route:
            try:
                self.text_docs = text_route_docs(unit)
            except Exception as exc:
                log(f"text route: document scan failed ({exc})")
                self.text_docs = []
            self.text_routed = bool(self.text_docs)
            if self.text_routed:
                t_probe = time.time()
                try:
                    self.text_probe_text = text_probe(unit, self.text_docs)
                except Exception as exc:
                    log(f"text route: probe failed ({type(exc).__name__}: {exc})")
                    self.text_probe_text = ""
                if self.text_probe_text:
                    context += "\n\n" + self.text_probe_text
                log(f"text route: {len(self.text_docs)} document(s); probe {len(self.text_probe_text)} chars in {time.time() - t_probe:.1f}s")
        if self.cfg.template_route:
            tf = template_route_file(unit)
            if tf is not None:
                try:
                    self.template_code = tf.path.read_text(encoding="utf-8", errors="replace").strip()
                    self.template_name = tf.rel or tf.path.name
                except OSError:
                    self.template_code = ""
            if self.template_code:
                context += self._template_section()
                log(f"template route: {self.template_name} ({len(self.template_code.splitlines())} lines) is the script to debug")
        if self.cfg.probe_plan:
            exts = {df.path.suffix.lower() for df in unit.data_files}
            if len(unit.data_files) >= 3 or exts & {".html", ".htm", ".txt", ".pdf", ".xlsx", ".xls", ".xml"}:
                facts = data_facts(unit)
                if facts:
                    context += "\n\nDATA FACTS (read from the actual input files; use these exact names, sheets, keys and shapes):\n" + facts
                    log(f"probe-plan: input structure added to the plan context ({len(facts)} chars)")

        if not self.house.available:
            log("MODEL_ENDPOINT is not set: no model access; nothing can be generated (expected under --network=none)")
            return

        code = ""
        for compact in (0, 1, 2):
            if compact:
                # The served context window was too small for the full context: shrink it and retry.
                previews = build_previews(unit, total_chars=4000 if compact == 1 else 1500)
                context = build_context(unit, self.out_dir, previews, compact=compact, memory_toc=memory_toc)
                if compact == 1 and self.text_probe_text:
                    context += "\n\n" + self.text_probe_text[:4000]
                if compact == 1 and self.template_code:
                    context += self._template_section(20000)
                log(f"retrying with a compact context, level {compact} ({len(context)} chars)")
                self.messages = []
            plan_json = None
            have_plan = False
            try:
                if self.cfg.do_plan and self._can_call(45):
                    plan_json = self.plan(context, with_memory=bool(memory_toc) and compact < 2)
                    self.plan_json = plan_json if isinstance(plan_json, dict) else None
                    have_plan = len(self.messages) >= 3
                    if self.cfg.conventions and self.cfg.conv_vote and have_plan:
                        self._conventions_vote(context)
                    planned = deliverables_from_plan(plan_json)
                    if self.cfg.chunk:
                        self.n_parts, self.part_groups = chunk_count(plan_json, len(merge_deliverables(static_deliverables, planned or [])))
                    if planned:
                        self.deliverables = merge_deliverables(static_deliverables, planned)
                        log("deliverables (with plan): " + ", ".join(d.file for d in self.deliverables))
                        if self.contract is not None and parse_contract is not None:
                            try:  # restrict the spec checks to files we now know are deliverables
                                self.contract = parse_contract(unit.instruction, [d.file for d in self.deliverables])
                            except Exception:
                                pass
                # choose the reference material: the model's picks, else the BM25 leaders
                self.mem_picks = []
                if self.memory is not None and self.mem_labels:
                    if plan_json is not None and isinstance(plan_json, dict) and "memory" in plan_json:
                        self.mem_picks = parse_plan_memory(plan_json, self.mem_labels, max_solutions=self.cfg.mem_solutions)
                        if not self.mem_picks:
                            log("memory: the plan chose no reference material")
                    else:
                        self.mem_picks = default_picks(*self.mem_candidates)
                if self.memory is not None and (self.cfg.mem_triggers or self.text_routed):
                    try:
                        names = [df.path.name for df in unit.data_files] + [d["name"].rsplit(":", 1)[-1] for d in self.text_docs]
                        trig = self.memory.triggered(unit.instruction, names, unit_id=unit.unit_id, task_dir_name=unit.task_dir.name)
                    except Exception as exc:
                        log(f"memory: trigger lookup failed ({exc})")
                        trig = []
                    added = [d for d in trig if all(d.id != getattr(p, "id", None) for p in self.mem_picks)]
                    if added:
                        self.mem_picks = list(self.mem_picks) + added[:3]
                        log("memory: triggered docs added: " + ", ".join(d.id for d in added[:3]))
                if not self._can_call(30):
                    log("no budget left for code generation")
                    return
                if self.cfg.loop:
                    self.task_context = context
                    self.agent_loop(context, have_plan, self._reference_material(compact))
                    return
                if self.cfg.two_track and len(self.deliverables) >= self.cfg.track_l_min:
                    self.task_context = context
                    self.track = "L"
                    log(f"track L: {len(self.deliverables)} deliverables; core script + deliverable groups")
                    if self._track_l(self._reference_material(compact)):
                        self.track_l_done = True
                        code = self.last_code or "# track L"
                        break
                    log("track L: core script failed; falling back to the single-script track")
                    self.track = "S"
                if self.template_code:
                    code = self._template_start(context, have_plan, self._reference_material(compact))
                else:
                    code = self.first_code(context, have_plan, self._reference_material(compact))
                break
            except ContextTooLong as exc:
                log(f"context too long: {exc}")
                if compact >= 2:
                    return
                continue
            except ModelError as exc:
                log(f"code generation failed: {exc}")
                return
        if not code:
            return
        self.task_context = context
        if self.track_l_done:
            log("track L: finished; the repair loop, review and vote are skipped (groups were self-checked)")
            return

        success = False
        last_issues: list[str] | None = None
        prefer_new = False   # template route: the fixed script scores like the template it replaces -> ties go to the fix
        for _ in range(self.cfg.max_attempts):
            val, res, feedback = self.run_attempt(code, f"a{self.attempt + 1}", prefer_new_on_tie=prefer_new)
            prefer_new = False
            if val.ok and self.template_pending_fix:
                # the shipped template runs and validates by construction; the bugs are still in it
                self.template_pending_fix = False
                if not self._can_call(45):
                    success = True
                    break
                log("template route: the template runs; asking for the bug fixes")
                fb = TEMPLATE_FIX_FEEDBACK + "\n\n" + summarize_outputs(self.out_dir, self.initial_files)
                if res is not None and res.stdout.strip():
                    fb += "\n\nSTDOUT (tail):\n" + truncate(res.stdout.strip(), 1200, tail=900)
                try:
                    code = self.repair(fb)
                except ModelError as exc:
                    log(f"template route: fix request failed ({exc}); keeping the template's outputs")
                    success = True
                    break
                prefer_new = True
                continue
            if val.ok:
                success = True
                break
            if self.cfg.restart and not self.restarted and self.attempt == 1 and self._can_call(90):
                self.restarted = True
                new_code, val2, res2, fb2 = self._restart(feedback)
                if val2 is not None and val2.ok:
                    success = True
                    break
                if val2 is not None and val2.score >= val.score:
                    log("restart: continuing the repair loop from the fresh script")
                    code, val, res, feedback = new_code, val2, res2, fb2
                    for i in range(len(self.messages) - 1, -1, -1):
                        if self.messages[i]["role"] == "assistant":
                            self.messages[i] = {"role": "assistant", "content": "```python\n" + new_code + "\n```"}
                            break
            clean_run = res is not None and res.returncode == 0 and not res.timed_out
            if clean_run and last_issues is not None and val.issues == last_issues and val.present == val.expected:
                # The script runs and the model keeps producing the same disputed shape: stop arguing.
                log("same validation issues twice on a clean run; keeping the outputs as they are")
                success = True
                break
            last_issues = val.issues if clean_run else None
            if not self._can_call(45):
                log("stopping the repair loop: no time/requests left")
                break
            try:
                code = self.repair(feedback)
            except BudgetExhausted as exc:
                log(f"stopping: {exc}")
                break
            except ModelError as exc:
                log(f"repair failed: {exc}")
                break

        for step, enabled in (("selftest", self.cfg.selftest), ("nversion", self.cfg.nversion)):
            if success and enabled:
                try:
                    getattr(self, step)()
                except ModelError as exc:
                    log(f"{step} stopped: {exc}")
                except Exception as exc:  # an optional step must never cost the unit
                    log(f"{step} error: {type(exc).__name__}: {exc}")
                    try:
                        restore_outputs(self.out_dir, self.initial_files, self.best_dir)
                    except Exception:
                        pass

        if success and self.cfg.do_review and self._can_call(90) and self.house.requests_left() >= 2:
            try:
                new_code = self.review()
            except ModelError as exc:
                log(f"review failed: {exc}")
                new_code = None
            if new_code:
                log("review proposed a revised script; running it")
                val, res, feedback = self.run_attempt(new_code, "review", prefer_new_on_tie=True)
                if val.ok:
                    log("revised script accepted")
                else:
                    log("revised script failed validation; the previous outputs were kept")
            else:
                log("review verdict: OK (or no full replacement script)")

        steps = [("verify", lambda: self.verify(), lambda: self.cfg.verify and (success or self.best_ok)),
                 ("vote", lambda: self.vote(success or self.best_ok), lambda: self.cfg.vote)]
        if self.cfg.verify_fix2:
            steps.reverse()   # vote first: a confirmed answer skips the verification, an unconfirmed one gets the fix branch
        for step, fn, wanted in steps:
            if not wanted():
                continue
            try:
                fn()
            except ModelError as exc:
                log(f"{step} stopped: {exc}")
            except Exception as exc:  # an optional step must never cost the unit
                log(f"{step} error: {type(exc).__name__}: {exc}")
                try:
                    restore_outputs(self.out_dir, self.initial_files, self.best_dir)
                except Exception:
                    pass


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def write_fallback(out_dir: Path, deliverables: list[Deliverable], initial: set[str]) -> None:
    """If nothing was produced, write a minimal well-formed placeholder for the first deliverable.

    A placeholder never scores; it only keeps the unit's failure a plain 'wrong answer' rather than
    'no output', and it exercises the output mount.  JSON -> {}, CSV/TSV/TXT/HTML -> empty file,
    parquet -> an empty table.
    """
    if list_new_files(out_dir, initial):
        return
    name = deliverables[0].file if deliverables else "results.json"
    for d in deliverables:  # prefer a JSON deliverable when there is one
        if d.file.endswith(".json"):
            name = d.file
            break
    try:
        target = out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext == "json":
            target.write_text("{}\n", encoding="utf-8")
        elif ext in ("parquet", "pqt"):
            import pandas as pd

            pd.DataFrame().to_parquet(target)
        else:
            target.write_text("", encoding="utf-8")
        log(f"wrote placeholder {name}")
    except Exception:
        pass


def solve(task_dir: Path, out_dir: Path, cfg: Config) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(cfg.workdir) / f"{task_dir.name}-{uuid.uuid4().hex[:8]}"
    try:
        workdir.mkdir(parents=True, exist_ok=True)
    except Exception:
        workdir = Path("/tmp") / f"agent-{uuid.uuid4().hex[:8]}"
        workdir.mkdir(parents=True, exist_ok=True)
    unit = load_unit(task_dir, out_dir, cfg)
    # Soft budget: the configured per-unit allowance, scaled up (at most 1.6x) for units whose card grants
    # more time than the usual 1800 s, and never more than 85% of the card's own ceiling.
    scale = min(1.6, max(1.0, unit.agent_timeout / 1800.0))
    factor, why = (1.0, "")
    if cfg.time_typed:
        try:
            factor, why = typed_time_factor(unit)
        except Exception as exc:
            log(f"typed budget: classification failed ({exc}); using the plain budget")
    soft = min(cfg.time_budget * factor * scale, max(60.0, unit.agent_timeout * 0.85))
    hard = min(unit.agent_timeout - 45.0, soft + 120.0)
    budget = Budget(soft, hard)
    log(f"agent {AGENT_VERSION}: unit={unit.unit_id} card_timeout={unit.agent_timeout:.0f}s soft_budget={soft:.0f}s "
        f"model={cfg.model or '?'} endpoint={'set' if cfg.endpoint else 'unset'} out={out_dir}")
    if why:
        log(f"typed budget: x{factor:.2f} ({why})")

    solver = Solver(unit, cfg, budget, out_dir, workdir)

    def _alarm(signum, frame):  # pragma: no cover - last-resort guard
        raise TimeoutError("hard deadline reached")

    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(max(30, hard)))
    except Exception:
        pass
    try:
        solver.solve()
    except TimeoutError:
        log("hard deadline reached; finalising with the best outputs so far")
        kill_all_processes()
        if solver.best_score >= 0:
            restore_outputs(out_dir, solver.initial_files, solver.best_dir)
    except BudgetExhausted as exc:
        log(f"budget exhausted: {exc}")
    except Exception:
        log("unexpected error in solver:\n" + traceback.format_exc())
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass
        kill_all_processes()
    try:
        if solver.best_score >= 0 and not list_new_files(out_dir, solver.initial_files):
            restore_outputs(out_dir, solver.initial_files, solver.best_dir)
        scrub_outputs(out_dir, solver.initial_files, unit.canary)
        write_fallback(out_dir, solver.deliverables, solver.initial_files)
    except Exception:
        log("finalisation error:\n" + traceback.format_exc())
    make_outputs_readable(out_dir, solver.initial_files)
    files = list_new_files(out_dir, solver.initial_files)
    log(f"done in {budget.elapsed():.0f}s; requests used {solver.house.requests_made}; "
        f"output files: {[str(p.relative_to(out_dir)) for p in files][:20]}")
    try:
        log(solver.budget_report())
    except Exception:
        pass
    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent", description="Agenthon 2026 Track 1 coding agent")
    parser.add_argument("verb", nargs="?", default="solve")
    parser.add_argument("--task-dir", default="/input")
    parser.add_argument("--out", default="/app/output")
    parser.add_argument("--dump-context", action="store_true", help="print the model context for the unit and exit (debugging)")
    try:
        args, _unknown = parser.parse_known_args(argv)
    except SystemExit:
        return 2
    if args.verb != "solve":
        sys.stderr.write(f"unknown verb {args.verb!r}; only 'solve' is supported\n")
        return 2
    cfg = Config.from_env()
    if args.dump_context:
        unit = load_unit(Path(args.task_dir), Path(args.out), cfg)
        toc = ""
        if cfg.memory and MemoryIndex is not None:
            idx = MemoryIndex(Path(cfg.memory_dir) if cfg.memory_dir else Path(_HERE) / "memory", log=log)
            sols, plays = idx.retrieve(unit.instruction, unit_id=unit.unit_id, task_dir_name=unit.task_dir.name,
                                       data_names=[df.path.name for df in unit.data_files])
            toc, _ = idx.toc(sols, plays)
        print(build_context(unit, Path(args.out), build_previews(unit), memory_toc=toc))
        if cfg.selfcheck and parse_contract is not None:
            print("\n## SPEC CONTRACT: " + parse_contract(unit.instruction).summary())
        dl = merge_deliverables(deliverables_from_instruction(unit.instruction, {df.path.name for df in unit.data_files}),
                                deliverables_from_checks(unit.task_dir) if not cfg.ignore_checks else [])
        print("\n## STATIC DELIVERABLES: " + ", ".join(d.file for d in dl))
        return 0
    try:
        solve(Path(args.task_dir), Path(args.out), cfg)
    except Exception:
        log("fatal error:\n" + traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(main())
