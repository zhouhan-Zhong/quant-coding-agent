"""Shared helpers: repository paths, canary hygiene, JSON files, the task kit, the image configuration."""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

try:                                    # Python 3.11+
    import tomllib
except ModuleNotFoundError:             # Python 3.10 (the agent treats tomllib as optional the same way)
    try:
        import tomli as tomllib         # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None                  # type: ignore[assignment]

REPO = Path(__file__).resolve().parents[1]
AGENT_DIR = REPO / "agent"
MEMORY_DIR = AGENT_DIR / "memory"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

GUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_CANARY_LINE = re.compile(r"canary|BENCHMARK DATA", re.I)


def log(msg: str) -> None:
    print(f"[devloop {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scrub(text: str) -> str:
    """Drop canary lines and mask GUIDs.  Everything devloop sends to a model or writes to disk passes through here."""
    lines = [ln for ln in str(text).splitlines() if not _CANARY_LINE.search(ln)]
    return GUID_RE.sub("<guid>", "\n".join(lines))


def clip(text: str, limit: int, tail_share: float = 0.3) -> str:
    """Keep the head and the tail of a long text."""
    text = str(text)
    if len(text) <= limit:
        return text
    tail = int(limit * tail_share)
    head = max(0, limit - tail - 40)
    return text[:head] + "\n...[clipped]...\n" + (text[-tail:] if tail else "")


def read_text(path: Path, limit: int | None = None) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return clip(text, limit) if limit else text


def write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, obj: Any) -> Path:
    return write_text(path, json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> Path:
    return write_text(path, "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for ln in read_text(path).splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


def git_commit(repo: Path = REPO) -> str:
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def parse_env_pairs(pairs: Iterable[str]) -> dict[str, str]:
    """["AGENT_VOTE=0", ...] -> {"AGENT_VOTE": "0"} (also accepts one string with spaces)."""
    out: dict[str, str] = {}
    for item in pairs or ():
        for tok in shlex.split(item):
            if "=" not in tok:
                raise ValueError(f"expected KEY=VALUE, got {tok!r}")
            k, v = tok.split("=", 1)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
                raise ValueError(f"bad variable name {k!r}")
            out[k] = v
    return out


def dockerfile_env(path: Path = REPO / "Dockerfile") -> dict[str, str]:
    """The ENV configuration baked into the image (the host runner applies it, so both runners agree)."""
    text = read_text(path)
    joined = re.sub(r"\\\r?\n", " ", text)
    env: dict[str, str] = {}
    for line in joined.splitlines():
        m = re.match(r"^\s*ENV\s+(.*)$", line)
        if not m:
            continue
        body = m.group(1).strip()
        toks = shlex.split(body, comments=False)
        if toks and all("=" in t for t in toks):
            for t in toks:
                k, v = t.split("=", 1)
                env[k] = v
        elif len(toks) >= 2:  # legacy "ENV KEY value"
            env[toks[0]] = " ".join(toks[1:])
    return env


class Kit:
    """A clone of Agenthon-2026/track1-coding-public (units/<id>/instruction.md, card.toml, environment/, checks/)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        if not (self.root / "units").is_dir():
            raise FileNotFoundError(f"not a task kit (no units/): {self.root}")

    def units(self) -> list[str]:
        return sorted(p.name for p in (self.root / "units").iterdir() if p.is_dir() and p.name.startswith("t1-"))

    def unit_dir(self, unit: str) -> Path:
        return self.root / "units" / unit

    def instruction(self, unit: str) -> str:
        return read_text(self.unit_dir(unit) / "instruction.md")

    def card(self, unit: str) -> dict:
        text = read_text(self.unit_dir(unit) / "card.toml")
        try:
            return tomllib.loads(text) if tomllib is not None else mini_toml(text)
        except Exception:  # noqa: BLE001  (a malformed card only loses its metadata)
            return {}

    def category(self, unit: str) -> str:
        return str((self.card(unit).get("metadata") or {}).get("category") or "")

    def tags(self, unit: str) -> list[str]:
        tags = (self.card(unit).get("metadata") or {}).get("tags") or []
        return [str(t) for t in tags] if isinstance(tags, list) else []

    def checker(self, unit: str) -> str:
        return read_text(self.unit_dir(unit) / "checks" / "test_outputs.py")

    def data_dir(self, unit: str) -> Path:
        d = self.unit_dir(unit) / "environment" / "data"
        return d if d.is_dir() else self.unit_dir(unit) / "environment"

    def data_files(self, unit: str) -> list[Path]:
        d = self.data_dir(unit)
        return sorted(p for p in d.rglob("*") if p.is_file() and p.name != "Dockerfile") if d.is_dir() else []


_TOML_VALUE = re.compile(r"""^("(?:[^"\\]|\\.)*"|'[^']*'|\[.*\]|[^#]*)""")


def mini_toml(text: str) -> dict:
    """Enough TOML for task cards when neither tomllib nor tomli is available: [a.b] tables and one-line
    `key = "string" | 'string' | [list] | number | true/false` entries."""
    import ast
    data: dict = {}
    cur = data
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^\[([^\[\]]+)\]$", line)
        if m:
            cur = data
            for part in m.group(1).split("."):
                cur = cur.setdefault(part.strip().strip('"'), {})
            continue
        m = re.match(r"^([A-Za-z0-9_\-]+)\s*=\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), _TOML_VALUE.match(m.group(2).strip()).group(1).strip()
        if val in ("true", "false"):
            cur[key] = val == "true"
            continue
        try:
            cur[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            cur[key] = val.strip("\"'")
    return data


def preview_file(path: Path, limit: int = 1200) -> str:
    """A short, scrubbed look at an output or input file for a prompt."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        return f"{path.name}: unreadable"
    head = f"{path.name} ({size} bytes)"
    if path.suffix.lower() in (".json", ".csv", ".tsv", ".txt", ".md", ".html", ".xml", ".yaml", ".yml"):
        body = read_text(path)
        if path.suffix.lower() in (".csv", ".tsv"):
            lines = body.splitlines()
            body = "\n".join(lines[:15]) + (f"\n... ({len(lines)} lines)" if len(lines) > 15 else "")
        return head + ":\n" + clip(scrub(body), limit)
    return head
