"""A small OpenAI-compatible chat client for the teacher model (stdlib only), plus answer parsers.

The teacher is any OpenAI-compatible endpoint: the House model served by eval/serve_vllm.sh works, a stronger
model writes better lessons.  Configure with DEVLOOP_ENDPOINT / DEVLOOP_MODEL / DEVLOOP_TOKEN or the CLI flags.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .util import log


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    endpoint: str
    model: str = "house"
    token: str = "local"
    timeout: float = 600.0
    max_tokens: int = 4000
    temperature: float = 0.2
    retries: int = 4

    @classmethod
    def from_env(cls, endpoint: str | None = None, model: str | None = None, token: str | None = None) -> "LLMConfig | None":
        ep = (endpoint or os.environ.get("DEVLOOP_ENDPOINT") or "").strip()
        if not ep:
            return None
        return cls(endpoint=ep,
                   model=(model or os.environ.get("DEVLOOP_MODEL") or "house").strip(),
                   token=(token or os.environ.get("DEVLOOP_TOKEN") or "local").strip(),
                   max_tokens=int(os.environ.get("DEVLOOP_MAX_TOKENS", "4000")),
                   temperature=float(os.environ.get("DEVLOOP_TEMPERATURE", "0.2")))


def chat_url(endpoint: str) -> str:
    """Accept an origin (like MODEL_ENDPOINT), a .../v1 base URL, or the full completions URL."""
    ep = endpoint.rstrip("/")
    if ep.endswith("/chat/completions"):
        return ep
    if ep.endswith("/v1"):
        return ep + "/chat/completions"
    return ep + "/v1/chat/completions"


def _opener(url: str) -> urllib.request.OpenerDirector:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host in ("127.0.0.1", "localhost", "::1"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never send loopback traffic to a proxy
    return urllib.request.build_opener()


class ChatClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.url = chat_url(cfg.endpoint)
        self.calls = 0

    def chat(self, messages: list[dict], *, max_tokens: int | None = None, temperature: float | None = None) -> str:
        body = {"model": self.cfg.model, "messages": messages,
                "max_tokens": int(max_tokens or self.cfg.max_tokens),
                "temperature": self.cfg.temperature if temperature is None else float(temperature)}
        data = json.dumps(body).encode()
        delay = 2.0
        last = ""
        for attempt in range(self.cfg.retries + 1):
            req = urllib.request.Request(self.url, data=data, method="POST", headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self.cfg.token}"})
            try:
                with _opener(self.url).open(req, timeout=self.cfg.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                self.calls += 1
                choice = (payload.get("choices") or [{}])[0]
                content = str((choice.get("message") or {}).get("content") or "")
                return strip_reasoning(content)
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}: {exc.read()[:300].decode('utf-8', errors='replace')}"
                if exc.code not in (408, 409, 425, 429, 500, 502, 503, 504):
                    raise LLMError(last) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt < self.cfg.retries:
                log(f"teacher request failed ({last}); retrying in {delay:.0f}s")
                time.sleep(delay)
                delay *= 2
        raise LLMError(f"teacher request failed after {self.cfg.retries + 1} attempts: {last}")

    def chat_json(self, messages: list[dict], **kw) -> dict:
        answer = self.chat(messages, **kw)
        obj = extract_json(answer)
        if obj is None:
            follow = messages + [{"role": "assistant", "content": answer},
                                 {"role": "user", "content": "Reply with exactly one JSON object and nothing else."}]
            obj = extract_json(self.chat(follow, **kw))
        if obj is None:
            raise LLMError("the teacher did not return a JSON object")
        return obj


def strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:                    # reasoning whose opening tag was in the prompt template
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> dict | None:
    for m in re.finditer(r"```(?:json)?\s*\n(.*?)```", text, re.S):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except ValueError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def extract_code(text: str) -> str | None:
    """The longest ```python block (a reply may also quote snippets)."""
    blocks = [m.group(1) for m in re.finditer(r"```(?:python|py)\s*\n(.*?)```", text, re.S)]
    if not blocks:
        blocks = [m.group(1) for m in re.finditer(r"```\s*\n(.*?)```", text, re.S)]
    blocks = [b for b in blocks if b.strip()]
    return max(blocks, key=len).rstrip() + "\n" if blocks else None


_FILE_HEAD = re.compile(r"^###\s*FILE:\s*(?P<path>\S+)\s*\((?P<action>create|update)\)(?:\s*sources:\s*(?P<sources>.*))?\s*$",
                        re.I | re.M)


def extract_files(text: str) -> list[dict]:
    """Parse `### FILE: playbooks/<slug>.md (create|update) sources: t1-a, t1-b` headers, each followed by a fenced
    block of 3+ backticks (4 lets the document contain its own ``` snippets)."""
    out = []
    for m in _FILE_HEAD.finditer(text):
        rest = text[m.end():]
        fence = re.match(r"\s*\n(`{3,})[^\n]*\n", rest)
        if not fence:
            continue
        ticks = fence.group(1)
        body_start = fence.end()
        close = re.search(r"^" + ticks + r"\s*$", rest[body_start:], re.M)
        if not close:
            continue
        content = rest[body_start:body_start + close.start()]
        sources = [s.strip() for s in (m.group("sources") or "").split(",") if s.strip()]
        out.append({"path": m.group("path").strip(), "action": m.group("action").lower(),
                    "sources": sources, "content": content.rstrip() + "\n"})
    return out


def prompt(name: str) -> str:
    """A prompt template from devloop/prompts/<name>.md."""
    from .util import PROMPTS_DIR, read_text
    text = read_text(PROMPTS_DIR / f"{name}.md")
    if not text:
        raise FileNotFoundError(f"missing prompt template {name}.md")
    return text
