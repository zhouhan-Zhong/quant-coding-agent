"""Unit tests for agent 2.9.11: vote_max up to 10 and the local-only candidate dump (AGENT_VOTE_DUMP)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))
sys.path.insert(0, str(HERE))

import agent  # noqa: E402
from test_v22_features import FakeHouse, _solver  # noqa: E402


def test_vote_max_clamp_and_dump_flag(monkeypatch):
    monkeypatch.setenv("AGENT_VOTE_MAX", "8")
    monkeypatch.setenv("AGENT_VOTE_DUMP", "/vote_dump")
    c = agent.Config.from_env()
    assert c.vote_max == 8 and c.vote_dump == "/vote_dump"
    monkeypatch.setenv("AGENT_VOTE_MAX", "50")
    monkeypatch.delenv("AGENT_VOTE_DUMP")
    c = agent.Config.from_env()
    assert c.vote_max == 10 and c.vote_dump == ""


def test_dump_candidate_and_meta(tmp_path):
    s = _solver(tmp_path, n_deliv=1, vote_dump=str(tmp_path / "dump"))
    s.house = FakeHouse([], s.cfg)
    cand = tmp_path / "cand"
    cand.mkdir()
    (cand / "f0.json").write_text(json.dumps({"v": 1.5}))
    meta: list = []
    s._dump_candidate(0, 0, cand, meta)
    s._dump_candidate(1, 3, cand, meta)
    s._dump_meta(meta, success=True, attempts=3, support=[1, 1], winner=0, agree=[[0, 1]])
    d = tmp_path / "dump"
    assert json.loads((d / "c1" / "f0.json").read_text()) == {"v": 1.5}
    m = json.loads((d / "meta.json").read_text())
    assert [c["attempt"] for c in m["candidates"]] == [0, 3] and m["winner"] == 0 and m["unit"] == "t1-test"


def test_dump_off_writes_nothing(tmp_path):
    s = _solver(tmp_path, n_deliv=1)
    meta: list = []
    s._dump_candidate(0, 0, tmp_path, meta)
    s._dump_meta(meta, success=True)
    assert meta == [] and not (tmp_path / "meta.json").exists()


def test_vote_dumps_every_candidate(tmp_path, monkeypatch):
    s = _solver(tmp_path, n_deliv=1, vote=True, vote_max=3, vote_dump=str(tmp_path / "dump"))
    s.house = FakeHouse([], s.cfg)
    (s.out_dir / "f0.json").write_text(json.dumps({"v": 1.0}))
    s.last_code = "print(1)"
    dirs = []
    for k, v in ((1, 2.0), (2, 2.0), (3, 3.0)):
        d = tmp_path / f"cand{k}"
        d.mkdir()
        (d / "f0.json").write_text(json.dumps({"v": v}))
        dirs.append((d, f"code{k}"))
    monkeypatch.setattr(s, "_vote_batch", lambda ks, prompt: [dirs[k - 1] for k in ks])
    monkeypatch.setattr(s, "_can_call", lambda *a, **k: True)
    s.vote(True)
    m = json.loads((tmp_path / "dump" / "meta.json").read_text())
    assert [c["attempt"] for c in m["candidates"]] == [0, 1, 2, 3]    # the original is never confirmed -> up to vote_max
    assert m["support"] == [0, 1, 1, 0] and m["winner"] == 1 and m["agree"] == [[1, 2]]
    assert json.loads((tmp_path / "dump" / "c2" / "f0.json").read_text()) == {"v": 2.0}
    assert json.loads((s.out_dir / "f0.json").read_text()) == {"v": 2.0}   # replaced by the agreeing pair


def test_memory_v4_gate(monkeypatch):
    import memory
    root = HERE.parent / "agent" / "memory"
    v4 = set(json.loads((root / "v4_docs.json").read_text()))
    monkeypatch.setenv("AGENT_MEMORY_V4", "0")
    assert not v4 & {d.id for d in memory.MemoryIndex(root).docs}
    monkeypatch.setenv("AGENT_MEMORY_V4", "1")
    assert v4 <= {d.id for d in memory.MemoryIndex(root).docs}
