"""Unit tests for agent 2.8: second-generation memory documents (v2_docs.json gate), task-card category routing
(normalize_category + reserved category slots in retrieval) and file-type query tokens (AGENT_RAG2)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))

import memory  # noqa: E402

MEM = HERE.parent / "agent" / "memory"
V2 = json.loads((MEM / "v2_docs.json").read_text())
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def _doc_path(doc_id: str) -> Path:
    for sub in ("playbooks", "notes"):
        p = MEM / sub / f"{doc_id}.md"
        if p.exists():
            return p
    raise AssertionError(doc_id)


def test_v2_manifest_lists_existing_generic_documents():
    assert len(V2) >= 10
    for d in V2:
        text = _doc_path(d).read_text(encoding="utf-8")
        fm = text.split("\n---", 1)[0]
        for key in ("title:", "category:", "tags:", "summary:"):
            assert key in fm, (d, key)
        cat = re.search(r"^category:\s*(.+)$", fm, re.M).group(1).strip()
        assert cat in memory.LIBRARY_CATEGORIES or cat == "meta", (d, cat)
        n = len(text.splitlines())
        assert 50 <= n <= 130, (d, n)
        assert not _UUID.search(text) and "canary" not in text.lower(), d
        assert not re.search(r"\bt1-[a-z0-9]", text), d          # generic: no public unit ids
        if "/playbooks/" in str(_doc_path(d)):
            for head in ("## When this applies", "## Method and formulas", "## Conventions and pitfalls", "## Tested snippet"):
                assert head in text, (d, head)
            assert text.count("```") == 2, d


def test_v2_gate_hides_exactly_the_manifest(monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_V2", raising=False)
    full = memory.MemoryIndex(MEM)
    monkeypatch.setenv("AGENT_MEMORY_V2", "0")
    gated = memory.MemoryIndex(MEM)
    ids_full, ids_gated = {d.id for d in full.docs}, {d.id for d in gated.docs}
    assert set(V2) <= ids_full
    assert ids_full - ids_gated == set(V2)
    assert gated.hidden_v2 == set(V2) and not full.hidden_v2


@pytest.mark.parametrize("raw, expected", [
    ("derivatives-pricing", ("derivatives-pricing", "volatility")),
    ("fixed-income", ("rates-curves",)),
    ("credit", ("credit",)),
    ("factor-research", ("backtest-factors", "portfolio")),
    ("backtesting", ("backtest-factors",)),
    ("risk-management", ("risk", "portfolio")),
    ("microstructure", ("microstructure",)),
    ("fx", ("fx-commodities-crypto",)),
    ("nlp-on-finance", ("text-filings",)),
    ("cross-domain", ()),
    ("", ()),
    ("Credit_Risk", ("credit",)),
    (["fx", "credit"], ("fx-commodities-crypto", "credit")),
    ("event-driven-analysis, data-processing", ("text-filings", "backtest-factors", "data-engineering")),
    ("something-unknown", ()),
])
def test_normalize_category(raw, expected):
    assert memory.normalize_category(raw) == expected


def _mini_library(tmp_path: Path, n_a: int = 6, n_b: int = 3) -> Path:
    root = tmp_path / "mem"
    (root / "playbooks").mkdir(parents=True)
    (root / "solutions").mkdir()
    for i in range(n_a):
        (root / "playbooks" / f"alpha-{i}.md").write_text(
            f"---\ntitle: Alpha card {i}\ncategory: risk\ntags: alpha, zeta, omega\nsummary: alpha zeta omega\n---\n"
            "## Method\nalpha zeta omega alpha zeta omega gamma delta\n" * 3)
    for i in range(n_b):
        (root / "playbooks" / f"beta-{i}.md").write_text(
            f"---\ntitle: Beta card {i}\ncategory: credit\ntags: beta, zeta\nsummary: beta zeta\n---\n"
            "## Method\nbeta zeta gamma delta\n" * 3)
    return root


def test_category_slots_are_reserved_only_with_a_hint(tmp_path):
    idx = memory.MemoryIndex(_mini_library(tmp_path))
    q = "alpha zeta omega gamma delta " * 5
    _, plain = idx.retrieve(q, unit_id="u", task_dir_name="u", n_playbooks=6)
    assert [d.category for d, _ in plain].count("credit") == 0          # alpha cards outscore every beta card
    _, routed = idx.retrieve(q, unit_id="u", task_dir_name="u", n_playbooks=6, category_hint=("credit",))
    cats = [d.category for d, _ in routed]
    assert len(routed) == 6 and len({d.id for d, _ in routed}) == 6
    assert cats.count("credit") == 2 and cats[:4] == ["risk"] * 4       # two slots reserved, leaders untouched
    assert routed[0][0].id == plain[0][0].id
    _, hinted_other = idx.retrieve(q, unit_id="u", task_dir_name="u", n_playbooks=6, category_hint=("fx-commodities-crypto",))
    assert [d.id for d, _ in hinted_other] == [d.id for d, _ in plain]  # no document of that category: nothing changes


def test_reservation_respects_the_score_floor(tmp_path):
    idx = memory.MemoryIndex(_mini_library(tmp_path))
    q = "alpha omega " * 8                                              # beta cards share no query term -> score 0 -> never promoted
    _, routed = idx.retrieve(q, unit_id="u", task_dir_name="u", n_playbooks=6, category_hint=("credit",))
    assert all(d.category == "risk" for d, _ in routed)


def test_file_type_tokens_only_when_requested():
    q0 = memory.MemoryIndex.build_query("parse the filings", data_names=["a.xml", "b.tsv", "c.csv"])
    q1 = memory.MemoryIndex.build_query("parse the filings", data_names=["a.xml", "b.tsv", "c.csv"], file_types=True)
    assert "xml" not in q0 and "tsv" not in q0
    assert q1["xml"] == 2 and q1["tsv"] == 2 and "csv" not in q1


def test_real_library_routing_keeps_the_leader_and_the_count():
    idx = memory.MemoryIndex(MEM)
    instr = "Bootstrap a piecewise-constant hazard rate curve from par CDS spreads and report survival probabilities and the CVA of a swap."
    sols, plays = idx.retrieve(instr, unit_id="x", task_dir_name="x", category_hint=memory.normalize_category("credit"))
    assert len(plays) == 6 and len(sols) == 6
    assert any(d.id == "cds-hazard-bootstrap-cva" for d, _ in plays)
    assert sum(d.category == "credit" for d, _ in plays) >= 2
    assert sum(d.category.startswith("credit") for d, _ in sols) >= 1   # public credit solutions are weak matches for CDS text
