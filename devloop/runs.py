"""Evaluation runs on disk: the layout written by eval/local_eval.sh (and by the host runner).

    <run>/results.csv          unit,pass,agent_exit,seconds,requests,checker_summary,output_files
    <run>/logs/<unit>.agent.log
    <run>/logs/<unit>.check.log
    <run>/out/<unit>/...        the agent's deliverables
    <run>/run.json             devloop metadata (label, runner, image, AGENT_* env, memory dir, commit, times)

A *group* is a directory holding repeats r1/, r2/, ... of the same configuration.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .util import read_json, read_text

RESULTS_HEADER = ["unit", "pass", "agent_exit", "seconds", "requests", "checker_summary", "output_files"]


@dataclass
class Run:
    path: Path
    rows: dict[str, dict] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Run":
        path = Path(path)
        rows: dict[str, dict] = {}
        with open(path / "results.csv", newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                if row.get("unit"):
                    rows[row["unit"]] = row
        return cls(path=path, rows=rows, meta=read_json(path / "run.json", {}) or {})

    @property
    def label(self) -> str:
        return str(self.meta.get("label") or self.path.name)

    def units(self) -> list[str]:
        return sorted(self.rows)

    def passed(self) -> set[str]:
        return {u for u, r in self.rows.items() if str(r.get("pass", "")).strip() == "1"}

    @property
    def n_passed(self) -> int:
        return len(self.passed())

    def agent_log(self, unit: str) -> str:
        return read_text(self.path / "logs" / f"{unit}.agent.log")

    def check_log(self, unit: str) -> str:
        return read_text(self.path / "logs" / f"{unit}.check.log")

    def out_dir(self, unit: str) -> Path:
        return self.path / "out" / unit


def is_run(path: Path) -> bool:
    return (Path(path) / "results.csv").is_file()


def expand_runs(paths) -> list[Run]:
    """Run directories, or group directories whose r*/ subdirectories are runs."""
    runs: list[Run] = []
    for p in paths:
        p = Path(p)
        if is_run(p):
            runs.append(Run.load(p))
            continue
        subs = sorted((d for d in p.glob("r*") if d.is_dir() and is_run(d)), key=lambda d: (len(d.name), d.name))
        if not subs:
            raise FileNotFoundError(f"no results.csv in {p} or in its r*/ subdirectories")
        runs.extend(Run.load(d) for d in subs)
    return runs


def pass_rates(runs: list[Run]) -> dict[str, float]:
    """Mean pass over the runs, for the units every run attempted."""
    if not runs:
        return {}
    common = set(runs[0].rows)
    for r in runs[1:]:
        common &= set(r.rows)
    return {u: sum(u in r.passed() for r in runs) / len(runs) for u in sorted(common)}


def failures(runs: list[Run]) -> dict[str, Run]:
    """Units that failed in at least half of the runs, each with the first run in which it failed (for its logs)."""
    out: dict[str, Run] = {}
    for unit, rate in pass_rates(runs).items():
        if rate <= 0.5:
            out[unit] = next(r for r in runs if unit not in r.passed())
    return out
