"""The outer loop: development rounds and flag ablations.

A round:
  1. baseline   evaluate the agent with the current library (explicit copy, AGENT_PB_EXCLUDE=1), or reuse runs
  2. attribute  classify every failure (rules, then the teacher); write the mechanism report
  3. lessons    lesson cards for the failures that knowledge can fix (the teacher sees checker + verified solution)
  4. fold       general lessons -> playbook edits through the guards -> candidate library (+ pb_exclude provenance)
  5. verify     the candidate library (and any new solution against its checker)
  6. A/B        evaluate the candidate exactly like the baseline; keep it only if it gains >= min_gain units
  7. apply      with --apply, copy the changed files into agent/memory (review them with git diff before committing)
Rounds continue from the kept candidate and stop at the first round that keeps nothing.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .attribution import KNOWLEDGE, attribute, write_reports
from .compare import Comparison, compare, render
from .fold import apply_candidate, build_candidate, changed_files, fold
from .lessons import write_lessons
from .llm import ChatClient
from .runner import EvalSpec, run_eval
from .runs import Run, expand_runs, failures
from .util import MEMORY_DIR, Kit, log, read_text, write_json, write_text
from .verify import verify_memory


@dataclass
class LoopConfig:
    kit: Kit
    work: Path
    runner: str = "docker"
    image: str = ""
    units: list[str] | None = None
    repeats: int = 1
    jobs: int = 3
    env: dict[str, str] = field(default_factory=dict)
    model_endpoint: str = "http://127.0.0.1:8000"
    model_name: str = "house"
    model_token: str = "local"
    unit_timeout: int = 1800
    teacher: ChatClient | None = None
    min_gain: float = 2.0
    max_lessons: int = 12
    solve_attempts: int = 6
    apply: bool = False
    test_snippets: bool = True
    verify_solutions: bool = False
    target_memory: Path = MEMORY_DIR


def _spec(cfg: LoopConfig, memory_dir: Path | None, extra_env: dict | None = None) -> EvalSpec:
    env = {**cfg.env, "AGENT_PB_EXCLUDE": "1", **(extra_env or {})}
    return EvalSpec(kit=cfg.kit, runner=cfg.runner, image=cfg.image, units=cfg.units, jobs=cfg.jobs, env=env,
                    memory_dir=memory_dir, model_endpoint=cfg.model_endpoint, model_name=cfg.model_name,
                    model_token=cfg.model_token, unit_timeout=cfg.unit_timeout)


def evaluate(cfg: LoopConfig, group_dir: Path, memory_dir: Path | None, label: str, extra_env: dict | None = None) -> list[Run]:
    return [run_eval(_spec(cfg, memory_dir, extra_env), Path(group_dir) / f"r{k}", label=f"{label}-r{k}")
            for k in range(1, cfg.repeats + 1)]


def _copy_library(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


@dataclass
class RoundResult:
    n: int
    keep: bool
    stopped: str
    baseline: list[Run]
    candidate: list[Run] | None
    memory: Path | None
    comparison: Comparison | None
    report: Path


def run_round(cfg: LoopConfig, n: int, base_memory: Path, baseline: list[Run] | None) -> RoundResult:
    rdir = Path(cfg.work) / "rounds" / f"{n:02d}"
    rdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lines = [f"# devloop round {n}", "", f"- started {time.strftime('%Y-%m-%d %H:%M:%S')}, runner {cfg.runner}"
             + (f", image `{cfg.image}`" if cfg.image else ""), f"- base library: `{base_memory}`"]

    def finish(stopped: str, keep: bool = False, candidate=None, memory=None, cmp=None) -> RoundResult:
        lines.extend(["", f"**Result:** {stopped}", "", f"_{time.time() - t0:.0f} s_"])
        report = write_text(rdir / "report.md", "\n".join(lines) + "\n")
        log(f"round {n}: {stopped}")
        return RoundResult(n, keep, stopped, baseline, candidate, memory, cmp, report)

    base_copy = _copy_library(base_memory, rdir / "base-memory")
    if baseline is None:
        baseline = evaluate(cfg, rdir / "baseline", base_copy, label=f"round{n}-baseline")
    lines.append(f"- baseline: {', '.join(f'{r.label} {r.n_passed}/{len(r.rows)}' for r in baseline)}")

    recs = attribute(baseline, cfg.kit, cfg.teacher)
    write_reports(recs, baseline, rdir)
    counts: dict[str, int] = {}
    for r in recs:
        counts[r.category] = counts.get(r.category, 0) + 1
    lines += ["", "## Attribution", "", ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
              or "no failures", "", "Details: `attribution.md`; recurring mechanics: `mechanisms.md`."]
    if cfg.teacher is None:
        return finish("attribution only (no teacher configured: set DEVLOOP_ENDPOINT to write lessons)")
    if not any(r.category in KNOWLEDGE for r in recs):
        return finish("no failure that better knowledge could fix; see mechanisms.md")

    cards, new_solutions = write_lessons(recs, failures(baseline), cfg.kit, cfg.teacher, base_copy, rdir / "lessons",
                                         runner=cfg.runner, image=cfg.image, max_units=cfg.max_lessons,
                                         solve_attempts=cfg.solve_attempts)
    lines += ["", "## Lessons", "", f"{len(cards)} lesson card(s) in `lessons/`"
              + (f"; {len(new_solutions)} new verified solution(s): {', '.join(sorted(new_solutions))}" if new_solutions else "")]
    if not cards:
        return finish("no lesson card could be written")

    lessons = {u: read_text(p) for u, p in cards.items()}
    accepted, rejected = fold(lessons, base_copy, cfg.teacher, cfg.kit,
                              runner=cfg.runner if cfg.test_snippets else None, image=cfg.image, log_dir=rdir / "fold")
    write_json(rdir / "proposals.json", {
        "accepted": [{"path": e.path, "action": e.action, "sources": e.sources} for e in accepted],
        "rejected": [{"path": e.path, "action": e.action, "sources": e.sources, "findings": [str(f) for f in e.findings]}
                     for e in rejected]})
    lines += ["", "## Fold", "", f"accepted {len(accepted)}: " + (", ".join(f"{e.path} ({e.action})" for e in accepted) or "-"),
              f"rejected {len(rejected)}: " + (", ".join(e.path for e in rejected) or "-") + " (reasons in `proposals.json`)"]
    if not accepted and not new_solutions:
        return finish("nothing passed the guards")

    cand = build_candidate(base_copy, rdir / "memory", accepted, new_solutions, cfg.kit)
    rep = verify_memory(cand, base_memory=base_copy, kit=cfg.kit,
                        solutions=bool(new_solutions) or cfg.verify_solutions,
                        units=None if cfg.verify_solutions else (sorted(new_solutions) or None),
                        runner=cfg.runner, image=cfg.image)
    write_json(rdir / "verify.json", rep)
    lines += ["", "## Verify", "", f"{'ok' if rep['ok'] else 'FAILED'}; documents {rep['docs']}"
              + (f"; problems: {'; '.join(rep['problems'][:5])}" if rep["problems"] else "")]
    if not rep["ok"]:
        return finish("the candidate library failed verification", memory=cand)

    candidate = evaluate(cfg, rdir / "candidate", cand, label=f"round{n}-candidate")
    sources = sorted({u for e in accepted for u in e.sources} | set(new_solutions))
    cmp = compare(baseline, candidate, min_gain=cfg.min_gain, sources=sources, label_a="baseline", label_b="candidate")
    write_text(rdir / "compare.md", render(cmp))
    lines += ["", "## A/B", "", render(cmp).split("\n", 2)[2].strip()]
    if not cmp.keep:
        return finish(f"discarded ({cmp.delta:+g} units, below the +{cfg.min_gain:g} threshold)",
                      candidate=candidate, memory=cand, cmp=cmp)
    changed = changed_files(cand, base_copy)
    lines += ["", "## Kept", "", "Changed files: " + ", ".join(f"`{f}`" for f in changed)]
    if cfg.apply:
        apply_candidate(cand, cfg.target_memory, base_copy)
        lines.append(f"Applied to `{cfg.target_memory}`; review with `git diff` before committing.")
    else:
        lines.append(f"Not applied (run with --apply, or copy them from `{cand}`).")
    return finish(f"kept ({cmp.delta:+g} units)", keep=True, candidate=candidate, memory=cand, cmp=cmp)


def run_loop(cfg: LoopConfig, rounds: int = 1, baseline_paths: list[Path] | None = None,
             start_memory: Path | None = None) -> list[RoundResult]:
    base = Path(start_memory or cfg.target_memory)
    baseline = expand_runs(baseline_paths) if baseline_paths else None
    results: list[RoundResult] = []
    for n in range(1, rounds + 1):
        res = run_round(cfg, n, base, baseline)
        results.append(res)
        if not res.keep:
            break
        base = cfg.target_memory if cfg.apply else res.memory
        baseline = res.candidate
    summary = ["# devloop summary", "", "| round | result | report |", "|---|---|---|"]
    summary += [f"| {r.n} | {r.stopped} | rounds/{r.n:02d}/report.md |" for r in results]
    write_text(Path(cfg.work) / "summary.md", "\n".join(summary) + "\n")
    return results


def ablate(cfg: LoopConfig, variant_env: dict[str, str], label: str = "", baseline_paths: list[Path] | None = None,
           memory: Path | None = None) -> Comparison:
    """A/B one or more AGENT_* switches on the same library (an explicit copy, so a stale image cannot interfere)."""
    label = label or "-".join(f"{k}={v}" for k, v in sorted(variant_env.items())).replace("/", "_")
    adir = Path(cfg.work) / "ablations" / label
    lib = _copy_library(Path(memory or cfg.target_memory), adir / "memory")
    base = expand_runs(baseline_paths) if baseline_paths else evaluate(cfg, adir / "baseline", lib, label=f"{label}-baseline")
    var = evaluate(cfg, adir / "variant", lib, label=f"{label}-variant", extra_env=variant_env)
    cmp = compare(base, var, min_gain=cfg.min_gain, label_a="baseline", label_b=label)
    write_text(adir / "compare.md", render(cmp))
    write_json(adir / "ablation.json", {"variant_env": variant_env, "delta": cmp.delta, "p_value": cmp.p_value,
                                        "keep": cmp.keep, "gained": cmp.gained, "lost": cmp.lost})
    return cmp


def config_dict(cfg: LoopConfig) -> dict:
    d = {k: v for k, v in vars(cfg).items() if k != "teacher"}
    d["kit"] = str(cfg.kit.root)
    d["teacher"] = bool(cfg.teacher)
    return d
