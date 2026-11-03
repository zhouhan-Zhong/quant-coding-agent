"""python -m devloop <command> ...   (see devloop/README.md)

  eval       evaluate the agent on public units (eval/local_eval.sh, or the host runner)
  attribute  classify the failures of one or more runs; write attribution.md and mechanisms.md
  lessons    lesson cards for attributed failures (teacher sees the checker and a verified solution)
  fold       fold lesson cards into a candidate memory library through the guards
  verify     check a memory library (optionally re-run every solution against its checker)
  compare    A/B two groups of runs
  ablate     A/B one or more AGENT_* switches
  round      the whole loop: baseline -> attribute -> lessons -> fold -> verify -> A/B -> keep or discard
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .util import MEMORY_DIR, Kit, log, parse_env_pairs, read_text, write_json, write_text


def _units(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [u for u in value.replace(",", " ").split() if u]


def _teacher(args):
    from .llm import ChatClient, LLMConfig
    if getattr(args, "no_teacher", False):
        return None
    cfg = LLMConfig.from_env(args.teacher_endpoint, args.teacher_model, args.teacher_token)
    return ChatClient(cfg) if cfg else None


def _loop_config(args, teacher=None):
    from .loop import LoopConfig
    return LoopConfig(kit=Kit(args.kit), work=Path(args.work), runner=args.runner, image=args.image,
                      units=_units(args.units), repeats=args.repeats, jobs=args.jobs, env=parse_env_pairs(args.env),
                      model_endpoint=args.model_endpoint, model_name=args.model_name, model_token=args.model_token,
                      unit_timeout=args.unit_timeout, teacher=teacher, min_gain=args.min_gain)


def _add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--kit", required=True, help="clone of Agenthon-2026/track1-coding-public")
    p.add_argument("--work", default="devloop-work", help="work directory (default: ./devloop-work)")
    p.add_argument("--runner", choices=("docker", "host"), default="docker")
    p.add_argument("--image", default="", help="agent image for the docker runner")
    p.add_argument("--units", help="unit ids (space or comma separated); default: all")
    p.add_argument("--repeats", type=int, default=1, help="runs per arm (local noise is about +-2-3 units)")
    p.add_argument("--jobs", type=int, default=3, help="parallel units (docker runner)")
    p.add_argument("--env", action="append", default=[], help="AGENT_*=value for the agent (repeatable)")
    p.add_argument("--model-endpoint", default="http://127.0.0.1:8000", help="House model origin for the agent")
    p.add_argument("--model-name", default="house")
    p.add_argument("--model-token", default="local")
    p.add_argument("--unit-timeout", type=int, default=1800)
    p.add_argument("--min-gain", type=float, default=2.0, help="units an arm must gain to be kept")


def _add_teacher_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--teacher-endpoint", help="OpenAI-compatible endpoint (default: $DEVLOOP_ENDPOINT)")
    p.add_argument("--teacher-model", help="default: $DEVLOOP_MODEL or house")
    p.add_argument("--teacher-token", help="default: $DEVLOOP_TOKEN")


def cmd_eval(args) -> int:
    from .loop import evaluate
    cfg = _loop_config(args)
    label = args.label or time.strftime("eval-%Y%m%d-%H%M%S")
    env_extra = {} if args.pb_exclude is None else {"AGENT_PB_EXCLUDE": args.pb_exclude}
    runs = evaluate(cfg, Path(args.work) / "runs" / label, Path(args.memory).resolve() if args.memory else None,
                    label=label, extra_env=env_extra)
    print(f"{label}: " + ", ".join(f"{r.path.name} {r.n_passed}/{len(r.rows)}" for r in runs))
    return 0


def cmd_attribute(args) -> int:
    from .attribution import attribute, write_reports
    from .runs import expand_runs
    runs = expand_runs(args.runs)
    kit = Kit(args.kit) if args.kit else None
    recs = attribute(runs, kit, _teacher(args))
    out = Path(args.out) if args.out else Path(args.runs[0]) / "attribution"
    write_reports(recs, runs, out, min_units=args.min_units)
    print(read_text(out / "attribution.md"))
    print(f"written: {out}/attribution.md, attribution.jsonl, mechanisms.md")
    return 0


def cmd_lessons(args) -> int:
    from .attribution import load_attributions
    from .lessons import write_lessons
    from .runs import expand_runs, failures
    teacher = _teacher(args)
    if teacher is None:
        sys.exit("lessons need a teacher: set DEVLOOP_ENDPOINT or --teacher-endpoint")
    recs = load_attributions(Path(args.attribution))
    runs = expand_runs(args.runs)
    cards, sols = write_lessons(recs, failures(runs), Kit(args.kit), teacher, Path(args.memory), Path(args.out),
                                runner=args.runner, image=args.image, max_units=args.max_lessons,
                                solve_attempts=args.solve_attempts)
    for unit, code in sols.items():
        write_text(Path(args.out) / "solutions" / f"{unit}.py", code)
    print(f"{len(cards)} lesson card(s) and {len(sols)} new solution(s) in {args.out}")
    return 0


def cmd_fold(args) -> int:
    from .fold import build_candidate, fold
    teacher = _teacher(args)
    if teacher is None:
        sys.exit("fold needs a teacher: set DEVLOOP_ENDPOINT or --teacher-endpoint")
    lessons_dir = Path(args.lessons)
    lessons = {p.stem: read_text(p) for p in sorted(lessons_dir.glob("t1-*.md"))}
    sols = {p.stem: read_text(p) for p in sorted((lessons_dir / "solutions").glob("t1-*.py"))}
    kit = Kit(args.kit) if args.kit else None
    out = Path(args.out)
    accepted, rejected = fold(lessons, Path(args.memory), teacher, kit,
                              runner=None if args.no_snippets else args.runner, image=args.image,
                              log_dir=out.with_name(out.name + ".fold"))
    cand = build_candidate(Path(args.memory), out, accepted, sols, kit)
    write_json(out.with_name(out.name + ".proposals.json"), {
        "accepted": [{"path": e.path, "action": e.action, "sources": e.sources} for e in accepted],
        "rejected": [{"path": e.path, "sources": e.sources, "findings": [str(f) for f in e.findings]} for e in rejected]})
    print(f"candidate library: {cand} ({len(accepted)} accepted, {len(rejected)} rejected)")
    return 0


def cmd_verify(args) -> int:
    from .verify import verify_memory
    rep = verify_memory(Path(args.memory), base_memory=Path(args.base) if args.base else None,
                        kit=Kit(args.kit) if args.kit else None, solutions=args.solutions, selfcheck=args.selfcheck,
                        units=_units(args.units), runner=args.runner, image=args.image)
    print(json.dumps({k: v for k, v in rep.items() if k != "solutions"}, indent=1))
    if "solutions" in rep:
        ok = sum(1 for v in rep["solutions"].values() if v["passed"])
        print(f"solutions: {ok}/{len(rep['solutions'])} pass their checker")
    return 0 if rep["ok"] else 1


def cmd_compare(args) -> int:
    from .compare import compare, render
    from .runs import expand_runs
    c = compare(expand_runs(args.a), expand_runs(args.vs), min_gain=args.min_gain, sources=_units(args.sources) or [],
                label_a=args.label_a, label_b=args.label_b)
    print(render(c))
    return 0


def cmd_ablate(args) -> int:
    from .compare import render
    from .loop import ablate
    variant = parse_env_pairs(args.variant)
    if not variant:
        sys.exit("ablate needs at least one --variant AGENT_X=value")
    cmp = ablate(_loop_config(args), variant, label=args.label, baseline_paths=args.baseline,
                 memory=Path(args.memory) if args.memory else None)
    print(render(cmp))
    return 0


def cmd_round(args) -> int:
    from .loop import run_loop
    teacher = _teacher(args)
    cfg = _loop_config(args, teacher)
    cfg.max_lessons, cfg.solve_attempts, cfg.apply = args.max_lessons, args.solve_attempts, args.apply
    cfg.test_snippets, cfg.verify_solutions = not args.no_snippets, args.verify_solutions
    if teacher is None:
        log("no teacher configured: the round stops after the attribution (rules only)")
    results = run_loop(cfg, rounds=args.rounds, baseline_paths=args.baseline,
                       start_memory=Path(args.memory) if args.memory else None)
    print(read_text(Path(args.work) / "summary.md"))
    return 0 if results else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m devloop", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("eval", help="evaluate the agent")
    _add_eval_args(p)
    p.add_argument("--label")
    p.add_argument("--memory", help="memory library to mount (default: the image's own)")
    p.add_argument("--pb-exclude", choices=("0", "1"), help="AGENT_PB_EXCLUDE for this run (rounds always use 1)")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("attribute", help="classify failures")
    p.add_argument("runs", nargs="+", help="run directories or groups of repeats")
    p.add_argument("--kit", help="task kit (adds the task text to the teacher's evidence)")
    p.add_argument("--out")
    p.add_argument("--min-units", type=int, default=3, help="smallest recurring class reported as a mechanism")
    p.add_argument("--no-teacher", action="store_true", help="rules only")
    _add_teacher_args(p)
    p.set_defaults(fn=cmd_attribute)

    p = sub.add_parser("lessons", help="write lesson cards")
    p.add_argument("attribution", help="attribution.jsonl")
    p.add_argument("--runs", nargs="+", required=True, help="the runs that were attributed")
    p.add_argument("--kit", required=True)
    p.add_argument("--memory", default=str(MEMORY_DIR), help="library with the verified solutions")
    p.add_argument("--out", required=True)
    p.add_argument("--runner", choices=("docker", "host"), default="docker")
    p.add_argument("--image", default="")
    p.add_argument("--max-lessons", type=int, default=12)
    p.add_argument("--solve-attempts", type=int, default=6, help="checker-in-the-loop attempts for units without a solution")
    _add_teacher_args(p)
    p.set_defaults(fn=cmd_lessons)

    p = sub.add_parser("fold", help="fold lesson cards into a candidate library")
    p.add_argument("lessons", help="directory of lesson cards (t1-*.md, optional solutions/)")
    p.add_argument("--memory", default=str(MEMORY_DIR), help="base library")
    p.add_argument("--out", required=True, help="candidate library directory")
    p.add_argument("--kit", help="task kit (leakage checks against the source units)")
    p.add_argument("--runner", choices=("docker", "host"), default="docker", help="where new snippets run")
    p.add_argument("--image", default="")
    p.add_argument("--no-snippets", action="store_true", help="do not execute new snippets")
    _add_teacher_args(p)
    p.set_defaults(fn=cmd_fold)

    p = sub.add_parser("verify", help="check a memory library")
    p.add_argument("memory", nargs="?", default=str(MEMORY_DIR))
    p.add_argument("--base", help="base library (nothing may be lost relative to it)")
    p.add_argument("--kit")
    p.add_argument("--solutions", action="store_true", help="re-run every solution against its checker")
    p.add_argument("--selfcheck", action="store_true", help="calibrate the spec self-check on the passing outputs")
    p.add_argument("--units")
    p.add_argument("--runner", choices=("docker", "host"), default="docker")
    p.add_argument("--image", default="")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("compare", help="A/B two groups of runs")
    p.add_argument("a", nargs="+", help="arm A: runs or groups")
    p.add_argument("--vs", nargs="+", required=True, help="arm B: runs or groups")
    p.add_argument("--min-gain", type=float, default=2.0)
    p.add_argument("--sources", help="units the change was derived from (reports the held-out change)")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("ablate", help="A/B AGENT_* switches")
    _add_eval_args(p)
    p.add_argument("--variant", action="append", default=[], help="AGENT_X=value for the variant arm (repeatable)")
    p.add_argument("--label")
    p.add_argument("--baseline", nargs="+", help="reuse these baseline runs")
    p.add_argument("--memory", help="library for both arms (default: agent/memory)")
    p.set_defaults(fn=cmd_ablate)

    p = sub.add_parser("round", help="run development rounds")
    _add_eval_args(p)
    _add_teacher_args(p)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--baseline", nargs="+", help="reuse these runs as the first baseline (same library!)")
    p.add_argument("--memory", help="starting library (default: agent/memory)")
    p.add_argument("--max-lessons", type=int, default=12)
    p.add_argument("--solve-attempts", type=int, default=6)
    p.add_argument("--no-snippets", action="store_true")
    p.add_argument("--verify-solutions", action="store_true", help="re-run every library solution in the verify step")
    p.add_argument("--apply", action="store_true", help="copy a kept candidate into agent/memory")
    p.set_defaults(fn=cmd_round)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
