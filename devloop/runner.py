"""Running things: the agent on units (an evaluation run), scripts on a unit's data, snippets, and the checkers.

Two runners share one contract (the run layout in runs.py):

* ``docker`` (default) delegates evaluations to eval/local_eval.sh, i.e. the platform's container limits.  Scripts
  and snippets run in the agent image with ``--network none``.  The memory library under test is mounted read-only
  and selected with AGENT_MEMORY_DIR, so both A/B arms use an explicit library and a stale image cannot leak in.
* ``host`` runs the same steps without containers, for throwaway sandboxes and CI machines without Docker.  It must
  run as root because the checkers hard-code absolute paths (/app/output, /app/data, /tests/reference_data, ...)
  which are re-pointed with symlinks for every unit (the same shims as tests/solve_unit.sh).  Never use it on a
  machine whose /app, /input, /output or /tests you care about.
"""
from __future__ import annotations

import contextlib
import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .runs import RESULTS_HEADER, Run
from .util import AGENT_DIR, REPO, Kit, dockerfile_env, git_commit, log, read_text, write_json

AGENT_PREFIXES = ("AGENT_", "MODEL_", "QFBENCH_", "OPENAI_")


@dataclass
class EvalSpec:
    kit: Kit
    runner: str = "docker"
    image: str = ""
    units: list[str] | None = None
    jobs: int = 3
    env: dict[str, str] = field(default_factory=dict)      # AGENT_* switches for this arm
    memory_dir: Path | None = None
    model_endpoint: str = "http://127.0.0.1:8000"
    model_name: str = "house"
    model_token: str = "local"
    unit_timeout: int = 1800
    python: str = sys.executable

    def describe(self) -> dict:
        return {"runner": self.runner, "image": self.image, "kit": str(self.kit.root), "units": self.units,
                "env": dict(sorted(self.env.items())), "memory_dir": str(self.memory_dir) if self.memory_dir else "",
                "model_endpoint": self.model_endpoint, "model_name": self.model_name}


# ----------------------------------------------------------------------------------------------------------------
# evaluation runs
# ----------------------------------------------------------------------------------------------------------------

def run_eval(spec: EvalSpec, out_dir: Path, label: str = "") -> Run:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"label": label or out_dir.name, **spec.describe(), "commit": git_commit(), "started": time.time()}
    write_json(out_dir / "run.json", meta)
    log(f"eval {meta['label']}: runner={spec.runner} units={len(spec.units) if spec.units else 'all'} "
        f"env={meta['env'] or '-'} memory={spec.memory_dir or 'built-in'}")
    if spec.runner == "docker":
        _eval_docker(spec, out_dir)
    elif spec.runner == "host":
        _eval_host(spec, out_dir)
    else:
        raise ValueError(f"unknown runner {spec.runner!r}")
    run = Run.load(out_dir)
    meta.update(finished=time.time(), passed=run.n_passed, total=len(run.rows))
    write_json(out_dir / "run.json", meta)
    run.meta = meta
    log(f"eval {meta['label']}: passed {run.n_passed}/{len(run.rows)}")
    return run


def docker_agent_env(env: dict[str, str], memory_dir: Path | None) -> str:
    """The AGENT_ENV string for eval/local_eval.sh: `-e K=V` switches plus the read-only memory mount."""
    flags = []
    for k, v in sorted(env.items()):
        if re.search(r"\s", f"{k}{v}"):
            raise ValueError(f"AGENT_ENV values cannot contain whitespace: {k}={v!r}")
        flags.append(f"-e {k}={v}")
    if memory_dir is not None:
        mem = str(Path(memory_dir).resolve())
        if re.search(r"[\s:]", mem):
            raise ValueError(f"memory directory path cannot contain whitespace or ':': {mem}")
        flags += ["-e AGENT_MEMORY_DIR=/devloop/memory", f"-v {mem}:/devloop/memory:ro"]
    return " ".join(flags)


def _make_readable(path: Path) -> None:
    for p in [path, *path.rglob("*")]:
        try:
            mode = p.stat().st_mode
            p.chmod(mode | 0o444 | (0o111 if p.is_dir() else 0))
        except OSError:
            pass


def _eval_docker(spec: EvalSpec, out_dir: Path) -> None:
    if not spec.image:
        raise ValueError("the docker runner needs --image (build one with: docker build -t agenthon-t1-agent:dev .)")
    if spec.memory_dir:
        _make_readable(Path(spec.memory_dir))      # the agent runs as uid 65534
    env = os.environ.copy()
    env.update(JOBS=str(spec.jobs), AGENT_ENV=docker_agent_env(spec.env, spec.memory_dir),
               MODEL_ENDPOINT=spec.model_endpoint, MODEL_NAME=spec.model_name, MODEL_TOKEN=spec.model_token,
               UNIT_TIMEOUT=str(spec.unit_timeout))
    env.pop("UNITS", None)
    if spec.units:
        env["UNITS"] = " ".join(spec.units)
    cmd = ["bash", str(REPO / "eval" / "local_eval.sh"), spec.image, str(spec.kit.root), str(out_dir)]
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0 or not (out_dir / "results.csv").is_file():
        raise RuntimeError(f"eval/local_eval.sh failed (exit {rc})")


def _host_env(spec: EvalSpec, workdir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(AGENT_PREFIXES)}
    env.update(dockerfile_env())                       # the image's configuration
    env.update(HOME=str(workdir), AGENT_WORKDIR=str(workdir / "agent_work"), MPLCONFIGDIR=str(workdir / "mpl"),
               NUMBA_CACHE_DIR=str(workdir / "numba"), XDG_CACHE_HOME=str(workdir / "xdg"),
               MODEL_ENDPOINT=spec.model_endpoint, MODEL_NAME=spec.model_name, MODEL_TOKEN=spec.model_token,
               QFBENCH_SEED="0", QFBENCH_NETWORK="restricted", AGENT_IGNORE_CHECKS="1")
    env.update(spec.env)
    if spec.memory_dir:
        env["AGENT_MEMORY_DIR"] = str(Path(spec.memory_dir).resolve())
    return env


def _require_root() -> None:
    if os.geteuid() != 0:
        raise PermissionError("the host runner must run as root (it re-points /app, /input, /output, /tests); "
                              "use --runner docker otherwise")


def _eval_host(spec: EvalSpec, out_dir: Path) -> None:
    _require_root()
    units = spec.units or spec.kit.units()
    results = out_dir / "results.csv"
    with open(results, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(RESULTS_HEADER)
    for d in ("logs", "out", "in"):
        (out_dir / d).mkdir(exist_ok=True)
    passed = 0
    for unit in units:
        row = _host_unit(spec, unit, out_dir)
        passed += row[1] == 1
        with open(results, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(row)
        print(f"{unit:<46} {'PASS' if row[1] else 'fail'}  agent_exit={row[2]}  {row[3]:4d}s  req={row[4] or '?'}",
              flush=True)
    print(f"\npassed {passed}/{len(units)} = {passed / max(len(units), 1):.4f}", flush=True)


def _host_unit(spec: EvalSpec, unit: str, res: Path) -> list:
    src = spec.kit.unit_dir(unit)
    inp, out = res / "in" / unit, res / "out" / unit
    for d in (inp, out):
        shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(src, inp, ignore=lambda d, names: ["checks"] if Path(d) == src else [], symlinks=True)
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="devloop-agent-") as tmp, shim_lock():
        install_shims(src, inp, out)
        t0 = time.time()
        rc = run_logged([spec.python, str(AGENT_DIR / "agent.py"), "solve", "--task-dir", str(inp), "--out", str(out)],
                        env=_host_env(spec, Path(tmp)), cwd=tmp, timeout=spec.unit_timeout,
                        log_path=res / "logs" / f"{unit}.agent.log")
        secs = int(time.time() - t0)
        crc, tail = _pytest_checker(src, out, spec.python)
    (res / "logs" / f"{unit}.check.log").write_text(tail, encoding="utf-8")
    agent_log = read_text(res / "logs" / f"{unit}.agent.log")
    req = re.findall(r"requests used (\d+)", agent_log)
    summary = ""
    for ln in tail.splitlines():
        if re.search(r"[0-9]+ (passed|failed)|error", ln):
            summary = ln
    summary = summary.replace(",", ";").replace('"', "'")[:160]
    files = " ".join(sorted(os.listdir(out)))
    return [unit, 1 if crc == 0 else 0, rc, secs, req[-1] if req else "", summary, files]


# ----------------------------------------------------------------------------------------------------------------
# host shims and processes
# ----------------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def shim_lock(path: str = "/tmp/devloop-shims.lock"):
    import fcntl
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _rm(p: Path) -> None:
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)


def install_shims(unit: Path, input_dir: Path, out: Path) -> None:
    """Point the absolute paths of the platform at this unit: /input, /app/output, /output, /app/data (symlink tree),
    /app/<input file>, the Dockerfile COPY aliases and /tests/reference_data.  Root only; hold shim_lock()."""
    _require_root()
    for p in ("/app/output", "/output", "/input", "/tests/reference_data", "/app/data"):
        _rm(Path(p))
    for d in ("/app", "/tests", "/logs/verifier"):
        Path(d).mkdir(parents=True, exist_ok=True)
    for p in Path("/app").iterdir():
        if p.is_symlink():
            p.unlink()                                   # stale /app/<file> shims of an earlier unit
    os.symlink(out, "/app/output")
    os.symlink(out, "/output")
    os.symlink(input_dir, "/input")
    ref = unit / "checks" / "reference_data"
    if ref.is_dir():
        os.symlink(ref, "/tests/reference_data")
    data = unit / "environment" / "data"
    if not data.is_dir():
        data = unit / "environment"
    Path("/app/data").mkdir()
    if data.is_dir():
        for f in sorted(data.rglob("*")):
            if f.is_file():
                dst = Path("/app/data") / f.relative_to(data)
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(f, dst)
        for f in sorted(data.iterdir()):
            dst = Path("/app") / f.name
            if not dst.exists() and not dst.is_symlink():
                os.symlink(f, dst)
    try:
        import agent  # noqa: PLC0415  (renamed inputs from the unit's environment/Dockerfile)
        for alias, real in agent.parse_dockerfile_aliases(unit).items():
            p = Path(alias)
            if alias.startswith("/app/") and not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(real, p)
    except Exception as exc:  # noqa: BLE001
        log(f"alias shims skipped: {exc}")


def run_logged(cmd: list[str], *, env: dict, cwd: str | Path, timeout: float, log_path: Path | None = None) -> int:
    """Run a command in its own process group; output to log_path; 124 on timeout (like GNU timeout)."""
    fh = open(log_path, "w", encoding="utf-8") if log_path else subprocess.DEVNULL
    try:
        proc = subprocess.Popen(cmd, env=env, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return 124
    finally:
        if log_path:
            fh.close()


def _tail(text: str, n: int) -> str:
    return "\n".join(text.splitlines()[-n:])


def _pytest_checker(unit: Path, out: Path, python: str) -> tuple[int, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(AGENT_PREFIXES)}
    env.update(OUTPUT_DIR=str(out), PYTHONDONTWRITEBYTECODE="1")
    try:
        proc = subprocess.run([python, "-m", "pytest", "test_outputs.py", "-q", "--tb=short", "-p", "no:cacheprovider"],
                              cwd=unit / "checks", env=env, capture_output=True, text=True, timeout=900)
        return proc.returncode, _tail(proc.stdout + proc.stderr, 60)
    except subprocess.TimeoutExpired:
        return 124, "checker timed out"


def grade_script() -> str:
    """The grading script of eval/local_eval.sh (one source of truth for the docker runner)."""
    text = read_text(REPO / "eval" / "local_eval.sh")
    m = re.search(r"read -r -d '' GRADE <<'GRADE_EOF'\n(.*?)\nGRADE_EOF", text, re.S)
    if not m:
        raise RuntimeError("could not find the GRADE script in eval/local_eval.sh")
    return m.group(1)


# ----------------------------------------------------------------------------------------------------------------
# scripts on a unit's data (teacher solutions) and snippets
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class ScriptResult:
    rc: int
    log: str
    checker_rc: int | None = None
    checker_log: str = ""
    out_dir: Path | None = None

    @property
    def passed(self) -> bool:
        return self.rc == 0 and self.checker_rc == 0


def run_solution(script: Path, kit: Kit, unit: str, *, runner: str, image: str = "", timeout: int = 600,
                 python: str = sys.executable, keep_dir: Path | None = None) -> ScriptResult:
    """Run a candidate solution on the unit's data (no checker visible), then grade it with the unit's checker."""
    src = kit.unit_dir(unit)
    work = Path(keep_dir or tempfile.mkdtemp(prefix="devloop-sol-")).resolve()
    out, inp = work / "out", work / "in"
    for d in (out, inp):
        shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(src, inp, ignore=lambda d, names: ["checks"] if Path(d) == src else [], symlinks=True)
    out.mkdir(parents=True)
    script = Path(script).resolve()
    if runner == "host":
        env = {k: v for k, v in os.environ.items() if not k.startswith(AGENT_PREFIXES)}
        env.update(DATA_DIR="/app/data", OUTPUT_DIR=str(out), MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
        with shim_lock():
            install_shims(src, inp, out)
            rc = run_logged([python, str(script)], env=env, cwd=out, timeout=timeout, log_path=work / "solution.log")
            install_shims(src, src, out)
            crc, ctail = _pytest_checker(src, out, python)
    elif runner == "docker":
        data = kit.data_dir(unit)
        out.chmod(0o777)                                  # the script runs as uid 65534, like the agent
        _make_readable(inp)
        _make_readable(script.parent)
        rc = run_logged(["docker", "run", "--rm", "--network", "none", "--read-only", "--user", "65534:65534",
                         "--tmpfs", "/tmp:rw,size=64m", "-e", "HOME=/tmp", "-e", "MPLBACKEND=Agg",
                         "-e", "DATA_DIR=/app/data", "-e", "OUTPUT_DIR=/app/output",
                         "-v", f"{data}:/app/data:ro", "-v", f"{inp}:/input:ro", "-v", f"{out}:/app/output",
                         "-v", f"{out}:/output", "-v", f"{script}:/devloop/solution.py:ro", "-w", "/app/output",
                         "--entrypoint", "python", image, "/devloop/solution.py"],
                        env=os.environ.copy(), cwd=work, timeout=timeout, log_path=work / "solution.log")
        proc = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "bash",
                               "-v", f"{src}:/unit:ro", "-v", f"{src}:/input:ro", "-v", f"{out}:/out",
                               "-v", f"{out}:/app/output", "-v", f"{out}:/output", image, "-c", grade_script()],
                              capture_output=True, text=True, timeout=900)
        crc, ctail = proc.returncode, _tail(proc.stdout + proc.stderr, 60)
    else:
        raise ValueError(f"unknown runner {runner!r}")
    return ScriptResult(rc=rc, log=_tail(read_text(work / "solution.log"), 40), checker_rc=crc, checker_log=ctail,
                        out_dir=out)


def run_snippet(code: str, *, runner: str, image: str = "", timeout: int = 120, python: str = sys.executable) -> ScriptResult:
    """Execute a playbook's tested snippet on its own (no data, no network)."""
    with tempfile.TemporaryDirectory(prefix="devloop-snippet-") as tmp:
        path = Path(tmp) / "snippet.py"
        path.write_text(code, encoding="utf-8")
        _make_readable(Path(tmp))
        if runner == "docker":
            cmd = ["docker", "run", "--rm", "--network", "none", "--read-only", "--user", "65534:65534",
                   "--tmpfs", "/tmp:rw,size=64m", "-e", "HOME=/tmp", "-e", "MPLBACKEND=Agg",
                   "-v", f"{path}:/devloop/snippet.py:ro", "--entrypoint", "python", image, "/devloop/snippet.py"]
            env = os.environ.copy()
        else:
            cmd = [python, "-I", str(path)]
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmp, "MPLBACKEND": "Agg",
                   "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        rc = run_logged(cmd, env=env, cwd=tmp, timeout=timeout, log_path=Path(tmp) / "out.log")
        return ScriptResult(rc=rc, log=_tail(read_text(Path(tmp) / "out.log"), 25))
