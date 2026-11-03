#!/usr/bin/env bash
# Local Track 1 evaluation: run an agent IMAGE on the public units against a self-hosted House model,
# then grade every unit with its own official checker.  Needs Docker and a running model server
# (eval/serve_vllm.sh).  Produces <results>/results.csv and per-unit logs.
#
#   bash eval/local_eval.sh <image> <kit-dir> [results-dir]
#
#   env  MODEL_ENDPOINT  model server origin (default http://127.0.0.1:8000; the agent appends /v1/chat/completions)
#        MODEL_NAME      served model name (default house)
#        JOBS            units run in parallel (default 3)
#        UNITS           space-separated unit ids to run instead of all of them
#        AGENT_ENV       extra "-e KEY=VALUE" flags for the agent container (A/B switches, e.g. "-e AGENT_MEMORY=0")
#        UNIT_TIMEOUT    wall-clock cap per unit in seconds (default 1800)
#        RUNTIME         docker (default when docker exists) or apptainer (HPC nodes without docker: IMAGE is then
#                        a .sif converted from the same image; see "apptainer" below)
#        RESUME          1 = keep <results-dir>/results.csv and skip the units already graded in it (continue a run
#                        that a batch job's walltime cut short)
#        DEADLINE        epoch seconds after which no new unit is started (units already running finish);
#                        creating <results-dir>/STOP has the same effect at any time
#
# Fidelity to the platform: the unit is copied WITHOUT checks/ (the platform strips it; with it the agent would
# also read the checker source), the container runs as uid 65534 with a read-only root, a 64 MiB noexec /tmp and
# the same pid/fd limits and a 16-CPU quota (CPUS), and both /output and /app/output are bound to one directory.
# (No per-uid nproc ulimit here: parallel containers all run as uid 65534 and would share that count.)  Grading runs as root in
# the same image (the organizers' sandbox stack) with the path shims the checkers expect.  Pass = pytest exit 0.
#
# apptainer (unprivileged, no setuid): the agent runs as the invoking user instead of uid 65534 (no other uid can be
# mapped), on the read-only SIF root with --containall (fresh /tmp + $HOME on the 64 MiB session tmpfs, clean env, own
# PID/IPC namespaces), no host bind paths, host network (as --network host), nofile 1024, and --cpus/--pids-limit when
# the node's cgroups allow them (probed below).  Not reproduced: uid 65534, noexec on /tmp.  Grading runs with
# --fakeroot --writable-tmpfs (root in a user namespace over a tmpfs overlay) so the shims can be written.
set -uo pipefail
IMAGE="${1:?usage: local_eval.sh <image> <kit-dir> [results-dir]}"
KIT="$(cd "${2:?kit dir (a clone of Agenthon-2026/track1-coding-public)}" && pwd)"
RES="${3:-results-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RES/logs" "$RES/out" "$RES/in"
RES="$(cd "$RES" && pwd)"
export IMAGE KIT RES
export MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:8000}"
export MODEL_NAME="${MODEL_NAME:-house}"
export AGENT_ENV="${AGENT_ENV:-}"
export UNIT_TIMEOUT="${UNIT_TIMEOUT:-1800}"
JOBS="${JOBS:-3}"

code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$MODEL_ENDPOINT/v1/models")"
if [ "$code" = "000" ]; then
  echo "model server not reachable at $MODEL_ENDPOINT (start eval/serve_vllm.sh first)" >&2
  exit 2
fi
NPROC="$(nproc)"; export CPUS="${CPUS:-$(( NPROC < 16 ? NPROC : 16 ))}"   # platform quota is 16 CPUs
export RUNTIME="${RUNTIME:-$(command -v docker >/dev/null 2>&1 && echo docker || echo apptainer)}"
if [ "$RUNTIME" = docker ]; then
  docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE" || { echo "cannot pull $IMAGE" >&2; exit 2; }
else
  [ -f "$IMAGE" ] || { echo "RUNTIME=apptainer needs IMAGE to be a .sif file: $IMAGE" >&2; exit 2; }
  command -v apptainer >/dev/null 2>&1 || { echo "apptainer not on PATH (module load apptainer)" >&2; exit 2; }
  # Flags shared by both apptainer runs: contained, no host bind paths, GPUs hidden (the site forces --nv).
  export APPTAINER_BASE="--containall --no-mount bind-paths,cwd --env CUDA_VISIBLE_DEVICES= --env NVIDIA_VISIBLE_DEVICES=void"
  # cgroup limits need a delegated cgroup; keep them only if this node accepts them.
  export APPTAINER_LIMITS="--cpus $CPUS --pids-limit 256"
  # shellcheck disable=SC2086
  apptainer exec $APPTAINER_BASE $APPTAINER_LIMITS "$IMAGE" true >/dev/null 2>&1 \
    || { echo "note: apptainer --cpus/--pids-limit unavailable here; running without them" >&2; APPTAINER_LIMITS=""; }
  # AGENT_ENV is written for docker ("-e K=V"); apptainer spells it "--env K=V".
  AGENT_ENV="$(printf '%s' "$AGENT_ENV" | sed -E 's/(^|[[:space:]])-e[[:space:]]+/\1--env /g')"
fi

# Grading script, run as root inside the image with the unit at /unit (read-only) and the outputs at /out.
read -r -d '' GRADE <<'GRADE_EOF'
set -u
# /out, /app/output and /output are bind mounts of the outputs, /unit and /input of the unit (see docker run).
# The base image ships /app/data as an empty directory: rebuild it as a symlink tree so renamed inputs fit in.
rm -rf /app/data && mkdir -p /app/data /tests /logs/verifier
[ -d /unit/checks/reference_data ] && ln -sfn /unit/checks/reference_data /tests/reference_data
DATA=/unit/environment/data; [ -d "$DATA" ] || DATA=/unit/environment
cp -as "$DATA"/. /app/data/ 2>/dev/null
for f in "$DATA"/*; do b="$(basename "$f")"; [ -e "/app/$b" ] || ln -sfn "$f" "/app/$b"; done
python - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, "/agent")
try:
    import agent  # renamed inputs from the unit's environment/Dockerfile COPY lines
    for alias, real in agent.parse_dockerfile_aliases(Path("/unit")).items():
        p = Path(alias)
        if alias.startswith("/app/") and not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(real, p)
except Exception as exc:
    print("alias shims skipped:", exc)
PY
cd /unit/checks || exit 3
OUTPUT_DIR=/out PYTHONDONTWRITEBYTECODE=1 python -m pytest test_outputs.py -q --tb=short -p no:cacheprovider 2>&1 | tail -60
exit "${PIPESTATUS[0]}"
GRADE_EOF
export GRADE

run_one() {
  local unit="$1" id in out t0 t1 rc crc req files pass summary
  id="$(basename "$unit")"
  if { [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -ge "$DEADLINE" ]; } || [ -e "$RES/STOP" ]; then
    printf '%-46s not started (deadline/STOP)\n' "$id"; return 0
  fi
  in="$RES/in/$id"; out="$RES/out/$id"
  rm -rf "$in" "$out"; mkdir -p "$in" "$out"; chmod 777 "$out"
  (cd "$unit" && tar --exclude=./checks -cf - .) | (cd "$in" && tar -xf -)
  # Local selection studies only: AGENT_VOTE_DUMP in AGENT_ENV mounts a per-unit directory for the vote candidates.
  local dumpd="" dumpb=""
  case "$AGENT_ENV" in *AGENT_VOTE_DUMP=*)
    dumpd="${DUMP_ROOT:-/hpctmp/${USER:-$(id -un)}/agenthon/dump}/$(basename "$RES")/$id"
    rm -rf "$dumpd"; mkdir -p "$dumpd"; chmod 777 "$dumpd"
    if [ "$RUNTIME" = docker ]; then dumpb="-v $dumpd:/vote_dump"; else dumpb="--bind $dumpd:/vote_dump"; fi;;
  esac
  t0=$(date +%s)
  if [ "$RUNTIME" = docker ]; then
  # shellcheck disable=SC2086
  timeout "$UNIT_TIMEOUT" docker run --rm --network host \
    --read-only --user 65534:65534 --cap-drop ALL --security-opt no-new-privileges \
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m --pids-limit 256 --cpus "$CPUS" \
    --ulimit nofile=1024:1024 \
    -e MODEL_ENDPOINT="$MODEL_ENDPOINT" -e MODEL_NAME="$MODEL_NAME" -e MODEL_TOKEN="${MODEL_TOKEN:-local}" \
    -e QFBENCH_SEED=0 -e QFBENCH_NETWORK=restricted -e AGENT_IGNORE_CHECKS=1 $AGENT_ENV \
    -v "$in":/input:ro -v "$out":/output -v "$out":/app/output $dumpb \
    "$IMAGE" solve --task-dir /input --out /app/output >"$RES/logs/$id.agent.log" 2>&1
  rc=$?
  t1=$(date +%s)
  docker run --rm --network none --entrypoint bash \
    -v "$unit":/unit:ro -v "$unit":/input:ro \
    -v "$out":/out -v "$out":/app/output -v "$out":/output \
    "$IMAGE" -c "$GRADE" >"$RES/logs/$id.check.log" 2>&1
  crc=$?
  else
  # shellcheck disable=SC2086
  (ulimit -n 1024 && exec timeout "$UNIT_TIMEOUT" apptainer run $APPTAINER_BASE $APPTAINER_LIMITS --no-privs \
    --env MODEL_ENDPOINT="$MODEL_ENDPOINT" --env MODEL_NAME="$MODEL_NAME" --env MODEL_TOKEN="${MODEL_TOKEN:-local}" \
    --env QFBENCH_SEED=0 --env QFBENCH_NETWORK=restricted --env AGENT_IGNORE_CHECKS=1 $AGENT_ENV \
    --bind "$in":/input:ro --bind "$out":/output --bind "$out":/app/output $dumpb --pwd /app \
    "$IMAGE" solve --task-dir /input --out /app/output) >"$RES/logs/$id.agent.log" 2>&1
  rc=$?
  t1=$(date +%s)
  # shellcheck disable=SC2086
  apptainer exec $APPTAINER_BASE --fakeroot --writable-tmpfs --net --network none \
    --bind "$unit":/unit:ro --bind "$unit":/input:ro \
    --bind "$out":/out --bind "$out":/app/output --bind "$out":/output --pwd /app \
    "$IMAGE" bash -c "$GRADE" >"$RES/logs/$id.check.log" 2>&1
  crc=$?
  fi
  pass=0; [ "$crc" -eq 0 ] && pass=1
  req="$(grep -oE 'requests used [0-9]+' "$RES/logs/$id.agent.log" | tail -1 | grep -oE '[0-9]+$')"
  files="$(cd "$out" && ls -A | tr '\n' ' ')"
  summary="$(grep -E '[0-9]+ (passed|failed)|error' "$RES/logs/$id.check.log" | tail -1 | tr ',"' ';\047' | cut -c1-160)"
  printf '%s,%s,%s,%s,%s,"%s","%s"\n' "$id" "$pass" "$rc" "$((t1 - t0))" "${req:-}" "$summary" "$files" >>"$RES/results.csv"
  printf '%-46s %s  agent_exit=%s  %4ss  req=%s\n' "$id" "$([ $pass = 1 ] && echo PASS || echo fail)" "$rc" "$((t1 - t0))" "${req:-?}"
}
export -f run_one

if [ "${RESUME:-0}" != 1 ] || [ ! -s "$RES/results.csv" ]; then
  echo "unit,pass,agent_exit,seconds,requests,checker_summary,output_files" >"$RES/results.csv"
fi
if [ -n "${UNITS:-}" ]; then
  for u in $UNITS; do echo "$KIT/units/$u"; done
else
  ls -d "$KIT"/units/t1-*
fi | while read -r u; do
  # RESUME: a unit with a row in results.csv was graded by an earlier run.
  [ "${RESUME:-0}" = 1 ] && grep -q "^$(basename "$u")," "$RES/results.csv" && continue
  echo "$u"
done | xargs -P "$JOBS" -I{} bash -c 'run_one "$1"' _ {}

python3 - "$RES/results.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
n = len(rows); p = sum(r["pass"] == "1" for r in rows)
secs = [int(r["seconds"]) for r in rows if r["seconds"].isdigit()]
print(f"\npassed {p}/{n} = {p / max(n, 1):.4f}   mean {sum(secs) / max(len(secs), 1):.0f}s/unit   total {sum(secs) / 3600:.1f} unit-hours")
PY
echo "results: $RES/results.csv   logs: $RES/logs/"
