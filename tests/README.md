# tests/ — plumbing tests with a mock House server (no real model needed)

These were used to verify the agent's control flow end to end without model access:

- `mock_model_server.py` — OpenAI-compatible mock of the House route (`/v1/chat/completions`, bearer
  check, 4000-token cap, 25-request cap). Answers come from `solutions/<unit-id>.py`. Modes via
  `MOCK_MODE`: `ok`, `buggy` (first script raises → repair), `truncate` (first script cut at the token
  limit → continuation), `nocode`, `slowscript` (infinite loop → exec timeout → repair), `nanjson`
  (NaN in JSON → validation → repair), `ctxlong` (HTTP 400 context-length → compact retry),
  `specfail` (v2: first script writes bools/ints as strings → the spec self-check must trigger a repair),
  `memnone` (v2: the plan explicitly selects no reference material). When the plan prompt contains the
  v2 "REFERENCE LIBRARY" the mock answers with `"memory": {"solutions": ["S1"], "playbooks": ["P1","P2"]}`
  and logs how much reference material reached the code prompt.
- `solve_unit.sh <unit> <solution.py>` — run a hand-written solution for one public unit, then its checker
  (used to verify every file in `agent/memory/solutions/`); serialised with a lock, shims `/app/...` paths and
  the unit Dockerfile's renamed COPY targets.
- `test_memory.py` — pytest unit tests for `agent/memory.py` (loading, hygiene, same-unit exclusion by id and
  by content, plan parsing, rendering budget) and `agent/selfcheck.py` (template/column/row parsing, hard vs
  soft findings). Run: `python -m pytest tests/test_memory.py -q`.
- `dev_run.sh <unit> [mode]` — Linux/root only: runs `agent/agent.py` directly against the mock, then the
  unit's own `checks/test_outputs.py` with the paths it hardcodes shimmed via symlinks.
- `mock_ctl.sh start <mode> | stop` — background mock for Docker-based runs
  (`MODEL_ENDPOINT=http://host.docker.internal:8765`, set `MOCK_UNIT=<unit-id>` because inside Docker the
  unit is mounted at `/input` and its id is not in the prompt).

With Docker, the same flow is `scripts/smoke_test.sh <unit>` with `MODEL_ENDPOINT` pointing at the mock.

devloop (the outer loop, `devloop/`):

- `mock_teacher.py` — deterministic stand-in for devloop's teacher model (in-process `respond()`, or an
  OpenAI-compatible server with `MOCK_TEACHER_PORT`). It answers the four devloop prompts (attribute, lesson,
  fold, solve); `MOCK_TEACHER_BAD=1` makes its first fold answer carry a GUID so the revision path runs.
- `test_devloop.py` — log parsers, failure rules, guards, the sign test, folding into a candidate library, the
  docker runner's commands, whole rounds with a fake evaluator. Opt-in, as root and with a kit:
  `DEVLOOP_E2E_KIT=<kit> python -m pytest tests/test_devloop.py -k e2e` runs a real round on five units with the
  host runner, the mock House and the mock teacher (about 80 s).
