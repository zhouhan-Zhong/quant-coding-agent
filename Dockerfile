# Agenthon 2026 - Track 1 coding agent image.
#
# Build the shared sandbox base ONCE from the public kit, then this thin image on top of it:
#   docker build -t finance-bench-sandbox:latest -f <kit>/docker/sandbox.Dockerfile <kit>
#   docker build --platform linux/amd64 -t agenthon-t1-agent:dev .
#
# The base already carries numpy/pandas/scipy/statsmodels/arch/polars/pyarrow/matplotlib/plotly/
# openpyxl/requests/pytest ...; only genuine extras are added here (pinned).
FROM finance-bench-sandbox:latest

# numba: the exemplar unit pins it and the graded stack uses 0.67.0; lxml/bs4: HTML/XML filings.
RUN pip install --no-cache-dir numba==0.67.0 lxml==5.3.0 beautifulsoup4==4.12.3

# The platform runs the container as uid 65534 with a read-only root filesystem and a 64 MiB
# noexec tmpfs at /tmp.  Everything that wants a writable home/cache is pointed at /tmp.
# Final configuration (agent 2.9.16, 2026-10-27, ablation rounds 4-18): low-effort thinking, SEARCH/REPLACE repairs,
# independent-solution vote (up to 8 candidates, 2 requested concurrently, 429 -> one at a time), 480-s soft budget per
# unit (scaled for longer cards; hard stop +120 s; ~8 h for 86 units), textbook playbooks v3/v4 + trigger-pinned notes,
# referential-integrity check of identifier columns, debug-the-script tasks fixed by edits of the shipped script.
# AGENT_PB_EXCLUDE=0: the Final phase grades new units, so the whole memory library is available (same-unit solutions
# stay excluded by id and by content, rule 8).  For a Development-board build set AGENT_PB_EXCLUDE=1.
ENV HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl \
    NUMBA_CACHE_DIR=/tmp/numba \
    XDG_CACHE_HOME=/tmp/xdg \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    AGENT_WORKDIR=/tmp/agent_work \
    AGENT_TIME_BUDGET=480 \
    AGENT_MAX_REQUESTS=25 \
    AGENT_MAX_ATTEMPTS=12 \
    AGENT_PLAN=1 \
    AGENT_REVIEW=1 \
    AGENT_THINKING=0 \
    AGENT_LOW_EFFORT=1 \
    AGENT_PATCH_REPAIR=1 \
    AGENT_PB_EXCLUDE=0 \
    AGENT_VOTE=1 \
    AGENT_VOTE_MAX=8 \
    AGENT_VOTE_PARALLEL=2 \
    AGENT_VOTE_MAX_MB=8 \
    AGENT_MEMORY=1 \
    AGENT_MEMORY_CHARS=22000 \
    AGENT_MEMORY_V3=1 \
    AGENT_MEMORY_V4=1 \
    AGENT_MEM_TRIGGERS=1 \
    AGENT_KEY_CHECK=1 \
    AGENT_TEMPLATE_ROUTE=1 \
    AGENT_TEXT_ROUTE=0 \
    AGENT_TIME_TYPED=0 \
    AGENT_VERIFY=0 \
    AGENT_STAGED=0 \
    AGENT_SELFCHECK=1

# agent/ includes the v2 memory library (agent/memory: verified solutions to public units, playbooks, notes,
# instruction fingerprints for same-unit exclusion).  Build-time guard: no canary GUID may travel in the image.
COPY agent/ /agent/
COPY LICENSE /agent/LICENSE
RUN chmod -R a+rX /agent \
    && python -m py_compile /agent/agent.py /agent/memory.py /agent/selfcheck.py \
    && rm -rf /agent/__pycache__ \
    && ! grep -rIlE "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}" /agent/memory \
    && python -c "import sys; sys.path.insert(0,'/agent'); from memory import MemoryIndex; i=MemoryIndex('/agent/memory'); assert len(i.docs) > 50, len(i.docs); print('memory docs:', len(i.docs))"

LABEL qfbench2.interface_version="2.0" \
      org.opencontainers.image.title="agenthon-t1-agent" \
      org.opencontainers.image.licenses="Apache-2.0"

# The verb arrives as the first argument after the image reference (SUBMISSION_CLI.md).
ENTRYPOINT ["python", "/agent/agent.py"]
CMD ["solve", "--task-dir", "/input", "--out", "/app/output"]
