# Quant-Finance Coding Agent

An autonomous LLM agent that reads quantitative-finance programming tasks, then writes, runs and repairs
Python until its deliverables pass hidden tests. Built for **NeurIPS 2026 Agenthon, Track 1 (Coding)**.

> **Status:** the competition is still running (Final phase until 25 October 2026; results in November 2026).
> The source code will be published here after the results are announced. Until then this repository
> documents the problem, the design and the engineering.

## The problem

Each task gives the agent a natural-language specification and its data (CSV, Parquet, JSON, Excel, XML/XBRL
filings). The agent has to produce the deliverables (JSON, CSV, Parquet, HTML) with exact file names, schemas
and numbers. A hidden pytest checker grades every task, with a median of 24 assertions per task, and a task
counts only if every assertion passes.

The tasks span most of a quant desk:

- **Derivatives:** Black–Scholes Greeks, American, Asian, barrier and lookback options, jump diffusion, Heston,
  local volatility, variance swaps
- **Market risk:** VaR and ES, extreme-value theory, GARCH and DCC-GARCH, copulas, stress testing
- **Fixed income and FX:** curve bootstrapping, immunization, Hull–White swaptions, cross-currency basis swaps
- **Credit:** rating migration, CreditMetrics, portfolio credit VaR
- **Strategies:** backtesting, factor and alpha research, execution and transaction-cost analysis
- **Filings and text:** SEC 10-K (XBRL), 8-K, 13F and Form 4, and FOMC statements

Every team works under the same constraints:

- **Model:** one shared open model for everyone (NVIDIA Nemotron 3 Super, 120B-A12B, FP8).
- **Budget:** at most 25 model requests per task and 4,000 output tokens per request.
- **Sandbox:** offline. Non-root user, read-only root filesystem, 64 MiB `/tmp`, and no network except the
  model endpoint.
- **Time:** a time limit for each task, plus a shared allowance for the whole task set.

## How it works

```
 task spec + data
        │
        ▼
 contract extraction ──► plan ──► generate ──► execute ──► validate ──► done
 (files, schemas,          ▲        program      in a        (spec self-
  row counts, order)       │                     sandbox      checks)
                           │                                   │
 retrieval: method ────────┘        patch repair ◄─────────────┘
 playbooks + solved                (SEARCH/REPLACE)
 examples of OTHER tasks
                                   independent candidates ──► vote on key outputs
```

- **Output contract.** Parses each task's text for the deliverables it must produce: file names, JSON
  templates, CSV columns, row counts and sort orders. Every run is checked against this contract, and any
  violation becomes concrete repair feedback.
- **Budget-aware loop.** Plan, generate, execute, validate and repair, under soft and hard deadlines derived
  from each task's own time limit. The agent never crashes, and it always leaves its best outputs so far.
- **Working within 4,000 output tokens.** Repairs are SEARCH/REPLACE patches rather than full rewrites, and a
  truncated program is continued exactly where it stopped.
- **Independent solutions and voting.** Several candidate programs are generated, some concurrently, and the
  answer is chosen by agreement on their key outputs.
- **Retrieval-augmented memory.** BM25 retrieval over curated method playbooks (pricing, curves, risk,
  backtest conventions) and verified solutions to *other* tasks. Two leakage guards keep this fair:
  - **Same-task exclusion:** matching by task ID and by a 5-gram Jaccard fingerprint means a task never sees
    its own solution.
  - **Canary scrubbing:** canary GUIDs are removed from prompts and outputs.
- **Robust model client.**
  - **Budget accounting:** requests are counted correctly even when several run concurrently.
  - **Reasoning mode:** a low-effort reasoning setting is used for model calls.
  - **Retries without waste:** transient errors, proxy refusals and rate limits are retried without using up
    the request budget.

## Evaluation and engineering

- **Local evaluation harness.** Reproduces the platform: the same container limits, offline execution, and the
  official checkers. The model runs on vLLM (FP8) on 2× NVIDIA H200. Per-task failure analysis drove each
  round of ablations.
- **CI/CD on GitHub Actions.**
  - **Build and publish:** builds `linux/amd64` images, pushes them to the registry pinned by digest, and
    checks that they can be pulled anonymously.
  - **Conformance and packaging:** runs an offline conformance sweep over all public tasks and packages
    HMAC-signed submissions.
- **Tests.** 140 unit tests, plus a mock model server for end-to-end runs.

**Stack:** Python 3.13 (NumPy, pandas, SciPy, statsmodels), Docker, vLLM, GitHub Actions, GHCR.

## Timeline

| When | What |
|---|---|
| Aug – Oct 2026 | Development phase: iterative versions evaluated on the 86 public tasks |
| 13 – 25 Oct 2026 | Final phase on hidden tasks |
| Nov 2026 | Results; source code release in this repository |

## Links

- Agenthon 2026: https://www.agenthon.net/
- Track 1 public kit: https://github.com/Agenthon-2026/track1-coding-public
