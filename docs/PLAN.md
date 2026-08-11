# Nexus Forge — Course of Action Plan

> Derived from a repo-wide SWOT review (2026-08-11). This document is the working
> roadmap: what to fix, in what order, and which Claude Code agent/skill to use
> for each workstream. It is meant to be re-run as a prompt list — each item is
> scoped so a single agent session can pick it up with no extra context.

## Context

The repo (`nexus-forge` / `textSummarizer`) is a single-maintainer, feature-rich
multimodal summarization platform (CLI + FastAPI + MCP server + HF Space) with
solid engineering hygiene (CI, linting, pre-commit, Docker, PyPI releases) but
three structural gaps:

1. **Correctness bugs in the flagship "grading loop" feature** — the thing that
   differentiates this project from a plain summarizer wrapper is currently
   buggy in ways that undercut its own pitch.
2. **No enforced quality bar** — tests exist (~1,374 lines) but coverage isn't
   gated, so regressions can land silently.
3. **A well-triaged but untouched backlog** — 33 open issues, all labeled with
   priority/difficulty/category, 0 open PRs. This is unusually ready for
   systematic burn-down, including handing "good first issue" items to outside
   contributors.

This plan sequences the backlog into batches with a clear "why now" for each,
and assigns the Claude Code agent type/skill best suited to execute it.

## Priority 1 — Fix correctness bugs in the grading loop

These sit in `src/textSummarizer/grading/` and directly contradict the
README's claims about the summarize → grade → refine loop.

| Issue | Problem | File(s) to start from |
|---|---|---|
| #37 | `SummarizationLoop` refinement doesn't use judge feedback | `src/textSummarizer/grading/loop.py` |
| #30 | Custom YAML rubric dimensions are never scored | `src/textSummarizer/grading/rubric_loader.py`, `rubric.py`, `config/rubrics/` |
| #33 | `/train` blocks the asyncio event loop | `src/textSummarizer/serving/app.py` |

**Agent:** `general-purpose` (each issue in its own session/branch) — these
need to read the existing loop/rubric/serving code, reproduce the bug via the
existing test suite (`tests/unit/test_grading.py`,
`tests/unit/test_geval_rubric.py`), fix it, and add a regression test. Use
`Explore` first inside that session if the bug's root cause isn't obvious from
the file alone, but a full `Plan`-agent detour is unnecessary — each bug is a
single-file, well-scoped fix.

**Verification:** `uv run pytest -m "not gpu and not slow and not network"`
plus a targeted run of the affected test file.

## Priority 2 — Enforce a quality bar

| Issue | Problem |
|---|---|
| #28 | Add coverage badge and 80% threshold |
| #26 | Add CLI smoke tests for `text-summarizer` |
| #27 | Add training pipeline stage tests |

**Agent:** `general-purpose`. Wire `--cov-fail-under=80` (or the agreed number)
into `pyproject.toml`'s `[tool.coverage]` / CI step, then add the missing
smoke/stage tests to actually clear that bar — don't just lower the bar to
fit existing coverage. This should land as one PR: threshold + tests, so CI
never goes red on the same commit that introduces the gate.

## Priority 3 — Supply-chain security

| Issue | Problem |
|---|---|
| #55 | PyPI publish: Sigstore provenance + SBOM (priority: high) |

**Agent:** `general-purpose`, but this one benefits from `WebSearch` /
`WebFetch` to confirm current Sigstore/`sigstore-python` and SBOM
(`cyclonedx-py` or `syft`) integration steps for GitHub Actions before editing
`.github/workflows/publish.yml`. Do this as its own PR — it's infrastructure,
not application code, and should be reviewable independently of Priority 1/2.

## Priority 4 — CI hygiene

- Investigate the stuck/queued workflow run (2026-08-06) and the intermittent
  red runs on feature branches (`fix/nexus-day2-security-docs`,
  `security/nexus-forge-46-image-audio-upload-validation`).
- Confirm whether these are flaky tests, cache issues, or infra — fix or file
  a tracked issue if the cause is external (e.g., GitHub Actions runner
  flakiness).

**Agent:** `general-purpose` with `mcp__github__actions_get` /
`get_job_logs` to pull the actual failure logs before guessing at a fix.

## Priority 5 — Backlog burn-down / contributor onboarding

The remaining ~25 open issues split cleanly:

- **Good first issues** (#25, #40, #41, #56, #60) — package naming docs,
  Docker multimodal deps, docker-compose GPU healthcheck parity, nbstripout
  pre-commit, demo script consolidation. These are genuinely suitable to hand
  to external contributors rather than have an agent do them — the point of
  Priority 5 is opening the repo up, not closing every issue solo.
- **Feature work** (#29 Redis rate limiting, #35 true SSE streaming, #36/#38
  expose loop/RAG params via API, #48 LangChain multimodal tools, #49
  observability) — larger, each deserves its own `Plan` agent pass before
  implementation once picked up.
- **Performance** (#34 cache SentenceTransformer, #42 evict stale rate-limiter
  state, #50 HF Space cold-start caching) — small, contained; `general-purpose`
  agent, one PR each.

**Recommended action:** don't batch-assign these to agents in this session.
Instead, tag the 5 good-first-issues clearly (they already are) and leave them
for external contributors; queue the feature/performance items one at a time
as separate sessions when picked up, each starting with a fresh `Plan` agent
pass for the larger items.

## Execution notes

- Each Priority 1–4 item should be its **own branch and PR** — keep the
  correctness fixes separable from the coverage-gate change and separable
  from the supply-chain change, so a revert or review doesn't take out
  unrelated work.
- Use `security-review` skill on anything touching `serving/`, `mcp/`, or file
  upload/sandbox code before merging (this repo has a real security track
  record worth protecting — see the sandboxing and upload-validation PRs in
  history).
- Re-run `code-review` skill (medium effort) on each PR before requesting
  merge, given there's currently no second human reviewer in the loop.
