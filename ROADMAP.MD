# Roadmap

## Project summary

Graphent: an extensible, production-grade agent framework for building, orchestrating, and deploying autonomous agents and agent-assisted applications. Graphent provides a lightweight runtime, planner, memory and tool integration primitives, SDKs (Python-first), observability, and safety controls so teams can iterate from prototypes to scalable production deployments.

Goals:
- Fast MVP for Python-first agent development.
- Clear separation of runtime, planner, and tools for extensibility.
- Opinionated defaults that are easy to override.
- Secure, observable, and testable agent behavior suitable for production.

Audience: researchers, ML engineers, backend engineers, and product teams building chatbots, automation workflows, or agentified services.

Scope (what's included):
- Core runtime for executing agents and plans.
- Planner interfaces (local & remote), memory stores, and tool adapters.
- SDK for building agents in Python and an HTTP API for polyglot integration.
- Testing harness, CI pipelines, and reference apps.

Out of scope for initial release:
- First-class support for languages other than Python (provided via HTTP API).
- Complex MLOps workflows (deferred to integrations).

---

## Core features (MVP → later)

MVP (alpha):
- Lightweight runtime that executes step-based plans with pluggable tool calls.
- Planner API and a simple built-in planner (heuristic / step sequencer).
- Memory primitives with an in-memory store and a simple Redis backend adapter.
- Tool adapters: HTTP tool, shell tool, and a stubbed LLM tool interface.
- Developer SDK (Python) with clear agent and tool abstractions.
- Local CLI to scaffold an agent, run a dev server, and replay traces.
- Basic tracing/observability (structured logs + trace IDs).
- Unit and integration tests for core pieces; CI-based checks.

Beta (post-MVP):
- External planner integrations (LangChain-style planner, search-based planners).
- Persistent memory backends (Postgres, vector DB adapters for retrieval-augmented memory).
- Tool sandboxing and permissioning; credential management patterns.
- Web UI for trace visualization and debugging.
- Rate-limiting, concurrency controls, and horizontal scaling patterns.
- SDK improvements (type hints, async-first APIs, helper patterns).

GA (polish & scale):
- Production-grade observability (metrics, metrics dashboards, OpenTelemetry support).
- Enterprise security features (RBAC, audit logs, policy engine for tool use).
- Templates and example applications (customer support agent, automation worker, data extraction agent).
- Official language SDKs (TypeScript/Node, Java) or maintained API specs.

---

## Architecture overview

High-level components:
- Agent Runtime: executes plans and handles step orchestration, retries, and error handling.
- Planner: component that converts goals or user intents into executable plans.
- Tools: pluggable interfaces representing external capabilities (LLMs, APIs, DBs, shell).
- Memory Store: retrieval & persistence layer for short/long-term memory.
- SDK / CLI / HTTP API: developer-facing tooling and integration points.
- Observability: logs, traces, metrics, and a simple trace viewer.

Design principles:
- Small core surface area with rich extension points.
- Explicit contracts between planner, runtime, and tools.
- Testability-first: provide simulation harnesses and deterministic replay.
- Security-by-default: least privilege for tools and clear credential boundaries.

---

## Development phases, milestones & acceptance criteria

Phase A — Alpha (6–8 weeks)
- Deliverables:
  - Core runtime with step execution and error handling.
  - Built-in simple planner and Python SDK.
  - In-memory memory store + Redis adapter.
  - CLI scaffold + basic examples (echo agent, todo agent).
  - Automated tests (unit + integration) with CI checks.
- Acceptance criteria:
  - All core components have unit tests and a reproducible dev flow.
  - Can run an example agent end-to-end locally and replay traces deterministically.

Phase B — Beta (10–14 weeks)
- Deliverables:
  - Pluggable planner integrations and example remote planner adapter.
  - Persistent memory adapters (Postgres + vector DB integration examples).
  - Tool permissioning model and credential injection pattern.
  - Basic web trace viewer and enhanced logs.
- Acceptance criteria:
  - Beta users can deploy agents with persistent memory and inspect traces in the UI.
  - Performance benchmarks for typical workflows; documented scaling guidance.

Phase C — GA (8–12 weeks)
- Deliverables:
  - OpenTelemetry metrics, metrics dashboard examples, RBAC patterns.
  - Production hardening: concurrency, rate-limiting, and backpressure.
  - SDK polish, documentation, and developer templates.
- Acceptance criteria:
  - Production load tests pass on defined SLOs.
  - Security review completed; docs and templates available for production onboarding.

---

## Suggested timeline (quarter-based, starting 2025-11-24)

Q4 2025 (Nov 24 — Dec 31): Alpha development
- Weeks 1–6: Runtime, planner, Python SDK, CLI scaffolds, in-memory memory store.
- Week 7–8: Tests, CI pipeline, and 1–2 example agents.

Q1 2026: Beta foundations
- Weeks 1–6: Memory adapters (Postgres, vector DB), planner integrations.
- Weeks 7–12: Tool permissioning, credential patterns, initial web trace viewer prototype.

Q2 2026: Beta stabilization & GA prep
- Weeks 1–6: Observability (OpenTelemetry), metrics, and dashboards.
- Weeks 7–12: Performance tuning, RBAC design, and docs/templates.

Q3 2026: GA and language SDK expansion
- Weeks 1–8: Final hardening, security audits, release candidate.
- Weeks 9–12: Documentation push, example apps, GA release.

---

## Testing strategy

- Unit tests: core runtime, planner, tool adapters, and memory contracts.
- Integration tests: run example agents end-to-end against test doubles (mock LLMs, sandboxed tools).
- E2E / simulation tests: deterministic replay harness that replays traces and validates outputs.
- Performance tests: synthetic workload generator for concurrency and latency profiling.
- Security tests: dependency scans, credential leakage checks, and policy fuzzing.

CI: run unit/integration on every PR; nightly runs for E2E/simulation and performance benchmarks.

---

## CI/CD & release plan

- GitHub Actions (or equivalent) with matrixed Python versions and linting.
- PR checks: lint, typecheck (optional), unit tests, integration tests with mocks.
- Nightly pipeline: full E2E simulation and performance benchmarks on a small cluster.
- Release cadence: Alpha/Beta snapshots every 2–4 weeks; GA release once acceptance criteria met.

---

## Documentation & examples

- Docs site (MkDocs or Sphinx) with quickstart, API reference, and cookbook.
- Example apps:
  - Chat assistant with memory and tool calls.
  - Automation worker that performs multi-step tasks.
  - Data extraction agent that writes to Postgres and a vector DB.
- Recipes: testing agents, sandboxing tools, memory tuning, and scaling guidance.

---

## Developer experience (DX)

- CLI: project scaffold, run local dev server, replay traces, run smoke tests.
- SDK: Python-first, async-friendly, typed where possible, small surface area.
- Templates & starter kits: minimal agent, agent+tool template, agent+DB template.
- Local developer mode: mock LLM and sandbox tools for rapid iteration.

---

## Security & governance

- Least-privilege tool execution and a credential manager integration pattern.
- Audit logging for tool invocations and decision points.
- Dependency scanning and a security checklist for contributions.
- Policy engine (deferred): allow admins to set policies for tool usage (GA+).

---

## Risk register & mitigations

- Risk: LLM unpredictability → Mitigation: deterministic planners for critical flows, guardrails & input validation.
- Risk: Secret leakage via tools → Mitigation: redact logs by default, credential vault integrations, review tools that can access secrets.
- Risk: Complexity explosion (many adapters) → Mitigation: focus on a few battle-tested adapters and provide clear extension guides.
- Risk: Slow adoption outside Python → Mitigation: well-documented HTTP API and concise SDK bindings.

---

## Metrics & success criteria

- Short-term (Alpha): ability to run 3 example agents; 80% unit test coverage for core modules.
- Mid-term (Beta): 100+ CI runs for E2E simulations; successful integration with at least two persistent memory backends.
- Long-term (GA): production users running agents with documented SLOs; error rate < 1% on synthetic workload; adoption metrics (stars, downloads, integrations).

---

## Next steps (first 30 days)

1. Finalize Python-first scope and confirm packaging (pyproject).  
2. Implement Alpha skeleton: runtime + simple planner + Python SDK.  
3. Create CLI scaffold and 2 minimal example agents.  
4. Set up CI with unit tests and PR checks.  
5. Draft docs quickstart and a minimal security checklist.


If you want, I can now:
- Commit this roadmap into `ROADMAP.MD` (done),
- Create initial issue templates / project board entries for Alpha tasks,
- Scaffold the Python package structure under `lib/` with minimal modules and tests.

Tell me which of those you'd like next.
