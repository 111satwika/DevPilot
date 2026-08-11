# DevPilot AI - Local AI Engineering Copilot (MCP)

## Vision

DevPilot AI is a local AI Software Engineering Copilot that understands
your development workspace and uses **Model Context Protocol (MCP)**
servers to analyze projects, execute tools, debug applications, generate
documentation, optimize code, and assist developers through intelligent
multi-tool orchestration.

------------------------------------------------------------------------

# Goals

-   Learn MCP deeply through a practical project.
-   Build a production-style local engineering assistant.
-   Understand tool calling and orchestration before moving to
    multi-agent systems.

------------------------------------------------------------------------

# Core Functionalities

## Repository Intelligence

-   Understand folder structure
-   Detect framework (React, FastAPI, Spring, Node, etc.)
-   Generate project summary
-   Dependency graph
-   Architecture overview

## Code Understanding

-   Explain files/functions/classes
-   Trace API flow
-   Find entry points
-   Find dead code
-   Search symbols

## Debugging Assistant

-   Run project
-   Capture stack traces
-   Read logs
-   Search official docs
-   Suggest fixes
-   Apply fixes after approval

## Terminal Assistant

-   Run build commands
-   Execute tests
-   Install packages
-   Check environment
-   Git operations

## Documentation Generator

-   README
-   API documentation
-   Architecture docs
-   Sequence diagrams
-   Flow diagrams

## Database Assistant

-   Execute SQL
-   Explain schema
-   Optimize queries
-   Generate ER summary

## Docker Assistant

-   Build images
-   Run containers
-   View logs
-   Inspect containers

## Workspace Search

-   Semantic search
-   Keyword search
-   Find configuration
-   Find endpoints
-   Find environment variables

------------------------------------------------------------------------

# High-Level Architecture

``` text
                User
                  │
                  ▼
          LLM + MCP Client
                  │
          Intent Understanding
                  │
          Tool Selection Layer
                  │
 ┌────────────────┼─────────────────────┐
 │                │                     │
Filesystem     Terminal             GitHub
 │                │                     │
 ├───────┐        │                     │
 ▼       ▼        ▼                     ▼
Browser Database Docker             Git
 │
 ▼
Aggregated Context
 │
 ▼
Final Response
```

------------------------------------------------------------------------

# MCP Servers

  MCP Server   Purpose
  ------------ -------------------------------
  Filesystem   Read/write/search files
  Terminal     Execute commands
  GitHub       Repository analysis
  Browser      Official documentation search
  Database     SQL execution
  Docker       Containers
  Git          Version control

------------------------------------------------------------------------

# MCP Concepts Learned

-   MCP Client
-   MCP Server
-   Tool Registration
-   Tool Discovery
-   Tool Invocation
-   Context Sharing
-   Tool Chaining
-   Multi-server Communication
-   Error Handling
-   Tool Permissions

------------------------------------------------------------------------

# AI Concepts Used

-   Function Calling
-   Prompt Engineering
-   Context Engineering
-   Tool Selection
-   Sequential Tool Orchestration
-   Chain-of-Thought Planning (internal)
-   Retrieval from Local Files
-   Code Analysis
-   Local Search
-   Explainability

------------------------------------------------------------------------

# Example Workflows

## Debug FastAPI

1.  Read project
2.  Detect framework
3.  Run application
4.  Capture traceback
5.  Search documentation
6.  Explain root cause
7.  Suggest patch
8.  Apply after approval
9.  Re-run

## Repository Explanation

1.  Read folder tree
2.  Parse dependencies
3.  Find entry point
4.  Build architecture summary
5.  Generate README

------------------------------------------------------------------------

# Suggested Folder Structure

``` text
devpilot-ai/
│
├── frontend/
├── backend/
├── mcp_client/
├── mcp_servers/
│   ├── filesystem/
│   ├── terminal/
│   ├── github/
│   ├── browser/
│   ├── database/
│   ├── docker/
│   └── git/
├── prompts/
├── services/
├── config/
├── tests/
├── infra/
│   ├── docker-compose.yml
│   └── ci/
├── logs/
└── docs/
```

------------------------------------------------------------------------

# How to Build

Each phase is scoped so the MCP concepts you're trying to learn
([MCP Concepts Learned](#mcp-concepts-learned)) map to something you're
actually forced to build, not read about. Production concerns are
introduced at the phase where they first become necessary — sandboxing
is the one exception, pulled into Phase 1, because Filesystem/Terminal
MCP servers are unsafe by default from the moment they exist.

## Phase 1 — MCP Foundations

**Concepts exercised:** MCP Client, MCP Server, Tool Registration, Tool
Discovery, Tool Invocation, Tool Permissions (basic)

-   MCP Client — connects to servers, discovers tools, invokes them
-   Filesystem MCP — **with a workspace-root boundary and path-traversal
    rejection from the first commit**, not added later
-   Terminal MCP — **with command allow-listing from the first commit**
    (npm, pytest, git only; no raw shell strings)
-   Typed config (pydantic-settings) for workspace root, allow-list, and
    environment — one place, not scattered env reads

## Phase 2 — Multi-Server Orchestration

**Concepts exercised:** Multi-server Communication, Context Sharing,
Error Handling, Tool Permissions (read-only data access)

-   GitHub MCP
-   Browser MCP
-   Database MCP — **read-only by default**; write/DDL is a separate,
    explicitly-approved mode
-   Aggregate context from three servers into a single response — this
    is where "Context Sharing" stops being a concept and becomes code
-   Cache expensive reads (dependency graph, repo scan) with mtime/hash
    invalidation
-   Structured logging (request/session IDs) — introduce it here, not
    later; debugging multi-server calls without it is painful enough
    that you'll want it before Phase 3, not after

## Phase 3 — Scaling the Orchestration

**Concepts exercised:** Tool Chaining, Multi-server Communication
(concurrent), Error Handling (graceful degradation)

-   Docker MCP — route through a background job queue (**Arq**, backed
    by **Redis**) since builds/runs are long-lived; this is also the
    server that most needs the sandboxing from Phase 1, since it
    effectively grants host-level power
-   Workspace search (semantic + keyword)
-   Documentation generation — your first real **Tool Chaining**
    workflow (read repo → parse deps → summarize → generate docs)
-   Health checks per MCP server + graceful degradation when one is
    down (e.g., Docker daemon unreachable shouldn't kill the session)

## Phase 4 — Intelligence + Production Hardening

**Concepts exercised:** Tool Permissions (full approval workflow),
Error Handling (timeouts/retries), all prior concepts under real
orchestration load

-   Intelligent tool selection — with a **max-iteration cap** to
    prevent runaway tool-call loops (a common MCP failure mode)
-   Tool chaining refinement across all seven servers
-   Approval workflow — extend the "apply fixes" gate to every
    destructive action across every server (writes, git push, DB
    writes, container stop/rm)
-   Session memory — Redis-backed, not in-process, so any MCP server
    can restart without losing context
-   Audit log of every tool invocation (args + result, append-only)
-   Contract tests per server + adversarial/security tests (path
    traversal, SQL/shell injection attempts), wired into CI

------------------------------------------------------------------------

# Technology Stack

**Frontend**
-   React

**Backend / API**
-   FastAPI
-   MCP Python SDK
-   pydantic-settings — typed, environment-based config

**LLM**
-   Claude / GPT

**Data & State**
-   PostgreSQL — persistent data (repo metadata, session/audit history)
-   Redis — shared session/context state, job queue broker, response
    cache (introduced in Phase 3, not upfront)

**Background Orchestration**
-   Arq — async job queue for long-running tools (Docker builds, test
    runs); fits FastAPI's async model better than Celery

**Observability**
-   structlog — structured JSON logs with request/session/trace IDs
-   prometheus-client — tool latency, error rate, and LLM token-spend
    metrics

**Infra**
-   Docker — both a deployment target *and* the sandbox boundary for
    Terminal/Docker MCP servers
-   GitHub API

> Redis, Arq, and prometheus-client aren't needed until Phase 3/4 —
> add them when the corresponding phase needs them rather than
> provisioning everything upfront.

------------------------------------------------------------------------

# Production & Scalability Considerations

Your MCP servers wrap Terminal, Docker, Database, and Filesystem access —
this is effectively remote code execution as a feature. Treat security and
reliability as core design constraints from Phase 1, not hardening you add
later. The sections below are organized so you can build them alongside
the existing phases rather than as a rewrite.

## 1. Security & Sandboxing (highest priority)

-   **Sandbox Terminal/Docker MCP servers** in a container or VM, never
    directly on the host. Use a disposable Docker-in-Docker or gVisor/
    Firecracker sandbox per session so a bad command can't touch the real
    filesystem or network.
-   **Filesystem MCP: enforce a workspace root.** Resolve every path,
    reject `..` traversal, and deny symlinks that escape the root. Never
    let the LLM pass an absolute path straight to `open()`.
-   **Command allow-listing for Terminal MCP.** Maintain an explicit list
    of permitted binaries/commands (npm, pytest, git, docker) instead of
    executing arbitrary shell strings. Block shell metacharacters
    (`;`, `&&`, `|`, backticks) unless you've deliberately chosen to
    support pipelines, in which case parse and validate each stage.
-   **Database MCP: read-only by default.** Require a separate,
    explicitly-approved mode for write/DDL statements. Use a connection
    role with minimal grants, and parameterize/validate any
    LLM-generated SQL before execution (never string-concatenate).
-   **Human-in-the-loop approval for destructive actions** (file writes,
    git push, `rm`, DB writes, container stop/rm) — you already planned
    this for "apply fixes"; extend the same approval gate to every MCP
    server, not just the debugging workflow.
-   **Secrets management.** Never let MCP servers read `.env` files or
    credentials directly into LLM context. Use a secrets broker
    (OS keychain, Vault, or at minimum a `.env` that's filtered out of
    any file-read tool) and redact known secret patterns (API keys,
    tokens, connection strings) from tool output before it reaches the
    model.
-   **Audit log every tool invocation**: who/what triggered it, the
    exact arguments, and the result — write-only, append-only log,
    separate from application logs, so you can reconstruct "what did the
    AI actually do" after the fact.

## 2. Scalability Architecture

-   **Keep MCP servers stateless and horizontally replaceable.** Session
    state (conversation history, pending approvals, workspace context)
    belongs in a shared store (Redis/Postgres), not in server process
    memory — this lets you restart or scale any MCP server without
    losing context.
-   **Isolate long-running/blocking tools** (Docker builds, test runs,
    large repo scans) behind a job queue (e.g., Celery/RQ/Arq) with
    progress polling, instead of holding an MCP request open. Prevents
    one slow tool call from blocking the whole orchestration loop.
-   **Cache expensive read operations** (dependency graph, architecture
    summary, symbol index) with file-hash or mtime-based invalidation so
    repeated questions about an unchanged repo don't re-scan it.
-   **Rate-limit and budget LLM calls.** Track token usage per session
    and per tool-chain; add a max-iterations cap on the tool-selection
    loop to prevent runaway orchestration (a common MCP failure mode is
    the model looping tool calls indefinitely).
-   **Design MCP servers to be independently deployable services**
    (each with its own `Dockerfile` and health endpoint) even though
    you'll run them locally at first — this is what makes "evolve into
    DeployGenie" (your own Future Enhancement) actually feasible later
    instead of a rewrite.

## 3. Reliability & Observability

-   **Structured logging** (JSON logs with request/session/trace IDs)
    across MCP client and every server, so a single user action can be
    traced end-to-end.
-   **Health checks + readiness probes** per MCP server (DB
    connectivity, Docker daemon reachable, filesystem root writable) —
    surface these in a `/status` dashboard rather than failing silently
    mid-workflow.
-   **Graceful degradation.** If one MCP server is down (say, Docker),
    the orchestrator should report that capability as unavailable and
    continue, not crash the whole session.
-   **Timeouts + retries with backoff** on every tool call, with a clear
    distinction between "tool errored" (surface to LLM, let it retry/
    adjust) and "tool infrastructure down" (surface to user directly).
-   **Metrics**: tool call latency/error-rate per MCP server, LLM token
    spend, approval-rate for destructive actions — even a simple
    Prometheus + Grafana setup pays off once you're debugging "why did
    that workflow take 40 seconds."

## 4. Testing Strategy

-   **Contract tests per MCP server** (tool schemas validate, mock
    inputs produce expected outputs) independent of the LLM.
-   **Golden-path integration tests** for each Example Workflow (Debug
    FastAPI, Repository Explanation) using a fixture repo, so
    refactors don't silently break orchestration.
-   **Adversarial/red-team tests** specifically for the sandboxing
    rules above: path traversal attempts, injection in SQL/shell
    arguments, oversized file reads — these are the tests most MCP demo
    projects skip and where real incidents happen.

## 5. Config, Versioning & Deployment

-   **Version your MCP tool schemas.** When you change a tool's
    input/output shape, bump a version field so the client and server
    can detect mismatches instead of failing opaquely.
-   **Environment-based config** (dev/staging/local) via a single
    typed config module (e.g., pydantic-settings) rather than scattered
    env var reads — add a `tests/` and `infra/` (docker-compose, CI
    workflow) directory alongside the existing `mcp_servers/` structure.
-   **CI pipeline**: lint + type-check + contract tests + adversarial
    tests on every PR, before "Docker build" is ever wired into an
    automated approval flow.
-   **Pin dependency and MCP protocol versions** explicitly — MCP is a
    young, fast-moving spec; unpinned upgrades are a common source of
    silent breakage.

------------------------------------------------------------------------

# Future Enhancements

-   Add lightweight planner
-   Add long-term memory
-   Add local RAG
-   Add code editing
-   Add voice interface
-   Evolve into DeployGenie AI Release Engineering Platform

------------------------------------------------------------------------

# Resume Summary

Developed **DevPilot AI**, a Local AI Engineering Copilot powered by the
Model Context Protocol (MCP). Integrated Filesystem, Terminal, GitHub,
Browser, Database, Docker, and Git MCP servers to analyze repositories,
investigate runtime failures, orchestrate developer tools, generate
documentation, explain codebases, validate environments, and automate
engineering workflows through intelligent multi-tool orchestration.
