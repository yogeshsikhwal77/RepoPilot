# RepoPilot v3
https://repopilot-six.vercel.app/

A verified multi-agent coding assistant for large repositories, now with a React frontend and a full deployment pipeline. The architecture picture and the folder tree come first; the written details for each are at the bottom.

## Architecture

## Folder structure

```
repopilot/
├── README.md
├── pyproject.toml
├── docker-compose.yml             # api, web, postgres, redis, qdrant
├── docker-compose.prod.yml        # production overrides
├── .env.example
├── .github/workflows/             # ci.yml, eval.yml, deploy.yml
├── configs/
│   ├── agents.yaml                # models, step limits per agent
│   ├── budget.yaml                # token, cost, loop caps
│   ├── mcp_servers.yaml           # servers and tool scopes
│   ├── retrieval.yaml             # chunking, top-k, hop limit
│   └── routing.yaml               # small/large model rules
├── src/repopilot/
│   ├── api/                       # main.py, routes.py, schemas.py, sse.py, auth.py
│   ├── orchestrator/              # planner, graph, scheduler, state, budget
│   ├── agents/                    # base, retriever, navigator, coder, tester, reflector, verifier
│   ├── indexing/                  # parser, symbol_graph, coverage_map, embedder, bm25, incremental, linker
│   ├── retrieval/                 # hybrid, reranker, graph_expand, impact
│   ├── mcp/
│   │   ├── client.py
│   │   └── servers/               # github, fs, lsp, issues, sandbox
│   ├── memory/                    # short_term, long_term, staleness, policies
│   ├── verification/              # citation, executable_claims, entailment, diff_checker
│   ├── sandbox/                   # runner, test_detect
│   ├── security/                  # permissions, injection_guard, quarantine
│   ├── llm/
│   │   ├── client.py
│   │   ├── router.py
│   │   └── prompts/               # planner, coder, reflector, verifier (.md)
│   └── observability/             # tracing.py, cost.py
├── web/                           # React frontend (Vite + TypeScript)
│   ├── package.json
│   ├── vite.config.ts
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── .env.example
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api/                   # client.ts, sse.ts, types.ts (mirrors schemas.py)
│       ├── pages/                 # Home, RunDetail, Repos, Memory, Eval, Settings
│       ├── components/            # ChatPanel, TaskGraphView, DiffViewer, TestReport, CitationList, CostMeter, ApprovalPrompt
│       ├── hooks/                 # useRunStream, useRepos
│       ├── store/                 # run state (Zustand)
│       └── styles/                # theme.css
├── deploy/
│   ├── docker/                    # api.Dockerfile, worker.Dockerfile
│   ├── k8s/                       # Helm chart: api, web, worker, ingress, hpa, secrets
│   ├── terraform/                 # cloud network, cluster, database (optional)
│   └── monitoring/                # otel-collector.yaml, grafana dashboards, alerts
├── cli/repopilot.py
├── extensions/vscode/
├── eval/
│   ├── datasets/                  # swebench_subset, custom_questions, injection_attacks (.jsonl)
│   ├── baselines/                 # plain_rag.py, single_call.py
│   ├── ablations/                 # no_graph, no_memory, no_verifier, no_coverage_map
│   ├── metrics.py
│   ├── run_eval.py
│   └── reports/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/                       # Playwright tests for the web app
│   └── fixtures/sample_repo/
├── scripts/                       # index_repo.py, seed_memory.py, build_coverage.py
└── docs/                          # architecture.md, evaluation.md, deployment.md, failure_analysis.md, demo.gif
```

## Details

### 1. Architecture details

The system is seven layers. The React web app, CLI and VS Code extension (interfaces) call the API. The API starts the orchestrator, which drives specialist agents. Agents reach the outside world only through the MCP tool layer, and read evidence from the retrieval and index layer. Memory and security wrap everything. The deployment layer packages, ships and monitors all of it.

One task runs through these states:

1. **Plan:** the Planner reads memory and builds a task graph of dependent sub-questions.
2. **Retrieve:** Retriever and Navigator run in parallel and produce evidence and the impact set (code plus the tests that cover it).
3. **Draft:** Coder writes the smallest diff, using only retrieved evidence and any reflection notes.
4. **Test:** Tester runs impacted tests first, then the full suite, in a no-network sandbox.
5. **Reflect:** on failure, Reflector writes a short note on why it failed and the flow returns to Draft, up to the loop cap in `budget.py`.
6. **Verify:** the Verifier runs executable checks (AST, graph query, sandbox run) first and falls back to an LLM judge. Unsupported claims are removed.
7. **Remember:** verified findings are stored in long-term memory with hashes of their source files.
8. **Respond:** return the diff, test report, cited claims and run cost. Each state change is also streamed to the web app as it happens.

Design rules that hold across the system:

- **Typed hand-offs.** Agents exchange only `Evidence`, `TaskNode`, `Claim`, `Patch`, `TestReport`, `Verdict` and `Answer` objects, so each agent can be tested alone.
- **Untrusted text.** Repo files, issues and PR comments are quoted as data. A tool call built from untrusted text is blocked unless the Planner approved it.
- **Least privilege.** Write tools stay off until the Verifier stage. The agent opens a draft PR and never merges.
- **Cost control.** The router sends easy steps to a small model and escalates on low confidence. The budget caps tokens, dollars, tool calls and loops, and returns a partial answer instead of looping.
- **Self-healing memory.** When a source file's hash changes, the memory entry is re-verified, downgraded or deleted.
- **One contract for UI and API.** `web/src/api/types.ts` is generated from `schemas.py`, so the frontend and backend cannot drift apart.

### 2. Folder structure details

| Folder | What lives there | Key files |
| --- | --- | --- |
| `configs/` | All tunable settings, so behavior changes without code changes | `agents`, `budget`, `mcp_servers`, `retrieval`, `routing` (.yaml) |
| `api/` | HTTP interface to start and watch runs, with login and a live event stream | `main.py`, `routes.py`, `schemas.py`, `sse.py`, `auth.py` |
| `orchestrator/` | Plans the work and drives the agent loop | `planner`, `graph` (LangGraph), `scheduler`, `state`, `budget` |
| `agents/` | Specialists with narrow jobs and fixed tool access | `retriever`, `navigator`, `coder`, `tester`, `reflector`, `verifier` |
| `indexing/` | Turns a repo into searchable structure: chunks, symbol graph, test coverage map | `parser`, `symbol_graph`, `coverage_map`, `embedder`, `bm25`, `incremental`, `linker` |
| `retrieval/` | Finds the right evidence, not just similar text | `hybrid`, `reranker`, `graph_expand`, `impact` |
| `mcp/` | One client plus tool servers for GitHub, files, LSP, issues and the sandbox | `client.py`, `servers/*` |
| `memory/` | Per-run scratchpad and long-term conventions, with staleness checks | `short_term`, `long_term`, `staleness`, `policies` |
| `verification/` | Decides which claims and patches are allowed out | `citation`, `executable_claims`, `entailment`, `diff_checker` |
| `sandbox/` | Docker runner with no network and resource limits; detects the test framework | `runner`, `test_detect` |
| `security/` | Permissions, prompt-injection defense, quarantine of untrusted input | `permissions`, `injection_guard`, `quarantine` |
| `llm/` | Model client, cost-aware router, prompt files | `client`, `router`, `prompts/*.md` |
| `observability/` | OpenTelemetry traces and token and cost accounting | `tracing`, `cost` |
| `web/` | React single-page app to submit tasks, watch runs live and review results | `pages/`, `components/`, `hooks/useRunStream`, `api/` |
| `deploy/` | Everything needed to build, ship and watch the system in production | `docker/`, `k8s/`, `terraform/`, `monitoring/` |
| `.github/workflows/` | Automatic checks and releases on every push | `ci.yml`, `eval.yml`, `deploy.yml` |
| `eval/` | Datasets, baselines, ablations and metrics that prove each upgrade helps | `run_eval.py`, `metrics.py`, `ablations/` |
| `tests/` | Unit, integration and end-to-end tests plus a small sample repo | `unit/`, `integration/`, `e2e/`, `fixtures/` |
| `scripts/` | One-off setup commands | `index_repo`, `seed_memory`, `build_coverage` |
| `cli/`, `extensions/vscode/`, `docs/` | Command-line tool, editor extension, written docs and demo | `repopilot.py` |

### 3. Frontend details (React)

The web app is a Vite + TypeScript single-page app. It talks to the API over REST for actions and Server-Sent Events (SSE) for live progress, so there is no polling.

| Screen | What the user does | Main components |
| --- | --- | --- |
| Home | Type a bug, issue link or feature request and start a run | `ChatPanel` |
| Run detail | Watch the plan, agents and tests update live; approve or reject the draft PR | `TaskGraphView`, `DiffViewer`, `TestReport`, `CitationList`, `CostMeter`, `ApprovalPrompt` |
| Repos | Connect a repo, start indexing, see index status | repo list, index progress bar |
| Memory | Browse stored conventions and delete stale ones | memory table with source hashes |
| Eval | View resolve rate, regression rate, cost and ablation results | charts from `eval/reports/` |
| Settings | Choose models, budget caps and tool permissions | forms that write to `configs/` via the API |

- **Live stream.** `useRunStream` opens one SSE connection per run and updates the store as each state (Plan, Retrieve, Draft, Test, Reflect, Verify) finishes.
- **Human approval.** Opening the draft PR needs a click on `ApprovalPrompt`. This keeps the "never merges on its own" rule visible to the user.
- **Citations you can click.** Every claim in `CitationList` links to `repo@commit:lines`, shown in `DiffViewer`.
- **Auth.** Login through GitHub OAuth; the API issues a short-lived token that the app keeps in memory, not in local storage.
- **Styling.** The same white, blue, red, purple, green and orange theme as this page, defined once in `styles/theme.css`.

### 4. Deployment details

| Piece | How it works |
| --- | --- |
| Local | `docker compose up` starts api, web, postgres, redis and qdrant with one command. |
| Images | Three images: `api`, `worker` (runs the agent loop and sandbox) and `web` (React build served by nginx). |
| CI | `ci.yml` runs lint, unit, integration and Playwright tests. `eval.yml` runs a small eval set and fails the build if resolve rate drops. |
| CD | `deploy.yml` builds images, pushes them to a registry, then rolls out to staging. Production needs a manual approval. |
| Runtime | Kubernetes with a Helm chart: api and web behind an ingress with TLS; workers scale with queue length (HPA on Redis depth). |
| Sandbox safety | Sandbox containers run on a separate node pool with no network and strict CPU, memory and time limits. |
| Secrets | API keys and the GitHub token live in a secret manager and are injected at runtime, never in images or `.env` files. |
| Monitoring | OpenTelemetry collector sends traces to Grafana. Alerts fire on error rate, cost per run above the cap and queue backlog. |

## Work split for two people

Both people work on the same kinds of files and logic: each writes indexing code, a retrieval agent, an action agent, tool servers, memory, verification, security and evaluation. Nobody is stuck on only "infrastructure" or only "agents". The frontend and deployment work is split the same way.

**Shared in the first 2 days:** create the repo, `pyproject.toml`, `docker-compose.yml` and the empty folder skeleton (including `web/` and `deploy/`); write the data contracts together in `orchestrator/state.py`; add the sample repo in `tests/fixtures/`.

### Who owns what

What improved: every producer and its consumer sit with different people, so the contracts get tested from week 1; the agent count is balanced (A has the Retriever and Coder side, B the Navigator and Tester side, and the Verifier goes to B because B owns its checks); and each person owns the tests for their own files.

| Folder | Person A | Person B |
| --- | --- | --- |
| `indexing/` | `parser`, `symbol_graph`, `incremental` | `coverage_map`, `embedder`, `bm25`, `linker` |
| `retrieval/` | `hybrid`, `graph_expand` | `reranker`, `impact` |
| `agents/` | `base`, `retriever`, `coder`, `reflector` | `navigator`, `tester`, `verifier` |
| `mcp/` | `client`, `github_server`, `issues_server` | `fs_server`, `lsp_server`, `sandbox_server` |
| `memory/` | `short_term`, `policies` | `long_term`, `staleness` |
| `orchestrator/` | `planner`, `budget` | `graph`, `scheduler` |
| `verification/` | `citation`, `diff_checker` | `executable_claims`, `entailment` |
| `security/` | `permissions`, `quarantine` | `injection_guard` |
| `sandbox/` | `runner` | `test_detect` |
| `llm/` | `client`, planner and coder prompts | `router`, reflector and verifier prompts |
| `observability/`, `api/`, `cli/` | `tracing`, `api/` (`routes`, `schemas`, `auth`) | `cost`, `cli/`, `api/sse.py` |
| `web/` | App shell, `api/`, `useRunStream`, `ChatPanel`, `TaskGraphView`, Home and Run detail pages | `DiffViewer`, `TestReport`, `CitationList`, `CostMeter`, `ApprovalPrompt`, Repos, Memory, Eval and Settings pages |
| `deploy/`, `.github/` | `docker/`, `k8s/` Helm chart, `ci.yml`, `deploy.yml` | `monitoring/`, `terraform/`, `eval.yml`, `docker-compose.prod.yml` |
| `tests/e2e/` | Home and Run detail flows | Approval, Repos and Settings flows |
| `configs/` | `agents`, `budget`, `mcp_servers` | `retrieval`, `routing` |
| `scripts/` | `index_repo`, `seed_memory` | `build_coverage` |
| Evaluation: metrics | Resolve rate, recall@k, citation precision | Regression rate, false-approval rate, attack success rate |
| Evaluation: other | Baselines, datasets, `no_graph` and `no_memory` ablations | `run_eval.py`, cost and latency, `no_verifier` and `no_coverage_map` ablations |

### Week by week

Every week both people work on the same layer, so they can review each other's code and agree on the contracts. Frontend and deployment add a seventh week.

| Week | Layer | Vibhav-j | Yogesh |
| --- | --- | --- | --- |
| 1 | Index | `parser`, `symbol_graph` | `embedder`, `bm25`, `coverage_map` |
| 2 | Tools | `client`, `github_server`, `issues_server`, `runner`, llm `client` | `fs_server`, `lsp_server`, `sandbox_server`, `test_detect`, `router` |
| 3 | Retrieve and remember | `hybrid`, `graph_expand`, `base`, Retriever, `short_term`, `policies` | `reranker`, `impact`, Navigator, `long_term`, `staleness` |
| 4 | Plan and act | `planner`, `budget`, Coder, Reflector, `incremental` | `graph`, `scheduler`, Tester, `linker` |
| 5 | Verify and protect | `citation`, `diff_checker`, `permissions`, `quarantine` | `executable_claims`, `entailment`, Verifier, `injection_guard` |
| 6 | Ship and prove | `tracing`, `api/`, A's evals and ablations, README results table | `cost`, `cli/`, B's evals and ablations, failure analysis, demo video |
| 7 | Frontend and deploy | `web/` shell, `useRunStream`, `ChatPanel`, `TaskGraphView`, Dockerfiles, Helm chart, `ci.yml`, `deploy.yml` | `DiffViewer`, `TestReport`, `ApprovalPrompt`, `api/sse.py`, `eval.yml`, monitoring, Playwright tests |

Rules: every file is reviewed by the other person; each person writes unit tests for their own files; connected files agree on the contracts in `state.py` and `types.ts` before coding; merge and run one end-to-end task every Friday.
