# RepoPilot v4.1
https://repopilot-six.vercel.app/

A verified multi-agent coding assistant for large repositories. This version proves it with numbers (evaluation first), makes the core algorithms visible (AST, call graph, hybrid search, graph traversal), hardens the sandbox and verifier, and measures cost control. Memory stays a stretch goal. Kubernetes, Terraform, Grafana, LSP and the VS Code extension are cut.

## Architecture

See the architecture diagram on the project page above. In short: a React app calls the API (REST plus a live SSE stream). The API starts the orchestrator, which drives specialist agents. Agents get evidence from the index and retrieval layer and act only through the tool layer. Every LLM call goes through the router and budget. An evaluation harness measures the whole system, and Docker, CI with an eval gate, a free-tier live demo and OTel tracing ship and watch it.

## Results

Fill this table only with numbers from `eval/reports/`. Every row comes from 3 runs on a fixed SWE-bench Lite subset.

| Config | Resolve rate | Regression rate | Cost per resolved task |
| --- | --- | --- | --- |
| Single call | [Z]% | [ ] | [ ] |
| Plain RAG | [Y]% | [ ] | [ ] |
| RepoPilot (full) | [X]% | [ ] | [ ] |
| no_graph | [ ]% | [ ] | [ ] |
| no_coverage_map | [ ]% | [ ] | [ ] |
| no_verifier | [ ]% | [ ] | [ ] |
| no_router | [ ]% | [ ] | [ ] |

Also reported: recall@k with and without the call graph, false approvals with and without the verifier, and prompt-injection attacks blocked versus not blocked.

## Folder structure

```
repopilot/
├── README.md                      # claim, results table, demo gif, architecture, limitations
├── Makefile                       # make test, make eval, make demo
├── eval/                          # the proof: built first, shown first
│   ├── datasets/                  # task_ids.txt (fixed SWE-bench Lite subset), custom_questions, injection_attacks (.jsonl)
│   ├── baselines/                 # single_call.py, plain_rag.py
│   ├── ablations/                 # no_graph, no_coverage_map, no_verifier, no_router, no_memory
│   ├── cache/                     # cached model outputs so anyone can re-check the table
│   ├── metrics.py                 # resolve rate, regression rate, recall@k, false approvals, cost per resolved task
│   ├── run_eval.py                # 3 repeats per config, reports mean and spread
│   └── reports/                   # results.md, ablation_table.md, failure_cases.csv
├── pyproject.toml
├── docker-compose.yml             # api, web, postgres, redis, qdrant
├── .env.example
├── .github/workflows/             # ci.yml (lint, tests, eval gate), eval.yml (full eval, manual)
├── configs/
│   ├── agents.yaml                # models, step limits per agent
│   ├── budget.yaml                # token, dollar, tool-call and loop caps
│   ├── routing.yaml               # cheap model first, escalation rules
│   ├── mcp_servers.yaml           # servers and tool scopes
│   └── retrieval.yaml             # chunking, top-k, hop limit, fusion weights
├── src/repopilot/
│   ├── indexing/                  # the algorithms live here
│   │   ├── parser.py              # tree-sitter AST for each file
│   │   ├── symbol_graph.py        # functions, classes, calls and imports as a directed graph
│   │   ├── coverage_map.py        # symbol to covering tests
│   │   ├── bm25.py                # own implementation, checked against a library
│   │   └── embedder.py
│   ├── retrieval/
│   │   ├── hybrid.py              # BM25 + vectors with rank fusion
│   │   ├── graph_expand.py        # BFS over the call graph with a hop limit
│   │   ├── reranker.py
│   │   └── impact.py              # reverse traversal: changed symbols to affected tests
│   ├── orchestrator/              # planner, graph (LangGraph), scheduler, state, budget
│   ├── agents/                    # base, retriever, navigator, coder, tester, reflector, verifier
│   ├── sandbox/
│   │   ├── runner.py              # Docker runner, no network
│   │   ├── limits.py              # CPU, memory, process and time limits
│   │   └── test_detect.py
│   ├── verification/              # citation, executable_claims, entailment, diff_checker
│   ├── security/                  # permissions, injection_guard, quarantine
│   ├── llm/
│   │   ├── client.py
│   │   ├── router.py              # cheap model first, escalate on failure or low confidence
│   │   └── prompts/               # planner, coder, reflector, verifier (.md)
│   ├── mcp/
│   │   ├── client.py
│   │   └── servers/               # github, fs, sandbox
│   ├── memory/                    # short_term (core); long_term, staleness (stretch)
│   ├── api/                       # main.py, routes.py, schemas.py, sse.py
│   └── observability/             # tracing.py, cost.py
├── web/                           # React frontend (Vite + TypeScript)
│   ├── package.json, vite.config.ts, Dockerfile, nginx.conf
│   └── src/
│       ├── api/                   # client.ts, sse.ts, types.ts (generated from schemas.py)
│       ├── pages/                 # Home, RunDetail, Repos, Eval
│       ├── components/            # ChatPanel, TaskGraphView, DiffViewer, TestReport, CitationList, CostMeter, ApprovalPrompt
│       ├── hooks/                 # useRunStream
│       └── store/                 # Zustand run state
├── deploy/
│   ├── docker/                    # api.Dockerfile, worker.Dockerfile
│   ├── demo/                      # free-tier host config for the live demo
│   └── tracing/                   # otel-collector.yaml
├── cli/repopilot.py
├── tests/
│   ├── unit/                      # includes bm25 vs library, graph traversal, sandbox no-network test
│   ├── integration/
│   ├── e2e/                       # Playwright
│   └── fixtures/sample_repo/
├── scripts/                       # index_repo.py, build_coverage.py, seed_memory.py
└── docs/                          # architecture.md, evaluation.md, design_decisions.md, failure_analysis.md, limitations.md, writeup.md, demo.gif
```

## Details

### 1. How it works

One task runs through these states. Every LLM call goes through the router and budget.

1. **Plan:** the Planner splits the task into a graph of dependent sub-questions.
2. **Retrieve:** Retriever (hybrid search) and Navigator (call graph, coverage map) run in parallel and produce evidence plus the impact set: changed code and the tests that cover it.
3. **Draft:** Coder writes the smallest diff from retrieved evidence only.
4. **Test:** Tester runs impacted tests first, then the full suite, in a no-network sandbox with CPU, memory and time limits.
5. **Reflect:** on failure, Reflector writes a short note and the flow returns to Draft, up to the loop cap in `budget.yaml`.
6. **Verify:** executable checks first (AST, graph query, sandbox run), LLM judge only as fallback. Unsupported claims are removed.
7. **Respond:** diff, test report, claims cited to `repo@commit:lines`, and run cost, streamed live to the web app. (Remembering verified findings is the stretch goal.)

Design rules:

- **Evidence first.** Every feature has an ablation and a metric. If it does not move a number, it gets cut.
- **Typed hand-offs.** Agents exchange only `Evidence`, `TaskNode`, `Claim`, `Patch`, `TestReport`, `Verdict` and `Answer`, defined in `orchestrator/state.py`, so each agent is testable alone.
- **Untrusted text.** Repo files, issues and PR comments are quoted as data. A tool call built from untrusted text is blocked unless the Planner approved it.
- **Least privilege.** Write tools stay off until the Verifier stage. It opens a draft PR and never merges.
- **Measured cost control.** Cheap model first, escalate on failure or low confidence. The budget caps tokens, dollars, tool calls and loops, and returns a partial answer instead of looping.
- **One contract.** `web/src/api/types.ts` is generated from `schemas.py`.

### 2. Folder details

| Folder | Purpose | Key files |
| --- | --- | --- |
| `eval/` | Fixed task subset, baselines, ablations and metrics; cached outputs so the table can be re-checked | `run_eval.py`, `metrics.py` |
| `indexing/` | Turns a repo into an AST, a symbol graph and a test coverage map | `parser`, `symbol_graph`, `coverage_map`, `bm25` |
| `retrieval/` | Finds evidence, not just similar text: fusion search, graph BFS, reverse impact traversal | `hybrid`, `graph_expand`, `impact` |
| `orchestrator/`, `agents/` | LangGraph state machine and narrow specialists with fixed tool access | `planner`, `graph`, `coder`, `verifier` |
| `sandbox/` | No-network Docker runner with resource limits; detects the test framework | `runner`, `limits`, `test_detect` |
| `verification/` | Decides which claims and patches may leave the system | `executable_claims`, `diff_checker` |
| `security/` | Permissions, prompt-injection guard, quarantine of untrusted input | `injection_guard`, `permissions` |
| `llm/`, `configs/` | Cost-aware router, prompt files, all tunable settings | `router`, `budget.yaml` |
| `mcp/` | One client and three tool servers (GitHub, files, sandbox) | `client.py`, `servers/*` |
| `api/`, `web/` | REST plus SSE backend and the React app | `sse.py`, `useRunStream` |
| `deploy/`, `.github/` | Docker images, free-tier demo config, tracing, CI with an eval gate | `ci.yml`, `demo/` |

### 3. Frontend (React)

Vite + TypeScript. REST for actions, Server-Sent Events for live progress, so there is no polling. Four screens only.

| Screen | What the user does | Main components |
| --- | --- | --- |
| Home | Enter a bug, issue link or feature request and start a run | `ChatPanel` |
| Run detail | Watch the plan, agents and tests live; approve or reject the draft PR | `TaskGraphView`, `DiffViewer`, `TestReport`, `CitationList`, `CostMeter`, `ApprovalPrompt` |
| Repos | Connect a repo and watch indexing progress | repo list, progress bar |
| Eval | See resolve rate, regression rate, cost and the ablation table | charts from `eval/reports/` |

- **Human approval.** Opening a draft PR needs a click on `ApprovalPrompt`, which keeps "never merges on its own" visible.
- **Citations you can click.** Every citation links to `repo@commit:lines` in `DiffViewer`.
- **Cut:** the Memory and Settings pages and the OAuth login. The demo uses a single token from the environment.

### 4. Deployment

| Piece | How it works |
| --- | --- |
| Local | `docker compose up` starts api, web, postgres, redis and qdrant. |
| Images | `api`, `worker` (agent loop and sandbox) and `web` (React build behind nginx). |
| CI | `ci.yml` runs lint, unit, integration and Playwright tests plus a small eval gate that fails if resolve rate drops. `eval.yml` runs the full eval by hand. |
| Live demo | One free-tier host from `deploy/demo/`, with a hard budget cap so the demo cannot overspend. |
| Sandbox safety | No network, CPU, memory, process and time limits, covered by a no-network unit test. |
| Secrets | Injected through environment variables, never baked into images. |
| Tracing | OpenTelemetry traces and a per-run cost log. |

## Work split for two people

Both people write core algorithms, an agent, tests and evaluation code. Every producer and its consumer sit with different people, so contracts get tested from week 1, and each person owns the tests for their own files.

**Shared in the first 2 days:** repo, `pyproject.toml`, `docker-compose.yml`, `Makefile`, folder skeleton, data contracts in `orchestrator/state.py`, the sample repo in `tests/fixtures/`, and the fixed task list in `eval/datasets/task_ids.txt`.

### Who owns what

| Area | Vibhav-j | Yogesh |
| --- | --- | --- |
| `indexing/` | `parser`, `symbol_graph` | `bm25` (plus test against a library), `embedder`, `coverage_map` |
| `retrieval/` | `hybrid`, `graph_expand` | `reranker`, `impact` |
| `agents/` | `base`, Retriever, Coder, Reflector | Navigator, Tester, Verifier |
| `orchestrator/` | `planner`, `budget` | `graph`, `scheduler` |
| `sandbox/` | `runner`, `limits` | `test_detect`, no-network test |
| `verification/`, `security/` | `citation`, `diff_checker`, `permissions`, `quarantine` | `executable_claims`, `entailment`, `injection_guard` |
| `llm/`, `mcp/` | llm `client`, planner and coder prompts, mcp `client`, github server | `router`, reflector and verifier prompts, fs and sandbox servers |
| `api/`, `observability/`, `cli/` | `routes`, `schemas`, `tracing` | `sse.py`, `cost`, `cli/` |
| `web/` | App shell, `useRunStream`, Home and Run detail, `ChatPanel`, `TaskGraphView` | `DiffViewer`, `TestReport`, `CitationList`, `CostMeter`, `ApprovalPrompt`, Repos and Eval pages |
| `eval/` | `datasets`, baselines, `no_graph` and `no_router` ablations, recall@k, resolve rate, cost per resolved task | `run_eval.py`, `cache`, `no_verifier` and `no_coverage_map` ablations, regression rate, false approvals, attack success |
| Deploy and CI | Dockerfiles, `ci.yml`, `demo/` | `eval.yml`, `otel-collector.yaml`, Playwright tests |

### Week by week

Evaluation starts in week 1, so every later feature is measured the week it lands. Stretch memory only starts if week 6 finishes early.

| Week | Focus | Vibhav-j | Yogesh |
| --- | --- | --- | --- |
| 1 | Index and eval skeleton | `parser`, `symbol_graph`, task list, `single_call` baseline | `bm25`, `embedder`, `coverage_map`, `metrics.py`, `run_eval.py` |
| 2 | Tools and sandbox | mcp `client`, github server, llm `client`, `runner`, `limits` | fs and sandbox servers, `test_detect`, `router`, no-network test |
| 3 | Retrieve | `hybrid`, `graph_expand`, Retriever, `plain_rag` baseline | `reranker`, `impact`, Navigator; first recall@k table |
| 4 | Plan and act | `planner`, `budget`, Coder, Reflector | `graph`, `scheduler`, Tester; first end-to-end run on the sample repo |
| 5 | Verify and protect | `citation`, `diff_checker`, `permissions`, `quarantine`, `no_graph`, `no_router` | `executable_claims`, `entailment`, Verifier, `injection_guard`, `no_verifier`, `no_coverage_map`, attack suite |
| 6 | Interface, ship and write-up | React shell, Home and Run detail, `routes`, Dockerfiles, `ci.yml`, `demo/`, README results table | Diff, test and approval components, Repos and Eval pages, `sse.py`, `eval.yml`, failure analysis, limitations, demo gif |

Rules: every file is reviewed by the other person; contracts in `state.py` and `types.ts` are agreed before coding; one end-to-end task runs every Friday; the full eval runs 3 times per config before any number goes into the README.

## Cut and stretch

Cut: Kubernetes and Helm, Terraform, Grafana dashboards, LSP server, VS Code extension, issues server, OAuth login, Memory and Settings pages. Stretch: long-term memory with staleness checks, only after the `no_memory` ablation can show it helps. Whatever is cut or unfinished goes into `docs/limitations.md` instead of being implied.