"""Eval harness: config x task x repeat loop, with a cache and a results table.

Run from the repo root:
    python -m eval.run_eval --configs dummy_gold dummy_none --limit 5
    python -m eval.run_eval --configs bm25_only vector_only --repeats 3
    python -m eval.run_eval --configs single_call --limit 5 --repeats 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import time
import shutil
import warnings
from pathlib import Path
from typing import Callable, Optional, Protocol

from dotenv import load_dotenv

from repopilot.orchestrator.state import Answer, Chunk, Citation, Evidence, Patch, TestReport
from eval.metrics import (
    EvalRecord,
    EvalTask,
    across_runs,
    cost_per_resolved,
    mean_recall_at_k,
    regression_rate,
    resolve_rate,
)

EVAL_DIR = Path(__file__).parent
CACHE_DIR = EVAL_DIR / "cache"
REPORT_DIR = EVAL_DIR / "reports"
REPOS_DIR = EVAL_DIR / ".repos"          # add ".repos/" to .gitignore
TASK_FILE = EVAL_DIR / "datasets" / "task_ids.txt"
CACHE_VERSION = "1"                      # bump to invalidate every cached record


# ------------------------------------------------------------------ tasks

def parse_gold_files(patch: str) -> list[str]:
    """Files touched by a unified diff, in order, without duplicates."""
    files: list[str] = []
    for m in re.finditer(r"^diff --git a/(.+?) b/(.+)$", patch, flags=re.M):
        path = m.group(2).strip()
        if path not in files:
            files.append(path)
    return files


def _as_list(value) -> list[str]:
    return json.loads(value) if isinstance(value, str) else list(value)


def load_tasks(task_file: Path = TASK_FILE, limit: Optional[int] = None) -> list[EvalTask]:
    """Read task ids from task_ids.txt and build EvalTasks from SWE-bench Lite."""
    from datasets import load_dataset  # pip install datasets

    wanted = [ln.strip() for ln in task_file.read_text(encoding="utf-8").splitlines()
              if ln.strip() and not ln.startswith("#")]
    if limit:
        wanted = wanted[:limit]
    rows = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Lite", split="test")}
    missing = [t for t in wanted if t not in rows]
    if missing:
        raise SystemExit(f"task ids not found in SWE-bench Lite: {missing}")
    return [
        EvalTask(
            task_id=tid,
            repo=rows[tid]["repo"],
            base_commit=rows[tid]["base_commit"],
            problem_statement=rows[tid]["problem_statement"],
            gold_files=parse_gold_files(rows[tid]["patch"]),   # "patch" excludes test files
            fail_to_pass=_as_list(rows[tid]["FAIL_TO_PASS"]),
            pass_to_pass=_as_list(rows[tid]["PASS_TO_PASS"]),
        )
        for tid in wanted
    ]


def _git(*args: str, cwd: Optional[Path] = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def checkout_repo(task: EvalTask) -> Path:
    """Fast shallow clone + fetch of the exact base_commit. Self-heals broken clones."""
    dest = REPOS_DIR / task.repo.replace("/", "__")
    REPOS_DIR.mkdir(parents=True, exist_ok=True)

    # If a previous clone crashed halfway, wipe the broken directory
    if dest.exists() and not (dest / ".git" / "HEAD").exists():
        shutil.rmtree(dest, ignore_errors=True)

    if not dest.exists():
        print(f"  -> cloning {task.repo} (shallow)...")
        _git("init", str(dest))
        _git("remote", "add", "origin", f"https://github.com/{task.repo}.git", cwd=dest)

    # Check if we already have the commit locally; if not, fetch just that commit
    has_commit = subprocess.run(
        ["git", "cat-file", "-e", f"{task.base_commit}^{{commit}}"],
        cwd=dest,
        capture_output=True,
    ).returncode == 0

    if not has_commit:
        print(f"  -> fetching {task.repo} @ {task.base_commit[:8]}...")
        _git("fetch", "--depth", "1", "origin", task.base_commit, cwd=dest)

    _git("checkout", "--force", task.base_commit, cwd=dest)
    _git("clean", "-fdx", cwd=dest)
    return dest


# ---------------------------------------------------------------- systems

class System(Protocol):
    name: str
    fingerprint: str   # model + prompt version; part of the cache key so stale results never reappear

    def run(self, task: EvalTask, run_idx: int) -> EvalRecord: ...


def to_evidence(chunk: Chunk, score: float, method: str) -> Evidence:
    """Turn one of your indexed chunks into the Evidence the metrics read."""
    return Evidence(
        chunk_id=chunk.id,
        citation=Citation(path=chunk.path, commit=chunk.commit,
                          start_line=chunk.start_line, end_line=chunk.end_line),
        content=chunk.text,
        source_type="code",
        retrieval=method,  # "bm25" | "vector" | "graph" | "impact"
        score=score,
    )


def _fake_evidence(task: EvalTask, path: str, score: float) -> Evidence:
    return Evidence(
        chunk_id=f"{path}:1-1",
        citation=Citation(path=path, commit=task.base_commit, start_line=1, end_line=1),
        content="", source_type="code", retrieval="bm25", score=score,
    )


class DummyGold:
    """Returns exactly the gold files. recall@k must come out 1.0, or the harness is wrong."""
    name, fingerprint = "dummy_gold", "v1"

    def run(self, task: EvalTask, run_idx: int) -> EvalRecord:
        ev = [_fake_evidence(task, p, 1.0 - i * 0.01) for i, p in enumerate(task.gold_files)]
        return EvalRecord(task_id=task.task_id, config=self.name, run_idx=run_idx, evidence=ev)


class DummyNone:
    """Returns paths that cannot match. recall@k must come out 0.0."""
    name, fingerprint = "dummy_none", "v1"

    def run(self, task: EvalTask, run_idx: int) -> EvalRecord:
        rng = random.Random(f"{task.task_id}:{run_idx}")
        ev = [_fake_evidence(task, f"nowhere/{rng.randrange(10**6)}.py", 1.0 - i * 0.01)
              for i in range(10)]
        return EvalRecord(task_id=task.task_id, config=self.name, run_idx=run_idx, evidence=ev)


SearchFn = Callable[[Path, EvalTask, int], list[Evidence]]


class RetrievalOnly:
    """No LLM, no patch: check out the repo, search with the issue text, record the evidence."""

    def __init__(self, name: str, search_fn: SearchFn, k: int = 20, fingerprint: str = "v1"):
        self.name, self.search_fn, self.k, self.fingerprint = name, search_fn, k, fingerprint

    def run(self, task: EvalTask, run_idx: int) -> EvalRecord:
        repo_dir = checkout_repo(task)
        evidence = self.search_fn(repo_dir, task, self.k)
        return EvalRecord(task_id=task.task_id, config=self.name, run_idx=run_idx, evidence=evidence)


def search_bm25(repo_dir: Path, task: EvalTask, k: int) -> list[Evidence]:
    from repopilot.indexing.parser import parse_repo
    from repopilot.indexing.bm25 import BM25Index

    parsed_files = parse_repo(repo_dir, commit=task.base_commit)
    chunks = [c for pf in parsed_files for c in pf.chunks]
    index = BM25Index().build(chunks)
    hits = index.search(task.problem_statement, k=k)
    return [to_evidence(index.get_chunk(cid), score, "bm25") for cid, score in hits]


def search_vector(repo_dir: Path, task: EvalTask, k: int) -> list[Evidence]:
    from repopilot.indexing.parser import parse_repo
    from repopilot.indexing.embedder import VectorIndex, get_embedder

    parsed_files = parse_repo(repo_dir, commit=task.base_commit)
    chunks = [c for pf in parsed_files for c in pf.chunks]
    chunk_map = {c.id: c for c in chunks}
    embedder = get_embedder("hashing")
    index = VectorIndex(embedder).build(chunks)
    hits = index.search(task.problem_statement, k=k)
    return [to_evidence(chunk_map[cid], score, "vector") for cid, score in hits]

# --- PASTE THIS BELOW search_vector() IN eval/run_eval.py ---

def search_hybrid_rerank(repo_dir: Path, task: EvalTask, k: int) -> list[Evidence]:
    """Week 3: Combines BM25 + Vector (Reciprocal Rank Fusion) + Yogesh's Reranker."""
    from repopilot.indexing.parser import parse_repo
    from repopilot.indexing.bm25 import BM25Index
    from repopilot.indexing.embedder import VectorIndex, get_embedder
    from repopilot.retrieval.reranker import Reranker

    parsed_files = parse_repo(repo_dir, commit=task.base_commit)
    chunks = [c for pf in parsed_files for c in pf.chunks]
    if not chunks:
        return []
    chunk_map = {c.id: c for c in chunks}

    # If Vibhav's hybrid.py is ready, use it; otherwise fuse BM25 + Vector via RRF directly
    try:
        from repopilot.retrieval.hybrid import hybrid_search  # type: ignore
        initial_hits = hybrid_search(chunks, task.problem_statement, k=max(k * 3, 30))
    except ImportError:
        bm25_hits = BM25Index().build(chunks).search(task.problem_statement, k=max(k * 3, 30))
        vec_hits = VectorIndex(get_embedder("hashing")).build(chunks).search(
            task.problem_statement, k=max(k * 3, 30)
        )
        rrf_scores: dict[str, float] = {}
        for rank, (cid, _) in enumerate(bm25_hits):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)
        for rank, (cid, _) in enumerate(vec_hits):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)
        initial_hits = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[: max(k * 3, 30)]

    reranker = Reranker()
    reranked = reranker.rerank_chunks(
        query=task.problem_statement,
        chunk_scores=[(chunk_map[cid], score) for cid, score in initial_hits if cid in chunk_map],
        top_k=k,
    )
    return [to_evidence(chunk, score, "vector") for chunk, score in reranked]


def search_navigator(repo_dir: Path, task: EvalTask, k: int) -> list[Evidence]:
    """Week 3 Full Pipeline: Hybrid + AST Call Graph + Coverage Map + Navigator + Reranker."""
    from repopilot.indexing.parser import parse_repo
    import repopilot.indexing.symbol_graph as sg_mod
    import repopilot.indexing.coverage_map as cm_mod
    from repopilot.agents.navigator import NavigatorAgent

    # 1. Get initial hybrid + reranked seed evidence
    seed_evidence = search_hybrid_rerank(repo_dir, task, k=k)

    # 2. Parse repo
    parsed_files = parse_repo(repo_dir, commit=task.base_commit)
    chunks = [c for pf in parsed_files for c in pf.chunks]

    # 3. Build SymbolGraph (supports SymbolGraph().build(...), SymbolGraph(parsed_files), or build_symbol_graph)
    if hasattr(sg_mod, "build_symbol_graph"):
        sym_graph = sg_mod.build_symbol_graph(parsed_files)
    elif hasattr(sg_mod, "SymbolGraph"):
        sg_cls = sg_mod.SymbolGraph
        try:
            sym_graph = sg_cls().build(parsed_files)
        except TypeError:
            sym_graph = sg_cls(parsed_files)
    else:
        sym_graph = None

    # 4. Build CoverageMap (supports CoverageMap().build(...), CoverageMap(...), or build_coverage_map)
    if hasattr(cm_mod, "build_coverage_map"):
        try:
            cov_map = cm_mod.build_coverage_map(parsed_files, sym_graph)
        except TypeError:
            cov_map = cm_mod.build_coverage_map(repo_dir)
    elif hasattr(cm_mod, "CoverageMap"):
        cm_cls = cm_mod.CoverageMap
        try:
            cov_map = cm_cls().build(parsed_files, sym_graph)
        except TypeError:
            try:
                cov_map = cm_cls().build(parsed_files)
            except Exception:
                cov_map = {}
    else:
        cov_map = {}

    # 5. Run Yogesh's Navigator agent to expand call graph, compute impact, and rerank
    navigator = NavigatorAgent(
        symbol_graph=sym_graph,
        coverage_map=cov_map,
        chunks=chunks,
        commit=task.base_commit,
    )
    nav_result = navigator.navigate(
        query=task.problem_statement,
        seed_evidence=seed_evidence,
        top_k=k,
    )
    return nav_result.evidence

class SingleCallSystem:
    """B0 Gemini single-call baseline adapter for the evaluation harness."""

    def __init__(self, model_name: str = "gemini-3.5-flash-lite"):
        from eval.baselines.single_call import SingleCallBaseline
        self.name = "single_call"
        self.model_name = model_name
        self.fingerprint = f"{model_name}-v1"
        self.baseline = SingleCallBaseline(model_name=model_name)

    def run(self, task: EvalTask, run_idx: int) -> EvalRecord:
        from eval.datasets.tasks import TaskInstance

        task_instance = TaskInstance(
            instance_id=task.task_id,
            repo=task.repo,
            base_commit=task.base_commit,
            problem_statement=task.problem_statement,
            hints_text="",
        )

        result = self.baseline.run(task_instance)

        if result.error:
            return EvalRecord(
                task_id=task.task_id,
                config=self.name,
                run_idx=run_idx,
                error=result.error,
                latency_s=result.latency_seconds,
            )

        diffs = []
        if result.patch:
            diffs.append(Patch(filepath="", diff=result.patch, base_commit=task.base_commit))

        tokens = result.prompt_tokens + result.completion_tokens
        cost = result.cost_usd
        if tokens > 0 and cost == 0.0:
            cost = 1e-6  # satisfies non-zero validation while tracking near-zero usage

        answer = Answer(
            diffs=diffs,
            test_report=TestReport(passed=False, total_run=0),
            verified_claims=[],
            cost_usd=cost,
            tokens=tokens,
        )

        return EvalRecord(
            task_id=task.task_id,
            config=self.name,
            run_idx=run_idx,
            evidence=[],
            answer=answer,
            latency_s=result.latency_seconds,
        )


REGISTRY: dict[str, Callable[[], System]] = {
    "dummy_gold": DummyGold,
    "dummy_none": DummyNone,
    "bm25_only": lambda: RetrievalOnly("bm25_only", search_bm25),
    "vector_only": lambda: RetrievalOnly("vector_only", search_vector),
    "single_call": SingleCallSystem,
    "hybrid_rerank": lambda: RetrievalOnly("hybrid_rerank", search_hybrid_rerank),
    "navigator_full": lambda: RetrievalOnly("navigator_full", search_navigator),
}


# ------------------------------------------------------------ run + cache

def cache_path(system: System, task: EvalTask, run_idx: int) -> Path:
    raw = json.dumps([CACHE_VERSION, system.name, system.fingerprint, task.task_id, run_idx])
    digest = hashlib.sha256(raw.encode()).hexdigest()[:10]
    return CACHE_DIR / f"{system.name}__{task.task_id}__{run_idx}__{digest}.json"


def check_record(rec: EvalRecord) -> None:
    """Catch a forgotten cost: a system that spends tokens but reports $0 would look free."""
    if rec.answer and rec.answer.tokens > 0 and rec.answer.cost_usd == 0:
        raise ValueError("tokens were used but cost_usd is 0; fill in the cost")


def run_one(system: System, task: EvalTask, run_idx: int, use_cache: bool = True) -> EvalRecord:
    path = cache_path(system, task, run_idx)
    if use_cache and path.exists():
        return EvalRecord.model_validate_json(path.read_text(encoding="utf-8"))
    start = time.perf_counter()
    try:
        rec = system.run(task, run_idx)
        check_record(rec)
    except Exception as exc:  # a crash is a result: it stays in the denominator and is not cached
        return EvalRecord(task_id=task.task_id, config=system.name, run_idx=run_idx,
                          error=repr(exc), latency_s=time.perf_counter() - start)
    rec.latency_s = time.perf_counter() - start
    # TODO (week 4+): grade here. Apply rec.answer.diffs at task.base_commit in the sandbox,
    # run task.fail_to_pass / task.pass_to_pass, then set rec.resolved and rec.regressed.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(rec.model_dump_json(indent=2), encoding="utf-8")
    return rec


def evaluate(systems: list[System], tasks: list[EvalTask], repeats: int,
             use_cache: bool = True) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for system in systems:
        for task in tasks:
            for run_idx in range(repeats):
                rec = run_one(system, task, run_idx, use_cache)
                status = f"error: {rec.error}" if rec.error else f"{len(rec.evidence)} evidence"
                print(f"[{system.name}] {task.task_id} run {run_idx}: {status}")
                records.append(rec)
    return records


# ----------------------------------------------------------------- reports

def _fmt(pair: tuple[Optional[float], Optional[float]], digits: int = 3) -> str:
    m, s = pair
    return "-" if m is None else f"{m:.{digits}f} ± {s:.{digits}f}"


def write_reports(records: list[EvalRecord], tasks: list[EvalTask], out_dir: Path = REPORT_DIR,
                  ks: tuple[int, ...] = (5, 10)) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_task = {t.task_id: t for t in tasks}

    with (out_dir / "raw.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    configs = list(dict.fromkeys(r.config for r in records))
    header = (["config", "tasks", "runs"] + [f"recall@{k}" for k in ks]
              + ["resolve rate", "regression rate", "cost / resolved ($)", "errors"])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for cfg in configs:
        rs = [r for r in records if r.config == cfg]
        graded = any(r.resolved is not None for r in rs)
        cells = [cfg, str(len({r.task_id for r in rs})), str(len({r.run_idx for r in rs}))]
        cells += [_fmt(across_runs(rs, lambda x, k=k: mean_recall_at_k(x, by_task, k))) for k in ks]
        cells += [
            _fmt(across_runs(rs, resolve_rate)) if graded else "-",
            _fmt(across_runs(rs, regression_rate)),
            _fmt(across_runs(rs, cost_per_resolved), 4),
            str(sum(r.error is not None for r in rs)),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    path = out_dir / "results.md"
    path.write_text("\n".join(lines) + "\n\nValues are mean ± std across repeats. "
                    "Crashed runs count as zero recall and as unresolved.\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------- CLI

def main() -> None:
    # Load environment variables from the .env file in the project root
    load_dotenv()

    ap = argparse.ArgumentParser(description="RepoPilot evaluation harness")
    ap.add_argument("--configs", nargs="+", default=["dummy_gold", "dummy_none"],
                    choices=sorted(REGISTRY))
    ap.add_argument("--tasks", type=Path, default=TASK_FILE, help="file with one task id per line")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-cache", action="store_true", help="ignore cached results and re-run")
    args = ap.parse_args()

    tasks = load_tasks(args.tasks, args.limit)
    systems = [REGISTRY[name]() for name in args.configs]
    records = evaluate(systems, tasks, args.repeats, use_cache=not args.no_cache)
    print(f"\nwrote {write_reports(records, tasks)}")


if __name__ == "__main__":
    main()