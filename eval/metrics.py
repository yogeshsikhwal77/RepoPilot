"""Eval metrics and the two records the harness passes around.

Everything here is a pure function: no file or network access, so it is easy to test.
Failed or crashed runs stay in every denominator; dropping them would flatter the numbers.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Callable, Optional

from pydantic import BaseModel, Field

from repopilot.orchestrator.state import Answer, Evidence


class EvalTask(BaseModel):
    task_id: str
    repo: str                        # "owner/name"
    base_commit: str
    problem_statement: str
    gold_files: list[str]            # files touched by the gold patch (repo-relative, forward slashes)
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)


class EvalRecord(BaseModel):
    task_id: str
    config: str                      # "bm25_only", "single_call", "full", ...
    run_idx: int                     # 0..repeats-1
    evidence: list[Evidence] = Field(default_factory=list)
    answer: Optional[Answer] = None  # None for retrieval-only runs and crashes
    resolved: Optional[bool] = None  # set by the harness grader, never by the system itself
    regressed: Optional[bool] = None
    error: Optional[str] = None
    latency_s: float = 0.0


# ---------------------------------------------------------------- retrieval

def ranked_files(evidence: list[Evidence], method: Optional[str] = None) -> list[str]:
    """Unique file paths, best score first.

    Only compare scores inside one retrieval method: BM25 and cosine scores are on
    different scales. For mixed evidence, make sure `score` holds the fused score.
    """
    items = [e for e in evidence if method is None or e.retrieval == method]
    items.sort(key=lambda e: e.score, reverse=True)  # stable sort keeps input order on ties
    seen: set[str] = set()
    out: list[str] = []
    for e in items:
        if e.citation.path not in seen:
            seen.add(e.citation.path)
            out.append(e.citation.path)
    return out


def recall_at_k(evidence: list[Evidence], gold_files: list[str], k: int,
                method: Optional[str] = None) -> Optional[float]:
    """Fraction of gold files found in the top-k files. None if there are no gold files."""
    gold = set(gold_files)
    if not gold:
        return None
    top = set(ranked_files(evidence, method)[:k])
    return len(top & gold) / len(gold)


def mean_recall_at_k(records: list[EvalRecord], tasks_by_id: dict[str, EvalTask], k: int,
                     method: Optional[str] = None) -> Optional[float]:
    vals = [recall_at_k(r.evidence, tasks_by_id[r.task_id].gold_files, k, method) for r in records]
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


# ------------------------------------------------------------------ outcomes

def resolve_rate(records: list[EvalRecord]) -> Optional[float]:
    """Crashes and ungraded runs count as unresolved."""
    return sum(bool(r.resolved) for r in records) / len(records) if records else None


def regression_rate(records: list[EvalRecord]) -> Optional[float]:
    """Among graded runs, the share that broke a previously passing test."""
    graded = [r for r in records if r.regressed is not None]
    return sum(bool(r.regressed) for r in graded) / len(graded) if graded else None


def cost_per_resolved(records: list[EvalRecord]) -> Optional[float]:
    """Total spend (failures included) divided by resolved tasks."""
    resolved = sum(bool(r.resolved) for r in records)
    total = sum(r.answer.cost_usd for r in records if r.answer)
    return total / resolved if resolved else None


# ------------------------------------------------------------------- repeats

def across_runs(records: list[EvalRecord],
                metric_fn: Callable[[list[EvalRecord]], Optional[float]]
                ) -> tuple[Optional[float], Optional[float]]:
    """Apply metric_fn to each repeat separately, return (mean, population std)."""
    by_run: dict[int, list[EvalRecord]] = defaultdict(list)
    for r in records:
        by_run[r.run_idx].append(r)
    vals = [v for v in (metric_fn(rs) for rs in by_run.values()) if v is not None]
    if not vals:
        return None, None
    return mean(vals), pstdev(vals)