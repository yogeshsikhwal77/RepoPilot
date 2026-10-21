"""B1 Baseline: Plain RAG with dense vector retrieval, single LLM call, no graph, no verifier."""
from __future__ import annotations

import logging
import re
import time
from typing import Any, List, Optional, Protocol, Sequence

from eval.metrics import EvalRecord, EvalTask
from repopilot.orchestrator.state import Answer, Evidence, Patch, TestReport
from repopilot.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

PLAIN_RAG_PROMPT_TEMPLATE = """You are an expert software engineer resolving an issue in a repository.

### Problem Statement
{problem_statement}

### Retrieved Context Code
{context}

### Instructions
1. Analyze the problem statement and context carefully.
2. Produce a valid unified git diff that resolves the issue.
3. Output ONLY the git diff enclosed inside a ```diff code block.
4. Do not include commentary outside the diff block.

```diff
"""


class Baseline(Protocol):
    """Protocol for standalone baselines producing a final Answer."""

    def run(self, problem_statement: str) -> Answer:
        ...


def extract_patches_from_diff_text(diff_text: str, default_commit: str = "HEAD") -> list[Patch]:
    """Parse fenced or raw unified diff text into individual Patch objects per file."""
    # 1. Extract content inside ```diff ... ``` if present
    m = re.search(r"```diff\n(.*?)```", diff_text, flags=re.DOTALL)
    if m:
        content = m.group(1).strip()
    elif "diff --git" in diff_text:
        content = diff_text[diff_text.find("diff --git") :].strip()
    elif "--- " in diff_text and "+++ " in diff_text:
        content = diff_text[diff_text.find("--- ") :].strip()
    else:
        return []

    # 2. Split into per-file patches
    patches: list[Patch] = []
    file_chunks = re.split(r"(?=diff --git |(?<!\w)--- (?:a/|[^\n]+))", content)

    for chunk in file_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # Find file path from +++ b/... or --- a/... or diff --git a/... b/...
        file_path: Optional[str] = None
        for line in chunk.splitlines():
            if line.startswith("+++ "):
                target = line[4:].strip()
                if target.startswith("b/"):
                    target = target[2:]
                if target != "/dev/null":
                    file_path = target
                    break
            elif line.startswith("--- ") and not file_path:
                target = line[4:].strip()
                if target.startswith("a/"):
                    target = target[2:]
                if target != "/dev/null":
                    file_path = target

        if file_path:
            norm_path = file_path.replace("\\", "/")
            patches.append(
                Patch(
                    filepath=norm_path,
                    diff=chunk,
                    base_commit=default_commit,
                )
            )

    return patches


class PlainRag:
    """Plain RAG baseline: dense retrieval -> single LLM generation -> Answer."""

    name: str = "plain_rag"
    fingerprint: str = "v1"

    def __init__(
        self,
        hybrid: HybridRetriever,
        llm: Any,
        k: int = 8,
        model_name: str = "gpt-4o-mini",
    ) -> None:
        self.hybrid = hybrid
        self.llm = llm
        self.k = k
        self.model_name = model_name

        # Enforce dense-only vector mode
        if self.hybrid.mode != "vector":
            self.hybrid.mode = "vector"

    def run(self, problem_statement: str) -> Answer:
        """Run dense vector retrieval and single LLM call to produce an Answer."""
        start_time = time.perf_counter()

        # 1. Dense retrieval only
        evidence_list = self.hybrid.search(problem_statement, k=self.k)

        # 2. Format context
        context_blocks = []
        default_commit = "HEAD"
        for ev in evidence_list:
            if ev.citation.commit:
                default_commit = ev.citation.commit
            context_blocks.append(
                f"# File: {ev.citation.path} (lines {ev.citation.start_line}-{ev.citation.end_line})\n"
                f"{ev.content}\n"
            )
        context_text = "\n".join(context_blocks) if context_blocks else "# No context retrieved.\n"

        prompt = PLAIN_RAG_PROMPT_TEMPLATE.format(
            problem_statement=problem_statement,
            context=context_text,
        )

        tokens = 0
        cost_usd = 0.0
        patches: list[Patch] = []

        # 3. Single LLM call
        try:
            resp = self.llm.generate(
                model=self.model_name,
                prompt=prompt,
                temperature=0.0,
                max_tokens=2048,
            )
            raw_text = resp.get("text", "") if hasattr(resp, "get") else getattr(resp, "text", str(resp))
            p_tok = resp.get("prompt_tokens", 0) if hasattr(resp, "get") else getattr(resp, "prompt_tokens", 0)
            c_tok = resp.get("completion_tokens", 0) if hasattr(resp, "get") else getattr(resp, "completion_tokens", 0)
            tokens = p_tok + c_tok
            cost_usd = resp.get("cost_usd", 0.0) if hasattr(resp, "get") else getattr(resp, "cost_usd", 0.0)

            # 4. Parse diff(s)
            patches = extract_patches_from_diff_text(raw_text, default_commit=default_commit)

        except Exception as exc:
            logger.warning("PlainRag LLM invocation failed: %s", exc)
            patches = []

        return Answer(
            diffs=patches,
            test_report=TestReport(passed=False, total_run=0),
            verified_claims=[],
            cost_usd=round(cost_usd, 6),
            tokens=tokens,
        )

    def run_task(self, task: EvalTask, run_idx: int = 0) -> EvalRecord:
        """Adapter method for eval/run_eval.py harness."""
        start_time = time.perf_counter()
        answer = self.run(task.problem_statement)
        latency = time.perf_counter() - start_time

        # Retrieve evidence for record
        evidence_list = self.hybrid.search(task.problem_statement, k=self.k)

        return EvalRecord(
            task_id=task.task_id,
            config=self.name,
            run_idx=run_idx,
            evidence=evidence_list,
            answer=answer,
            latency_s=round(latency, 3),
        )
