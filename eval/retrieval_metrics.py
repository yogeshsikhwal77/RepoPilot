"""Evaluation metrics for code retrieval (recall@k at file and span levels)."""
from __future__ import annotations

import re
from typing import List, Literal, Sequence, Set

from pydantic import BaseModel, ConfigDict

from repopilot.orchestrator.state import Evidence
from repopilot.retrieval.hybrid import HybridResult


class GoldSpan(BaseModel):
    """Line span in a file touched by a gold patch."""

    model_config = ConfigDict(frozen=True)

    path: str
    start_line: int
    end_line: int


def _normalize_diff_path(raw_path: str) -> str:
    path = raw_path.strip()
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path.replace("\\", "/")


def gold_locations_from_patch(diff: str) -> list[GoldSpan]:
    """Extract old-file pre-image spans from unified-diff hunk headers."""
    spans: list[GoldSpan] = []
    current_path: str | None = None

    lines = diff.splitlines()
    for line in lines:
        if line.startswith("--- "):
            raw = line[4:].strip()
            if raw != "/dev/null":
                current_path = _normalize_diff_path(raw)
        elif line.startswith("+++ ") and current_path is None:
            raw = line[4:].strip()
            if raw != "/dev/null":
                current_path = _normalize_diff_path(raw)
        elif line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                current_path = _normalize_diff_path(parts[2])
        elif line.startswith("@@ ") and current_path:
            # e.g. @@ -10,4 +10,6 @@
            m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                if old_count == 0:
                    # Pure addition becomes a 1-line span at the anchor line
                    spans.append(
                        GoldSpan(
                            path=current_path,
                            start_line=old_start,
                            end_line=old_start,
                        )
                    )
                else:
                    spans.append(
                        GoldSpan(
                            path=current_path,
                            start_line=old_start,
                            end_line=old_start + old_count - 1,
                        )
                    )

    return spans


def recall_at_k(
    retrieved: Sequence[Evidence],
    gold: Sequence[GoldSpan],
    k: int,
    level: Literal["file", "span"] = "file",
) -> float:
    """Compute recall@k for retrieved evidence against gold ground-truth locations."""
    if not gold:
        raise ValueError("Gold locations cannot be empty for recall@k.")

    top_ev = retrieved[:k]

    if level == "file":
        gold_files = {g.path for g in gold}
        retrieved_files = {ev.citation.path for ev in top_ev}
        hits = len(gold_files & retrieved_files)
        return hits / len(gold_files)

    elif level == "span":
        hits = 0
        for g in gold:
            is_hit = False
            for ev in top_ev:
                if ev.citation.path == g.path:
                    # Check line interval overlap
                    if max(ev.citation.start_line, g.start_line) <= min(
                        ev.citation.end_line, g.end_line
                    ):
                        is_hit = True
                        break
            if is_hit:
                hits += 1
        return hits / len(gold)

    raise ValueError(f"Unknown recall level: {level}")


def recall_by_method(
    detailed: HybridResult,
    gold: Sequence[GoldSpan],
    k: int,
    level: Literal["file", "span"] = "file",
) -> dict[str, float]:
    """Compute recall@k for each individual retrieval method and fused results."""
    return {
        "bm25": recall_at_k(detailed.bm25, gold, k=k, level=level),
        "vector": recall_at_k(detailed.vector, gold, k=k, level=level),
        "fused": recall_at_k(detailed.fused, gold, k=k, level=level),
    }
