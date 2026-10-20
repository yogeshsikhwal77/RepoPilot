"""Reranker for filtering and ordering hybrid + graph retrieval candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from repopilot.orchestrator.state import Chunk


@dataclass(frozen=True)
class CandidateChunk:
    """Represents a retrieved code candidate prior to or after reranking."""

    chunk_id: str
    file_path: str
    symbol_name: str
    content: str
    start_line: int
    end_line: int
    base_score: float = 0.0
    hop_distance: int = 0
    commit_sha: str = "HEAD"
    metadata: dict[str, Any] = field(default_factory=dict)


class Reranker:
    """Scores and re-orders retrieved candidates using structural and lexical signals."""

    def __init__(
        self,
        exact_symbol_boost: float = 0.35,
        path_match_boost: float = 0.25,
        call_distance_decay: float = 0.12,
        min_score_threshold: float = 0.0,
    ) -> None:
        self.exact_symbol_boost = exact_symbol_boost
        self.path_match_boost = path_match_boost
        self.call_distance_decay = call_distance_decay
        self.min_score_threshold = min_score_threshold

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Splits camelCase, snake_case, and punctuation into lowercase tokens."""
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        tokens = re.findall(r"[a-zA-Z0-9_]+", spaced.lower())
        expanded: set[str] = set()
        for tok in tokens:
            if len(tok) > 1:
                expanded.add(tok)
            for sub in tok.split("_"):
                if len(sub) > 1:
                    expanded.add(sub)
        return expanded

    def score_candidate(self, query: str, candidate: CandidateChunk) -> float:
        """Computes a composite relevance score for a single candidate chunk."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return candidate.base_score

        symbol_tokens = self._tokenize(candidate.symbol_name)
        path_tokens = self._tokenize(candidate.file_path)
        content_tokens = self._tokenize(candidate.content)

        # 1. Exact or token-level symbol name boost
        symbol_overlap = len(query_tokens & symbol_tokens) / max(len(symbol_tokens), 1)
        exact_symbol = (
            1.0
            if candidate.symbol_name
            and len(candidate.symbol_name) > 2
            and candidate.symbol_name.lower() in query.lower()
            else symbol_overlap
        )
        symbol_bonus = self.exact_symbol_boost * exact_symbol

        # 2. File/module path match boost
        path_overlap = len(query_tokens & path_tokens) / max(len(path_tokens), 1)
        path_bonus = self.path_match_boost * path_overlap

        # 3. Content token overlap
        content_overlap = len(query_tokens & content_tokens) / len(query_tokens)
        lexical_bonus = 0.25 * content_overlap

        # 4. Hop distance penalty (direct hits favored over distant graph neighbors)
        hop_penalty = self.call_distance_decay * max(candidate.hop_distance, 0)

        final_score = (
            candidate.base_score
            + symbol_bonus
            + path_bonus
            + lexical_bonus
            - hop_penalty
        )
        return round(max(final_score, 0.0), 6)

    def rerank(
        self,
        query: str,
        candidates: Iterable[CandidateChunk],
        top_k: int = 10,
    ) -> list[CandidateChunk]:
        """Deduplicates, scores, filters, and returns the top-k CandidateChunks."""
        deduped: dict[tuple[str, int, int], CandidateChunk] = {}
        for cand in candidates:
            key = (cand.file_path, cand.start_line, cand.end_line)
            if key not in deduped or cand.base_score > deduped[key].base_score:
                deduped[key] = cand

        scored: list[CandidateChunk] = []
        for cand in deduped.values():
            new_score = self.score_candidate(query, cand)
            if new_score >= self.min_score_threshold:
                scored.append(
                    CandidateChunk(
                        chunk_id=cand.chunk_id,
                        file_path=cand.file_path,
                        symbol_name=cand.symbol_name,
                        content=cand.content,
                        start_line=cand.start_line,
                        end_line=cand.end_line,
                        base_score=new_score,
                        hop_distance=cand.hop_distance,
                        commit_sha=cand.commit_sha,
                        metadata={**cand.metadata, "raw_score": cand.base_score},
                    )
                )

        scored.sort(
            key=lambda c: (-c.base_score, c.hop_distance, c.file_path, c.start_line)
        )
        return scored[: max(top_k, 0)]

    def rerank_chunks(
        self,
        query: str,
        chunk_scores: Sequence[tuple[Chunk, float]],
        hop_distances: dict[str, int] | None = None,
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """Directly reranks state.Chunk objects used by run_eval.py."""
        hops = hop_distances or {}
        by_id: dict[str, Chunk] = {}
        candidates: list[CandidateChunk] = []

        for chunk, score in chunk_scores:
            by_id[chunk.id] = chunk
            symbol_name = getattr(chunk, "symbol", None) or getattr(chunk, "name", "") or ""
            if not symbol_name:
                # Extract def/class name from the first line of chunk.text if present
                m = re.search(r"\b(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", chunk.text)
                symbol_name = m.group(1) if m else ""

            candidates.append(
                CandidateChunk(
                    chunk_id=chunk.id,
                    file_path=chunk.path,
                    symbol_name=symbol_name,
                    content=chunk.text,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    base_score=score,
                    hop_distance=hops.get(chunk.id, 0),
                    commit_sha=chunk.commit,
                )
            )

        reranked = self.rerank(query, candidates, top_k=top_k)
        return [(by_id[c.chunk_id], c.base_score) for c in reranked if c.chunk_id in by_id]