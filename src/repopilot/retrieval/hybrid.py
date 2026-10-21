"""Hybrid retrieval combining lexical (BM25) and dense vector search with Reciprocal Rank Fusion."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, List, Literal, Mapping, Optional, Protocol, Sequence, Tuple, Union

import yaml

from repopilot.orchestrator.state import Chunk, Citation, Evidence

logger = logging.getLogger(__name__)


class LexicalIndex(Protocol):
    """Protocol for BM25 / keyword index."""

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Search lexical index, returning (chunk_id, score) pairs ordered best-first."""
        ...


class DenseIndex(Protocol):
    """Protocol for vector embeddings index."""

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Search dense index, returning (chunk_id, score) pairs ordered best-first."""
        ...


def is_test_file(path: str) -> bool:
    """Return True if path represents a test file or is inside a test directory."""
    p = PurePosixPath(path)
    return (
        any(part in {"tests", "test"} for part in p.parts[:-1])
        or p.name.startswith("test_")
        or p.name.endswith("_test.py")
    )


def chunk_to_evidence(
    chunk: Chunk,
    retrieval: Literal["bm25", "vector", "graph", "impact"],
    score: float,
) -> Evidence:
    """Convert an indexed Chunk into a typed Evidence model."""
    return Evidence(
        chunk_id=chunk.id,
        citation=Citation(
            path=chunk.path,
            commit=chunk.commit,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        ),
        content=chunk.text,
        source_type="test" if is_test_file(chunk.path) else "code",
        retrieval=retrieval,
        score=score,
    )


@dataclass(frozen=True)
class HybridResult:
    """Detailed retrieval result containing fused, bm25, and vector candidates."""

    fused: list[Evidence]
    bm25: list[Evidence]
    vector: list[Evidence]


class HybridRetriever:
    """Hybrid code retriever fusing lexical and dense vectors via RRF."""

    def __init__(
        self,
        chunks: Mapping[str, Chunk],
        lexical: LexicalIndex,
        dense: DenseIndex,
        *,
        mode: Literal["hybrid", "bm25", "vector"] = "hybrid",
        rrf_k: int = 60,
        w_lexical: float = 1.0,
        w_dense: float = 1.0,
        candidate_k: int = 50,
        config_path: Optional[Union[str, Path]] = "configs/retrieval.yaml",
    ) -> None:
        self.chunks = chunks
        self.lexical = lexical
        self.dense = dense
        self.mode = mode
        self.candidate_k = candidate_k

        # Load weights and rrf_k from config if present
        cfg_rrf_k = None
        cfg_w_lex = None
        cfg_w_dense = None

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                # Check for "fusion weights" or "retrieval.weights"
                ret_sec = data.get("retrieval", data)
                if isinstance(ret_sec, dict):
                    cfg_rrf_k = ret_sec.get("fusion_k", ret_sec.get("rrf_k"))
                    w = ret_sec.get("fusion_weights", ret_sec.get("weights", {}))
                    if isinstance(w, dict):
                        cfg_w_lex = w.get("bm25", w.get("lexical"))
                        cfg_w_dense = w.get("vector", w.get("dense"))
            except Exception as exc:
                logger.warning("Could not load fusion config from %s: %s", config_path, exc)

        self.rrf_k = cfg_rrf_k if cfg_rrf_k is not None else rrf_k
        self.w_lexical = cfg_w_lex if cfg_w_lex is not None else w_lexical
        self.w_dense = cfg_w_dense if cfg_w_dense is not None else w_dense

    def search(self, query: str, k: int = 10) -> list[Evidence]:
        """Search chunks according to configured mode and return top-k Evidence."""
        return self.search_many([query], k=k)

    def search_detailed(self, query: str, k: int = 10) -> HybridResult:
        """Search and return fused, bm25-only, and vector-only top-k Evidence lists."""
        fused = self.search(query, k=k)

        bm25_hits = self.lexical.search(query, k=k)
        bm25_ev = []
        for cid, sc in bm25_hits:
            if cid in self.chunks:
                bm25_ev.append(chunk_to_evidence(self.chunks[cid], "bm25", sc))

        dense_hits = self.dense.search(query, k=k)
        dense_ev = []
        for cid, sc in dense_hits:
            if cid in self.chunks:
                dense_ev.append(chunk_to_evidence(self.chunks[cid], "vector", sc))

        return HybridResult(fused=fused, bm25=bm25_ev, vector=dense_ev)

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[Evidence]:
        """Perform search across multiple queries and fuse results using RRF."""
        if not queries:
            return []

        # Single-mode shortcuts
        if self.mode == "bm25":
            return self._search_single_mode(self.lexical, "bm25", queries, k)
        elif self.mode == "vector":
            return self._search_single_mode(self.dense, "vector", queries, k)

        # Hybrid RRF fusion across queries
        rrf_scores: dict[str, float] = {}
        best_lex_rank: dict[str, int] = {}
        best_dense_rank: dict[str, int] = {}

        for q in queries:
            lex_hits = self.lexical.search(q, k=self.candidate_k)
            dense_hits = self.dense.search(q, k=self.candidate_k)

            for rank, (cid, _) in enumerate(lex_hits, start=1):
                if cid not in self.chunks:
                    logger.warning("Unknown chunk_id '%s' in lexical search; skipping.", cid)
                    continue
                score = self.w_lexical / (self.rrf_k + rank)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score
                if cid not in best_lex_rank or rank < best_lex_rank[cid]:
                    best_lex_rank[cid] = rank

            for rank, (cid, _) in enumerate(dense_hits, start=1):
                if cid not in self.chunks:
                    logger.warning("Unknown chunk_id '%s' in dense search; skipping.", cid)
                    continue
                score = self.w_dense / (self.rrf_k + rank)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score
                if cid not in best_dense_rank or rank < best_dense_rank[cid]:
                    best_dense_rank[cid] = rank

        # Sort by (-rrf_score, chunk_id) for deterministic tie-breaking
        sorted_cids = sorted(
            rrf_scores.keys(),
            key=lambda cid: (-rrf_scores[cid], cid),
        )

        results: list[Evidence] = []
        for cid in sorted_cids[:k]:
            chunk = self.chunks[cid]
            lex_r = best_lex_rank.get(cid, float("inf"))
            dense_r = best_dense_rank.get(cid, float("inf"))
            # Label rule: method that ranked better; tie -> "bm25"
            retrieval_method: Literal["bm25", "vector"] = (
                "vector" if dense_r < lex_r else "bm25"
            )
            score = round(rrf_scores[cid], 6)
            results.append(chunk_to_evidence(chunk, retrieval_method, score))

        return results

    def _search_single_mode(
        self,
        index: Union[LexicalIndex, DenseIndex],
        method: Literal["bm25", "vector"],
        queries: Sequence[str],
        k: int,
    ) -> list[Evidence]:
        """Search a single index across queries, breaking ties deterministically."""
        scores: dict[str, float] = {}
        for q in queries:
            hits = index.search(q, k=k)
            for cid, sc in hits:
                if cid not in self.chunks:
                    logger.warning("Unknown chunk_id '%s' in %s search; skipping.", cid, method)
                    continue
                if cid not in scores or sc > scores[cid]:
                    scores[cid] = sc

        sorted_cids = sorted(scores.keys(), key=lambda cid: (-scores[cid], cid))
        results: list[Evidence] = []
        for cid in sorted_cids[:k]:
            results.append(chunk_to_evidence(self.chunks[cid], method, scores[cid]))
        return results
