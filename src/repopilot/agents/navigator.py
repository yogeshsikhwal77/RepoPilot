"""Navigator specialist agent for call-graph exploration and test impact discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from repopilot.orchestrator.state import Chunk, Citation, Evidence
from repopilot.retrieval.impact import ImpactAnalyzer, ImpactSet
from repopilot.retrieval.reranker import Reranker


@dataclass(frozen=True)
class NavigationReport:
    """Output of the Navigator agent: reranked Evidence list + ImpactSet."""

    query: str
    evidence: list[Evidence]
    impact_set: ImpactSet
    metadata: dict[str, Any] = field(default_factory=dict)


class NavigatorAgent:
    """Answers 'Who calls this?' and 'Which tests cover this?' using AST graph + coverage map."""

    def __init__(
        self,
        symbol_graph: Any,
        coverage_map: Any,
        chunks: Sequence[Chunk] | None = None,
        reranker: Reranker | None = None,
        max_hops: int = 2,
        commit: str = "HEAD",
    ) -> None:
        self.symbol_graph = symbol_graph
        self.coverage_map = coverage_map
        self.chunks = list(chunks) if chunks else []
        self.reranker = reranker or Reranker()
        self.impact_analyzer = ImpactAnalyzer(
            symbol_graph=symbol_graph,
            coverage_map=coverage_map,
            max_hops=max_hops,
        )
        self.commit = commit

        # Index chunks by symbol name and by chunk_id for fast lookup
        self._chunk_by_id: dict[str, Chunk] = {c.id: c for c in self.chunks}
        self._chunks_by_symbol: dict[str, list[Chunk]] = {}
        for c in self.chunks:
            sym = self._extract_symbol_name(c.text, getattr(c, "symbol", ""))
            if sym:
                self._chunks_by_symbol.setdefault(sym, []).append(c)

    @staticmethod
    def _extract_symbol_name(text: str, fallback: str = "") -> str:
        if fallback:
            return fallback
        m = re.search(r"\b(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", text)
        return m.group(1) if m else ""

    def navigate(
        self,
        query: str,
        seed_evidence: Iterable[Evidence],
        top_k: int = 20,
    ) -> NavigationReport:
        """Expands seed Evidence across the call graph, computes test impact, and reranks."""
        seeds = list(seed_evidence)
        seed_symbols: list[str] = []

        chunk_scores: dict[str, tuple[Chunk, float]] = {}
        hop_distances: dict[str, int] = {}

        for ev in seeds:
            sym = self._extract_symbol_name(ev.content)
            if sym:
                seed_symbols.append(sym)
            chunk = self._chunk_by_id.get(
                ev.chunk_id,
                Chunk(
                    id=ev.chunk_id,
                    path=ev.citation.path,
                    commit=ev.citation.commit,
                    start_line=ev.citation.start_line,
                    end_line=ev.citation.end_line,
                    text=ev.content,
                ),
            )
            chunk_scores[chunk.id] = (chunk, ev.score)
            hop_distances[chunk.id] = 0

        # 1. Reverse traversal to find upstream callers and covering tests
        impact_set = self.impact_analyzer.compute_impact(seed_symbols)

        # 2. Expand candidate pool with chunks of impacted callers
        base_seed_score = max((ev.score for ev in seeds), default=0.5)
        for sym in impact_set.impacted_symbols:
            hop = impact_set.hop_by_symbol.get(sym, 1)
            if hop == 0:
                continue
            for caller_chunk in self._chunks_by_symbol.get(sym, []):
                if caller_chunk.id not in chunk_scores:
                    chunk_scores[caller_chunk.id] = (caller_chunk, base_seed_score * 0.8)
                    hop_distances[caller_chunk.id] = hop

        # 3. Rerank all candidates (seeds + graph-expanded callers)
        reranked = self.reranker.rerank_chunks(
            query=query,
            chunk_scores=list(chunk_scores.values()),
            hop_distances=hop_distances,
            top_k=top_k,
        )

        # 4. Package back into typed state.Evidence objects
        final_evidence: list[Evidence] = []
        for chunk, score in reranked:
            hop = hop_distances.get(chunk.id, 0)
            final_evidence.append(
                Evidence(
                    chunk_id=chunk.id,
                    citation=Citation(
                        path=chunk.path,
                        commit=chunk.commit,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                    ),
                    content=chunk.text,
                    source_type="code",
                    retrieval="bm25" if hop == 0 else "graph",
                    score=score,
                )
            )

        return NavigationReport(
            query=query,
            evidence=final_evidence,
            impact_set=impact_set,
            metadata={
                "seed_count": len(seeds),
                "expanded_count": len(chunk_scores),
                "covering_tests": impact_set.covering_tests,
            },
        )