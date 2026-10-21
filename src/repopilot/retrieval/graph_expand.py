"""Call-graph expansion of retrieved code seeds using SymbolGraph."""
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field

from repopilot.indexing.symbol_graph import CONFIDENCE_RANK, SymbolGraph
from repopilot.orchestrator.state import Chunk, Citation, Evidence

logger = logging.getLogger(__name__)


class ExpandConfig(BaseModel):
    """Configuration for call-graph expansion."""

    enabled: bool = True  # Ablation switch (no_graph)
    hops: int = 2
    max_nodes: int = 20
    decay: float = 0.5
    worst_conf: Literal["exact", "heuristic", "ambiguous"] = "heuristic"
    kinds: tuple[str, ...] = ("calls", "inherits")
    direction: Literal["out", "in", "both"] = "both"
    conf_weight: dict[str, float] = Field(
        default_factory=lambda: {"exact": 1.0, "heuristic": 0.6, "ambiguous": 0.3}
    )
    include_overrides: bool = False

    @classmethod
    def from_yaml(cls, path: Union[str, Path] = "configs/retrieval.yaml") -> ExpandConfig:
        cfg_path = Path(path)
        hops = 2
        worst_conf = "heuristic"
        max_nodes = 20
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                # Support both top-level and nested sections
                g_sec = data.get("graph", data)
                if isinstance(g_sec, dict):
                    hops = g_sec.get("hop_limit", g_sec.get("max_hops", hops))
                    worst_conf = g_sec.get("graph_worst_confidence", worst_conf)
                    max_nodes = g_sec.get("max_graph_nodes", max_nodes)
            except Exception as exc:
                logger.warning("Could not read graph config from %s: %s", cfg_path, exc)
        return cls(hops=hops, worst_conf=worst_conf, max_nodes=max_nodes)


def _find_subclasses(graph: SymbolGraph, cls_q: str) -> list[str]:
    """Find all classes that directly or indirectly inherit from cls_q."""
    subclasses: list[str] = []
    visited: set[str] = set()
    queue = deque([cls_q])
    while queue:
        parent = queue.popleft()
        for e in graph.in_edges.get(parent, ()):
            if e.kind == "inherits" and e.src not in visited:
                visited.add(e.src)
                subclasses.append(e.src)
                queue.append(e.src)
    return subclasses


def expand(
    graph: SymbolGraph,
    seeds: Sequence[Evidence],
    cfg: Optional[ExpandConfig] = None,
) -> list[Evidence]:
    """Expand retrieved Evidence seeds along the call graph up to hop limit.

    Maps each seed chunk to a symbol via graph.symbol_at, performs weighted BFS
    tracking edge confidence decay, and returns ranked Evidence chunks.
    """
    if cfg is None:
        cfg = ExpandConfig.from_yaml()

    if not cfg.enabled or not seeds:
        return []

    seed_chunk_ids: set[str] = {s.chunk_id for s in seeds}
    worst_limit = CONFIDENCE_RANK[cfg.worst_conf]

    # Map each seed to its symbol qualname
    seed_symbols: dict[str, float] = {}  # qualname -> best seed score
    for s in seeds:
        q = graph.symbol_at(s.citation.path, s.citation.start_line)
        if q:
            seed_symbols[q] = max(seed_symbols.get(q, 0.0), s.score)

    if not seed_symbols:
        return []

    # BFS from all seed symbols
    best_scores: dict[str, float] = {}
    best_hops: dict[str, int] = {}

    for start_q, start_score in seed_symbols.items():
        # queue item: (node_qualname, current_hop, weakest_confidence_on_path)
        queue: deque[tuple[str, int, str]] = deque([(start_q, 0, "exact")])
        visited_dist: dict[str, int] = {start_q: 0}

        while queue:
            curr, hop, weakest_conf = queue.popleft()

            if curr != start_q:
                conf_w = cfg.conf_weight.get(weakest_conf, 1.0)
                path_score = start_score * (cfg.decay ** hop) * conf_w
                if curr not in best_scores or path_score > best_scores[curr]:
                    best_scores[curr] = path_score
                    best_hops[curr] = hop

            if hop >= cfg.hops:
                continue

            # Traverse outgoing and incoming edges per config
            next_edges = []
            if cfg.direction in ("out", "both"):
                next_edges.extend(
                    [
                        (e.dst, e.conf)
                        for e in graph.out_edges.get(curr, ())
                        if e.kind in cfg.kinds and CONFIDENCE_RANK[e.conf] <= worst_limit
                    ]
                )
            if cfg.direction in ("in", "both"):
                next_edges.extend(
                    [
                        (e.src, e.conf)
                        for e in graph.in_edges.get(curr, ())
                        if e.kind in cfg.kinds and CONFIDENCE_RANK[e.conf] <= worst_limit
                    ]
                )

            for nxt, edge_conf in next_edges:
                nxt_weakest = (
                    edge_conf
                    if CONFIDENCE_RANK[edge_conf] > CONFIDENCE_RANK[weakest_conf]
                    else weakest_conf
                )
                if nxt not in visited_dist or hop + 1 < visited_dist[nxt]:
                    visited_dist[nxt] = hop + 1
                    queue.append((nxt, hop + 1, nxt_weakest))

    # Optional: include subclass method overrides (both reached methods and seed methods)
    if cfg.include_overrides:
        override_scores: dict[str, float] = {}
        for q, sc in list(best_scores.items()) + list(seed_symbols.items()):
            sym = graph.symbols.get(q)
            if sym and sym.kind == "method" and "." in sym.local_name:
                cls_q = sym.parent
                method_name = sym.local_name.split(".")[-1]
                subclasses = _find_subclasses(graph, cls_q)
                for sub_cls in subclasses:
                    sub_m_q = f"{sub_cls}.{method_name}"
                    if sub_m_q in graph.symbols and sub_m_q not in best_scores:
                        override_scores[sub_m_q] = sc
        best_scores.update(override_scores)

    # Convert reached symbols to Evidence
    candidate_evidence: dict[str, tuple[Evidence, float, str]] = {}

    for q, score in best_scores.items():
        if q in seed_symbols:
            continue
        sym = graph.symbols.get(q)
        if not sym:
            continue
        chunks = graph.chunks_for(q)
        for chunk in chunks:
            if chunk.id in seed_chunk_ids:
                continue
            ev = Evidence(
                chunk_id=chunk.id,
                citation=Citation(
                    path=chunk.path,
                    commit=chunk.commit,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                ),
                content=chunk.text,
                source_type="test" if "test" in chunk.path else "code",
                retrieval="graph",
                score=round(score, 6),
            )
            # Deduplicate by chunk_id keeping highest score
            if chunk.id not in candidate_evidence or score > candidate_evidence[chunk.id][1]:
                candidate_evidence[chunk.id] = (ev, score, q)

    # Order by score descending, tie-break by qualname then chunk_id
    sorted_items = sorted(
        candidate_evidence.values(),
        key=lambda item: (-item[1], item[2], item[0].chunk_id),
    )

    return [item[0] for item in sorted_items[: cfg.max_nodes]]
