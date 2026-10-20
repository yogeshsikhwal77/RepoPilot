"""Reverse call-graph traversal and test impact analysis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class ImpactSet:
    """Output of reverse traversal: changed symbols, upstream callers, and covering tests."""

    seed_symbols: list[str]
    impacted_symbols: list[str]
    covering_tests: list[str]
    hop_by_symbol: dict[str, int] = field(default_factory=dict)


class ImpactAnalyzer:
    """Computes upstream affected symbols and covering tests for a set of code changes."""

    def __init__(
        self,
        symbol_graph: Any,
        coverage_map: Any,
        max_hops: int = 2,
    ) -> None:
        self.symbol_graph = symbol_graph
        self.coverage_map = coverage_map
        self.max_hops = max_hops

    def _get_direct_callers(self, symbol: str) -> list[str]:
        """Adapts to SymbolGraph methods in src/repopilot/indexing/symbol_graph.py."""
        for method_name in ("get_callers", "callers_of", "callers"):
            fn = getattr(self.symbol_graph, method_name, None)
            if callable(fn):
                res = fn(symbol)
                return [
                    getattr(item, "name", None) or getattr(item, "id", None) or str(item)
                    for item in res
                ]

        if hasattr(self.symbol_graph, "reverse_edges") and isinstance(
            self.symbol_graph.reverse_edges, dict
        ):
            return list(self.symbol_graph.reverse_edges.get(symbol, []))
        return []

    def _get_tests(self, symbol: str) -> list[str]:
        """Adapts to CoverageMap methods in src/repopilot/indexing/coverage_map.py."""
        for method_name in ("get_tests_for_symbol", "tests_for", "get_covering_tests", "tests_for_symbol"):
            fn = getattr(self.coverage_map, method_name, None)
            if callable(fn):
                return [str(t) for t in fn(symbol)]

        if isinstance(self.coverage_map, dict):
            return [str(t) for t in self.coverage_map.get(symbol, [])]
        return []

    def compute_impact(
        self,
        seed_symbols: Iterable[str],
        max_hops: int | None = None,
    ) -> ImpactSet:
        """Runs reverse BFS from seed_symbols up the call graph and collects covering tests."""
        hop_limit = self.max_hops if max_hops is None else max_hops
        seeds = sorted({s for s in seed_symbols if s})

        visited: dict[str, int] = {s: 0 for s in seeds}
        queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)

        while queue:
            current_symbol, depth = queue.popleft()
            if depth >= hop_limit:
                continue

            for caller in sorted(self._get_direct_callers(current_symbol)):
                if caller and caller not in visited:
                    visited[caller] = depth + 1
                    queue.append((caller, depth + 1))

        impacted_symbols = sorted(visited.keys(), key=lambda s: (visited[s], s))
        tests: set[str] = set()
        for sym in impacted_symbols:
            for test_id in self._get_tests(sym):
                tests.add(test_id)

        return ImpactSet(
            seed_symbols=seeds,
            impacted_symbols=impacted_symbols,
            covering_tests=sorted(tests),
            hop_by_symbol=visited,
        )