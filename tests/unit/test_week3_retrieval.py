"""Unit tests for Yogesh's Week 3 deliverables: reranker, impact, and navigator."""

from __future__ import annotations

from repopilot.agents.navigator import NavigatorAgent
from repopilot.retrieval.impact import ImpactAnalyzer
from repopilot.retrieval.reranker import CandidateChunk, Reranker


class DummySymbolGraph:
    """Mock AST call graph mimicking calculator.py relationships."""

    def __init__(self) -> None:
        self.reverse_edges = {
            "divide": ["calculate_ratio", "run_batch"],
            "calculate_ratio": ["main_pipeline"],
        }
        self.symbols = {
            "calculate_ratio": {
                "file_path": "calculator.py",
                "start_line": 20,
                "end_line": 28,
                "content": "def calculate_ratio(a, b):\n    return divide(a, b)",
            },
            "run_batch": {
                "file_path": "calculator.py",
                "start_line": 30,
                "end_line": 42,
                "content": "def run_batch(pairs):\n    return [divide(x, y) for x, y in pairs]",
            },
            "main_pipeline": {
                "file_path": "app.py",
                "start_line": 5,
                "end_line": 15,
                "content": "def main_pipeline(data):\n    return calculate_ratio(*data)",
            },
        }

    def get_callers(self, symbol: str) -> list[str]:
        return self.reverse_edges.get(symbol, [])

    def get_symbol(self, symbol: str) -> dict | None:
        return self.symbols.get(symbol)


class DummyCoverageMap:
    """Mock coverage map linking calculator symbols to pytest functions."""

    def __init__(self) -> None:
        self.mapping = {
            "divide": ["tests/test_calculator.py::test_divide_by_zero"],
            "calculate_ratio": ["tests/test_calculator.py::test_calculate_ratio"],
            "main_pipeline": ["tests/test_app.py::test_main_pipeline"],
        }

    def get_tests_for_symbol(self, symbol: str) -> list[str]:
        return self.mapping.get(symbol, [])


def test_reranker_prioritizes_exact_symbol_and_closer_hops() -> None:
    reranker = Reranker()
    query = "ZeroDivisionError in divide function inside calculator.py"

    noisy_chunk = CandidateChunk(
        chunk_id="c1",
        file_path="utils/Formatting.py",
        symbol_name="format_output",
        content="def format_output(val): return str(val)",
        start_line=1,
        end_line=5,
        base_score=0.55,
        hop_distance=2,
    )
    target_chunk = CandidateChunk(
        chunk_id="c2",
        file_path="calculator.py",
        symbol_name="divide",
        content="def divide(a, b):\n    return a / b",
        start_line=10,
        end_line=14,
        base_score=0.50,
        hop_distance=0,
    )

    ranked = reranker.rerank(query, [noisy_chunk, target_chunk], top_k=2)
    assert len(ranked) == 2
    assert ranked[0].symbol_name == "divide"
    assert ranked[0].base_score > ranked[1].base_score


def test_impact_analyzer_reverse_bfs_and_covering_tests() -> None:
    graph = DummySymbolGraph()
    cov = DummyCoverageMap()
    analyzer = ImpactAnalyzer(symbol_graph=graph, coverage_map=cov, max_hops=2)

    impact = analyzer.compute_impact(["divide"])

    assert impact.seed_symbols == ["divide"]
    # Hop 0: divide, Hop 1: calculate_ratio & run_batch, Hop 2: main_pipeline
    assert set(impact.impacted_symbols) == {
        "divide",
        "calculate_ratio",
        "run_batch",
        "main_pipeline",
    }
    assert impact.hop_by_symbol["divide"] == 0
    assert impact.hop_by_symbol["calculate_ratio"] == 1
    assert impact.hop_by_symbol["main_pipeline"] == 2
    assert set(impact.covering_tests) == {
        "tests/test_calculator.py::test_divide_by_zero",
        "tests/test_calculator.py::test_calculate_ratio",
        "tests/test_app.py::test_main_pipeline",
    }


def test_navigator_agent_produces_citations_and_impact_set() -> None:
    graph = DummySymbolGraph()
    cov = DummyCoverageMap()
    navigator = NavigatorAgent(
        symbol_graph=graph,
        coverage_map=cov,
        max_hops=2,
        repo_name="sample_repo",
        commit_sha="ec14b64",
    )

    seed = CandidateChunk(
        chunk_id="seed1",
        file_path="calculator.py",
        symbol_name="divide",
        content="def divide(a, b):\n    return a / b",
        start_line=10,
        end_line=14,
        base_score=0.80,
        hop_distance=0,
    )

    report = navigator.navigate(
        query="fix divide by zero in calculator.py",
        seed_candidates=[seed],
        top_k=5,
    )

    assert len(report.evidence) >= 2
    assert (
        report.evidence[0].citation
        == "sample_repo@ec14b64:calculator.py:10-14"
    )
    assert "tests/test_calculator.py::test_divide_by_zero" in report.impact_set.covering_tests