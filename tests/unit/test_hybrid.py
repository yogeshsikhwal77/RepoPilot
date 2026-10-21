import logging
import pytest

from repopilot.orchestrator.state import Chunk
from repopilot.retrieval.hybrid import (
    DenseIndex,
    HybridResult,
    HybridRetriever,
    LexicalIndex,
    chunk_to_evidence,
    is_test_file,
)


class FakeLexical(LexicalIndex):
    def __init__(self, mapping: dict[str, list[tuple[str, float]]]):
        self.mapping = mapping

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.mapping.get(query, [])[:k]


class FakeDense(DenseIndex):
    def __init__(self, mapping: dict[str, list[tuple[str, float]]]):
        self.mapping = mapping

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.mapping.get(query, [])[:k]


@pytest.fixture
def sample_chunks():
    return {
        "c1": Chunk(id="c1", path="pkg/a.py", start_line=1, end_line=10, symbol="A", text="class A:", commit="rev1"),
        "c2": Chunk(id="c2", path="pkg/b.py", start_line=1, end_line=5, symbol="B", text="def b():", commit="rev1"),
        "c3": Chunk(id="c3", path="tests/test_a.py", start_line=1, end_line=10, symbol="test_a", text="def test():", commit="rev1"),
    }


def test_chunk_to_evidence(sample_chunks):
    ev_code = chunk_to_evidence(sample_chunks["c1"], "bm25", 0.95)
    assert ev_code.chunk_id == "c1"
    assert ev_code.source_type == "code"
    assert ev_code.retrieval == "bm25"
    assert str(ev_code.citation) == "pkg/a.py@rev1:1-10"

    ev_test = chunk_to_evidence(sample_chunks["c3"], "vector", 0.85)
    assert ev_test.source_type == "test"
    assert is_test_file("tests/test_a.py")
    assert not is_test_file("pkg/a.py")


def test_hand_computed_rrf_and_label_rule(sample_chunks):
    # c1: lex rank 1, dense rank 2
    # c2: lex rank 2, dense rank 1
    # rrf_k = 60, w_lex = 1.0, w_dense = 1.0
    # c1 score: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.032522
    # c2 score: 1/(60+2) + 1/(60+1) = 0.032522
    # Since scores are equal, tie-breaker is chunk_id: "c1" < "c2"
    # c1 lex_rank (1) < dense_rank (2) -> label is "bm25"
    # c2 dense_rank (1) < lex_rank (2) -> label is "vector"
    lex = FakeLexical({"q": [("c1", 10.0), ("c2", 5.0)]})
    dense = FakeDense({"q": [("c2", 0.9), ("c1", 0.8)]})

    retriever = HybridRetriever(
        chunks=sample_chunks,
        lexical=lex,
        dense=dense,
        rrf_k=60,
        w_lexical=1.0,
        w_dense=1.0,
        config_path=None,
    )

    results = retriever.search("q", k=10)
    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].retrieval == "bm25"
    assert abs(results[0].score - (1 / 61 + 1 / 62)) < 1e-4

    assert results[1].chunk_id == "c2"
    assert results[1].retrieval == "vector"
    assert abs(results[1].score - (1 / 61 + 1 / 62)) < 1e-4


def test_tie_in_ranks_defaults_to_bm25(sample_chunks):
    # c1: lex rank 1, dense rank 1 -> tie -> retrieval label must be "bm25"
    lex = FakeLexical({"q": [("c1", 10.0)]})
    dense = FakeDense({"q": [("c1", 0.9)]})

    retriever = HybridRetriever(
        chunks=sample_chunks,
        lexical=lex,
        dense=dense,
        config_path=None,
    )
    res = retriever.search("q", k=10)
    assert res[0].retrieval == "bm25"


def test_modes_bm25_and_vector(sample_chunks):
    lex = FakeLexical({"q": [("c1", 5.0)]})
    dense = FakeDense({"q": [("c2", 0.9)]})

    r_bm25 = HybridRetriever(sample_chunks, lex, dense, mode="bm25", config_path=None)
    res_bm25 = r_bm25.search("q", k=10)
    assert len(res_bm25) == 1
    assert res_bm25[0].chunk_id == "c1"
    assert res_bm25[0].retrieval == "bm25"

    r_vec = HybridRetriever(sample_chunks, lex, dense, mode="vector", config_path=None)
    res_vec = r_vec.search("q", k=10)
    assert len(res_vec) == 1
    assert res_vec[0].chunk_id == "c2"
    assert res_vec[0].retrieval == "vector"


def test_search_detailed(sample_chunks):
    lex = FakeLexical({"q": [("c1", 5.0)]})
    dense = FakeDense({"q": [("c2", 0.9)]})

    retriever = HybridRetriever(sample_chunks, lex, dense, config_path=None)
    det = retriever.search_detailed("q", k=10)
    assert isinstance(det, HybridResult)
    assert len(det.bm25) == 1 and det.bm25[0].chunk_id == "c1"
    assert len(det.vector) == 1 and det.vector[0].chunk_id == "c2"
    assert len(det.fused) == 2


def test_search_many_fuses_multiple_queries(sample_chunks):
    lex = FakeLexical({"q1": [("c1", 10.0)], "q2": [("c2", 8.0)]})
    dense = FakeDense({"q1": [("c1", 0.9)], "q2": []})

    retriever = HybridRetriever(sample_chunks, lex, dense, config_path=None)
    res = retriever.search_many(["q1", "q2"], k=10)
    cids = [r.chunk_id for r in res]
    assert "c1" in cids and "c2" in cids
    # c1 appeared in both lex and dense for q1, so it should rank first
    assert cids[0] == "c1"


def test_unknown_chunk_id_logged_and_skipped(sample_chunks, caplog):
    lex = FakeLexical({"q": [("unknown_chunk_id", 10.0), ("c1", 5.0)]})
    dense = FakeDense({"q": []})

    retriever = HybridRetriever(sample_chunks, lex, dense, config_path=None)
    with caplog.at_level(logging.WARNING):
        res = retriever.search("q", k=10)
    assert len(res) == 1
    assert res[0].chunk_id == "c1"
    assert "unknown_chunk_id" in caplog.text


def test_k_cap_respected(sample_chunks):
    lex = FakeLexical({"q": [("c1", 10.0), ("c2", 5.0), ("c3", 2.0)]})
    dense = FakeDense({"q": []})

    retriever = HybridRetriever(sample_chunks, lex, dense, config_path=None)
    res = retriever.search("q", k=2)
    assert len(res) == 2
