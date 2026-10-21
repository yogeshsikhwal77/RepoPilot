from pathlib import Path
from textwrap import dedent

import pytest

from repopilot.agents.base import AgentConfig
from repopilot.agents.retriever import RetrievalConfig, RetrieverAgent, extract_queries
from repopilot.indexing.parser import parse_repo
from repopilot.indexing.symbol_graph import SymbolGraph
from repopilot.orchestrator.state import AgentState, Chunk, Citation, Evidence
from repopilot.retrieval.hybrid import DenseIndex, HybridRetriever, LexicalIndex

REPO = {
    "math_mod.py": """
        def divide(a, b):
            if b == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return a / b

        def calculate_ratio(x, y):
            return divide(x, y)
    """
}


class MockLexical(LexicalIndex):
    def __init__(self, hits):
        self.hits = hits

    def search(self, query: str, k: int):
        return self.hits[:k]


class MockDense(DenseIndex):
    def __init__(self, hits):
        self.hits = hits

    def search(self, query: str, k: int):
        return self.hits[:k]


class FakeLLM:
    def __init__(self, response_text="ratio_calc, divide", raise_exc=False):
        self.response_text = response_text
        self.raise_exc = raise_exc

    def generate(self, model, prompt, **kwargs):
        if self.raise_exc:
            raise RuntimeError("API timeout")
        return {
            "text": self.response_text,
            "prompt_tokens": 15,
            "completion_tokens": 5,
            "cost_usd": 0.0001,
        }


@pytest.fixture
def setup_retriever(tmp_path):
    for rel, src in REPO.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(dedent(src).lstrip("\n"), encoding="utf-8")
    files = parse_repo(tmp_path, "c1")
    graph = SymbolGraph.build(files)
    chunks = {c.id: c for f in files for c in f.chunks}

    # Find divide chunk
    divide_chunk_id = next(c.id for c in chunks.values() if c.symbol == "divide")
    ratio_chunk_id = next(c.id for c in chunks.values() if c.symbol == "calculate_ratio")

    lex = MockLexical([(divide_chunk_id, 10.0)])
    dense = MockDense([(ratio_chunk_id, 0.9)])
    hybrid = HybridRetriever(chunks, lex, dense, config_path=None)

    return graph, hybrid, divide_chunk_id, ratio_chunk_id


def test_extract_queries():
    text = "Fix ZeroDivisionError in math_mod.py: `divide` fails when b=0. Call calculate_ratio."
    queries = extract_queries(text)
    assert "math_mod.py" in queries
    assert "ZeroDivisionError" in queries
    assert "divide" in queries
    assert "calculate_ratio" in queries


def test_retriever_delta_only_and_no_duplicates(setup_retriever):
    graph, hybrid, divide_cid, _ = setup_retriever
    existing_ev = Evidence(
        chunk_id=divide_cid,
        citation=Citation(path="math_mod.py", commit="c1", start_line=1, end_line=4),
        content="def divide",
        source_type="code",
        retrieval="bm25",
        score=1.0,
    )
    state = AgentState(
        task_input="Fix ZeroDivisionError in math_mod.py",
        collected_evidence=[existing_ev],
    )
    agent = RetrieverAgent(hybrid, graph, RetrievalConfig(top_k=5), use_graph=True)
    delta = agent(state)

    assert "collected_evidence" in delta
    new_evidence = delta["collected_evidence"]
    # Existing divide_cid must NOT be re-added
    assert not any(e.chunk_id == divide_cid for e in new_evidence)


def test_retriever_without_llm_deterministic(setup_retriever):
    graph, hybrid, _, _ = setup_retriever
    state = AgentState(task_input="Fix divide in math_mod.py")
    agent = RetrieverAgent(hybrid, graph, RetrievalConfig(top_k=5), llm=None)
    delta1 = agent(state)
    delta2 = agent(state)

    cids1 = [e.chunk_id for e in delta1["collected_evidence"]]
    cids2 = [e.chunk_id for e in delta2["collected_evidence"]]
    assert cids1 == cids2


def test_retriever_fake_llm_and_usage_deltas(setup_retriever):
    graph, hybrid, _, _ = setup_retriever
    fake_llm = FakeLLM("new_query_keyword")
    agent = RetrieverAgent(hybrid, graph, RetrievalConfig(top_k=5), llm=fake_llm)
    state = AgentState(task_input="Fix issue in math_mod.py")
    delta = agent(state)

    assert delta["tokens_used"] == 20
    assert delta["cost_usd"] == 0.0001
    assert delta["tool_calls"] == 1


def test_retriever_llm_exception_fallback(setup_retriever):
    graph, hybrid, _, _ = setup_retriever
    failing_llm = FakeLLM(raise_exc=True)
    agent = RetrieverAgent(hybrid, graph, RetrievalConfig(top_k=5), llm=failing_llm)
    state = AgentState(task_input="Fix issue in math_mod.py")
    delta = agent(state)

    assert "collected_evidence" in delta
    # Did not crash, silently fell back
    assert len(delta["collected_evidence"]) >= 1


def test_use_graph_false_yields_no_graph_evidence(setup_retriever):
    graph, hybrid, _, _ = setup_retriever
    agent = RetrieverAgent(hybrid, graph, RetrievalConfig(top_k=5), use_graph=False)
    state = AgentState(task_input="Fix divide in math_mod.py")
    delta = agent(state)

    for ev in delta["collected_evidence"]:
        assert ev.retrieval != "graph"
