from pathlib import Path
from textwrap import dedent

import pytest

langgraph = pytest.importorskip("langgraph")
from langgraph.graph import END, START, StateGraph

from repopilot.agents.base import AgentConfig
from repopilot.agents.retriever import RetrievalConfig, RetrieverAgent
from repopilot.indexing.parser import parse_repo
from repopilot.indexing.symbol_graph import SymbolGraph
from repopilot.orchestrator.state import AgentState, Chunk, Citation, Evidence
from repopilot.retrieval.hybrid import DenseIndex, HybridRetriever, LexicalIndex


class DummyLexical(LexicalIndex):
    def __init__(self, cids):
        self.cids = cids

    def search(self, query: str, k: int):
        return [(cid, 10.0) for cid in self.cids[:k]]


class DummyDense(DenseIndex):
    def search(self, query: str, k: int):
        return []


def test_parallel_evidence_aggregation_in_langgraph(tmp_path):
    # Setup mini-repo
    repo_file = tmp_path / "module.py"
    repo_file.write_text(
        dedent("""
        def func_a():
            return 1

        def func_b():
            return 2
    """).lstrip("\n"),
        encoding="utf-8",
    )
    files = parse_repo(tmp_path, "c1")
    graph = SymbolGraph.build(files)
    chunks = {c.id: c for f in files for c in f.chunks}

    chunk_a = next(c for c in chunks.values() if c.symbol == "func_a")
    chunk_b = next(c for c in chunks.values() if c.symbol == "func_b")

    lex = DummyLexical([chunk_a.id])
    dense = DummyDense()
    hybrid = HybridRetriever(chunks, lex, dense, config_path=None)

    # Retriever agent node
    retriever_agent = RetrieverAgent(
        hybrid=hybrid,
        graph=graph,
        cfg=RetrievalConfig(top_k=2),
        llm=None,
        use_graph=False,
        agent_config=AgentConfig(max_steps=5),
    )

    # Dummy navigator node running in parallel returning delta for collected_evidence & budget
    def dummy_navigator_node(state: AgentState) -> dict:
        ev = Evidence(
            chunk_id=chunk_b.id,
            citation=Citation(path=chunk_b.path, commit="c1", start_line=chunk_b.start_line, end_line=chunk_b.end_line),
            content=chunk_b.text,
            source_type="code",
            retrieval="impact",
            score=0.75,
        )
        return {
            "collected_evidence": [ev],
            "impact_set": ["tests/test_module.py::test_func_b"],
            "tokens_used": 25,
            "cost_usd": 0.0005,
            "tool_calls": 1,
        }

    # Construct LangGraph with parallel branches from START
    builder = StateGraph(AgentState)
    builder.add_node("retriever", retriever_agent)
    builder.add_node("navigator", dummy_navigator_node)

    builder.add_edge(START, "retriever")
    builder.add_edge(START, "navigator")
    builder.add_edge("retriever", END)
    builder.add_edge("navigator", END)

    app = builder.compile()

    initial_state = AgentState(
        task_input="Locate and test func_a and func_b in module.py",
        tokens_used=10,
        cost_usd=0.0001,
        tool_calls=1,
    )

    final_state = app.invoke(initial_state)

    # 1. Assert no InvalidUpdateError occurred and state returned is dictionary-like
    collected = final_state["collected_evidence"]
    assert len(collected) == 2
    cids = {e.chunk_id for e in collected}
    assert chunk_a.id in cids
    assert chunk_b.id in cids

    # 2. Check impact_set delta from navigator
    assert "tests/test_module.py::test_func_b" in final_state["impact_set"]

    # 3. Budget counters must add up correctly:
    # initial: tokens=10, cost=0.0001, tools=1
    # navigator delta: tokens=25, cost=0.0005, tools=1
    # retriever delta: tokens=0, cost=0.0, tools=0
    assert final_state["tokens_used"] == 35
    assert abs(final_state["cost_usd"] - 0.0006) < 1e-6
    assert final_state["tool_calls"] == 2
