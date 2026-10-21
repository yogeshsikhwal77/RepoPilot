import pytest

from eval.baselines.plain_rag import PlainRag, extract_patches_from_diff_text
from repopilot.orchestrator.state import Answer, Chunk
from repopilot.retrieval.hybrid import DenseIndex, HybridRetriever, LexicalIndex


class DummyLexical(LexicalIndex):
    def search(self, query: str, k: int):
        return []


class DummyDense(DenseIndex):
    def __init__(self, cids):
        self.cids = cids

    def search(self, query: str, k: int):
        return [(cid, 0.9 - i * 0.1) for i, cid in enumerate(self.cids[:k])]


class FakeRagLLM:
    def __init__(self, text: str, tokens: int = 150, cost_usd: float = 0.002):
        self.text = text
        self.tokens = tokens
        self.cost_usd = cost_usd

    def generate(self, model, prompt, **kwargs):
        return {
            "text": self.text,
            "prompt_tokens": self.tokens // 2,
            "completion_tokens": self.tokens // 2,
            "cost_usd": self.cost_usd,
        }


@pytest.fixture
def rag_setup():
    chunks = {
        f"c{i}": Chunk(
            id=f"c{i}",
            path=f"src/file_{i}.py",
            start_line=1,
            end_line=20,
            symbol=f"fn_{i}",
            text=f"def fn_{i}(): pass",
            commit="base_sha",
        )
        for i in range(10)
    }
    dense = DummyDense([f"c{i}" for i in range(10)])
    hybrid = HybridRetriever(chunks, DummyLexical(), dense, mode="vector", config_path=None)
    return hybrid


def test_plain_rag_single_file_diff(rag_setup):
    diff_text = """```diff
--- a/src/file_0.py
+++ b/src/file_0.py
@@ -1,3 +1,3 @@
-def fn_0(): pass
+def fn_0(): return 42
```"""
    llm = FakeRagLLM(diff_text, tokens=100, cost_usd=0.001)
    rag = PlainRag(rag_setup, llm, k=5)
    ans = rag.run("fix fn_0")

    assert isinstance(ans, Answer)
    assert len(ans.diffs) == 1
    assert ans.diffs[0].filepath == "src/file_0.py"
    assert "return 42" in ans.diffs[0].diff
    assert ans.tokens == 100
    assert ans.cost_usd == 0.001
    assert not ans.test_report.passed


def test_plain_rag_multi_file_diff(rag_setup):
    diff_text = """```diff
--- a/src/file_0.py
+++ b/src/file_0.py
@@ -1 +1 @@
-pass
+return 1
--- a/src/file_1.py
+++ b/src/file_1.py
@@ -1 +1 @@
-pass
+return 2
```"""
    llm = FakeRagLLM(diff_text)
    rag = PlainRag(rag_setup, llm, k=4)
    ans = rag.run("fix files")

    assert len(ans.diffs) == 2
    paths = [p.filepath for p in ans.diffs]
    assert "src/file_0.py" in paths
    assert "src/file_1.py" in paths


def test_plain_rag_malformed_output_no_crash(rag_setup):
    llm = FakeRagLLM("I cannot solve this problem, sorry! No diff here.")
    rag = PlainRag(rag_setup, llm, k=4)
    ans = rag.run("impossible task")

    assert isinstance(ans, Answer)
    assert ans.diffs == []


def test_plain_rag_k_cap_respected(rag_setup):
    llm = FakeRagLLM("```diff\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-1\n+2\n```")
    rag = PlainRag(rag_setup, llm, k=3)
    
    # Check that search called inside rag retrieved at most k=3
    evs = rag.hybrid.search("test", k=rag.k)
    assert len(evs) == 3
