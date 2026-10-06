import numpy as np

from repopilot.indexing.embedder import CachedEmbedder, HashingEmbedder, VectorIndex, get_embedder
from repopilot.orchestrator.state import Chunk


def test_hashing_is_deterministic_normalised_and_shaped():
    e = HashingEmbedder(dim=128)
    a, b = e.embed(["def add_item(x): pass"]), e.embed(["def add_item(x): pass"])
    assert a.shape == (1, 128) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5
    assert e.embed([]).shape == (0, 128)


def test_similar_text_scores_higher_than_unrelated():
    e = HashingEmbedder()
    q, close, far = e.embed(["add item to cart", "def add_item(cart, item)", "render the login page css"])
    assert q @ close > q @ far


class _Counting:
    name, dim = "count", 8

    def __init__(self):
        self.seen = []

    def embed(self, texts):
        self.seen.extend(texts)
        out = np.array([[len(t)] + [1] * 7 for t in texts], dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_cache_only_embeds_misses(tmp_path):
    inner = _Counting()
    c = CachedEmbedder(inner, tmp_path / "c.sqlite")
    first = c.embed(["aa", "bbb", "aa"])
    assert inner.seen == ["aa", "bbb"]            # duplicate embedded once
    second = c.embed(["bbb", "aa", "new"])
    assert inner.seen == ["aa", "bbb", "new"]     # only the new text hit the model
    assert np.allclose(second[0], first[1]) and np.allclose(second[1], first[0])
    assert c.hits == 2 and c.misses == 4   # call 1: 3 misses; call 2: 2 hits + 1 miss ('new')
    c.close()
    # persisted: a fresh wrapper reuses stored vectors
    inner2 = _Counting()
    c2 = CachedEmbedder(inner2, tmp_path / "c.sqlite")
    c2.embed(["aa", "new"])
    assert inner2.seen == []
    c2.close()


def test_vector_index_retrieves_relevant_chunk():
    def ch(i, sym, text):
        return Chunk(id=f"f.py:{i}-{i+1}", path="f.py", start_line=i, end_line=i + 1, symbol=sym, text=text, commit="c")

    chunks = [ch(1, "add_item", "def add_item(cart, item): cart.items.append(item)"),
              ch(5, "render_page", "def render_page(): return html template css"),
              ch(9, "apply_discount", "def apply_discount(total, pct): return total * pct")]
    idx = VectorIndex(HashingEmbedder()).build(chunks)
    assert idx.search("add item to cart", k=2)[0][0] == chunks[0].id
    assert idx.search("discount percentage total", k=1)[0][0] == chunks[2].id
    assert VectorIndex(HashingEmbedder()).search("x") == []


def test_factory():
    assert get_embedder("hashing").dim == 384
    try:
        get_embedder("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")