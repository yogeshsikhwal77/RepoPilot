import random

import pytest

from repopilot.indexing.bm25 import BM25, BM25Index, tokenize
from repopilot.orchestrator.state import Chunk

rank_bm25 = pytest.importorskip("rank_bm25")


def test_tokenize_splits_identifiers_and_keeps_full_name():
    assert tokenize("add_item") == ["add_item", "add", "item"]
    assert tokenize("HTTPServer") == ["httpserver", "http", "server"]
    assert tokenize("x = foo(1)") == ["x", "foo", "1"]


def _random_corpus(seed=7, n_docs=200, vocab=60):
    rng = random.Random(seed)
    words = [f"w{i}" for i in range(vocab)]
    # skewed weights so some words are in >50% of docs (negative okapi idf)
    weights = [1.0 / (i + 1) for i in range(vocab)]
    return [rng.choices(words, weights, k=rng.randint(5, 40)) for _ in range(n_docs)]


@pytest.mark.parametrize("k1,b", [(1.5, 0.75), (1.2, 0.5), (2.0, 0.9)])
def test_matches_rank_bm25_exactly(k1, b):
    corpus = _random_corpus()
    ours = BM25(corpus, k1=k1, b=b, idf_variant="okapi")
    ref = rank_bm25.BM25Okapi(corpus, k1=k1, b=b)
    rng = random.Random(1)
    for _ in range(25):
        query = rng.choices([f"w{i}" for i in range(65)], k=rng.randint(1, 5))  # includes unseen words
        assert ours.get_scores(query) == pytest.approx(list(ref.get_scores(query)), rel=1e-9, abs=1e-9)


def test_lucene_idf_is_never_negative_and_ranks_rare_term_higher():
    corpus = [["common", "rare"]] + [["common", "x"] for _ in range(9)]
    bm = BM25(corpus, idf_variant="lucene")
    assert all(v >= 0 for v in bm.idf.values())
    assert bm.idf["rare"] > bm.idf["common"]
    assert bm.top_n(["rare"], 3)[0][0] == 0


def _chunk(path, symbol, text, s=1, e=3):
    return Chunk(id=f"{path}:{s}-{e}", path=path, start_line=s, end_line=e, symbol=symbol, text=text, commit="abc")


def test_index_finds_exact_symbol_and_natural_language():
    chunks = [
        _chunk("shop/cart.py", "Cart.add_item", "def add_item(self, name, price): ..."),
        _chunk("shop/cart.py", "Cart.total", "def total(self): return sum(...)", 5, 7),
        _chunk("shop/pricing.py", "apply_discount", "def apply_discount(total, pct): ...", 1, 2),
    ]
    idx = BM25Index().build(chunks)
    assert idx.search("Cart.add_item")[0][0] == chunks[0].id
    assert idx.search("apply discount")[0][0] == chunks[2].id
    assert idx.search("zzz_not_present") == []
    assert len(idx.search("total", k=1)) == 1


def test_save_load_roundtrip(tmp_path):
    chunks = [_chunk("a.py", "f", "def f(): pass"), _chunk("b.py", "g", "def g(): pass")]
    idx = BM25Index().build(chunks)
    idx.save(tmp_path / "bm25.pkl")
    assert BM25Index.load(tmp_path / "bm25.pkl").search("g") == idx.search("g")


def test_search_before_build_raises():
    with pytest.raises(RuntimeError):
        BM25Index().search("x")