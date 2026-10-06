# Indexing: BM25, embedder and coverage map

Author: Yogesh. Week 1. Status: done, 18 unit tests passing (Python 3.12 and 3.14).

This document covers the three files that turn a repository's code chunks into something the retrieval layer can search and the tester can use:

| File | One-line job |
| --- | --- |
| `src/repopilot/indexing/bm25.py` | Keyword search over chunks (own BM25 implementation) |
| `src/repopilot/indexing/embedder.py` | Text to vectors, caching, and in-memory vector search |
| `src/repopilot/indexing/coverage_map.py` | Which tests cover which symbols |

All three import only `Chunk` from `orchestrator/state.py`. Nothing in `state.py` was changed.

## Where they sit in the pipeline

```
parser.py / symbol_graph.py (Vibhav)
        |
        v
   list[Chunk]
        |
        +--> bm25.py ------+
        |                  +--> retrieval/hybrid.py --> Retriever --> Evidence
        +--> embedder.py --+
        |
        +--> coverage_map.py --> retrieval/impact.py --> Navigator --> Tester
                                 (changed symbols -> tests to run first)
```

- `bm25.py` and `embedder.py` answer "which code should the agent read?".
- `coverage_map.py` answers "which tests should run after this change?".

## Contracts with the rest of the project

1. **`Chunk` shape** (from `state.py`): `id`, `path`, `start_line`, `end_line`, `symbol`, `text`, `commit`.
2. **Chunk ids** are `f"{path}:{start_line}-{end_line}"`. BM25 and the vector index both return these ids, so `hybrid.py` can fuse the two result lists by id.
3. **`Chunk.symbol`** must be filled in, using `Class.method` for methods (for example `Cart.add_item`). Chunks with `symbol=None` are skipped by the coverage map and can only be found by text search.
4. **Symbol id format** used by the coverage map: `f"{path}::{symbol}"`, for example `shop/cart.py::Cart.add_item`. `symbol_graph.py` must produce the same ids, or `impact.py` cannot join graph nodes to the coverage map. Use `symbol_id()` from `coverage_map.py`.
5. **Test id format** is pytest node-id style: `tests/test_cart.py::test_total` or `tests/test_cart.py::TestCart::test_add`. This matches `TestReport.failed_tests` and `AgentState.impact_set`.

---

## 1. `bm25.py`

### Purpose
Exact-word search. It is strong where vectors are weak: function names, error strings and rare identifiers. The README requires "own implementation, checked against a library", and the unit test against `rank_bm25` is that check.

### Score formula
For document `d` and query tokens `t`:

```
score(d) = sum over t of  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
```

Defaults: `k1 = 1.5`, `b = 0.75`.

Two IDF variants:

| Variant | Formula | Use |
| --- | --- | --- |
| `"lucene"` (default) | `log(1 + (N - df + 0.5) / (df + 0.5))`, never negative | Production search |
| `"okapi"` | `log(N - df + 0.5) - log(df + 0.5)`; negatives replaced by `epsilon * average_idf` | Matches `rank_bm25.BM25Okapi` exactly, used to verify correctness |

### Tokenizer
`tokenize(text)` is code-aware. It keeps the full identifier and also its parts:

```
"add_item"   -> ["add_item", "add", "item"]
"HTTPServer" -> ["httpserver", "http", "server"]
"x = foo(1)" -> ["x", "foo", "1"]
```

An exact symbol query matches the full token strongly, and a plain-English query ("add an item") still hits the parts.

### API

```python
from repopilot.indexing.bm25 import BM25Index

index = BM25Index(k1=1.5, b=0.75, idf_variant="lucene").build(chunks)
hits = index.search("Cart.add_item", k=10)   # [(chunk_id, score), ...] best first
chunk = index.get_chunk(hits[0][0])
index.save("indexes/bm25.pkl")
index = BM25Index.load("indexes/bm25.pkl")
```

| Name | Notes |
| --- | --- |
| `tokenize(text)` | Returns a list of lowercase tokens |
| `BM25(corpus, k1, b, idf_variant, epsilon)` | Core algorithm on pre-tokenised documents; has `scores`, `get_scores` (dense list) and `top_n` |
| `chunk_to_text(chunk)` | What BM25 indexes: `symbol + path + text` |
| `BM25Index.build(chunks)` | Builds the inverted index; returns `self` |
| `BM25Index.search(query, k)` | Returns `[(chunk_id, score)]`; empty list if nothing matches; raises `RuntimeError` if `build()` was not called |
| `BM25Index.save` / `load` | Pickle. Only load files you wrote yourself |

### Design decisions
- **Inverted index.** Only documents sharing a term with the query are scored, so search cost does not grow with the whole corpus.
- **Repeated query terms count repeatedly.** This matches `rank_bm25`, which is what makes exact comparison possible.
- **Deterministic ties.** Equal scores are ordered by document position, so results are stable across runs (important for repeatable eval numbers).
- **Symbol and path are indexed with the text**, so searching `Cart.add_item` or `shop/cart.py` finds the right chunk.

### Tests (`tests/unit/test_bm25.py`, 8 tests)
- Tokenizer splitting.
- Scores identical to `rank_bm25.BM25Okapi` on a random 200-document corpus, for three `(k1, b)` settings, with unseen query words and skewed term frequencies (so negative-IDF handling is exercised).
- Lucene IDF is never negative and ranks rarer terms higher.
- Exact symbol and natural-language queries find the right chunk; no match returns `[]`.
- Save/load round trip; searching before building raises.

### Limitations
- Lexical only. It cannot match "remove" to "delete"; that is the embedder's job.
- No stop-word removal and no stemming.
- Pickle persistence is simple but not safe for untrusted files.

---

## 2. `embedder.py`

### Purpose
Turns text into vectors so meaning-based search works ("add an item to the cart" finds `Cart.add_item` even with different words). All vectors are L2-normalised, so cosine similarity equals a dot product.

### Components

| Class | Role |
| --- | --- |
| `Embedder` (Protocol) | Interface: `name`, `dim`, `embed(texts) -> float32 array (n, dim)` |
| `HashingEmbedder(dim=384)` | No model, no network. Signed feature hashing over code-aware tokens with sublinear tf. Lexical, not semantic. Used for tests and CI |
| `SentenceTransformerEmbedder(model_name, batch_size, device)` | Real local model, default `sentence-transformers/all-MiniLM-L6-v2` (384 dims). Imported lazily; needs `pip install sentence-transformers` |
| `CachedEmbedder(inner, cache_path)` | SQLite cache keyed by `sha256(model name + text)`. Only cache misses reach the model. Exposes `hits`, `misses`, `close()` |
| `VectorIndex(embedder)` | In-memory brute-force cosine search over chunks |
| `get_embedder(name, cache_path)` | Factory: `"hashing"` or `"minilm"`, optionally wrapped in the cache |

### API

```python
from repopilot.indexing.embedder import get_embedder, VectorIndex

embedder = get_embedder("minilm", cache_path=".cache/embeddings.sqlite")
index = VectorIndex(embedder).build(chunks)
hits = index.search("add an item to the cart", k=10)   # [(chunk_id, cosine), ...]
```

### Design decisions
- **Stable hashing.** `HashingEmbedder` uses `blake2b`, never Python's built-in `hash()` (which changes between processes), so vectors are identical across machines and runs.
- **Cache key includes the model name**, so switching models never returns stale vectors.
- **Duplicate texts are embedded once** per call.
- **Cache is persistent.** Re-indexing and re-running the eval three times per configuration does not re-embed unchanged text, which saves time and money.
- **`VectorIndex` is a stand-in for Qdrant.** It lets `plain_rag` and recall@k run without a running Qdrant service. `hybrid.py` can move to Qdrant later without changing the `Embedder`.
- **Embedded text** per chunk is `path + symbol + text` (`chunk_to_embed_text`).

### Tests (`tests/unit/test_embedder.py`, 5 tests)
- Deterministic, normalised, correct shape and dtype; empty input gives shape `(0, dim)`.
- Similar text scores higher than unrelated text.
- Cache embeds only misses, handles duplicates, reports hits and misses, and persists across a new wrapper.
- `VectorIndex` retrieves the relevant chunk; empty index returns `[]`.
- Factory returns the right backend and rejects unknown names.

The MiniLM backend is not exercised by the unit tests (it needs a model download). Test it manually before relying on it, and compare it with the hashing baseline using the eval harness.

### Limitations
- `VectorIndex` is O(n) per query; fine for tens of thousands of chunks, not for millions.
- MiniLM is a general-purpose text model. A code-specific model (for example `jinaai/jina-embeddings-v2-base-code`) may do better; decide with the eval harness, not by guessing.
- The cache is one SQLite file with no size limit or eviction.

---

## 3. `coverage_map.py`

### Purpose
Records which tests execute which symbols. Used by `impact.py` to turn "these symbols changed" into "run these tests first", which makes bad patches fail fast. It also drives the regression-rate metric and the `no_coverage_map` ablation.

### Two ways to build it

**Dynamic (precise).** Runs the test suite once with a coverage context per test:

```
python -m pytest --cov=. --cov-context=test
```

`from_coverage_data()` then maps each executed line back to the chunk (and so the symbol) containing it, and records the test id from the context label (the trailing `|run`, `|setup` or `|teardown` is stripped; lines executed at import time have no test and are skipped).

**Static (fallback, Python only).** `from_static_analysis()` parses test files with `ast`, collects every name each test mentions, and links the test to symbols with that short name. `Class.__init__` is linked when the class name is mentioned. Helper methods in a `Test*` class count for every test in that class.

The static build over-approximates (it can add tests that do not really cover a symbol) but never executes code and does not miss obvious links. For impact analysis, running a few extra tests is the safe direction.

`build_coverage_map(repo_root, chunks, coverage_file=None)` uses the dynamic data when the file exists, otherwise static.

### API

```python
from repopilot.indexing.coverage_map import (
    CoverageMap, symbol_id, collect_coverage, build_coverage_map,
)

cov_file = collect_coverage(repo_root, "indexes/.coverage")   # trusted repos only
cmap = build_coverage_map(repo_root, chunks, cov_file)

cmap.tests_for([symbol_id("shop/cart.py", "Cart.total")])
# ['tests/test_cart.py::test_total']
cmap.save("indexes/coverage_map.json")
cmap = CoverageMap.load("indexes/coverage_map.json")
```

| Name | Notes |
| --- | --- |
| `symbol_id(path, symbol)` | `"{path}::{symbol}"` |
| `is_test_path(path)` | True for `test_*.py`, `*_test.py`, or files under `tests/` or `test/` |
| `CoverageMap.tests_for(symbols)` | Tests covering any of the symbols; sorted, de-duplicated |
| `CoverageMap.symbols_for(test_id)` | Reverse lookup |
| `CoverageMap.uncovered(all_symbols)` | Symbols no test touches (useful for failure analysis) |
| `CoverageMap.stats()` | Source, symbols covered, tests seen, average tests per symbol |
| `CoverageMap.add` / `merge` / `save` / `load` | Build and persist (JSON) |
| `collect_coverage(repo_root, out_file, pytest_args, timeout)` | Runs pytest with contexts; returns the coverage file path |
| `from_coverage_data(coverage_file, repo_root, chunks)` | Dynamic build |
| `from_static_analysis(repo_root, chunks)` | Static build |

### Design decisions and bugs found
- **Test files are never treated as sources.** Chunks under test paths are skipped, so tests do not appear as covered code.
- **Safe tracer on Python 3.14.** Coverage's default `sys.monitoring` core records a line only once, so a line shared by several tests was credited to the first test only (for example `add_item` showed `test_add_item` but missed `test_total`). That would make impact analysis skip tests it should run. `collect_coverage()` now sets `COVERAGE_CORE` to `ctrace` (or `pytrace` if the C tracer is missing). If you run `pytest --cov-context=test` by hand on Python 3.14, set `COVERAGE_CORE=ctrace` yourself.
- **Safety.** `collect_coverage()` executes the repository's code on the host machine. Use it only on trusted repos (the sample repo, SWE-bench containers). For untrusted repos, run the same command inside `sandbox/runner.py` and pass the resulting `.coverage` file to `from_coverage_data()`.

### Tests (`tests/unit/test_coverage_map.py`, 5 tests)
Uses a small sample repo (`shop/cart.py`, `shop/pricing.py` and two test files) written by `helpers.py`. Both the static build and a real end-to-end `pytest --cov-context` run must produce the same expected links, including the case where one symbol is covered by two tests. Also covers context-label cleaning, test-path detection, fallback to static when the coverage file is missing, and query/stats/save/load/merge.

### Limitations
- Static analysis is Python only, and name-based: a common name such as `get` can link extra tests. It ignores `conftest.py` fixtures.
- Dynamic coverage needs `pytest` and `pytest-cov` and a suite that runs; a broken suite gives an incomplete map.
- Parametrised tests keep their full ids (`test_x[1-2]`), which can make the map larger.
- A class chunk is credited with every test that touches any of its methods, because its line range includes them.
- Accuracy depends on chunk line ranges and `symbol` names from the parser.

---

## Running the tests

```powershell
pip install -e . numpy coverage pytest pytest-cov rank-bm25
pytest tests/unit -q
```

If the project is not installed in editable mode:

```powershell
$env:PYTHONPATH="src;."
pytest tests/unit -q
```

Expected: `18 passed`. File layout:

```
src/repopilot/indexing/{bm25,embedder,coverage_map}.py
tests/__init__.py
tests/unit/{__init__,helpers,test_bm25,test_embedder,test_coverage_map}.py
```

`tests/unit/helpers.py` contains a small stdlib chunker used only by the tests; it stands in for Vibhav's tree-sitter parser.

## Dependencies

| Package | Needed for |
| --- | --- |
| `numpy` | embedder |
| `coverage`, `pytest`, `pytest-cov` | dynamic coverage map and its test |
| `rank-bm25` | BM25 verification test (dev only) |
| `sentence-transformers` | MiniLM backend (optional) |

## Open items
- Confirm with Vibhav that `symbol_graph.py` uses the `"{path}::{symbol}"` id format.
- Run `SentenceTransformerEmbedder` once with real chunks and compare it with `HashingEmbedder` in the week-3 recall@k table.
- Replace the stand-in chunker in `helpers.py` with Vibhav's parser output once available.