# Indexing: parser, symbol graph, BM25, embedder and coverage map

Authors: Yogesh (BM25, embedder, coverage map) and Vibhav (parser, symbol graph). Week 1. Status: done, 60 unit tests passing (18 + 42) on Python 3.14; Yogesh's 18 also on 3.12.

This document covers the five files that turn a repository's code into something the retrieval layer can search and the tester can use:

| File | One-line job |
| --- | --- |
| `src/repopilot/indexing/parser.py` | Python source to chunks, imports, call references and base classes (stdlib `ast`) |
| `src/repopilot/indexing/symbol_graph.py` | Directed graph over parsed symbols: contains, imports, inherits, calls |
| `src/repopilot/indexing/bm25.py` | Keyword search over chunks (own BM25 implementation) |
| `src/repopilot/indexing/embedder.py` | Text to vectors, caching, and in-memory vector search |
| `src/repopilot/indexing/coverage_map.py` | Which tests cover which symbols |

`bm25.py`, `embedder.py` and `coverage_map.py` import only `Chunk` from `orchestrator/state.py`. `parser.py` imports `Chunk` too, and `symbol_graph.py` also imports `symbol_id()` from `coverage_map.py`. Nothing in `state.py` was changed: it is the standard, and every file here follows it.

## Where they sit in the pipeline

```
repo files (*.py)
        |
        v
   parser.py   parse_repo(root, commit)
        |
        v
   list[ParsedFile]  (chunks, symbols, imports)
        |
        +--> symbol_graph.py --> retrieval/graph_expand.py --> Navigator
        |         |             (BFS out: what does this code call?)
        |         |
        |         +--> to_symbol_ids() --+
        |         (BFS in: what is affected?)
        v                                |
   list[Chunk]                           |
        |                                v
        +--> bm25.py ------+        coverage_map.py --> retrieval/impact.py --> Navigator --> Tester
        |                  +--> retrieval/hybrid.py     (changed symbols -> tests to run first)
        +--> embedder.py --+         --> Retriever --> Evidence
        |
        +--> coverage_map.py (chunk line ranges map coverage data to symbols)
```

- `parser.py` answers "what code units exist, and where?".
- `symbol_graph.py` answers "who calls this?", "what does this call?" and "what breaks if I change this?".
- `bm25.py` and `embedder.py` answer "which code should the agent read?".
- `coverage_map.py` answers "which tests should run after this change?".

## Contracts with the rest of the project

1. **`Chunk` shape** (from `state.py`): `id`, `path`, `start_line`, `end_line`, `symbol`, `text`, `commit`.
2. **Chunk ids** are `f"{path}:{start_line}-{end_line}"`. BM25 and the vector index both return these ids, so `hybrid.py` can fuse the two result lists by id. The parser produces them.
3. **`Chunk.symbol`** is local to the file and always filled in by the parser, using `Class.method` for methods (for example `Cart.add_item`) and `outer.inner` for nested functions. It is never module-qualified. Chunks with `symbol=None` are skipped by the coverage map and can only be found by text search.
4. **Symbol id format** used by the coverage map: `f"{path}::{symbol}"`, for example `shop/cart.py::Cart.add_item`. `symbol_graph.py` produces the same ids through `symbol_id()` from `coverage_map.py` (see `SymbolGraph.symbol_id_of()` and `to_symbol_ids()`), so `impact.py` can join graph nodes to the coverage map.
5. **Test id format** is pytest node-id style: `tests/test_cart.py::test_total` or `tests/test_cart.py::TestCart::test_add`. This matches `TestReport.failed_tests` and `AgentState.impact_set`.
6. **Graph node names** are module-qualified (`shop.cart.Cart.add_item`, `ParsedSymbol.qualname`). They exist only inside `symbol_graph.py`; convert at the boundary and use `Chunk.symbol` or symbol ids everywhere else.
7. **One root for everything.** Paths are relative to the root passed to `parse_repo`, with forward slashes. `build_coverage_map(repo_root, chunks, ...)` must receive chunks parsed from the same root, otherwise ids differ by a prefix (`src/shop/cart.py` against `shop/cart.py`) and `tests_for` returns nothing.

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

## 4. `parser.py`

### Purpose
Cuts each Python file into one chunk per class, function and method, and records what the graph needs: imports, the calls each function makes, and each class's base classes. It uses Python's own `ast` module, so it needs no extra dependency. It handles Python only, which is what the SWE-bench Lite subset needs.

### Data model

| Type | Fields | Notes |
| --- | --- | --- |
| `ImportRef` (frozen) | `module`, `name`, `alias`, `level` | `from ..x import y as z` is `module="x", name="y", alias="z", level=2` |
| `ParsedSymbol` (frozen) | `chunk`, `kind`, `qualname`, `calls`, `bases` | `kind` is `"class"`, `"method"` or `"function"` |
| `ParsedFile` | `path`, `module`, `sha256`, `chunks`, `symbols`, `imports`, `is_package` | `chunks` is always `[s.chunk for s in symbols]` |

`kind` is `"method"` only for a function defined directly in a class body. A function nested inside a method is a `"function"`.

### API

```python
from pathlib import Path
from repopilot.indexing.parser import parse_repo, parse_file

files = parse_repo(Path("."), commit="abc123")          # list[ParsedFile], sorted by path
chunks = [c for pf in files for c in pf.chunks]          # feed to BM25Index / VectorIndex / build_coverage_map

pf = parse_file(Path("shop/cart.py"), Path("."), "abc123")
pf.module                                                # "shop.cart"
pf.symbols[1].chunk.symbol, pf.symbols[1].qualname       # ("Cart.add_item", "shop.cart.Cart.add_item")
```

| Name | Notes |
| --- | --- |
| `parse_file(path, root, commit)` | Parses one file. Raises `ValueError` if `commit` is empty, and `SyntaxError` on invalid code |
| `parse_repo(root, commit)` | Parses every `*.py` under `root`, skipping `IGNORED_DIRS`. Files that fail to parse are skipped with a logged warning |
| `IGNORED_DIRS` | `.git`, `.venv`, `venv`, `__pycache__`, `node_modules`, `build`, `dist` |
| `_module_name(relative_path, is_package)` | `a/b.py` becomes `a.b`; `pkg/__init__.py` becomes `pkg`; a root `__init__.py` becomes `""` |

### Worked example

Source `shop/cart.py`:

```python
 1  from shop.pricing import apply_discount
 2
 3  class Cart:
 4      def add_item(self, name, price):
 5          self.items.append((name, price))
 6
 7      def subtotal(self):
 8          return sum(p for _, p in self.items)
 9
10      def total(self):
11          return apply_discount(self.subtotal(), 10)
```

| `Chunk.id` | `Chunk.symbol` | `qualname` | `kind` | `calls` |
| --- | --- | --- | --- | --- |
| `shop/cart.py:3-11` | `Cart` | `shop.cart.Cart` | class | none |
| `shop/cart.py:4-5` | `Cart.add_item` | `shop.cart.Cart.add_item` | method | `self.items.append` |
| `shop/cart.py:7-8` | `Cart.subtotal` | `shop.cart.Cart.subtotal` | method | `sum` |
| `shop/cart.py:10-11` | `Cart.total` | `shop.cart.Cart.total` | method | `apply_discount`, `self.subtotal` |

The `Cart` chunk id spans the whole class (lines 3-11), but its text is only the header line, because the method bodies already have their own chunks. Imports recorded: `ImportRef(module="shop.pricing", name="apply_discount")`.

### Design decisions
- **Decorators belong to the chunk.** `start_line` is the first decorator line, and the chunk text includes `@property`, `@app.route(...)` and so on, so citations point at the lines a reader needs.
- **Class chunks do not repeat method bodies.** Text is the class header, docstring and class-level fields. Without this, BM25 and the vectors would hold every method twice. The id still spans the whole class, so line-based lookups (the coverage map) behave as before.
- **Chunk text is dedented**, so embeddings and BM25 tokens are free of indentation noise.
- **Calls come from the function's own body only.** Calls inside a nested function or class belong to that nested symbol. Lambdas and comprehensions count for the enclosing function. Calls are sorted and de-duplicated.
- **Callee forms kept:** plain names (`helper`) and attribute chains (`self.repo.save`, `json.dumps`). Calls on other expressions, such as `f()()` or `d["k"]()`, are skipped.
- **Module names come from the path**, relative to the root you pass. Parsing from the repo root gives `src.repopilot.indexing.parser`; the graph handles the `src.` prefix with its suffix index.
- **BOM-safe reading.** Files are read as `utf-8-sig`, so a Windows byte-order mark does not make `ast.parse` fail.
- **Skipped files are logged**, not silent: a `SyntaxError` or `UnicodeDecodeError` produces a `skipped <path>: <reason>` warning.
- **`sha256` of the source** is stored per file, for change detection when re-indexing.

### Tests (`tests/unit/test_parser.py`, 17 tests)
- Chunk contract: local `Chunk.symbol`, qualified `qualname`, id format, kinds, commit stored, `chunks` equals the symbols' chunks; `commit` is required.
- Decorators are part of the span and text; method text is dedented; class chunks exclude method bodies but keep their full id span.
- Nested-function calls do not leak to the outer function; call collection (sorted, comprehension, async, attribute chains); base classes.
- Imports for all four forms; `_module_name` (4 cases); package `__init__` flags; `sha256` tracks content; a UTF-8 BOM file parses.
- `parse_repo` skips ignored directories and logs broken and binary files.

### Limitations
- Module-level code (constants, `if __name__ == "__main__":`, config) is not chunked, so retrieval cannot find it. A "module" chunk per file would fix this and is a candidate if the eval shows missed lookups.
- Python only. Other languages would need a tree-sitter-based parser.
- Imports are recorded per file, including ones inside functions, so all count as file-level scope.
- Calls made at module level are not attributed to anything.
- A syntactically invalid file is skipped entirely, so none of its symbols exist.

---

## 5. `symbol_graph.py`

### Purpose
Builds a directed graph over the parsed symbols so the agent can follow code instead of only searching it. It supports reverse traversal for impact analysis and forward traversal for context expansion. The README requires BFS over an AST-derived call graph with a hop limit; `neighbors()` is that BFS.

Resolution is static and name-based, with no type inference. Every edge carries a confidence, so callers can trade precision for recall.

### Nodes and edges
Nodes are module-qualified names: a module (`shop.cart`) or a symbol (`shop.cart.Cart.total`).

| Edge kind | From | To | Meaning |
| --- | --- | --- | --- |
| `contains` | module or class | class, function or method | Structural nesting |
| `imports` | module | module | Repo-internal imports only (external packages are ignored) |
| `inherits` | class | class | Base class resolved inside the repo |
| `calls` | function or method | function, method or class `__init__` | A call that could be resolved |

### Confidence levels

| Confidence | Rule | Example |
| --- | --- | --- |
| `exact` | Resolved by scope: `self.`/`cls.` call (including inherited methods), same-module name, or an imported name | `self.total()`, `helper(x)` after `from .util import helper` |
| `heuristic` | Receiver unknown, and exactly one method in the repo has that name | `cart.add(1)` where only `Cart.add` exists |
| `ambiguous` | Receiver unknown, and 2 to `max_ambiguity` (default 4) methods share the name | `x.go()` with `A.go` and `B.go` |
| (no edge) | Zero matches, more than `max_ambiguity` matches, or the call is external | `json.dumps(...)`, `sum(...)` |

If the same edge is found more than once, the best confidence wins.

### Call resolution order
For each call reference in a function, the graph tries these in order and stops at the first hit:

1. `self.x` or `cls.x`: look for `x` on the enclosing class, then up its resolved base classes (depth-first approximation of the MRO). **exact**.
2. A dotted name visible from the module: the module's own symbols, `from m import n [as k]`, `import m as k`, `import a.b.c`. Calling a class resolves to its `__init__` when it has one, otherwise to the class. **exact**.
3. Heuristic by method name, but only if the call's head is not imported from outside the repo, so `json.dumps(...)` never links to a repo method named `dumps`. **heuristic** or **ambiguous**.
4. Nothing: counted in `unresolved_calls` (builtins, external libraries, dynamic calls).

### Module suffix index
Files parsed from the repo root get module names such as `src.repopilot.indexing.parser`, while code imports them as `repopilot.indexing.parser`. The graph indexes every dotted suffix of every module name, so both spellings resolve. A suffix shared by two modules is treated as ambiguous and not matched.

### API

```python
from repopilot.indexing.parser import parse_repo
from repopilot.indexing.symbol_graph import SymbolGraph, EdgeKind

graph = SymbolGraph.build(parse_repo(root, commit), include_heuristic=True, max_ambiguity=4)

graph.callers("shop.pricing.apply_discount")             # [Edge, ...] who calls it
graph.callees("shop.cart.Cart.total", worst="exact")     # what it calls, exact edges only
graph.neighbors("shop.pricing.apply_discount", hops=2, direction="in")
# {"shop.cart.Cart.total": 1}   {node: hop distance}
graph.lookup("Cart.total")                               # [Chunk, ...] by qualname or dotted suffix
graph.stats()
```

| Name | Notes |
| --- | --- |
| `SymbolGraph.build(files, *, include_heuristic=True, max_ambiguity=4)` | Builds the graph from `list[ParsedFile]` |
| `edges(kind=None)` | All edges, optionally of one `EdgeKind` |
| `callers(q, worst="heuristic")` / `callees(q, worst="heuristic")` | One-hop call edges at or better than `worst` |
| `neighbors(node, hops=2, kinds=(CALLS,), direction="both", worst="heuristic")` | BFS. `direction` is `"out"`, `"in"` or `"both"`. Returns `{node: distance}`, excluding the start node. Cycles are safe |
| `lookup(name)` | Chunks whose qualname equals `name` or ends with `"." + name`; `Chunk.symbol` stays local |
| `symbol_id_of(qualname)` | `"shop/cart.py::Cart.total"`, or `None` for modules and unknown names |
| `qualname_of(symbol_id)` | The reverse mapping, or `None` |
| `to_symbol_ids(qualnames)` | Sorted, de-duplicated coverage-map ids. Module nodes are dropped |
| `stats()` | Module and symbol counts, edge counts by `kind/confidence`, `unresolved_calls` |
| `symbols`, `modules`, `files`, `in_edges`, `out_edges`, `unresolved_calls` | Public attributes |

`worst` filters by confidence: `"exact"` keeps exact edges only, `"heuristic"` (default) adds heuristic edges, `"ambiguous"` keeps everything.

### How the other modules use it

**Impact analysis** (`retrieval/impact.py`): changed symbols to tests to run first.

```python
reached = graph.neighbors(qualname, hops=2, direction="in")      # who depends on it
ids = graph.to_symbol_ids([qualname, *reached])                  # "path::Class.method" ids
tests = coverage_map.tests_for(ids)
```

**Graph expansion** (`retrieval/graph_expand.py`): turn graph nodes into `Evidence` without parsing names.

```python
chunk = graph.symbols[qualname].chunk
# Evidence(chunk_id=chunk.id,
#          citation=Citation(path=chunk.path, commit=chunk.commit,
#                            start_line=chunk.start_line, end_line=chunk.end_line),
#          content=chunk.text, source_type="code", retrieval="graph")
```

### Worked example
The `Cart` file from section 4, plus `shop/pricing.py` with `def apply_discount(total, pct)`:

| Edge | Kind | Confidence | Why |
| --- | --- | --- | --- |
| `shop.cart` to `shop.pricing` | imports | exact | `from shop.pricing import ...` |
| `shop.cart` to `shop.cart.Cart` | contains | exact | structure |
| `shop.cart.Cart` to `shop.cart.Cart.total` | contains | exact | structure |
| `Cart.total` to `Cart.subtotal` | calls | exact | `self.subtotal()` on the same class |
| `Cart.total` to `shop.pricing.apply_discount` | calls | exact | imported name |
| (none) | | | `sum(...)` and `self.items.append(...)` find no target, so `unresolved_calls` is 2 |

`graph.neighbors("shop.pricing.apply_discount", hops=2, direction="in")` returns `{"shop.cart.Cart.total": 1}`, and `to_symbol_ids(...)` turns that into `["shop/cart.py::Cart.total"]`.

### Design decisions
- **Two names for a symbol, one at each boundary.** The graph resolves with dotted qualnames because imports and `self.` calls need them. The rest of the project uses `Chunk.symbol` and coverage-map ids; the bridge methods convert.
- **Confidence on every edge, not a global switch.** The eval can run `worst="exact"` against `worst="heuristic"` and report the effect on recall@k and impact precision.
- **External calls are never guessed.** If a call's head name is imported from outside the repo, the heuristic step is skipped, so `os.path.join` does not link to a repo method called `join`. Relative imports always count as in-repo.
- **Heuristic matches are capped** by `max_ambiguity`, so a common method name (`get`, `run`) produces no edge instead of noise.
- **Redefinitions collapse.** Property setters and conditional redefinitions share one qualname, so they become one node and one id, matching what the coverage map sees.
- **Self-calls are not edges**, and `neighbors()` never returns its start node.
- **Root-level `__init__.py` symbols** get no `contains` edge from an empty-named parent.

### Tests (`tests/unit/test_symbol_graph.py`, 25 tests)
Uses a small `pkg` fixture (`base`, `util`, `shop`) written to a temp directory.
- **Call resolution:** `self.` calls, inherited methods, imported functions through relative imports, class call to `__init__` or to the class, unknown receiver as heuristic, external calls not matching repo methods.
- **Other edges:** inherits, imports (external and self excluded), contains, root `__init__` has no empty parent.
- **`stats()` snapshot:** exact counts for the fixture. Deliberately strict, so update it only when a behavior change is intended.
- **Confidence:** ambiguous edges and the `worst` filter, `max_ambiguity` and `include_heuristic` switches, best confidence wins.
- **Queries:** `lookup` by suffix, `callers` and `callees`, `neighbors` out and in (the impact direction, with the exact-only walk), start node excluded, cycles handled.
- **Bridge:** ids match the `symbol_id()` format, `to_symbol_ids` drops modules and sorts, an impact walk yields coverage-map ids, every symbol has a unique id.
- **Contract:** every graph symbol satisfies the `Chunk` contract.
- **Self-index smoke test:** the graph built over RepoPilot itself links `symbol_graph` to `parser`, which proves `repopilot.x` imports match files parsed as `src.repopilot.x`.

### Limitations
- **No type inference.** `cart.add(1)` where `cart` is a parameter resolves by method name only, so it is at best `heuristic`. This is the main source of missed and extra edges.
- **`super().__init__()` and similar calls** go through the heuristic step. If several classes define `__init__` (up to `max_ambiguity`), this produces `ambiguous` edges, which the default `worst="heuristic"` filters out.
- **Re-exports are not followed.** `from pkg import Y` fails if `Y` lives in a submodule and is re-exported by `pkg/__init__.py`, so the call is counted as unresolved.
- **`from x import *`, `getattr`, wrapping decorators, callbacks and dynamic dispatch** are not resolved.
- **Module-level calls** are not attributed to any node.
- **Class nodes** link to their methods by `contains`; coverage credited to a class chunk includes every method's tests.
- `lookup()` is a linear scan, fine for tens of thousands of symbols.

---

## Running the tests

```powershell
pip install -e . numpy coverage pytest pytest-cov rank-bm25
python -m pytest tests/unit -q
```

If the project is not installed in editable mode:

```powershell
$env:PYTHONPATH="src;."
python -m pytest tests/unit -q
```

Expected: `60 passed` (18 from BM25, embedder and coverage map, 17 from the parser, 25 from the symbol graph). File layout:

```
src/repopilot/indexing/{parser,symbol_graph,bm25,embedder,coverage_map}.py
tests/__init__.py
tests/unit/{__init__,helpers,test_parser,test_symbol_graph,test_bm25,test_embedder,test_coverage_map}.py
```

`tests/unit/helpers.py` contains a small stdlib chunker used only by the coverage-map and index tests; it stands in for `parse_repo` output.

Notes for Windows: save source files as UTF-8 without a BOM (the parser tolerates one, other tools may not), and use `python -m pytest` so the venv's interpreter is the one that runs.

## Dependencies

| Package | Needed for |
| --- | --- |
| standard library only (`ast`, `hashlib`, `logging`, `dataclasses`) | `parser.py`, `symbol_graph.py` |
| `numpy` | embedder |
| `coverage`, `pytest`, `pytest-cov` | dynamic coverage map and its test |
| `rank-bm25` | BM25 verification test (dev only) |
| `sentence-transformers` | MiniLM backend (optional) |

`symbol_graph.py` imports `symbol_id` from `coverage_map.py`, so the two files must be on the same branch.

## Open items
- Closed: `symbol_graph.py` uses the `"{path}::{symbol}"` id format, through `symbol_id()` in `to_symbol_ids()`.
- Replace the stand-in chunker in `helpers.py` with `parse_repo` output, which now follows the contracts above.
- Build `retrieval/graph_expand.py` and `retrieval/impact.py` as thin wrappers over `neighbors()` and `to_symbol_ids()`; the traversal already exists and should not be written twice.
- Use one root for the parser and the coverage map when running the pipeline (contract 7).
- Decide with the eval whether to add a module-level chunk per file, and whether to follow `__init__.py` re-exports.
- Run `SentenceTransformerEmbedder` once with real chunks and compare it with `HashingEmbedder` in the week-3 recall@k table.