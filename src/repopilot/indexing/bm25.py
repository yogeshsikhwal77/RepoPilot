"""Own BM25 implementation (Okapi BM25 over an inverted index).

Two layers:
- `BM25`      : the algorithm on pre-tokenised documents. Pure Python, no deps.
                Checked against `rank_bm25` in tests/unit/test_bm25.py.
- `BM25Index` : the thing the rest of RepoPilot uses. Indexes `Chunk`s and
                returns (chunk_id, score) pairs for `retrieval/hybrid.py`.

Score of document d for query q:

    sum over query tokens t:  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

Two IDF variants:
- "lucene": log(1 + (N - df + 0.5) / (df + 0.5)). Never negative. Default.
- "okapi" : log(N - df + 0.5) - log(df + 0.5), negative values replaced by
            epsilon * average_idf. This is exactly what `rank_bm25.BM25Okapi`
            does, so the tests can demand identical scores.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

from repopilot.orchestrator.state import Chunk

# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokenize(text: str) -> list[str]:
    """Code-aware tokenizer.

    "add_item" -> ["add_item", "add", "item"]
    "HTTPServer" -> ["httpserver", "http", "server"]
    "x" -> ["x"]

    The full identifier is kept so an exact symbol name matches strongly; the
    parts are kept so natural-language queries ("add an item") still hit.
    """
    tokens: list[str] = []
    for ident in _IDENT.findall(text):
        parts = [p.lower() for p in _PARTS.findall(ident)]
        if len(parts) > 1:
            tokens.append(ident.lower())
        tokens.extend(parts)
    return tokens


# --------------------------------------------------------------------------- #
# Core algorithm
# --------------------------------------------------------------------------- #
class BM25:
    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
        idf_variant: Literal["lucene", "okapi"] = "lucene",
        epsilon: float = 0.25,
    ) -> None:
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.idf_variant = idf_variant
        self.n_docs = len(corpus)
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

        # inverted index: term -> [(doc_index, term_frequency), ...]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(corpus):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))

        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        n = self.n_docs
        if self.idf_variant == "lucene":
            return {
                t: math.log(1.0 + (n - len(p) + 0.5) / (len(p) + 0.5))
                for t, p in self.postings.items()
            }
        raw = {
            t: math.log(n - len(p) + 0.5) - math.log(len(p) + 0.5)
            for t, p in self.postings.items()
        }
        if not raw:
            return {}
        floor = self.epsilon * (sum(raw.values()) / len(raw))
        return {t: (v if v >= 0 else floor) for t, v in raw.items()}

    def scores(self, query: Sequence[str]) -> dict[int, float]:
        """Sparse scores: only documents that share at least one term."""
        acc: dict[int, float] = defaultdict(float)
        if self.avgdl == 0:
            return acc
        for term in query:  # repeated query terms count repeatedly (as rank_bm25 does)
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                length_norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                acc[i] += idf * tf * (self.k1 + 1) / (tf + length_norm)
        return acc

    def get_scores(self, query: Sequence[str]) -> list[float]:
        """Dense scores, one per document (same shape as rank_bm25.get_scores)."""
        sparse = self.scores(query)
        return [sparse.get(i, 0.0) for i in range(self.n_docs)]

    def top_n(self, query: Sequence[str], k: int = 10) -> list[tuple[int, float]]:
        ranked = sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]


# --------------------------------------------------------------------------- #
# Chunk-level index used by retrieval/hybrid.py
# --------------------------------------------------------------------------- #
def chunk_to_text(chunk: Chunk) -> str:
    """What BM25 sees for a chunk: symbol and path help exact-name queries."""
    return f"{chunk.symbol or ''} {chunk.path} {chunk.text}"


class BM25Index:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        idf_variant: Literal["lucene", "okapi"] = "lucene",
    ) -> None:
        self.k1, self.b, self.idf_variant = k1, b, idf_variant
        self._chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self._bm25: Optional[BM25] = None

    def build(self, chunks: Iterable[Chunk]) -> "BM25Index":
        self._chunks = list(chunks)
        self._by_id = {c.id: c for c in self._chunks}
        corpus = [tokenize(chunk_to_text(c)) for c in self._chunks]
        self._bm25 = BM25(corpus, k1=self.k1, b=self.b, idf_variant=self.idf_variant)
        return self

    def __len__(self) -> int:
        return len(self._chunks)

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Returns [(chunk_id, bm25_score)] best first. Empty if nothing matches."""
        if self._bm25 is None:
            raise RuntimeError("BM25Index.build() must be called before search()")
        hits = self._bm25.top_n(tokenize(query), k)
        return [(self._chunks[i].id, s) for i, s in hits]

    # pickle is fine here: the index is our own artefact, never user input
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)