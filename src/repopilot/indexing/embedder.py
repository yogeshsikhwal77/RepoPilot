"""Text -> vector, behind one small interface.

Backends:
- HashingEmbedder           : no model, no network, deterministic. Used in unit
                              tests and CI so they stay fast and offline.
- SentenceTransformerEmbedder: real local model (default all-MiniLM-L6-v2, 384 dims).
- CachedEmbedder            : wraps any embedder, stores vectors in SQLite so
                              re-indexing and re-running evals never re-embeds.

All vectors are L2-normalised, so cosine similarity == dot product.

`VectorIndex` is a tiny in-memory brute-force index (numpy). It lets the
plain_rag baseline and recall@k run without Qdrant; hybrid.py can swap in
Qdrant later without changing the Embedder.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np

from repopilot.indexing.bm25 import tokenize
from repopilot.orchestrator.state import Chunk


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Returns float32 array of shape (len(texts), dim), rows L2-normalised."""
        ...


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


# --------------------------------------------------------------------------- #
class HashingEmbedder:
    """Feature hashing over the code-aware tokens (signed buckets, sublinear tf).

    Not semantic, only lexical, but stable across processes and machines
    (uses blake2b, never Python's randomised hash()).
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok, tf in Counter(tokenize(text)).items():
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[row, h % self.dim] += sign * (1.0 + math.log(tf))
        return _normalise(out)


# --------------------------------------------------------------------------- #
class SentenceTransformerEmbedder:
    """Local embedding model. Imported lazily so the base install stays light.

    pip install sentence-transformers
    Code-specific alternative to try later (compare with the eval harness):
    "jinaai/jina-embeddings-v2-base-code" (needs trust_remote_code=True).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)


# --------------------------------------------------------------------------- #
class CachedEmbedder:
    """SQLite cache keyed by sha256(model name + text). Only misses hit the model."""

    def __init__(self, inner: Embedder, cache_path: str | Path = ".cache/embeddings.sqlite") -> None:
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self.hits = 0
        self.misses = 0
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute("CREATE TABLE IF NOT EXISTS vec (k TEXT PRIMARY KEY, v BLOB NOT NULL)")
        self._db.commit()

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.inner.name}\x00{text}".encode()).hexdigest()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):  # stay under SQLite's variable limit
            batch = keys[i : i + 500]
            marks = ",".join("?" * len(batch))
            for k, blob in self._db.execute(f"SELECT k, v FROM vec WHERE k IN ({marks})", batch):
                found[k] = np.frombuffer(blob, dtype=np.float32)

        missing = [i for i, k in enumerate(keys) if k not in found]
        if missing:
            # de-duplicate identical texts so each is embedded once
            unique = list(dict.fromkeys(texts[i] for i in missing))
            new = self.inner.embed(unique)
            for text, vec in zip(unique, new):
                k = self._key(text)
                found[k] = vec
                self._db.execute("INSERT OR REPLACE INTO vec VALUES (?, ?)", (k, vec.tobytes()))
            self._db.commit()
        self.misses += len(missing)
        self.hits += len(keys) - len(missing)

        if not keys:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([found[k] for k in keys]).astype(np.float32)

    def close(self) -> None:
        self._db.close()


# --------------------------------------------------------------------------- #
def chunk_to_embed_text(chunk: Chunk) -> str:
    return f"{chunk.path} {chunk.symbol or ''}\n{chunk.text}"


class VectorIndex:
    """In-memory cosine-similarity index over chunks."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._ids: list[str] = []
        self._mat = np.zeros((0, embedder.dim), dtype=np.float32)

    def build(self, chunks: Iterable[Chunk]) -> "VectorIndex":
        chunks = list(chunks)
        self._ids = [c.id for c in chunks]
        self._mat = self.embedder.embed([chunk_to_embed_text(c) for c in chunks])
        return self

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Returns [(chunk_id, cosine)] best first."""
        if not self._ids:
            return []
        sims = self._mat @ self.embedder.embed([query])[0]
        k = min(k, len(self._ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(self._ids[i], float(sims[i])) for i in top]


def get_embedder(name: str = "hashing", cache_path: str | Path | None = None) -> Embedder:
    """Factory used by configs/retrieval.yaml: embedder: hashing | minilm."""
    if name == "hashing":
        emb: Embedder = HashingEmbedder()
    elif name == "minilm":
        emb = SentenceTransformerEmbedder()
    else:
        raise ValueError(f"unknown embedder {name!r} (use 'hashing' or 'minilm')")
    return CachedEmbedder(emb, cache_path) if cache_path else emb