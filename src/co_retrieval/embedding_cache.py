"""FAISS-based embedding cache for candidate chunks.

Encodes ``CodeChunk`` objects into dense vectors via an external encoder
callable, indexes them with FAISS for fast nearest-neighbour retrieval,
and supports disk persistence so the expensive encoding step only runs
once per repository snapshot.

If ``faiss`` is not installed the module falls back to brute-force cosine
similarity on NumPy arrays — slower but functionally identical.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from co_retrieval.chunking import CodeChunk

logger = logging.getLogger(__name__)

# ── Optional FAISS import ─────────────────────────────────────────────────────

try:
    import faiss  # type: ignore[import-untyped]

    _HAS_FAISS = True
except ImportError:
    faiss = None  # type: ignore[assignment]
    _HAS_FAISS = False


# ── Helpers ───────────────────────────────────────────────────────────────────


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (in-place safe)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return vectors / norms


def _chunks_fingerprint(chunks: Sequence[CodeChunk]) -> str:
    """Deterministic hash over chunk IDs for cache invalidation."""
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk.chunk_id.encode("utf-8"))
    return hasher.hexdigest()[:16]


# ── EmbeddingCache ────────────────────────────────────────────────────────────


class EmbeddingCache:
    """Dense vector index over ``CodeChunk`` objects.

    Parameters
    ----------
    dim : int
        Embedding dimension (must match the encoder output).
    use_faiss : bool
        If *True* and FAISS is installed, use a FAISS index.  Otherwise
        fall back to brute-force NumPy search.
    """

    def __init__(self, dim: int = 256, use_faiss: bool = True) -> None:
        self.dim = dim
        self._use_faiss = use_faiss and _HAS_FAISS

        # chunk_id → sequential index
        self.chunk_ids: List[str] = []
        self.chunk_map: Dict[str, int] = {}

        # Dense matrix (num_chunks, dim) – always kept as np.float32
        self._vectors: Optional[np.ndarray] = None
        self._faiss_index: Optional[object] = None  # faiss.Index

        # Metadata for cache invalidation
        self._fingerprint: Optional[str] = None

    # ── Build ─────────────────────────────────────────────────────────────

    def build_from_chunks(
        self,
        chunks: Sequence[CodeChunk],
        encode_fn: Callable[[List[str]], np.ndarray],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> None:
        """Encode *chunks* and populate the index.

        Parameters
        ----------
        encode_fn
            A callable that takes a list of text strings and returns an
            ``np.ndarray`` of shape ``(len(texts), dim)`` with float32
            vectors.  The ``DenseRetriever.encode_texts`` method satisfies
            this contract.
        batch_size
            How many chunks to encode in a single call to *encode_fn*.
        """
        texts = [chunk.retrieval_text() for chunk in chunks]
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.chunk_map = {cid: idx for idx, cid in enumerate(self.chunk_ids)}

        all_vecs: List[np.ndarray] = []
        total = len(texts)
        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            vecs = encode_fn(batch)
            all_vecs.append(np.asarray(vecs, dtype=np.float32))
            if show_progress and (start // batch_size) % 10 == 0:
                logger.info(
                    "EmbeddingCache: encoded %d / %d chunks", start + len(batch), total
                )

        self._vectors = _l2_normalize(np.concatenate(all_vecs, axis=0))
        self._fingerprint = _chunks_fingerprint(chunks)
        self._build_faiss_index()
        logger.info(
            "EmbeddingCache: built index with %d vectors (dim=%d, faiss=%s)",
            len(self.chunk_ids),
            self.dim,
            self._use_faiss,
        )

    def _build_faiss_index(self) -> None:
        """(Re)build the FAISS index from ``_vectors``."""
        if not self._use_faiss or self._vectors is None:
            self._faiss_index = None
            return
        index = faiss.IndexFlatIP(self.dim)
        index.add(self._vectors)  # type: ignore[arg-type]
        self._faiss_index = index

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[float, str]]:
        """Return the *top_k* nearest chunk IDs by cosine similarity.

        Parameters
        ----------
        query_vec
            1-D float32 array of shape ``(dim,)`` — **must already be
            L2-normalised**.

        Returns
        -------
        list of (score, chunk_id)
            Sorted descending by score.
        """
        if self._vectors is None or len(self.chunk_ids) == 0:
            return []

        query = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        k = min(top_k, len(self.chunk_ids))

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query, k)  # type: ignore[union-attr]
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                results.append((float(score), self.chunk_ids[idx]))
            return results

        # Brute-force fallback
        scores = (query @ self._vectors.T).flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.chunk_ids[i]) for i in top_indices]

    def get_vectors_by_ids(self, chunk_ids: Sequence[str]) -> np.ndarray:
        """Return the cached vectors for the given chunk IDs.

        Returns an ``(len(chunk_ids), dim)`` float32 array.  Unknown IDs
        are mapped to zero vectors.
        """
        out = np.zeros((len(chunk_ids), self.dim), dtype=np.float32)
        for i, cid in enumerate(chunk_ids):
            idx = self.chunk_map.get(cid)
            if idx is not None and self._vectors is not None:
                out[i] = self._vectors[idx]
        return out

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """Write the index and metadata to *directory*."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        if self._vectors is not None:
            np.save(str(path / "vectors.npy"), self._vectors)

        meta = {
            "dim": self.dim,
            "chunk_ids": self.chunk_ids,
            "fingerprint": self._fingerprint,
        }
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        logger.info("EmbeddingCache: saved to %s", path)

    def load(self, directory: str | Path) -> bool:
        """Load a previously saved index.  Returns *True* on success."""
        path = Path(directory)
        meta_path = path / "meta.json"
        vec_path = path / "vectors.npy"

        if not meta_path.exists() or not vec_path.exists():
            return False

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("dim") != self.dim:
            logger.warning(
                "EmbeddingCache: dimension mismatch (expected %d, got %d)",
                self.dim,
                meta.get("dim"),
            )
            return False

        self.chunk_ids = meta["chunk_ids"]
        self.chunk_map = {cid: idx for idx, cid in enumerate(self.chunk_ids)}
        self._vectors = np.load(str(vec_path)).astype(np.float32)
        self._fingerprint = meta.get("fingerprint")
        self._build_faiss_index()
        logger.info(
            "EmbeddingCache: loaded %d vectors from %s", len(self.chunk_ids), path
        )
        return True

    def is_valid_for(self, chunks: Sequence[CodeChunk]) -> bool:
        """Check whether the cached index matches the given chunks."""
        if self._fingerprint is None:
            return False
        return self._fingerprint == _chunks_fingerprint(chunks)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    @property
    def is_empty(self) -> bool:
        return self.size == 0
