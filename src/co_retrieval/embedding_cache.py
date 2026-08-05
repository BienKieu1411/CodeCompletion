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
import uuid
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
    """Deterministic hash over chunk identity and retrieval content."""
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk.chunk_id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(chunk.retrieval_text().encode("utf-8"))
        hasher.update(b"\0")
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
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        texts = [chunk.retrieval_text() for chunk in chunks]
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.chunk_map = {cid: idx for idx, cid in enumerate(self.chunk_ids)}

        if not texts:
            self._vectors = None
            self._fingerprint = _chunks_fingerprint(chunks)
            self._build_faiss_index()
            return

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
        k = min(max(0, int(top_k)), len(self.chunk_ids))
        if k == 0:
            return []

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

        chunk_ids = meta.get("chunk_ids")
        if not isinstance(chunk_ids, list) or not all(
            isinstance(chunk_id, str) for chunk_id in chunk_ids
        ):
            logger.warning("EmbeddingCache: invalid chunk_ids metadata in %s", path)
            return False

        vectors = np.load(str(vec_path), allow_pickle=False).astype(np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(chunk_ids), self.dim):
            logger.warning(
                "EmbeddingCache: invalid vector shape %s for %d ids (expected (%d, %d))",
                vectors.shape,
                len(chunk_ids),
                len(chunk_ids),
                self.dim,
            )
            return False

        self.chunk_ids = chunk_ids
        self.chunk_map = {cid: idx for idx, cid in enumerate(self.chunk_ids)}
        self._vectors = vectors
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

    def clear(self) -> None:
        """Discard vectors and metadata while retaining the configured dimension."""
        self.chunk_ids = []
        self.chunk_map = {}
        self._vectors = None
        self._faiss_index = None
        self._fingerprint = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    @property
    def is_empty(self) -> bool:
        return self.size == 0


class ShardedEmbeddingCache:
    """Disk-backed CPU cache split into bounded NumPy-mmap shards.

    This cache is intended for evaluation corpora that are too large to keep
    in one in-memory matrix.  It stores one ``.npy`` memmap per shard and
    keeps only the chunk-id routing table in Python memory.  Retrieval still
    scores the sample's candidate chunks, so sharding does not change the
    candidate pool or introduce a cross-sample retrieval shortcut.
    """

    FORMAT_VERSION = 1

    def __init__(self, dim: int = 256, shard_size: int = 50_000) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.dim = int(dim)
        self.shard_size = int(shard_size)
        self.directory: Optional[Path] = None
        self.chunk_ids: List[str] = []
        self.chunk_map: Dict[str, Tuple[int, int]] = {}
        self._shards: List[np.ndarray] = []
        self._fingerprint: Optional[str] = None

    def build_from_chunks(
        self,
        chunks: Sequence[CodeChunk],
        encode_fn: Callable[[List[str]], np.ndarray],
        batch_size: int = 32,
        directory: str | Path = "cache/eval_index",
        show_progress: bool = False,
    ) -> None:
        """Encode chunks into bounded, CPU-readable shard files."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.clear()
        self.directory = path

        total = len(chunks)
        shard_entries: List[Dict[str, object]] = []
        total_encoded = 0

        for shard_index, start in enumerate(range(0, total, self.shard_size)):
            shard_chunks = list(chunks[start : start + self.shard_size])
            token = uuid.uuid4().hex[:12]
            filename = f"vectors-{token}-{shard_index:05d}.npy"
            final_path = path / filename
            temp_path = path / f".{filename}.tmp"
            vectors = np.lib.format.open_memmap(
                str(temp_path),
                mode="w+",
                dtype=np.float32,
                shape=(len(shard_chunks), self.dim),
            )

            shard_ids = [chunk.chunk_id for chunk in shard_chunks]
            for offset in range(0, len(shard_chunks), batch_size):
                batch_chunks = shard_chunks[offset : offset + batch_size]
                encoded = np.asarray(
                    encode_fn([chunk.retrieval_text() for chunk in batch_chunks]),
                    dtype=np.float32,
                )
                expected = (len(batch_chunks), self.dim)
                if encoded.shape != expected:
                    raise ValueError(
                        f"Encoder returned shape {encoded.shape}; expected {expected}"
                    )
                vectors[offset : offset + len(batch_chunks)] = _l2_normalize(
                    encoded
                )
                total_encoded += len(batch_chunks)
                if show_progress and total_encoded % (batch_size * 10) == 0:
                    logger.info(
                        "ShardedEmbeddingCache: encoded %d / %d chunks",
                        total_encoded,
                        total,
                    )

            vectors.flush()
            del vectors
            os.replace(temp_path, final_path)
            shard_entries.append(
                {
                    "file": filename,
                    "chunk_ids": shard_ids,
                    "count": len(shard_ids),
                }
            )
        manifest = {
            "format_version": self.FORMAT_VERSION,
            "dim": self.dim,
            "shard_size": self.shard_size,
            "total_chunks": total,
            "fingerprint": _chunks_fingerprint(chunks),
            "shards": shard_entries,
        }
        manifest_tmp = path / f".manifest-{uuid.uuid4().hex[:12]}.tmp"
        with open(manifest_tmp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False)
        os.replace(manifest_tmp, path / "manifest.json")
        self.load(path)
        logger.info(
            "ShardedEmbeddingCache: built %d chunks in %d CPU mmap shards at %s",
            total,
            len(shard_entries),
            path,
        )

    def load(self, directory: str | Path) -> bool:
        """Load a shard manifest and memory-map its vector files."""
        path = Path(directory)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("format_version") != self.FORMAT_VERSION:
                return False
            if int(manifest.get("dim", -1)) != self.dim:
                return False
            shards = manifest.get("shards")
            if not isinstance(shards, list):
                return False

            loaded_shards: List[np.ndarray] = []
            chunk_ids: List[str] = []
            chunk_map: Dict[str, Tuple[int, int]] = {}
            for shard_index, entry in enumerate(shards):
                if not isinstance(entry, dict):
                    return False
                filename = entry.get("file")
                ids = entry.get("chunk_ids")
                if not isinstance(filename, str) or not isinstance(ids, list):
                    return False
                vector_path = path / filename
                if not vector_path.exists():
                    return False
                vectors = np.load(str(vector_path), mmap_mode="r", allow_pickle=False)
                if vectors.ndim != 2 or vectors.shape != (len(ids), self.dim):
                    return False
                for row, chunk_id in enumerate(ids):
                    if not isinstance(chunk_id, str):
                        return False
                    chunk_map[chunk_id] = (shard_index, row)
                    chunk_ids.append(chunk_id)
                loaded_shards.append(vectors)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            logger.exception("Failed to load sharded embedding cache from %s", path)
            return False

        self.clear()
        self.directory = path
        self.chunk_ids = chunk_ids
        self.chunk_map = chunk_map
        self._shards = loaded_shards
        self._fingerprint = manifest.get("fingerprint")
        self.shard_size = int(manifest.get("shard_size", self.shard_size))
        logger.info(
            "ShardedEmbeddingCache: loaded %d chunks from %d CPU mmap shards at %s",
            len(self.chunk_ids),
            len(self._shards),
            path,
        )
        return True

    def is_valid_for(self, chunks: Sequence[CodeChunk]) -> bool:
        return bool(self._fingerprint) and self._fingerprint == _chunks_fingerprint(
            chunks
        )

    def get_vectors_by_ids(self, chunk_ids: Sequence[str]) -> np.ndarray:
        """Read only the requested vectors from the memory-mapped shards."""
        output = np.zeros((len(chunk_ids), self.dim), dtype=np.float32)
        for row, chunk_id in enumerate(chunk_ids):
            location = self.chunk_map.get(chunk_id)
            if location is None:
                continue
            shard_index, shard_row = location
            output[row] = self._shards[shard_index][shard_row]
        return output

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[float, str]]:
        """Search all CPU shards without materialising a global matrix."""
        if not self._shards or top_k <= 0:
            return []
        query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        candidates: List[Tuple[float, str]] = []
        for shard_index, vectors in enumerate(self._shards):
            scores = np.asarray(vectors @ query, dtype=np.float32)
            local_k = min(int(top_k), len(scores))
            if local_k <= 0:
                continue
            indices = np.argpartition(scores, -local_k)[-local_k:]
            candidates.extend(
                (float(scores[row]), self.chunk_ids[self._chunk_offset(shard_index) + int(row)])
                for row in indices
            )
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[: int(top_k)]

    def _chunk_offset(self, shard_index: int) -> int:
        return sum(len(shard) for shard in self._shards[:shard_index])

    def clear(self) -> None:
        self.directory = None
        self.chunk_ids = []
        self.chunk_map = {}
        self._shards = []
        self._fingerprint = None

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    @property
    def is_empty(self) -> bool:
        return self.size == 0
