
import json
import logging
import threading
from pathlib import Path

import faiss
import numpy as np

from backend.config import get_config

logger = logging.getLogger(__name__)

_config = get_config()

_index_lock = threading.RLock()
_index: faiss.Index | None = None
_metadata: dict[str, dict] | None = None


class IndexerError(Exception):
    """Raised on any FAISS index or metadata store failure."""


def _index_path() -> Path:
    return Path(_config.FAISS_INDEX_PATH)


def _metadata_path() -> Path:
    return Path(_config.FAISS_METADATA_PATH)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalizes rows so inner product search behaves as cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero for degenerate zero vectors
    return vectors / norms


def _create_empty_index(dim: int) -> faiss.Index:
    flat = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap2(flat)


def load_or_create_index(dim: int | None = None) -> faiss.Index:
    """
    Loads the FAISS index from disk if present, otherwise creates a new
    empty IndexIDMap2(IndexFlatIP(dim)) index. Cached in-process after
    first load.

    Args:
        dim: embedding dimensionality to use if creating a new index.
             Defaults to config.FAISS_EMBEDDING_DIM.

    Raises:
        IndexerError: if the on-disk index file exists but fails to load.
    """
    global _index
    dim = dim or _config.FAISS_EMBEDDING_DIM

    with _index_lock:
        if _index is not None:
            return _index

        path = _index_path()
        if path.exists():
            try:
                _index = faiss.read_index(str(path))
                logger.info("Loaded FAISS index from %s (%d vectors)", path, _index.ntotal)
            except Exception as exc:  # noqa: BLE001
                raise IndexerError(f"Failed to load FAISS index from {path}: {exc}") from exc
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            _index = _create_empty_index(dim)
            logger.info("Created new empty FAISS index (dim=%d) at %s", dim, path)

        return _index


def save_index(index: faiss.Index) -> None:
    """Persists the FAISS index to disk."""
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        faiss.write_index(index, str(path))
    except Exception as exc:  # noqa: BLE001
        raise IndexerError(f"Failed to save FAISS index to {path}: {exc}") from exc


def load_metadata() -> dict[str, dict]:
    """Loads the metadata store (id -> metadata dict) from disk, caching in-process."""
    global _metadata
    if _metadata is not None:
        return _metadata

    path = _metadata_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise IndexerError(f"Failed to load metadata store from {path}: {exc}") from exc
    else:
        _metadata = {}

    return _metadata


def save_metadata(metadata: dict[str, dict]) -> None:
    """Persists the metadata store to disk."""
    path = _metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise IndexerError(f"Failed to save metadata store to {path}: {exc}") from exc


def add_vectors(
    ids: list[int],
    vectors: list[list[float]],
    metadatas: list[dict],
    index: faiss.Index | None = None,
) -> None:
    """
    Adds vectors to the FAISS index under the given IDs, and stores their
    metadata. Persists both the index and metadata store to disk.

    Args:
        ids: list of stable integer IDs (typically Document.id), one per vector.
        vectors: list of embedding vectors, same length/order as `ids`.
        metadatas: list of metadata dicts, same length/order as `ids`. Each
                   is stored under str(id) in the metadata store.
        index: optional pre-loaded index (avoids a redundant load). If
               omitted, load_or_create_index() is called.

    Raises:
        IndexerError: on shape mismatches or FAISS/storage failures.
    """
    if not (len(ids) == len(vectors) == len(metadatas)):
        raise IndexerError(
            f"ids ({len(ids)}), vectors ({len(vectors)}), and metadatas "
            f"({len(metadatas)}) must be the same length."
        )
    if not ids:
        return

    with _index_lock:
        idx = index or load_or_create_index()
        metadata_store = load_metadata()

        np_vectors = np.array(vectors, dtype="float32")
        if np_vectors.ndim != 2 or np_vectors.shape[1] != idx.d:
            raise IndexerError(
                f"Vector dimensionality {np_vectors.shape[-1] if np_vectors.ndim == 2 else '?'} "
                f"does not match index dimensionality {idx.d}."
            )

        np_vectors = _normalize(np_vectors)
        np_ids = np.array(ids, dtype="int64")

        try:
            idx.add_with_ids(np_vectors, np_ids)
        except Exception as exc:  # noqa: BLE001
            raise IndexerError(f"Failed to add vectors to FAISS index: {exc}") from exc

        for doc_id, meta in zip(ids, metadatas):
            metadata_store[str(doc_id)] = meta

        save_index(idx)
        save_metadata(metadata_store)

    logger.info("Added %d vector(s) to the FAISS index.", len(ids))


def remove_vectors(ids: list[int], index: faiss.Index | None = None) -> int:
    """
    Removes vectors by ID from the FAISS index and metadata store.

    Args:
        ids: list of document IDs to remove.
        index: optional pre-loaded index.

    Returns:
        Number of vectors actually removed from the FAISS index.
    """
    if not ids:
        return 0

    with _index_lock:
        idx = index or load_or_create_index()
        metadata_store = load_metadata()

        np_ids = np.array(ids, dtype="int64")
        try:
            n_removed = idx.remove_ids(np_ids)
        except Exception as exc:  # noqa: BLE001
            raise IndexerError(f"Failed to remove vectors from FAISS index: {exc}") from exc

        for doc_id in ids:
            metadata_store.pop(str(doc_id), None)

        save_index(idx)
        save_metadata(metadata_store)

    logger.info("Removed %d vector(s) from the FAISS index.", n_removed)
    return int(n_removed)

def search(
    query_vector: list[float],
    top_k: int = 5,
    index: faiss.Index | None = None,
    min_score: float | None = None,
) -> list[dict]:
    """
    Runs top-K cosine similarity search against the index.

    Args:
        query_vector: the embedded query.
        top_k: number of results to return.
        index: optional pre-loaded index.
        min_score: if set, filters out results with cosine similarity below
                   this threshold (scores range roughly -1..1).

    Returns:
        List of dicts, best match first:
        {"id": int, "score": float, "metadata": {...}}
    """
    with _index_lock:
        idx = index or load_or_create_index()
        metadata_store = load_metadata()

    if idx.ntotal == 0:
        return []

    np_query = np.array([query_vector], dtype="float32")
    if np_query.shape[1] != idx.d:
        raise IndexerError(
            f"Query vector dimensionality {np_query.shape[1]} does not match "
            f"index dimensionality {idx.d}."
        )
    np_query = _normalize(np_query)

    k = min(top_k, idx.ntotal)
    scores, ids = idx.search(np_query, k)

    results = []
    for score, doc_id in zip(scores[0], ids[0]):
        if doc_id == -1:
            continue  # FAISS pads with -1 when fewer than k results exist
        if min_score is not None and score < min_score:
            continue
        meta = metadata_store.get(str(int(doc_id)))
        if meta is None:
            logger.warning(
                "FAISS returned id %d with no matching metadata entry — skipping.", doc_id
            )
            continue
        results.append({"id": int(doc_id), "score": float(score), "metadata": meta})

    return results


