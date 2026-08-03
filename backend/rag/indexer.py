
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
