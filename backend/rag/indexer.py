
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
