import logging
import random
import time

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from backend.config import get_config

logger = logging.getLogger(__name__)

_config = get_config()

DEFAULT_BATCH_SIZE = 20
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0
MAX_BACKOFF_SECONDS = 30.0
INTER_BATCH_DELAY_SECONDS = 0.2  # light pacing to be a polite API citizen

# 5xx (ServerError) is always worth retrying. 429 rate limiting arrives as a
# ClientError with code 429 and is also worth retrying with backoff.
RATE_LIMIT_STATUS_CODE = 429

_client: genai.Client | None = None


class EmbeddingError(Exception):
    """Raised when embedding generation fails after all retries, or on bad input."""


def _get_client() -> genai.Client:
    """Lazily builds and caches the genai Client for this process."""
    global _client
    if _client is None:
        if not _config.GEMINI_API_KEY:
            raise EmbeddingError("GEMINI_API_KEY is not configured. Set it in backend/.env")
        _client = genai.Client(api_key=_config.GEMINI_API_KEY)
    return _client


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == RATE_LIMIT_STATUS_CODE
    return False

