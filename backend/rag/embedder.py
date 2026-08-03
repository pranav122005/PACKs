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


def _embed_batch_with_retry(texts: list[str], task_type: str) -> list[list[float]]:
    """
    Embeds a batch of texts in a single API call, retrying transient
    failures (5xx, 429) with exponential backoff + jitter.

    Raises:
        EmbeddingError: if all retries are exhausted, or a non-retryable error occurs.
    """
    client = _get_client()
    backoff = INITIAL_BACKOFF_SECONDS
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=_config.GEMINI_EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            if not response.embeddings or len(response.embeddings) != len(texts):
                raise EmbeddingError(
                    f"Gemini API returned {len(response.embeddings or [])} embeddings "
                    f"for {len(texts)} inputs — mismatch."
                )
            return [item.values for item in response.embeddings]

        except (genai_errors.ServerError, genai_errors.ClientError) as exc:
            if not _is_retryable(exc) or attempt >= MAX_RETRIES:
                if _is_retryable(exc):
                    last_exc = exc
                    break
                raise EmbeddingError(f"Non-retryable error calling Gemini embed_content: {exc}") from exc
            last_exc = exc
            sleep_time = min(backoff, MAX_BACKOFF_SECONDS) + random.uniform(0, 0.5)
            logger.warning(
                "Gemini embed_content transient error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                MAX_RETRIES,
                exc,
                sleep_time,
            )
            time.sleep(sleep_time)
            backoff *= BACKOFF_MULTIPLIER

        except EmbeddingError:
            raise

        except Exception as exc:  # noqa: BLE001 - non-retryable, fail fast
            raise EmbeddingError(f"Unexpected error calling Gemini embed_content: {exc}") from exc

    raise EmbeddingError(
        f"Failed to embed batch of {len(texts)} after {MAX_RETRIES} retries: {last_exc}"
    ) from last_exc


def embed_documents(texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
    """
    Generates embeddings for a list of document chunks using
    task_type="RETRIEVAL_DOCUMENT" (asymmetric embedding for the "document
    side" of a retrieval pair). Texts are sent in true batched API calls
    (multiple contents per request), not one call per text.

    Args:
        texts: list of chunk texts to embed. Empty/whitespace-only entries
               are rejected up front.
        batch_size: how many texts to send per API call.

    Returns:
        List of embedding vectors (list[float]), in the same order as `texts`.

    Raises:
        EmbeddingError: if input is invalid, or embedding fails after retries.
    """
    if not texts:
        return []

    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError(f"texts[{i}] is empty or not a string; cannot embed.")

    embeddings: list[list[float]] = []
    total = len(texts)

    for batch_start in range(0, total, batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        logger.info(
            "Embedding batch %d-%d of %d chunks...",
            batch_start,
            min(batch_start + batch_size, total) - 1,
            total,
        )
        embeddings.extend(_embed_batch_with_retry(batch, task_type="RETRIEVAL_DOCUMENT"))

        if batch_start + batch_size < total:
            time.sleep(INTER_BATCH_DELAY_SECONDS)

    return embeddings


def embed_query(text: str) -> list[float]:
    """
    Generates an embedding for a user query using task_type="RETRIEVAL_QUERY"
    (asymmetric embedding for the "query side").

    Args:
        text: the user's question.

    Returns:
        A single embedding vector (list[float]).

    Raises:
        EmbeddingError: if input is invalid, or embedding fails after retries.
    """
    if not isinstance(text, str) or not text.strip():
        raise EmbeddingError("Query text is empty; cannot embed.")

    result = _embed_batch_with_retry([text.strip()], task_type="RETRIEVAL_QUERY")
    return result[0]
