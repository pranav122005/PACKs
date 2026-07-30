
import logging
import re

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50

# Splits on '.', '!', '?' followed by whitespace and a capital letter, digit,
# quote, or opening paren -- a reasonable heuristic without pulling in NLTK/spaCy.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])')
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    raw_sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in raw_sentences if s.strip()]


def _split_long_unit(unit: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Fallback for a single 'sentence' longer than chunk_size (e.g. a long
    unpunctuated CSV cell or run-on line). Splits on word boundaries.
    """
    words = unit.split(" ")
    pieces = []
    current = []
    current_len = 0

    for word in words:
        added_len = len(word) + (1 if current else 0)
        if current_len + added_len > chunk_size and current:
            pieces.append(" ".join(current))
            # carry trailing words up to `overlap` chars into the next piece
            overlap_words = []
            overlap_len = 0
            for w in reversed(current):
                if overlap_len + len(w) > overlap:
                    break
                overlap_words.insert(0, w)
                overlap_len += len(w) + 1
            current = overlap_words
            current_len = sum(len(w) + 1 for w in current)
        current.append(word)
        current_len += added_len

    if current:
        pieces.append(" ".join(current))

    return pieces

def _pack_sentences_into_chunks(
    sentences: list[str], chunk_size: int, overlap: int
) -> list[str]:
    """Greedily packs sentences into chunks of ~chunk_size chars with ~overlap carryover."""
    chunks = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        # A single sentence longer than the whole chunk budget needs its own split.
        if len(sentence) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.extend(_split_long_unit(sentence, chunk_size, overlap))
            continue

        added_len = len(sentence) + (1 if current else 0)
        if current_len + added_len > chunk_size and current:
            chunks.append(" ".join(current))

            # Carry trailing sentences (up to ~overlap chars) into the next chunk
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 1
            current = overlap_sentences
            current_len = sum(len(s) + 1 for s in current)

        current.append(sentence)
        current_len += added_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_text(
    text: str,
    filename: str,
    source_id: str,
    location: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    start_chunk_index: int = 0,
) -> list[dict]:
    """
    Splits `text` into sentence-aware chunks and attaches source metadata.

    Args:
        text: raw text to chunk (e.g. one PDF page, one CSV row, or a whole .txt file).
        filename: display name of the source file, e.g. "meal_plan.pdf".
        source_id: stable identifier for the source document, e.g. a Drive file ID
                   or local file path.
        location: human-readable location within the source, e.g. "page 3" or
                  "row 12". None if not applicable (e.g. plain .txt files).
        chunk_size: target max characters per chunk.
        overlap: target max characters carried over between consecutive chunks.
        start_chunk_index: starting value for the chunk_index counter, so callers
                            chunking a document segment-by-segment (e.g. page by
                            page) can keep a single continuous index across the
                            whole document.

    Returns:
        List of dicts, each:
        {
            "content": str,
            "filename": str,
            "source_id": str,
            "location": str | None,
            "chunk_index": int,
        }
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    normalized = _normalize_whitespace(text or "")
    if not normalized:
        return []

    sentences = _split_sentences(normalized)
    if not sentences:
        return []

    raw_chunks = _pack_sentences_into_chunks(sentences, chunk_size, overlap)

    results = []
    for i, content in enumerate(raw_chunks):
        results.append(
            {
                "content": content,
                "filename": filename,
                "source_id": source_id,
                "location": location,
                "chunk_index": start_chunk_index + i,
            }
        )

    return results


