
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

