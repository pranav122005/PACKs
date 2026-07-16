"""
backend/services/text_postprocessor.py

PACKS - AI Ingredient Intelligence Platform
====================================================
Cleans raw OCR text: fixes common OCR character/word confusions,
removes duplicate adjacent words, and normalizes spacing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from backend.utils.ocr_patterns import OCR_CHARACTER_CORRECTIONS, OCR_WORD_CORRECTIONS

logger = logging.getLogger("packs.text_postprocessor")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


@dataclass
class PostprocessResult:
    """Result of OCR text post-processing."""

    original_text: str
    cleaned_text: str
    corrections_applied: int


class TextPostprocessor:
    """
    Applies a sequence of cleanup steps to raw OCR text to improve
    downstream extraction/parsing accuracy.
    """

    def __init__(self) -> None:
        self._word_corrections = {k.lower(): v for k, v in OCR_WORD_CORRECTIONS.items()}
        self._char_corrections = OCR_CHARACTER_CORRECTIONS

    def process(self, raw_text: str) -> PostprocessResult:
        """
        Run the full post-processing pipeline on raw OCR text.

        Args:
            raw_text: Raw text as returned by the OCR engine.

        Returns:
            PostprocessResult containing the original text, cleaned
            text, and a count of corrections applied.
        """
        if not raw_text or not raw_text.strip():
            logger.warning("Empty text supplied to text postprocessor")
            return PostprocessResult(original_text=raw_text or "", cleaned_text="", corrections_applied=0)

        text = raw_text
        corrections = 0

        text, char_fixes = self._fix_character_confusions(text)
        corrections += char_fixes

        text, word_fixes = self._fix_word_confusions(text)
        corrections += word_fixes

        text, dup_fixes = self._remove_duplicate_words(text)
        corrections += dup_fixes

        text = self._normalize_spacing(text)

        logger.info("Text postprocessing applied %d corrections", corrections)
        return PostprocessResult(
            original_text=raw_text, cleaned_text=text, corrections_applied=corrections
        )

    def _fix_character_confusions(self, text: str) -> tuple[str, int]:
        """
        Replace known character-level OCR confusions (e.g. 'rn' -> 'm'
        within specific contexts is risky, so only safe, unambiguous
        substrings from the pattern table are replaced).

        Args:
            text: Input text.

        Returns:
            Tuple of (corrected text, number of substitutions made).
        """
        corrected = text
        total = 0
        for wrong, right in self._char_corrections.items():
            count = corrected.count(wrong)
            if count:
                corrected = corrected.replace(wrong, right)
                total += count
        return corrected, total

    def _fix_word_confusions(self, text: str) -> tuple[str, int]:
        """
        Replace known whole-word OCR misreads (e.g. 'ingrediants' ->
        'ingredients'), case-insensitively while preserving original
        casing style where reasonable.

        Args:
            text: Input text.

        Returns:
            Tuple of (corrected text, number of word substitutions made).
        """
        total = 0

        def _replace(match: re.Match[str]) -> str:
            nonlocal total
            word = match.group(0)
            lowered = word.lower()
            if lowered in self._word_corrections:
                total += 1
                replacement = self._word_corrections[lowered]
                if word.isupper():
                    return replacement.upper()
                if word[0].isupper():
                    return replacement.capitalize()
                return replacement
            return word

        corrected = re.sub(r"[A-Za-z]+", _replace, text)
        return corrected, total

    def _remove_duplicate_words(self, text: str) -> tuple[str, int]:
        """
        Remove immediately repeated words (case-insensitive), a common
        OCR artifact from smudged or double-printed text.

        Args:
            text: Input text.

        Returns:
            Tuple of (corrected text, number of duplicates removed).
        """
        tokens = text.split(" ")
        result: List[str] = []
        removed = 0

        for token in tokens:
            if (
                result
                and token.strip().lower() == result[-1].strip().lower()
                and token.strip() != ""
            ):
                removed += 1
                continue
            result.append(token)

        return " ".join(result), removed

    def _normalize_spacing(self, text: str) -> str:
        """
        Normalize whitespace: collapse multiple spaces, fix spacing
        around punctuation, and trim.

        Args:
            text: Input text.

        Returns:
            Text with normalized spacing.
        """
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s+,", ",", text)
        text = re.sub(r"\s+:", ":", text)
        text = re.sub(r",\s*,", ",", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()