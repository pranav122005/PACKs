"""
backend/services/ingredient_extractor.py

PACKS - AI Powered Ingredient Intelligence Platform
====================================================
Extracts the raw ingredient list from OCR'd packaging text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("packs.ingredient_extractor")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Header patterns that mark the start of an ingredient section.
_SECTION_HEADERS = [
    r"ingredients\s*:",
    r"ingredient\s*:",
    r"contains\s*:",
    r"made from\s*:",
    r"made\s*from\s*:",
]

# Header-like patterns that mark the end of the ingredient section
# (i.e. the start of a different section such as nutrition facts).
_STOP_HEADERS = [
    r"nutrition(al)?\s+(facts|information|value)",
    r"allergen\s+information",
    r"best before",
    r"manufactured by",
    r"marketed by",
    r"storage instructions",
    r"net (weight|wt)",
]

_BULLET_CHARS = ["•", "◦", "▪", "‣", "∙", "·", "*", "-"]


@dataclass
class IngredientExtractionResult:
    """Structured result of ingredient extraction."""

    raw_section_text: str
    ingredients: List[str] = field(default_factory=list)
    header_matched: Optional[str] = None


class IngredientExtractor:
    """
    Extracts a clean list of ingredient names from raw OCR text.

    Responsible only for locating the ingredient section and splitting
    it into discrete ingredient tokens (single responsibility).
    """

    def __init__(
        self,
        section_headers: Optional[List[str]] = None,
        stop_headers: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            section_headers: Regex patterns identifying the start of the
                ingredient section. Defaults to a standard set.
            stop_headers: Regex patterns identifying the end of the
                ingredient section. Defaults to a standard set.
        """
        self._section_headers = section_headers or _SECTION_HEADERS
        self._stop_headers = stop_headers or _STOP_HEADERS

    def extract(self, ocr_text: str) -> IngredientExtractionResult:
        """
        Extract the ingredient list from raw OCR text.

        Args:
            ocr_text: Raw text produced by the OCR engine.

        Returns:
            IngredientExtractionResult containing the raw section text
            and the parsed list of ingredient strings.
        """
        if not ocr_text or not ocr_text.strip():
            logger.warning("Empty OCR text supplied to ingredient extractor")
            return IngredientExtractionResult(raw_section_text="", ingredients=[])

        normalized_text = self._normalize_whitespace(ocr_text)
        section_text, header_matched = self._locate_section(normalized_text)

        if not section_text:
            logger.info("No ingredient section header found in OCR text")
            return IngredientExtractionResult(raw_section_text="", ingredients=[])

        ingredients = self._split_ingredients(section_text)
        logger.info("Extracted %d ingredients", len(ingredients))

        return IngredientExtractionResult(
            raw_section_text=section_text,
            ingredients=ingredients,
            header_matched=header_matched,
        )

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse excessive whitespace/newlines while keeping structure."""
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    def _locate_section(self, text: str) -> tuple[str, Optional[str]]:
        """
        Find the ingredient section between a start header and the next
        stop header (or end of text).

        Args:
            text: Normalized OCR text.

        Returns:
            Tuple of (section_text, matched_header) or ("", None) if not found.
        """
        start_pattern = re.compile(
            "|".join(f"({p})" for p in self._section_headers), re.IGNORECASE
        )
        match = start_pattern.search(text)
        if not match:
            return "", None

        start_idx = match.end()
        header_matched = match.group(0)

        remainder = text[start_idx:]

        stop_pattern = re.compile(
            "|".join(f"({p})" for p in self._stop_headers), re.IGNORECASE
        )
        stop_match = stop_pattern.search(remainder)

        section_text = remainder[: stop_match.start()] if stop_match else remainder
        return section_text.strip(), header_matched

    def _split_ingredients(self, section_text: str) -> List[str]:
        """
        Split section text into individual ingredient strings, handling
        commas, semicolons, bullet points, and line breaks.

        Args:
            section_text: Raw text of the ingredient section.

        Returns:
            List of cleaned, non-empty ingredient strings in order,
            with exact duplicates removed while preserving order.
        """
        text = section_text
        for bullet in _BULLET_CHARS:
            text = text.replace(bullet, ",")
        text = text.replace("\n", ",")

        # Split on commas and semicolons.
        raw_tokens = re.split(r"[;,]", text)

        cleaned: List[str] = []
        seen = set()
        for token in raw_tokens:
            item = self._clean_token(token)
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(item)

        return cleaned

    def _clean_token(self, token: str) -> str:
        """
        Clean an individual ingredient token: strip whitespace, trailing
        punctuation, parenthetical percentage annotations left dangling,
        and stray symbols.

        Args:
            token: Raw ingredient token.

        Returns:
            Cleaned ingredient string, or empty string if not meaningful.
        """
        item = token.strip()
        item = item.strip(".:;- \t")
        item = re.sub(r"\s+", " ", item)

        # Remove leading numbering like "1." or "(a)"
        item = re.sub(r"^\(?\d+\)?[\.\)]\s*", "", item)

        # Drop tokens that are purely numeric or too short to be meaningful.
        if not item or item.isdigit() or len(item) < 2:
            return ""

        return item