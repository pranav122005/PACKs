"""
backend/services/ingredient_section_detector.py

PACKS - AI Ingredient Intelligence Platform
====================================================
Extracts ONLY the ingredient section from raw OCR text, explicitly
excluding manufacturer/address, marketing text, and storage
instructions that commonly surround it on packaging.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from backend.utils.ocr_patterns import INGREDIENT_HEADER_PATTERNS, INGREDIENT_STOP_PATTERNS

logger = logging.getLogger("packs.ingredient_section_detector")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Lines matching these patterns are dropped from within an otherwise
# valid ingredient section (defense against noisy OCR bleeding
# non-ingredient text into the block).
_NOISE_LINE_PATTERNS: List[str] = [
    r"^www\.",
    r"^http",
    r"customer\s*care",
    r"toll\s*free",
    r"^\+?\d[\d\s\-]{7,}$",  # phone numbers
    r"marketed\s*by",
    r"manufactured\s*by",
    r"packed\s*by",
    r"distributed\s*by",
    r"registered\s*office",
    r"fssai\s*(lic(ense)?)?\s*(no)?\.?\s*\d*",
    r"best\s*before",
    r"store\s*in\s*a\s*cool",
    r"keep\s*away\s*from\s*sunlight",
]


@dataclass
class IngredientSectionResult:
    """Result of ingredient-section-only extraction."""

    section_text: str
    excluded_line_count: int = 0
    header_matched: Optional[str] = None
    lines: List[str] = field(default_factory=list)


class IngredientSectionDetector:
    """
    Locates and isolates the ingredient section from raw OCR text,
    filtering out manufacturer, marketing, and storage-instruction
    content that OCR often merges into the same block.
    """

    def __init__(
        self,
        section_headers: Optional[List[str]] = None,
        stop_headers: Optional[List[str]] = None,
        noise_patterns: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            section_headers: Regex patterns marking the start of the
                ingredient section. Defaults to the standard pattern set.
            stop_headers: Regex patterns marking the end of the
                ingredient section. Defaults to the standard pattern set.
            noise_patterns: Regex patterns for lines to exclude even
                when they fall within the detected section boundaries.
        """
        self._start_pattern = re.compile(
            "|".join(f"({p})" for p in (section_headers or INGREDIENT_HEADER_PATTERNS)),
            re.IGNORECASE,
        )
        self._stop_pattern = re.compile(
            "|".join(f"({p})" for p in (stop_headers or INGREDIENT_STOP_PATTERNS)),
            re.IGNORECASE,
        )
        self._noise_pattern = re.compile(
            "|".join(f"({p})" for p in (noise_patterns or _NOISE_LINE_PATTERNS)),
            re.IGNORECASE,
        )

    def detect(self, ocr_text: str) -> IngredientSectionResult:
        """
        Extract only the ingredient section from raw OCR text.

        Args:
            ocr_text: Raw OCR text from the full packaging image.

        Returns:
            IngredientSectionResult containing the isolated, noise-
            filtered ingredient section text.
        """
        if not ocr_text or not ocr_text.strip():
            logger.warning("Empty OCR text supplied to ingredient section detector")
            return IngredientSectionResult(section_text="")

        text = self._normalize(ocr_text)

        start_match = self._start_pattern.search(text)
        if not start_match:
            logger.info("No ingredient section header found")
            return IngredientSectionResult(section_text="")

        header_matched = start_match.group(0)
        remainder = text[start_match.end():]

        stop_match = self._stop_pattern.search(remainder)
        raw_section = remainder[: stop_match.start()] if stop_match else remainder

        cleaned_lines, excluded_count = self._filter_noise_lines(raw_section)
        section_text = " ".join(cleaned_lines).strip()

        logger.info(
            "Ingredient section detected: %d lines kept, %d lines excluded as noise",
            len(cleaned_lines),
            excluded_count,
        )

        return IngredientSectionResult(
            section_text=section_text,
            excluded_line_count=excluded_count,
            header_matched=header_matched,
            lines=cleaned_lines,
        )

    def _normalize(self, text: str) -> str:
        """Normalize newlines/whitespace while preserving line breaks."""
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _filter_noise_lines(self, section_text: str) -> tuple[List[str], int]:
        """
        Split the raw section into lines and drop any line matching a
        known noise pattern (manufacturer info, phone numbers, storage
        instructions, etc.).

        Args:
            section_text: Raw text of the detected ingredient section.

        Returns:
            Tuple of (kept lines, number of excluded lines).
        """
        candidate_lines = re.split(r"[\n]", section_text)
        kept: List[str] = []
        excluded = 0

        for line in candidate_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if self._noise_pattern.search(stripped):
                excluded += 1
                continue
            kept.append(stripped)

        return kept, excluded