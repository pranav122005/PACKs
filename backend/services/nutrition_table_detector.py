"""
backend/services/nutrition_table_detector.py

PACKS - AI Ingredient Intelligence Platform
====================================================
Extracts ONLY the nutrition facts table from raw OCR text and returns
it as a structured dictionary.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from backend.utils.ocr_patterns import (
    NUTRITION_FIELD_PATTERNS,
    NUTRITION_HEADER_PATTERNS,
    NUTRITION_STOP_PATTERNS,
)

logger = logging.getLogger("packs.nutrition_table_detector")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


@dataclass
class NutritionTableResult:
    """Structured result of nutrition-table-only extraction."""

    raw_table_text: str
    serving_size: Optional[str] = None
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    fat_g: Optional[float] = None
    saturated_fat_g: Optional[float] = None
    trans_fat_g: Optional[float] = None
    carbohydrates_g: Optional[float] = None
    sugar_g: Optional[float] = None
    added_sugar_g: Optional[float] = None
    fiber_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    salt_g: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        """Return the nutrition table result as a plain dictionary."""
        return asdict(self)


class NutritionTableDetector:
    """
    Isolates the nutrition facts table region of raw OCR text (ignoring
    ingredients, manufacturer info, and marketing text) and parses each
    known nutrient field from within that isolated region.
    """

    def __init__(self) -> None:
        self._start_pattern = re.compile(
            "|".join(f"({p})" for p in NUTRITION_HEADER_PATTERNS), re.IGNORECASE
        )
        self._stop_pattern = re.compile(
            "|".join(f"({p})" for p in NUTRITION_STOP_PATTERNS), re.IGNORECASE
        )
        self._field_patterns = NUTRITION_FIELD_PATTERNS

    def detect(self, ocr_text: str) -> NutritionTableResult:
        """
        Extract and parse only the nutrition facts table from raw OCR
        text.

        Args:
            ocr_text: Raw OCR text from the full packaging image.

        Returns:
            NutritionTableResult with the isolated raw table text and
            all parsed nutrient fields (None where not found).
        """
        if not ocr_text or not ocr_text.strip():
            logger.warning("Empty OCR text supplied to nutrition table detector")
            return NutritionTableResult(raw_table_text="")

        text = self._normalize(ocr_text)
        table_text = self._isolate_table(text)

        if not table_text:
            logger.info("No nutrition table region found in OCR text")
            return NutritionTableResult(raw_table_text="")

        result = NutritionTableResult(raw_table_text=table_text)
        result.serving_size = self._extract_serving_size(table_text)
        result.calories = self._extract_numeric(table_text, "calories")
        result.protein_g = self._extract_numeric(table_text, "protein_g")
        result.fat_g = self._extract_numeric(table_text, "fat_g")
        result.saturated_fat_g = self._extract_numeric(table_text, "saturated_fat_g")
        result.trans_fat_g = self._extract_numeric(table_text, "trans_fat_g")
        result.carbohydrates_g = self._extract_numeric(table_text, "carbohydrates_g")
        result.sugar_g = self._extract_numeric(table_text, "sugar_g")
        result.added_sugar_g = self._extract_numeric(table_text, "added_sugar_g")
        result.fiber_g = self._extract_numeric(table_text, "fiber_g")
        result.sodium_mg = self._extract_numeric(table_text, "sodium_mg")
        result.salt_g = self._extract_numeric(table_text, "salt_g")

        logger.info("Nutrition table extraction complete: %s", result.to_dict())
        return result

    def _normalize(self, text: str) -> str:
        """Normalize whitespace while preserving structure."""
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _isolate_table(self, text: str) -> str:
        """
        Locate the nutrition table region between its header and the
        next non-nutrition section (or end of text).

        Args:
            text: Normalized OCR text.

        Returns:
            The isolated raw nutrition table text, or empty string if
            no header was found.
        """
        start_match = self._start_pattern.search(text)
        if not start_match:
            return ""

        remainder = text[start_match.end():]
        stop_match = self._stop_pattern.search(remainder)
        table_text = remainder[: stop_match.start()] if stop_match else remainder
        return table_text.strip()

    def _extract_serving_size(self, table_text: str) -> Optional[str]:
        """Extract the serving size string from the isolated table text."""
        pattern = self._field_patterns["serving_size"]
        match = pattern.search(table_text)
        return match.group(1).strip() if match else None

    def _extract_numeric(self, table_text: str, field_name: str) -> Optional[float]:
        """
        Extract a numeric nutrient value from the isolated table text.

        Args:
            table_text: The isolated nutrition table text.
            field_name: Key into self._field_patterns.

        Returns:
            Parsed float value, or None if not found/unparseable.
        """
        pattern = self._field_patterns[field_name]
        match = pattern.search(table_text)
        if not match:
            return None

        raw_value = match.group(1)
        if raw_value is None:
            return None

        try:
            normalized = raw_value.replace(",", ".")
            if normalized.count(".") > 1:
                parts = normalized.split(".")
                normalized = parts[0] + "." + "".join(parts[1:])
            return float(normalized)
        except ValueError:
            logger.debug(
                "Failed to parse numeric value %r for field %s", raw_value, field_name
            )
            return None