"""
backend/services/nutrition_label_parser.py

PACKS - AI Powered Ingredient Intelligence Platform
====================================================
Parses nutrition facts table data from raw OCR text into a structured
dictionary of standardized nutrient values.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Pattern

logger = logging.getLogger("packs.nutrition_label_parser")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Matches a leading numeric value (int or float, optionally with a comma
# as thousands separator) followed by an optional unit.
_NUMBER_UNIT_PATTERN = r"([\d]+(?:[.,]\d+)?)\s*(kcal|kj|g|mg|mcg|µg|%)?"


@dataclass
class NutritionFacts:
    """Structured nutrition facts extracted from a product label."""

    serving_size: Optional[str] = None
    calories: Optional[float] = None
    energy_kj: Optional[float] = None
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

    def to_dict(self) -> Dict[str, Optional[float] | Optional[str]]:
        """Return the nutrition facts as a plain dictionary."""
        return asdict(self)


class NutritionLabelParser:
    """
    Parses free-form OCR text of a nutrition facts panel into a
    NutritionFacts dataclass.

    Each nutrient is matched independently via its own regex pattern set,
    keeping the parser resilient to missing or reordered fields.
    """

    def __init__(self) -> None:
        self._field_patterns: Dict[str, Pattern[str]] = {
            "serving_size": re.compile(
                r"serving\s*size\s*[:\-]?\s*([\d.,]+\s*(?:g|ml|kg|l|oz)?)",
                re.IGNORECASE,
            ),
            "calories": re.compile(
                r"(?:calories|energy)\s*[:\-]?\s*" + _NUMBER_UNIT_PATTERN
                + r"?\s*(?:kcal)?",
                re.IGNORECASE,
            ),
            "energy_kj": re.compile(
                r"energy\s*[:\-]?\s*[\d.,]+\s*kcal\s*[/,]?\s*"
                + _NUMBER_UNIT_PATTERN
                + r"\s*kj",
                re.IGNORECASE,
            ),
            "protein_g": re.compile(
                r"protein\s*[:\-]?\s*" + _NUMBER_UNIT_PATTERN, re.IGNORECASE
            ),
            "fat_g": re.compile(
                r"(?:total\s+fat|fat)\s*[:\-]?\s*" + _NUMBER_UNIT_PATTERN,
                re.IGNORECASE,
            ),
            "saturated_fat_g": re.compile(
                r"saturated\s*(?:fat|fatty\s*acids)?\s*[:\-]?\s*" + _NUMBER_UNIT_PATTERN,
                re.IGNORECASE,
            ),
            "trans_fat_g": re.compile(
                r"trans\s*(?:fat|fatty\s*acids)?\s*[:\-]?\s*" + _NUMBER_UNIT_PATTERN,
                re.IGNORECASE,
            ),
            "carbohydrates_g": re.compile(
                r"(?:total\s+carbohydrate[s]?|carbohydrate[s]?|carbs)\s*[:\-]?\s*"
                + _NUMBER_UNIT_PATTERN,
                re.IGNORECASE,
            ),
            "sugar_g": re.compile(
                r"(?<!added\s)(?<!added )sugar[s]?\s*[:\-]?\s*" + _NUMBER_UNIT_PATTERN,
                re.IGNORECASE,
            ),
            "added_sugar_g": re.compile(
                r"added\s*sugar[s]?\s*[:\-]?\s*" + _NUMBER_UNIT_PATTERN,
                re.IGNORECASE,
            ),
            "fiber_g": re.compile(
                r"(?:dietary\s*)?fib(?:er|re)\s*[:\-]?\s*" + _NUMBER_UNIT_PATTERN,
                re.IGNORECASE,
            ),
            "sodium_mg": re.compile(
                r"sodium\s*[:\-]?\s*" + _NUMBER_UNIT_PATTERN, re.IGNORECASE
            ),
            "salt_g": re.compile(
                r"salt\s*[:\-]?\s*" + _NUMBER_UNIT_PATTERN, re.IGNORECASE
            ),
        }

    def parse(self, ocr_text: str) -> NutritionFacts:
        """
        Parse nutrition facts from raw OCR text.

        Args:
            ocr_text: Raw OCR text, ideally containing a nutrition facts
                panel section.

        Returns:
            A populated NutritionFacts dataclass. Fields that could not
            be located remain None.
        """
        if not ocr_text or not ocr_text.strip():
            logger.warning("Empty OCR text supplied to nutrition label parser")
            return NutritionFacts()

        text = self._normalize_text(ocr_text)
        facts = NutritionFacts()

        facts.serving_size = self._extract_serving_size(text)
        facts.calories = self._extract_numeric(text, "calories")
        facts.energy_kj = self._extract_numeric(text, "energy_kj")
        facts.protein_g = self._extract_numeric(text, "protein_g")
        facts.fat_g = self._extract_numeric(text, "fat_g")
        facts.saturated_fat_g = self._extract_numeric(text, "saturated_fat_g")
        facts.trans_fat_g = self._extract_numeric(text, "trans_fat_g")
        facts.carbohydrates_g = self._extract_numeric(text, "carbohydrates_g")
        facts.sugar_g = self._extract_numeric(text, "sugar_g")
        facts.added_sugar_g = self._extract_numeric(text, "added_sugar_g")
        facts.fiber_g = self._extract_numeric(text, "fiber_g")
        facts.sodium_mg = self._extract_numeric(text, "sodium_mg")
        facts.salt_g = self._extract_numeric(text, "salt_g")

        logger.info("Parsed nutrition facts: %s", facts.to_dict())
        return facts

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and unify decimal separators for parsing."""
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _extract_serving_size(self, text: str) -> Optional[str]:
        """Extract the serving size as a raw string (e.g. '30g')."""
        match = self._field_patterns["serving_size"].search(text)
        if not match:
            return None
        return match.group(1).strip()

    def _extract_numeric(self, text: str, field_name: str) -> Optional[float]:
        """
        Extract a numeric nutrient value for the given field.

        Args:
            text: Normalized OCR text.
            field_name: Key into self._field_patterns.

        Returns:
            Parsed float value, or None if not found / unparseable.
        """
        pattern = self._field_patterns[field_name]
        match = pattern.search(text)
        if not match:
            return None

        raw_value = match.group(1)
        if raw_value is None:
            return None

        try:
            normalized = raw_value.replace(",", ".")
            # Guard against multiple dots from odd OCR artifacts.
            if normalized.count(".") > 1:
                parts = normalized.split(".")
                normalized = parts[0] + "." + "".join(parts[1:])
            return float(normalized)
        except ValueError:
            logger.debug("Failed to parse numeric value %r for field %s", raw_value, field_name)
            return None