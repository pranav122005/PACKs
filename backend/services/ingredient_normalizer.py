"""
backend/services/ingredient_normalizer.py

PACKS - AI Powered Ingredient Intelligence Platform
====================================================
Normalizes raw ingredient strings into a canonical form: resolving
aliases (INS/E-numbers, common names), normalizing casing/whitespace,
and removing duplicates.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("packs.ingredient_normalizer")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# Canonical alias map. Keys are normalized (lowercase, whitespace-collapsed)
# forms of known aliases; values are the canonical ingredient name.
_DEFAULT_ALIAS_MAP: Dict[str, str] = {
    # MSG family
    "ins621": "MSG",
    "e621": "MSG",
    "621": "MSG",
    "mono sodium glutamate": "MSG",
    "monosodium glutamate": "MSG",
    "msg": "MSG",
    # Sugar family
    "sugar syrup": "Sugar",
    "invert sugar syrup": "Sugar",
    "cane sugar": "Sugar",
    "refined sugar": "Sugar",
    # Palm oil family
    "palmolein": "Palm Oil",
    "palm olein": "Palm Oil",
    "palm kernel oil": "Palm Oil",
    "palm oil": "Palm Oil",
    # HFCS family
    "high fructose corn syrup": "HFCS",
    "high-fructose corn syrup": "HFCS",
    "corn syrup high fructose": "HFCS",
    "hfcs": "HFCS",
    # Sodium family
    "sodium chloride": "Salt",
    "common salt": "Salt",
    "table salt": "Salt",
    # Common additive aliases
    "ins211": "Sodium Benzoate",
    "e211": "Sodium Benzoate",
    "ins102": "Tartrazine",
    "e102": "Tartrazine",
    "ins330": "Citric Acid",
    "e330": "Citric Acid",
    "ins322": "Soy Lecithin",
    "e322": "Soy Lecithin",
    "lecithin": "Soy Lecithin",
}


@dataclass
class NormalizationResult:
    """Result of normalizing a list of raw ingredient strings."""

    normalized_ingredients: List[str] = field(default_factory=list)
    alias_map_applied: Dict[str, str] = field(default_factory=dict)


class IngredientNormalizer:
    """
    Normalizes ingredient names: alias resolution, whitespace/casing
    normalization, and de-duplication.

    Follows the Open/Closed Principle: the alias map can be extended or
    replaced at construction time without modifying this class.
    """

    def __init__(self, alias_map: Optional[Dict[str, str]] = None) -> None:
        """
        Args:
            alias_map: Optional custom alias map (normalized-alias -> canonical
                name). Defaults to the built-in PACKS alias map.
        """
        self._alias_map = {
            self._normalize_key(k): v for k, v in (alias_map or _DEFAULT_ALIAS_MAP).items()
        }

    def normalize(self, ingredients: List[str]) -> NormalizationResult:
        """
        Normalize a list of raw ingredient strings.

        Args:
            ingredients: Raw ingredient strings (e.g. from IngredientExtractor).

        Returns:
            NormalizationResult with deduplicated, alias-resolved,
            title-cased ingredient names in original order of first
            appearance.
        """
        if not ingredients:
            return NormalizationResult()

        normalized_list: List[str] = []
        applied_aliases: Dict[str, str] = {}
        seen = set()

        for raw in ingredients:
            candidate = self._strip_percentage_annotations(raw)
            candidate = self._normalize_whitespace(candidate)

            if not candidate:
                continue

            key = self._normalize_key(candidate)
            canonical = self._alias_map.get(key)

            if canonical:
                final_name = canonical
                applied_aliases[raw.strip()] = canonical
            else:
                final_name = self._title_case(candidate)

            dedupe_key = final_name.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized_list.append(final_name)

        logger.info(
            "Normalized %d raw ingredients into %d unique canonical ingredients",
            len(ingredients),
            len(normalized_list),
        )

        return NormalizationResult(
            normalized_ingredients=normalized_list,
            alias_map_applied=applied_aliases,
        )

    def _normalize_key(self, text: str) -> str:
        """Produce a lookup key: lowercase, collapsed whitespace, no punctuation."""
        text = text.strip().lower()
        text = re.sub(r"[().]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse internal whitespace and trim."""
        return re.sub(r"\s+", " ", text.strip())

    def _strip_percentage_annotations(self, text: str) -> str:
        """
        Remove trailing percentage annotations, e.g. 'Sugar (10%)' -> 'Sugar'.
        """
        text = re.sub(r"\(\s*\d+(?:\.\d+)?\s*%\s*\)", "", text)
        text = re.sub(r"\d+(?:\.\d+)?\s*%", "", text)
        return text.strip(" -,;:")

    def _title_case(self, text: str) -> str:
        """
        Apply a readable title case while preserving common all-caps
        abbreviations (already-resolved aliases bypass this).
        """
        words = text.split(" ")
        titled = [w if w.isupper() and len(w) <= 4 else w.capitalize() for w in words]
        return " ".join(titled)