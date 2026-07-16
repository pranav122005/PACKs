"""
backend/ocr/text_cleaner.py

Text Cleaner — turns raw, noisy OCR output into a clean ingredient list.

Responsibilities:
    - Fix common character-level OCR mistakes (0/O, 1/l/I, rn/m, etc.)
    - Normalize whitespace, punctuation and casing artifacts
    - Locate the "Ingredients:" section within a full label's OCR text
    - Split that section into individual, cleaned ingredient entries,
      correctly handling nested parenthetical sub-ingredients
    - Strip trailing boilerplate ("may contain traces of...", allergen
      warnings, nutrition facts headers, etc.) that OCR often merges in
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Common OCR character confusions seen on printed packaging fonts.
# Keys are applied only within alphabetic word contexts to avoid mangling
# genuine numbers (handled separately in `_fix_word_level_errors`).
_CHAR_SUBSTITUTIONS: Dict[str, str] = {
    "0": "o",
    "1": "l",
    "|": "l",
    "!": "i",
    "$": "s",
    "@": "a",
    "5": "s",
    "8": "b",
}

# Multi-character OCR confusions that need whole-fragment replacement.
_FRAGMENT_SUBSTITUTIONS: Dict[str, str] = {
    "rn": "m",
    "vv": "w",
    "ii": "ii",  # placeholder for explicit no-op documentation
}

# Section headers that mark the END of an ingredients list on most labels.
_INGREDIENT_SECTION_TERMINATORS = (
    "nutrition facts", "nutritional information", "nutrition information",
    "allerg", "may contain", "contains milk", "storage instructions",
    "best before", "manufactured by", "marketed by", "net weight", "net wt",
    "mfg date", "batch no", "customer care", "fssai", "barcode",
)

_INGREDIENTS_HEADER_PATTERN = re.compile(
    r"ingredients?\s*[:\-]?\s*", re.IGNORECASE
)


@dataclass
class CleanedTextResult:
    """Output of the text-cleaning + ingredient-extraction pipeline."""

    raw_text: str
    cleaned_text: str
    ingredients_section_found: bool
    ingredients_raw_text: str
    ingredients: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "ingredients_section_found": self.ingredients_section_found,
            "ingredients_raw_text": self.ingredients_raw_text,
            "ingredients": self.ingredients,
            "ingredient_count": len(self.ingredients),
        }


class TextCleaner:
    """Cleans raw OCR text and extracts a structured ingredient list."""

    def __init__(self, apply_ocr_char_fixes: bool = True) -> None:
        self._apply_ocr_char_fixes = apply_ocr_char_fixes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, raw_text: str) -> CleanedTextResult:
        """Run the full clean -> locate -> extract pipeline on raw OCR text."""
        raw_text = raw_text or ""
        cleaned_text = self.clean_text(raw_text)

        ingredients_raw = self.extract_ingredients_section(cleaned_text)
        ingredients_found = bool(ingredients_raw)
        ingredients = self.parse_ingredient_list(ingredients_raw) if ingredients_raw else []

        return CleanedTextResult(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            ingredients_section_found=ingredients_found,
            ingredients_raw_text=ingredients_raw,
            ingredients=ingredients,
        )

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """Normalize whitespace/punctuation and optionally fix OCR character noise."""
        text = text.replace("\r", "\n")
        # OCR line-wraps often break an ingredient list across lines with no
        # trailing comma (e.g. "...Tartrazine (E102)\nSodium Benzoate...").
        # Treat line breaks as item separators so they don't get silently
        # merged into a single ingredient during top-level comma splitting.
        text = text.replace("\n", ", ")
        text = re.sub(r"[ \t]+", " ", text)

        if self._apply_ocr_char_fixes:
            text = self._fix_word_level_errors(text)

        # Normalize weird bullet/dash artifacts OCR sometimes injects between items
        text = re.sub(r"[•●▪◦]", ",", text)
        text = re.sub(r"\s*,\s*", ", ", text)
        text = re.sub(r",{2,}", ",", text)
        return text.strip()

    def _fix_word_level_errors(self, text: str) -> str:
        """
        Apply character substitutions only to alphabetic tokens that are
        NOT purely numeric (so "100g" isn't corrupted into "loog").
        """
        def fix_token(match: "re.Match[str]") -> str:
            token = match.group(0)
            if token.isdigit():
                return token
            # Only fix tokens containing letters (avoid mangling numbers/units)
            if not re.search(r"[A-Za-z]", token):
                return token
            fixed = token
            for fragment, replacement in _FRAGMENT_SUBSTITUTIONS.items():
                if fragment != replacement:
                    fixed = fixed.replace(fragment, replacement)
            return fixed

        return re.sub(r"\S+", fix_token, text)

    # ------------------------------------------------------------------
    # Ingredient section extraction
    # ------------------------------------------------------------------

    def extract_ingredients_section(self, cleaned_text: str) -> str:
        """
        Locate the substring starting at the "Ingredients:" header and
        ending at the first known terminator section (nutrition facts,
        allergens, storage, etc.), or end of text if none is found.
        """
        match = _INGREDIENTS_HEADER_PATTERN.search(cleaned_text)
        if not match:
            return ""

        remainder = cleaned_text[match.end():]
        lower_remainder = remainder.lower()

        end_index = len(remainder)
        for terminator in _INGREDIENT_SECTION_TERMINATORS:
            idx = lower_remainder.find(terminator)
            if idx != -1:
                end_index = min(end_index, idx)

        section = remainder[:end_index].strip(" .,:;\n")
        return section

    # ------------------------------------------------------------------
    # Ingredient list parsing
    # ------------------------------------------------------------------

    def parse_ingredient_list(self, ingredients_text: str) -> List[str]:
        """
        Split an ingredients string into individual, cleaned entries.
        Respects nested parentheses (e.g. "Vegetable Oil (Palm, Sunflower)"
        stays a single top-level entry) and strips trailing percentages
        and asterisk/footnote markers.
        """
        items = self._split_top_level(ingredients_text)
        cleaned_items = []
        for item in items:
            item = self._clean_single_ingredient(item)
            if item:
                cleaned_items.append(item)
        return cleaned_items

    @staticmethod
    def _split_top_level(text: str) -> List[str]:
        items: List[str] = []
        depth = 0
        current: List[str] = []
        for char in text:
            if char in "([":
                depth += 1
                current.append(char)
            elif char in ")]":
                depth = max(0, depth - 1)
                current.append(char)
            elif char == "," and depth == 0:
                items.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            items.append("".join(current).strip())
        return [i for i in items if i]

    @staticmethod
    def _clean_single_ingredient(item: str) -> str:
        item = item.strip(" .;:\n")
        # Remove trailing percentage annotations, e.g. "Sugar 12%" -> "Sugar"
        item = re.sub(r"\s*\d+(\.\d+)?\s*%\s*$", "", item)
        # Remove footnote markers like *, †
        item = re.sub(r"[*†‡]+$", "", item).strip()
        # Collapse internal whitespace
        item = re.sub(r"\s{2,}", " ", item)
        return item.strip()
