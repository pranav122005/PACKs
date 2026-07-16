"""
backend/utils/ocr_patterns.py

PACKS - AI Ingredient Intelligence Platform
====================================================
Centralized regex pattern library used across OCR post-processing,
ingredient extraction, and nutrition table parsing.
"""

from __future__ import annotations

import re
from typing import Dict, List, Pattern

# ---------------------------------------------------------------------- #
# Section header patterns
# ---------------------------------------------------------------------- #
INGREDIENT_HEADER_PATTERNS: List[str] = [
    r"ingredients\s*:",
    r"ingredient\s*:",
    r"contains\s*:",
    r"made from\s*:",
    r"made\s*from\s*:",
    r"composition\s*:",
]

INGREDIENT_STOP_PATTERNS: List[str] = [
    r"nutrition(al)?\s+(facts|information|value)",
    r"allergen\s+information",
    r"best before",
    r"manufactured by",
    r"marketed by",
    r"storage instructions",
    r"net (weight|wt)",
    r"customer care",
    r"fssai",
    r"batch no",
]

NUTRITION_HEADER_PATTERNS: List[str] = [
    r"nutrition(al)?\s+(facts|information|value)\s*(panel|table)?",
    r"nutrition\s*facts",
    r"per\s*100\s*g",
    r"per\s*serving",
]

NUTRITION_STOP_PATTERNS: List[str] = [
    r"ingredients\s*:",
    r"allergen\s+information",
    r"manufactured by",
    r"marketed by",
    r"storage instructions",
    r"fssai",
    r"batch no",
]

MANUFACTURER_HEADER_PATTERNS: List[str] = [
    r"manufactured by",
    r"marketed by",
    r"packed by",
    r"customer care",
    r"distributed by",
]

BARCODE_HINT_PATTERNS: List[str] = [
    r"\b\d{8}\b",
    r"\b\d{12,13}\b",
]

# ---------------------------------------------------------------------- #
# Nutrition field value patterns (numeric + optional unit)
# ---------------------------------------------------------------------- #
_NUMBER_UNIT = r"([\d]+(?:[.,]\d+)?)\s*(kcal|kj|g|mg|mcg|\u00b5g|%)?"

SERVING_SIZE_PATTERN: Pattern[str] = re.compile(
    r"serving\s*size\s*[:\-]?\s*([\d.,]+\s*(?:g|ml|kg|l|oz)?)", re.IGNORECASE
)

ENERGY_PATTERN: Pattern[str] = re.compile(
    r"(?:energy|calories)\s*[:\-]?\s*" + _NUMBER_UNIT, re.IGNORECASE
)

CALORIES_PATTERN: Pattern[str] = re.compile(
    r"calories\s*[:\-]?\s*" + _NUMBER_UNIT, re.IGNORECASE
)

PROTEIN_PATTERN: Pattern[str] = re.compile(
    r"protein\s*[:\-]?\s*" + _NUMBER_UNIT, re.IGNORECASE
)

FAT_PATTERN: Pattern[str] = re.compile(
    r"(?:total\s+fat|fat)\s*[:\-]?\s*" + _NUMBER_UNIT, re.IGNORECASE
)

SATURATED_FAT_PATTERN: Pattern[str] = re.compile(
    r"saturated\s*(?:fat|fatty\s*acids)?\s*[:\-]?\s*" + _NUMBER_UNIT, re.IGNORECASE
)

TRANS_FAT_PATTERN: Pattern[str] = re.compile(
    r"trans\s*(?:fat|fatty\s*acids)?\s*[:\-]?\s*" + _NUMBER_UNIT, re.IGNORECASE
)

CARBOHYDRATE_PATTERN: Pattern[str] = re.compile(
    r"(?:total\s+carbohydrate[s]?|carbohydrate[s]?|carbs)\s*[:\-]?\s*" + _NUMBER_UNIT,
    re.IGNORECASE,
)

SUGAR_PATTERN: Pattern[str] = re.compile(
    r"(?<!added\s)(?<!added )sugar[s]?\s*[:\-]?\s*" + _NUMBER_UNIT, re.IGNORECASE
)

ADDED_SUGAR_PATTERN: Pattern[str] = re.compile(
    r"added\s*sugar[s]?\s*[:\-]?\s*" + _NUMBER_UNIT, re.IGNORECASE
)

FIBER_PATTERN: Pattern[str] = re.compile(
    r"(?:dietary\s*)?fib(?:er|re)\s*[:\-]?\s*" + _NUMBER_UNIT, re.IGNORECASE
)

SODIUM_PATTERN: Pattern[str] = re.compile(
    r"sodium\s*[:\-]?\s*" + _NUMBER_UNIT, re.IGNORECASE
)

SALT_PATTERN: Pattern[str] = re.compile(
    r"salt\s*[:\-]?\s*" + _NUMBER_UNIT, re.IGNORECASE
)

NUTRITION_FIELD_PATTERNS: Dict[str, Pattern[str]] = {
    "serving_size": SERVING_SIZE_PATTERN,
    "calories": CALORIES_PATTERN,
    "energy": ENERGY_PATTERN,
    "protein_g": PROTEIN_PATTERN,
    "fat_g": FAT_PATTERN,
    "saturated_fat_g": SATURATED_FAT_PATTERN,
    "trans_fat_g": TRANS_FAT_PATTERN,
    "carbohydrates_g": CARBOHYDRATE_PATTERN,
    "sugar_g": SUGAR_PATTERN,
    "added_sugar_g": ADDED_SUGAR_PATTERN,
    "fiber_g": FIBER_PATTERN,
    "sodium_mg": SODIUM_PATTERN,
    "salt_g": SALT_PATTERN,
}

# ---------------------------------------------------------------------- #
# Common OCR confusion corrections (character/word level)
# ---------------------------------------------------------------------- #
OCR_CHARACTER_CORRECTIONS: Dict[str, str] = {
    "0ils": "oils",
    "l0": "10",
    "|": "I",
    "rn": "m",
    "vv": "w",
}

OCR_WORD_CORRECTIONS: Dict[str, str] = {
    "ingrediants": "ingredients",
    "ingrdients": "ingredients",
    "ingredtents": "ingredients",
    "nutritton": "nutrition",
    "nutritionai": "nutritional",
    "caiories": "calories",
    "caloríes": "calories",
    "proteín": "protein",
    "carbohydrafes": "carbohydrates",
    "carbohydrate5": "carbohydrates",
    "flbre": "fibre",
    "flber": "fiber",
    "sodíum": "sodium",
    "sugai": "sugar",
    "suqar": "sugar",
    "fai": "fat",
    "sen1ing": "serving",
    "servlng": "serving",
}