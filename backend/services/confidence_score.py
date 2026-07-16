"""
backend/services/confidence_score.py

PACKS - AI Ingredient Intelligence Platform
====================================================
Computes confidence scores for each stage of the extraction pipeline
(OCR, ingredient extraction, nutrition parsing) and an overall combined
extraction confidence.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

logger = logging.getLogger("packs.confidence_score")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Weighting of each stage's contribution to overall confidence.
_WEIGHT_OCR = 0.35
_WEIGHT_INGREDIENT = 0.35
_WEIGHT_NUTRITION = 0.30

# Expected total nutrition fields tracked for coverage scoring.
_EXPECTED_NUTRITION_FIELDS = [
    "serving_size",
    "calories",
    "protein_g",
    "fat_g",
    "saturated_fat_g",
    "trans_fat_g",
    "carbohydrates_g",
    "sugar_g",
    "fiber_g",
    "sodium_mg",
]


@dataclass
class ConfidenceScoreResult:
    """Structured confidence score result across all pipeline stages."""

    ocr_confidence: float
    ingredient_confidence: float
    nutrition_confidence: float
    overall_confidence: float

    def to_dict(self) -> Dict[str, float]:
        """Return the confidence result as a plain dictionary."""
        return asdict(self)


class ConfidenceScorer:
    """
    Computes independent confidence scores for OCR quality, ingredient
    extraction completeness, and nutrition parsing coverage, then
    combines them into a single overall extraction confidence score.
    """

    def score_ocr(
        self,
        raw_text: str,
        corrections_applied: int,
    ) -> float:
        """
        Estimate OCR confidence based on text length, alphanumeric
        density, and the number of corrections the post-processor had
        to apply (more corrections implies noisier raw OCR output).

        Args:
            raw_text: Raw OCR text before post-processing.
            corrections_applied: Number of corrections applied by the
                text postprocessor.

        Returns:
            OCR confidence score in the range 0-100.
        """
        if not raw_text or not raw_text.strip():
            return 0.0

        alnum_chars = sum(1 for c in raw_text if c.isalnum() or c.isspace())
        alnum_density = alnum_chars / max(1, len(raw_text))

        length_score = min(len(raw_text) / 400.0, 1.0) * 100.0
        density_score = alnum_density * 100.0

        correction_penalty = min(corrections_applied * 3.0, 40.0)

        score = (length_score * 0.4) + (density_score * 0.6) - correction_penalty
        score = max(0.0, min(100.0, score))

        logger.debug(
            "OCR confidence computed: length_score=%.2f density_score=%.2f penalty=%.2f -> %.2f",
            length_score,
            density_score,
            correction_penalty,
            score,
        )
        return round(score, 2)

    def score_ingredient_extraction(
        self,
        ingredients: List[str],
        header_matched: Optional[str],
        excluded_line_count: int = 0,
    ) -> float:
        """
        Estimate confidence in the extracted ingredient list based on
        whether a section header was matched, the number of ingredients
        found, and how much noise had to be filtered out.

        Args:
            ingredients: Extracted/normalized ingredient list.
            header_matched: The matched ingredient section header text,
                or None if no header was found.
            excluded_line_count: Number of noise lines filtered out of
                the detected ingredient section.

        Returns:
            Ingredient extraction confidence score in the range 0-100.
        """
        if header_matched is None:
            return 0.0

        if not ingredients:
            return 15.0

        count_score = min(len(ingredients) / 10.0, 1.0) * 70.0
        header_bonus = 20.0
        noise_penalty = min(excluded_line_count * 2.0, 20.0)

        score = count_score + header_bonus - noise_penalty
        score = max(0.0, min(100.0, score))

        logger.debug(
            "Ingredient extraction confidence computed: count_score=%.2f header_bonus=%.2f penalty=%.2f -> %.2f",
            count_score,
            header_bonus,
            noise_penalty,
            score,
        )
        return round(score, 2)

    def score_nutrition_parsing(self, nutrition_facts: Dict[str, object]) -> float:
        """
        Estimate confidence in the parsed nutrition facts based on the
        fraction of expected fields that were successfully populated.

        Args:
            nutrition_facts: Dict of parsed nutrition fields (e.g. from
                NutritionLabelParser.to_dict() or
                NutritionTableDetector.to_dict()).

        Returns:
            Nutrition parsing confidence score in the range 0-100.
        """
        if not nutrition_facts:
            return 0.0

        found_count = 0
        for field_name in _EXPECTED_NUTRITION_FIELDS:
            value = nutrition_facts.get(field_name)
            if value is not None:
                found_count += 1

        coverage = found_count / len(_EXPECTED_NUTRITION_FIELDS)
        score = round(coverage * 100.0, 2)

        logger.debug(
            "Nutrition parsing confidence computed: %d/%d fields found -> %.2f",
            found_count,
            len(_EXPECTED_NUTRITION_FIELDS),
            score,
        )
        return score

    def score_overall(
        self,
        ocr_confidence: float,
        ingredient_confidence: float,
        nutrition_confidence: float,
    ) -> ConfidenceScoreResult:
        """
        Combine the three stage-level confidence scores into a single
        weighted overall confidence score.

        Args:
            ocr_confidence: OCR-stage confidence (0-100).
            ingredient_confidence: Ingredient-extraction confidence (0-100).
            nutrition_confidence: Nutrition-parsing confidence (0-100).

        Returns:
            ConfidenceScoreResult containing all stage scores plus the
            combined overall score.
        """
        overall = (
            (ocr_confidence * _WEIGHT_OCR)
            + (ingredient_confidence * _WEIGHT_INGREDIENT)
            + (nutrition_confidence * _WEIGHT_NUTRITION)
        )
        overall = round(max(0.0, min(100.0, overall)), 2)

        result = ConfidenceScoreResult(
            ocr_confidence=round(ocr_confidence, 2),
            ingredient_confidence=round(ingredient_confidence, 2),
            nutrition_confidence=round(nutrition_confidence, 2),
            overall_confidence=overall,
        )

        logger.info(
            "Overall extraction confidence: ocr=%.2f ingredient=%.2f nutrition=%.2f -> overall=%.2f",
            ocr_confidence,
            ingredient_confidence,
            nutrition_confidence,
            overall,
        )
        return result 