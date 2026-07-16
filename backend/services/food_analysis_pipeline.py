"""
backend/services/food_analysis_pipeline.py

PACKS - AI Powered Ingredient Intelligence Platform
====================================================
Main orchestrator for the full food analysis pipeline:

    Image
        -> Image Preprocessing
        -> OCR Engine
        -> Extract Ingredient Section
        -> Extract Nutrition Table
        -> Normalize Ingredients
        -> Query Ingredient Knowledge Database
        -> Nutrition Engine
        -> Disease Engine
        -> Additive Engine
        -> NOVA Engine
        -> Recommendation Engine
        -> Health Score
        -> Generate Final JSON

This module depends only on abstractions (Protocols) for the pluggable
sub-engines (image preprocessor, OCR engine, knowledge DB, nutrition
engine, disease engine, additive engine, NOVA engine, recommendation
engine), following the Dependency Inversion Principle. Concrete
implementations are injected at construction time.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from backend.services.ingredient_extractor import IngredientExtractor
from backend.services.ingredient_normalizer import IngredientNormalizer
from backend.services.macro_calculator import MacroCalculator, MacroProfile
from backend.services.nutrition_label_parser import NutritionFacts, NutritionLabelParser
from backend.services.health_score_service import HealthScoreResult, HealthScoreService

logger = logging.getLogger("packs.food_analysis_pipeline")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------- #
# Pluggable engine interfaces (Protocols)
# ---------------------------------------------------------------------- #
class ImagePreprocessor(Protocol):
    """Prepares a raw image for OCR (denoising, deskew, contrast, etc.)."""

    def preprocess(self, image_path: str) -> str:
        """Return the path to the preprocessed image."""
        ...


class OCREngine(Protocol):
    """Extracts raw text from a (preprocessed) image."""

    def extract_text(self, image_path: str) -> str:
        """Return raw OCR text extracted from the image."""
        ...


class IngredientKnowledgeDB(Protocol):
    """Provides enrichment data for known ingredients."""

    def lookup_many(self, ingredient_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Return a mapping of ingredient name -> knowledge record."""
        ...


class NutritionEngine(Protocol):
    """Scores overall nutritional quality."""

    def score(
        self, nutrition_facts: NutritionFacts, macro_profile: MacroProfile
    ) -> Dict[str, Any]:
        """Return a dict containing at least 'score', 'positives', 'warnings'."""
        ...


class DiseaseEngine(Protocol):
    """Flags ingredients/nutrients relevant to specific disease risks."""

    def score(
        self,
        ingredients: List[str],
        knowledge: Dict[str, Dict[str, Any]],
        nutrition_facts: NutritionFacts,
    ) -> Dict[str, Any]:
        """Return a dict containing at least 'score', 'positives', 'warnings'."""
        ...


class AdditiveEngine(Protocol):
    """Assesses risk from food additives (E-numbers, preservatives, etc.)."""

    def score(
        self, ingredients: List[str], knowledge: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return a dict containing at least 'score', 'positives', 'warnings'."""
        ...


class NovaEngine(Protocol):
    """Classifies the product on the NOVA processing scale."""

    def classify(
        self, ingredients: List[str], knowledge: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return a dict containing at least 'nova_group', 'score', 'positives', 'warnings'."""
        ...


class RecommendationEngine(Protocol):
    """Generates user-facing recommendations/alternatives."""

    def recommend(
        self,
        ingredients: List[str],
        nutrition_facts: NutritionFacts,
        health_score: HealthScoreResult,
    ) -> List[str]:
        """Return a list of recommendation strings."""
        ...


# ---------------------------------------------------------------------- #
# Default no-op engine implementations
# ---------------------------------------------------------------------- #
class NoOpImagePreprocessor:
    """Default preprocessor that passes the image through unchanged."""

    def preprocess(self, image_path: str) -> str:
        """Return the input path unchanged."""
        return image_path


class DefaultNutritionEngine:
    """Baseline nutrition scoring using macro balance as a proxy."""

    def score(
        self, nutrition_facts: NutritionFacts, macro_profile: MacroProfile
    ) -> Dict[str, Any]:
        """Compute a baseline nutrition score from macro balance."""
        score = macro_profile.macro_balance_score if macro_profile.macro_balance_score is not None else 50.0
        positives: List[str] = []
        warnings: List[str] = []

        if nutrition_facts.fiber_g is not None and nutrition_facts.fiber_g >= 3:
            positives.append(f"Contains {nutrition_facts.fiber_g}g of fiber")
        if nutrition_facts.sodium_mg is not None and nutrition_facts.sodium_mg > 600:
            warnings.append(f"High sodium content: {nutrition_facts.sodium_mg}mg")

        return {"score": score, "positives": positives, "warnings": warnings}


class DefaultDiseaseEngine:
    """Baseline disease-risk flagging using simple nutrient thresholds."""

    _DIABETES_SUGAR_THRESHOLD_G = 15.0
    _HYPERTENSION_SODIUM_THRESHOLD_MG = 600.0

    def score(
        self,
        ingredients: List[str],
        knowledge: Dict[str, Dict[str, Any]],
        nutrition_facts: NutritionFacts,
    ) -> Dict[str, Any]:
        """Compute a baseline disease-risk score."""
        score = 100.0
        positives: List[str] = []
        warnings: List[str] = []

        if nutrition_facts.sugar_g is not None and nutrition_facts.sugar_g >= self._DIABETES_SUGAR_THRESHOLD_G:
            score -= 35.0
            warnings.append(
                f"Sugar content ({nutrition_facts.sugar_g}g) may be a concern for diabetics"
            )
        if nutrition_facts.sodium_mg is not None and nutrition_facts.sodium_mg >= self._HYPERTENSION_SODIUM_THRESHOLD_MG:
            score -= 30.0
            warnings.append(
                f"Sodium content ({nutrition_facts.sodium_mg}mg) may be a concern for hypertension"
            )
        if nutrition_facts.trans_fat_g is not None and nutrition_facts.trans_fat_g > 0:
            score -= 25.0
            warnings.append("Contains trans fat, linked to cardiovascular risk")

        if not warnings:
            positives.append("No major disease-risk flags detected from available nutrient data")

        return {"score": max(0.0, score), "positives": positives, "warnings": warnings}


class DefaultAdditiveEngine:
    """Baseline additive risk scoring using a small known high-risk set."""

    _HIGH_RISK_ADDITIVES = {"msg", "tartrazine", "sodium benzoate"}

    def score(
        self, ingredients: List[str], knowledge: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute a baseline additive-risk score."""
        score = 100.0
        positives: List[str] = []
        warnings: List[str] = []

        flagged = [
            ingredient
            for ingredient in ingredients
            if ingredient.lower() in self._HIGH_RISK_ADDITIVES
        ]

        if flagged:
            score -= min(len(flagged) * 20.0, 60.0)
            for item in flagged:
                warnings.append(f"Contains additive of concern: {item}")
        else:
            positives.append("No high-risk additives detected")

        return {"score": max(0.0, score), "positives": positives, "warnings": warnings}


class DefaultNovaEngine:
    """Baseline NOVA classification using ultra-processed marker ingredients."""

    _ULTRA_PROCESSED_MARKERS = {"hfcs", "msg", "palm oil", "hydrogenated oil"}

    def classify(
        self, ingredients: List[str], knowledge: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Classify the product's NOVA group based on marker ingredients."""
        lowered = {i.lower() for i in ingredients}
        marker_hits = lowered.intersection(self._ULTRA_PROCESSED_MARKERS)

        if len(marker_hits) >= 2:
            nova_group = 4
            score = 30.0
            warnings = [f"Multiple ultra-processed markers detected: {sorted(marker_hits)}"]
            positives: List[str] = []
        elif len(marker_hits) == 1:
            nova_group = 3
            score = 60.0
            warnings = [f"Processed ingredient marker detected: {sorted(marker_hits)}"]
            positives = []
        else:
            nova_group = 2
            score = 85.0
            warnings = []
            positives = ["No strong ultra-processing markers detected"]

        return {
            "nova_group": nova_group,
            "score": score,
            "positives": positives,
            "warnings": warnings,
        }


class DefaultRecommendationEngine:
    """Baseline recommendation generator using health score grade."""

    def recommend(
        self,
        ingredients: List[str],
        nutrition_facts: NutritionFacts,
        health_score: HealthScoreResult,
    ) -> List[str]:
        """Generate simple grade-based recommendations."""
        recommendations: List[str] = []

        if health_score.grade in ("D", "E"):
            recommendations.append(
                "Consider choosing a product with a higher health grade and lower sugar/additive content"
            )
        if nutrition_facts.sodium_mg is not None and nutrition_facts.sodium_mg > 600:
            recommendations.append("Look for a lower-sodium alternative if consumed regularly")
        if nutrition_facts.sugar_g is not None and nutrition_facts.sugar_g > 15:
            recommendations.append("Look for a lower-sugar alternative if consumed regularly")
        if not recommendations:
            recommendations.append("This product fits reasonably within a balanced diet")

        return recommendations


class NoOpIngredientKnowledgeDB:
    """Default knowledge DB that returns empty enrichment records."""

    def lookup_many(self, ingredient_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Return empty knowledge records for each ingredient."""
        return {name: {} for name in ingredient_names}


# ---------------------------------------------------------------------- #
# Pipeline result & orchestrator
# ---------------------------------------------------------------------- #
@dataclass
class FoodAnalysisResult:
    """Final structured result of the full food analysis pipeline."""

    success: bool
    ingredients: List[str] = field(default_factory=list)
    nutrition_facts: Optional[Dict[str, Any]] = None
    macro_profile: Optional[Dict[str, Any]] = None
    nutrition_engine: Optional[Dict[str, Any]] = None
    disease_engine: Optional[Dict[str, Any]] = None
    additive_engine: Optional[Dict[str, Any]] = None
    nova_engine: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)
    health_score: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the result as a plain JSON-serializable dictionary."""
        return asdict(self)


class FoodAnalysisPipeline:
    """
    Orchestrates the full ingredient-intelligence analysis pipeline for
    a single product image.

    All sub-engines are injected via constructor parameters, defaulting
    to lightweight built-in implementations. This keeps the orchestrator
    itself free of engine-specific logic (Single Responsibility) and
    open to extension with more sophisticated engines (Open/Closed).
    """

    def __init__(
        self,
        image_preprocessor: Optional[ImagePreprocessor] = None,
        ocr_engine: Optional[OCREngine] = None,
        ingredient_extractor: Optional[IngredientExtractor] = None,
        nutrition_label_parser: Optional[NutritionLabelParser] = None,
        ingredient_normalizer: Optional[IngredientNormalizer] = None,
        knowledge_db: Optional[IngredientKnowledgeDB] = None,
        macro_calculator: Optional[MacroCalculator] = None,
        nutrition_engine: Optional[NutritionEngine] = None,
        disease_engine: Optional[DiseaseEngine] = None,
        additive_engine: Optional[AdditiveEngine] = None,
        nova_engine: Optional[NovaEngine] = None,
        recommendation_engine: Optional[RecommendationEngine] = None,
        health_score_service: Optional[HealthScoreService] = None,
    ) -> None:
        """
        Args:
            image_preprocessor: Prepares the raw image for OCR.
            ocr_engine: Extracts raw text from the (preprocessed) image.
            ingredient_extractor: Extracts the ingredient list from OCR text.
            nutrition_label_parser: Extracts structured nutrition facts.
            ingredient_normalizer: Normalizes/deduplicates ingredient names.
            knowledge_db: Enriches ingredients with knowledge-base data.
            macro_calculator: Computes macro-nutrient ratios/densities.
            nutrition_engine: Scores overall nutritional quality.
            disease_engine: Flags disease-relevant risks.
            additive_engine: Assesses additive risk.
            nova_engine: Classifies processing level (NOVA).
            recommendation_engine: Produces user-facing recommendations.
            health_score_service: Aggregates all engine outputs into a
                final health score.

        All parameters default to lightweight built-in implementations
        so the pipeline is fully functional out of the box, while
        remaining swappable for production-grade engines.
        """
        self._image_preprocessor = image_preprocessor or NoOpImagePreprocessor()
        self._ocr_engine = ocr_engine or self._default_ocr_engine()
        self._ingredient_extractor = ingredient_extractor or IngredientExtractor()
        self._nutrition_label_parser = nutrition_label_parser or NutritionLabelParser()
        self._ingredient_normalizer = ingredient_normalizer or IngredientNormalizer()
        self._knowledge_db = knowledge_db or NoOpIngredientKnowledgeDB()
        self._macro_calculator = macro_calculator or MacroCalculator()
        self._nutrition_engine = nutrition_engine or DefaultNutritionEngine()
        self._disease_engine = disease_engine or DefaultDiseaseEngine()
        self._additive_engine = additive_engine or DefaultAdditiveEngine()
        self._nova_engine = nova_engine or DefaultNovaEngine()
        self._recommendation_engine = recommendation_engine or DefaultRecommendationEngine()
        self._health_score_service = health_score_service or HealthScoreService()

    def analyze(self, image_path: str) -> FoodAnalysisResult:
        """
        Execute the full analysis pipeline for a given product image.

        Args:
            image_path: Filesystem path to the product image.

        Returns:
            A FoodAnalysisResult describing the outcome. On failure,
            `success` is False and `message` explains why.
        """
        try:
            path = self._validate_image_path(image_path)
        except ValueError as exc:
            logger.error("Image validation failed: %s", exc)
            return FoodAnalysisResult(success=False, message=str(exc))

        try:
            preprocessed_path = self._image_preprocessor.preprocess(str(path))
        except Exception:
            logger.exception("Image preprocessing failed for: %s", path)
            return FoodAnalysisResult(success=False, message="Image preprocessing failed")

        try:
            ocr_text = self._ocr_engine.extract_text(preprocessed_path)
        except Exception:
            logger.exception("OCR engine failed for: %s", preprocessed_path)
            return FoodAnalysisResult(success=False, message="OCR extraction failed")

        if not ocr_text or not ocr_text.strip():
            logger.warning("OCR produced no text for image: %s", path)
            return FoodAnalysisResult(success=False, message="No text detected on packaging")

        extraction_result = self._ingredient_extractor.extract(ocr_text)
        nutrition_facts = self._nutrition_label_parser.parse(ocr_text)

        if not extraction_result.ingredients and not any(
            v is not None for v in nutrition_facts.to_dict().values()
        ):
            logger.warning("No ingredients or nutrition data could be extracted from: %s", path)
            return FoodAnalysisResult(
                success=False, message="Could not extract ingredient or nutrition data"
            )

        normalization_result = self._ingredient_normalizer.normalize(
            extraction_result.ingredients
        )
        ingredients = normalization_result.normalized_ingredients

        try:
            knowledge = self._knowledge_db.lookup_many(ingredients)
        except Exception:
            logger.exception("Ingredient knowledge DB lookup failed")
            knowledge = {name: {} for name in ingredients}

        macro_profile = self._macro_calculator.calculate(nutrition_facts)

        nutrition_result = self._safe_call(
            "nutrition_engine",
            lambda: self._nutrition_engine.score(nutrition_facts, macro_profile),
        )
        disease_result = self._safe_call(
            "disease_engine",
            lambda: self._disease_engine.score(ingredients, knowledge, nutrition_facts),
        )
        additive_result = self._safe_call(
            "additive_engine",
            lambda: self._additive_engine.score(ingredients, knowledge),
        )
        nova_result = self._safe_call(
            "nova_engine",
            lambda: self._nova_engine.classify(ingredients, knowledge),
        )

        health_score = self._health_score_service.calculate(
            nutrition_score=(nutrition_result or {}).get("score"),
            disease_score=(disease_result or {}).get("score"),
            additive_score=(additive_result or {}).get("score"),
            nova_score=(nova_result or {}).get("score"),
            macro_profile=macro_profile,
            nutrition_insights=nutrition_result,
            disease_insights=disease_result,
            additive_insights=additive_result,
            nova_insights=nova_result,
        )

        recommendations = self._safe_call(
            "recommendation_engine",
            lambda: self._recommendation_engine.recommend(
                ingredients, nutrition_facts, health_score
            ),
        ) or []

        logger.info(
            "Food analysis pipeline completed successfully for image: %s (grade=%s)",
            path,
            health_score.grade,
        )

        return FoodAnalysisResult(
            success=True,
            ingredients=ingredients,
            nutrition_facts=nutrition_facts.to_dict(),
            macro_profile=macro_profile.to_dict(),
            nutrition_engine=nutrition_result,
            disease_engine=disease_result,
            additive_engine=additive_result,
            nova_engine=nova_result,
            recommendations=recommendations,
            health_score=health_score.to_dict(),
        )

    def _validate_image_path(self, image_path: str) -> Path:
        """
        Validate the provided image path exists and is a file.

        Args:
            image_path: Raw path string.

        Returns:
            A resolved pathlib.Path.

        Raises:
            ValueError: If the path is missing or invalid.
        """
        if not image_path:
            raise ValueError("No image path provided")

        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Image not found at path: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path

    def _safe_call(self, engine_name: str, func: Any) -> Optional[Dict[str, Any]]:
        """
        Invoke a sub-engine callable, catching and logging any exception
        so a single engine failure does not abort the whole pipeline.

        Args:
            engine_name: Name of the engine, used for logging context.
            func: A zero-argument callable to invoke.

        Returns:
            The callable's return value, or None if it raised.
        """
        try:
            return func()
        except Exception:
            logger.exception("%s raised an exception during analysis", engine_name)
            return None

    def _default_ocr_engine(self) -> OCREngine:
        """
        Build a default OCR engine adapter around the project's existing
        OCR service, if available.

        Returns:
            An OCREngine-compatible object.
        """
        try:
            from backend.ocr.ocr_service import OCRService

            class _OcrServiceAdapter:
                """Adapts backend.ocr.ocr_service.OCRService to the OCREngine protocol."""

                def __init__(self) -> None:
                    self._service = OCRService()

                def extract_text(self, image_path: str) -> str:
                    """Delegate text extraction to the underlying OCRService."""
                    return self._service.extract_text(image_path)

            return _OcrServiceAdapter()
        except Exception:
            logger.warning(
                "Could not load backend.ocr.ocr_service.OCRService; "
                "OCR extraction will fail until a custom ocr_engine is supplied."
            )

            class _UnavailableOcrEngine:
                """Fallback OCR engine that always fails clearly."""

                def extract_text(self, image_path: str) -> str:
                    raise RuntimeError(
                        "No OCR engine available. Inject an ocr_engine into "
                        "FoodAnalysisPipeline or ensure backend.ocr.ocr_service is importable."
                    )

            return _UnavailableOcrEngine()


def run_food_analysis_pipeline(image_path: str) -> Dict[str, Any]:
    """
    Convenience module-level function for one-off pipeline execution.

    Args:
        image_path: Filesystem path to the product image.

    Returns:
        The analysis result as a JSON-serializable dictionary.
    """
    pipeline = FoodAnalysisPipeline()
    return pipeline.analyze(image_path).to_dict()
