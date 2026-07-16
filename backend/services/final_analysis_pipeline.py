"""
backend/services/final_analysis_pipeline.py

PACKS - AI Ingredient Intelligence Platform
====================================================
This is THE single, canonical orchestrator for the entire product
analysis flow. It wires together every stage of the system — from raw
image to a final structured JSON report — and is the only module
callers should invoke to run a full analysis.

Flow:
    Image
        -> Image Preprocessing
        -> OCR
        -> Text Cleaning (TextPostprocessor)
        -> OCR Layout Detection
        -> Ingredient Section Detection
        -> Nutrition Table Detection
        -> Ingredient Extraction
        -> Ingredient Normalization
        -> Knowledge Service
        -> Ingredient Analysis
        -> Nutrition Analysis (+ Macro Calculator, Nutrition Engine)
        -> Disease Engine
        -> Additive Engine
        -> NOVA Engine
        -> Recommendation Engine
        -> Health Score Service
        -> AI Summary Service
        -> Confidence Scoring
        -> Final structured JSON
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from backend.services.text_postprocessor import TextPostprocessor
from backend.services.ocr_layout_detector import OCRLayoutDetector
from backend.services.ingredient_section_detector import IngredientSectionDetector
from backend.services.nutrition_table_detector import NutritionTableDetector
from backend.services.ingredient_extractor import IngredientExtractor
from backend.services.ingredient_normalizer import IngredientNormalizer
from backend.services.knowledge_service import KnowledgeService
from backend.services.ingredient_analysis_service import (
    IngredientAnalysisResult,
    IngredientAnalysisService,
)
from backend.services.nutrition_analysis_service import (
    NutritionAnalysisResult,
    NutritionAnalysisService,
)
from backend.services.nutrition_label_parser import NutritionFacts
from backend.services.macro_calculator import MacroCalculator, MacroProfile
from backend.services.health_score_service import HealthScoreResult, HealthScoreService
from backend.services.confidence_score import ConfidenceScorer, ConfidenceScoreResult
from backend.services.food_analysis_pipeline import (
    DefaultAdditiveEngine,
    DefaultDiseaseEngine,
    DefaultNovaEngine,
    DefaultRecommendationEngine,
    NoOpImagePreprocessor,
)

logger = logging.getLogger("packs.final_analysis_pipeline")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------- #
# Pluggable interfaces
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


class AISummaryEngine(Protocol):
    """Generates a natural-language summary of the analysis."""

    def summarize(
        self,
        ingredients: List[str],
        nutrition_analysis: Optional[NutritionAnalysisResult],
        ingredient_analysis: Optional[IngredientAnalysisResult],
        health_score: HealthScoreResult,
    ) -> str:
        """Return a human-readable summary string."""
        ...


class DefaultAISummaryEngine:
    """Rule-based fallback AI summary generator (no external LLM call)."""

    def summarize(
        self,
        ingredients: List[str],
        nutrition_analysis: Optional[NutritionAnalysisResult],
        ingredient_analysis: Optional[IngredientAnalysisResult],
        health_score: HealthScoreResult,
    ) -> str:
        """Build a concise rule-based natural-language summary."""
        parts: List[str] = []

        parts.append(
            f"This product received an overall health grade of {health_score.grade} "
            f"({health_score.overall_score}/100)."
        )

        if ingredient_analysis and ingredient_analysis.additives:
            preview = ", ".join(ingredient_analysis.additives[:5])
            parts.append(
                f"It contains {len(ingredient_analysis.additives)} detected additive(s), "
                f"including: {preview}."
            )
        else:
            parts.append("No significant additives were detected in the ingredient list.")

        if nutrition_analysis and nutrition_analysis.warnings:
            parts.append(
                f"Key nutrition concerns: {'; '.join(nutrition_analysis.warnings[:3])}."
            )

        if health_score.positive_points:
            parts.append(f"Positives: {'; '.join(health_score.positive_points[:3])}.")

        if not ingredients:
            parts.append("No ingredient list could be confidently extracted from the packaging.")

        return " ".join(parts)


# ---------------------------------------------------------------------- #
# Final report dataclass
# ---------------------------------------------------------------------- #
@dataclass
class FinalAnalysisReport:
    """Final structured report returned by the final analysis pipeline."""

    success: bool
    ingredients: List[str] = field(default_factory=list)
    nutrition: Dict[str, Any] = field(default_factory=dict)
    health_score: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    positives: List[str] = field(default_factory=list)
    nova: Dict[str, Any] = field(default_factory=dict)
    disease_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    ai_summary: str = ""
    confidence: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the final report as a plain JSON-serializable dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------- #
# Orchestrator
# ---------------------------------------------------------------------- #
class FinalAnalysisPipeline:
    """
    The single canonical orchestrator for PACKS product analysis.

    Wires together image preprocessing, OCR, text cleaning, layout
    detection, section-specific detectors, extraction, normalization,
    knowledge enrichment, all scoring engines, recommendations, health
    scoring, AI summarization, and confidence scoring into one
    coherent, resilient pipeline. Every sub-component is injected via
    the constructor (Dependency Inversion), defaulting to standard
    implementations so the pipeline is fully functional out of the box.
    """

    def __init__(
        self,
        image_preprocessor: Optional[ImagePreprocessor] = None,
        ocr_engine: Optional[OCREngine] = None,
        text_postprocessor: Optional[TextPostprocessor] = None,
        ocr_layout_detector: Optional[OCRLayoutDetector] = None,
        ingredient_section_detector: Optional[IngredientSectionDetector] = None,
        nutrition_table_detector: Optional[NutritionTableDetector] = None,
        ingredient_extractor: Optional[IngredientExtractor] = None,
        ingredient_normalizer: Optional[IngredientNormalizer] = None,
        knowledge_service: Optional[KnowledgeService] = None,
        ingredient_analysis_service: Optional[IngredientAnalysisService] = None,
        nutrition_analysis_service: Optional[NutritionAnalysisService] = None,
        macro_calculator: Optional[MacroCalculator] = None,
        disease_engine: Optional[Any] = None,
        additive_engine: Optional[Any] = None,
        nova_engine: Optional[Any] = None,
        recommendation_engine: Optional[Any] = None,
        health_score_service: Optional[HealthScoreService] = None,
        ai_summary_engine: Optional[AISummaryEngine] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
    ) -> None:
        """
        Args:
            image_preprocessor: Prepares the raw image for OCR.
            ocr_engine: Extracts raw text from the (preprocessed) image.
            text_postprocessor: Cleans raw OCR text.
            ocr_layout_detector: Detects coarse layout regions via OpenCV.
            ingredient_section_detector: Isolates ingredient-only text.
            nutrition_table_detector: Isolates nutrition-table-only text.
            ingredient_extractor: Extracts ingredient list from cleaned text.
            ingredient_normalizer: Normalizes/deduplicates ingredient names.
            knowledge_service: Provides ingredient knowledge enrichment.
            ingredient_analysis_service: Categorizes additives/risk.
            nutrition_analysis_service: Produces macro/nutrition report.
            macro_calculator: Computes macro-nutrient ratios/densities.
            disease_engine: Flags disease-relevant risks.
            additive_engine: Assesses additive risk.
            nova_engine: Classifies processing level (NOVA).
            recommendation_engine: Produces user-facing recommendations.
            health_score_service: Aggregates all engine outputs into a
                final health score.
            ai_summary_engine: Generates the natural-language summary.
            confidence_scorer: Computes stage-level and overall
                extraction confidence.
        """
        self._image_preprocessor = image_preprocessor or NoOpImagePreprocessor()
        self._ocr_engine = ocr_engine or self._default_ocr_engine()
        self._text_postprocessor = text_postprocessor or TextPostprocessor()
        self._ocr_layout_detector = ocr_layout_detector or OCRLayoutDetector()
        self._ingredient_section_detector = (
            ingredient_section_detector or IngredientSectionDetector()
        )
        self._nutrition_table_detector = nutrition_table_detector or NutritionTableDetector()
        self._ingredient_extractor = ingredient_extractor or IngredientExtractor()
        self._ingredient_normalizer = ingredient_normalizer or IngredientNormalizer()
        self._knowledge_service = knowledge_service or KnowledgeService()
        self._ingredient_analysis_service = (
            ingredient_analysis_service or IngredientAnalysisService()
        )
        self._nutrition_analysis_service = (
            nutrition_analysis_service or NutritionAnalysisService()
        )
        self._macro_calculator = macro_calculator or MacroCalculator()
        self._disease_engine = disease_engine or DefaultDiseaseEngine()
        self._additive_engine = additive_engine or DefaultAdditiveEngine()
        self._nova_engine = nova_engine or DefaultNovaEngine()
        self._recommendation_engine = recommendation_engine or DefaultRecommendationEngine()
        self._health_score_service = health_score_service or HealthScoreService()
        self._ai_summary_engine = ai_summary_engine or DefaultAISummaryEngine()
        self._confidence_scorer = confidence_scorer or ConfidenceScorer()

    def analyze(self, image_path: str) -> FinalAnalysisReport:
        """
        Execute the full, end-to-end analysis pipeline for a product
        image and return the final structured report.

        Args:
            image_path: Filesystem path to the product packaging image.

        Returns:
            FinalAnalysisReport describing the complete analysis
            outcome. On failure, `success` is False and `message`
            explains why.
        """
        try:
            path = self._validate_image_path(image_path)
        except ValueError as exc:
            logger.error("Image validation failed: %s", exc)
            return FinalAnalysisReport(success=False, message=str(exc))

        preprocessed_path = self._safe_call(
            "image_preprocessor", lambda: self._image_preprocessor.preprocess(str(path))
        )
        if preprocessed_path is None:
            return FinalAnalysisReport(success=False, message="Image preprocessing failed")

        raw_ocr_text = self._safe_call(
            "ocr_engine", lambda: self._ocr_engine.extract_text(preprocessed_path)
        )
        if not raw_ocr_text or not raw_ocr_text.strip():
            logger.warning("OCR produced no text for image: %s", path)
            return FinalAnalysisReport(success=False, message="No text detected on packaging")

        postprocess_result = self._safe_call(
            "text_postprocessor", lambda: self._text_postprocessor.process(raw_ocr_text)
        )
        cleaned_text = (
            postprocess_result.cleaned_text if postprocess_result else raw_ocr_text
        )
        corrections_applied = (
            postprocess_result.corrections_applied if postprocess_result else 0
        )

        # Layout detection is advisory/diagnostic; failures here must
        # not abort the pipeline since text-based detectors are the
        # primary extraction path.
        self._safe_call(
            "ocr_layout_detector", lambda: self._ocr_layout_detector.detect(str(path))
        )

        ingredient_section_result = self._safe_call(
            "ingredient_section_detector",
            lambda: self._ingredient_section_detector.detect(cleaned_text),
        )
        nutrition_table_result = self._safe_call(
            "nutrition_table_detector",
            lambda: self._nutrition_table_detector.detect(cleaned_text),
        )

        extraction_result = self._safe_call(
            "ingredient_extractor",
            lambda: self._ingredient_extractor.extract(cleaned_text),
        )
        raw_ingredients = extraction_result.ingredients if extraction_result else []
        header_matched = extraction_result.header_matched if extraction_result else None
        excluded_line_count = (
            ingredient_section_result.excluded_line_count if ingredient_section_result else 0
        )

        nutrition_facts = self._build_nutrition_facts(nutrition_table_result, cleaned_text)

        if not raw_ingredients and not any(v is not None for v in nutrition_facts.to_dict().values()):
            logger.warning("No ingredients or nutrition data extracted from: %s", path)
            return FinalAnalysisReport(
                success=False, message="Could not extract ingredient or nutrition data"
            )

        normalization_result = self._safe_call(
            "ingredient_normalizer",
            lambda: self._ingredient_normalizer.normalize(raw_ingredients),
        )
        ingredients = (
            normalization_result.normalized_ingredients if normalization_result else []
        )

        knowledge = self._safe_call(
            "knowledge_service", lambda: self._knowledge_service.lookup_many(ingredients)
        ) or {}

        ingredient_analysis = self._safe_call(
            "ingredient_analysis_service",
            lambda: self._ingredient_analysis_service.analyze(ingredients, knowledge),
        )

        nutrition_analysis = self._safe_call(
            "nutrition_analysis_service",
            lambda: self._nutrition_analysis_service.analyze(nutrition_facts),
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

        macro_profile = self._safe_call(
            "macro_calculator", lambda: self._macro_calculator.calculate(nutrition_facts)
        ) or MacroProfile()

        nutrition_score = (
            nutrition_analysis.nutrition_score
            if isinstance(nutrition_analysis, NutritionAnalysisResult)
            else None
        )

        health_score = self._health_score_service.calculate(
            nutrition_score=nutrition_score,
            disease_score=(disease_result or {}).get("score"),
            additive_score=(additive_result or {}).get("score"),
            nova_score=(nova_result or {}).get("score"),
            macro_profile=macro_profile,
            nutrition_insights={
                "positives": [],
                "warnings": (
                    nutrition_analysis.warnings
                    if isinstance(nutrition_analysis, NutritionAnalysisResult)
                    else []
                ),
            },
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

        ai_summary = self._safe_call(
            "ai_summary_engine",
            lambda: self._ai_summary_engine.summarize(
                ingredients, nutrition_analysis, ingredient_analysis, health_score
            ),
        ) or ""

        confidence = self._compute_confidence(
            raw_ocr_text=raw_ocr_text,
            corrections_applied=corrections_applied,
            ingredients=ingredients,
            header_matched=header_matched,
            excluded_line_count=excluded_line_count,
            nutrition_facts=nutrition_facts,
        )

        logger.info(
            "Final analysis pipeline completed successfully for: %s (grade=%s, overall_confidence=%.2f)",
            path,
            health_score.grade,
            confidence.overall_confidence,
        )

        return FinalAnalysisReport(
            success=True,
            ingredients=ingredients,
            nutrition=nutrition_facts.to_dict(),
            health_score=health_score.to_dict(),
            warnings=health_score.warnings,
            positives=health_score.positive_points,
            nova=nova_result or {},
            disease_analysis=disease_result or {},
            recommendations=recommendations,
            ai_summary=ai_summary,
            confidence=confidence.to_dict(),
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

    def _build_nutrition_facts(
        self, nutrition_table_result: Any, cleaned_text: str
    ) -> NutritionFacts:
        """
        Build a NutritionFacts object from the nutrition table
        detector's result, falling back to an empty NutritionFacts on
        failure.

        Args:
            nutrition_table_result: Result from NutritionTableDetector,
                or None if detection failed.
            cleaned_text: Full cleaned OCR text (unused fallback source,
                kept for future extension).

        Returns:
            A populated NutritionFacts dataclass.
        """
        if nutrition_table_result is None:
            return NutritionFacts()

        return NutritionFacts(
            serving_size=nutrition_table_result.serving_size,
            calories=nutrition_table_result.calories,
            energy_kj=None,
            protein_g=nutrition_table_result.protein_g,
            fat_g=nutrition_table_result.fat_g,
            saturated_fat_g=nutrition_table_result.saturated_fat_g,
            trans_fat_g=nutrition_table_result.trans_fat_g,
            carbohydrates_g=nutrition_table_result.carbohydrates_g,
            sugar_g=nutrition_table_result.sugar_g,
            added_sugar_g=nutrition_table_result.added_sugar_g,
            fiber_g=nutrition_table_result.fiber_g,
            sodium_mg=nutrition_table_result.sodium_mg,
            salt_g=nutrition_table_result.salt_g,
        )

    def _compute_confidence(
        self,
        raw_ocr_text: str,
        corrections_applied: int,
        ingredients: List[str],
        header_matched: Optional[str],
        excluded_line_count: int,
        nutrition_facts: NutritionFacts,
    ) -> ConfidenceScoreResult:
        """
        Compute stage-level and overall confidence scores for this
        analysis run.

        Args:
            raw_ocr_text: Raw OCR text prior to cleaning.
            corrections_applied: Number of corrections the text
                postprocessor applied.
            ingredients: Final normalized ingredient list.
            header_matched: Ingredient section header matched, if any.
            excluded_line_count: Number of noise lines filtered from
                the ingredient section.
            nutrition_facts: Parsed nutrition facts.

        Returns:
            ConfidenceScoreResult with OCR, ingredient, nutrition, and
            overall confidence values. Falls back to zeroed confidence
            on unexpected failure.
        """
        try:
            ocr_confidence = self._confidence_scorer.score_ocr(raw_ocr_text, corrections_applied)
            ingredient_confidence = self._confidence_scorer.score_ingredient_extraction(
                ingredients, header_matched, excluded_line_count
            )
            nutrition_confidence = self._confidence_scorer.score_nutrition_parsing(
                nutrition_facts.to_dict()
            )
            return self._confidence_scorer.score_overall(
                ocr_confidence, ingredient_confidence, nutrition_confidence
            )
        except Exception:
            logger.exception("Confidence scoring failed")
            return ConfidenceScoreResult(
                ocr_confidence=0.0,
                ingredient_confidence=0.0,
                nutrition_confidence=0.0,
                overall_confidence=0.0,
            )

    def _safe_call(self, name: str, func: Any) -> Optional[Any]:
        """
        Invoke a sub-component callable, catching and logging any
        exception so a single stage failure does not abort the whole
        pipeline (except for OCR, whose absence is treated as fatal by
        the caller checking its return value).

        Args:
            name: Component name, used for logging context.
            func: A zero-argument callable to invoke.

        Returns:
            The callable's return value, or None if it raised.
        """
        try:
            return func()
        except Exception:
            logger.exception("%s raised an exception during pipeline execution", name)
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
                        "FinalAnalysisPipeline or ensure backend.ocr.ocr_service is importable."
                    )

            return _UnavailableOcrEngine()


def run_final_analysis_pipeline(image_path: str) -> Dict[str, Any]:
    """
    Convenience module-level function for one-off pipeline execution.

    Args:
        image_path: Filesystem path to the product packaging image.

    Returns:
        The final analysis report as a JSON-serializable dictionary.
    """
    pipeline = FinalAnalysisPipeline()
    return pipeline.analyze(image_path).to_dict()