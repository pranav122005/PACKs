"""
backend/services/product_analysis_service.py

Product Analysis Service — the orchestrator that runs a product through
the full PACKS analysis pipeline:

    Product
      -> Nutrition Engine
      -> Disease Engine
      -> Additive Engine
      -> NOVA Engine
      -> Recommendation Engine
      -> Overall Score
      -> Final Report (delegated to ReportEngine)

Each engine is injected via the constructor (Dependency Inversion
Principle), so this service has zero knowledge of *how* any engine
computes its result — only that it conforms to the expected interface.
This makes the pipeline trivially unit-testable with mock engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.engines.additive_engine import AdditiveEngine, AdditiveReport
from backend.engines.disease_engine import DiseaseEngine, DiseaseReport
from backend.engines.nova_engine import NovaEngine, NovaReport
from backend.engines.nutrition_engine import NutritionEngine, NutritionReport
from backend.engines.recommendation_engine import (
    ProductRepositoryProtocol,
    RecommendationEngine,
    RecommendationReport,
)
from backend.schemas.common import Positive, Warning
from backend.utils.config_loader import HealthRulesConfig, get_health_rules_config


@dataclass
class AnalysisResult:
    """Raw, structured output of the full analysis pipeline for one product."""

    product: Dict[str, Any]
    nutrition: NutritionReport
    disease: DiseaseReport
    additives: AdditiveReport
    nova: NovaReport
    recommendations: RecommendationReport
    overall_score: float
    overall_band: str
    warnings: List[Warning]
    positives: List[Positive]


class ProductAnalysisService:
    """
    Coordinates all analysis engines to produce a single, unified
    AnalysisResult for a product. This is the only class that knows the
    pipeline order and how scores from independent engines are blended.
    """

    def __init__(
        self,
        nutrition_engine: Optional[NutritionEngine] = None,
        disease_engine: Optional[DiseaseEngine] = None,
        additive_engine: Optional[AdditiveEngine] = None,
        nova_engine: Optional[NovaEngine] = None,
        recommendation_engine: Optional[RecommendationEngine] = None,
        config: Optional[HealthRulesConfig] = None,
        product_repository: Optional[ProductRepositoryProtocol] = None,
    ) -> None:
        self._config = config or get_health_rules_config()
        self._nutrition_engine = nutrition_engine or NutritionEngine(self._config)
        self._disease_engine = disease_engine or DiseaseEngine(self._config)
        self._additive_engine = additive_engine or AdditiveEngine(self._config)
        self._nova_engine = nova_engine or NovaEngine(self._config)
        self._recommendation_engine = recommendation_engine or RecommendationEngine(
            self._config, product_repository=product_repository
        )
        self._weights = self._config.section("health_score_weights")
        self._bands = self._config.section("health_score_bands")

    def analyze(self, product: Dict[str, Any]) -> AnalysisResult:
        """
        Run the full pipeline for a single product and return a unified
        AnalysisResult. `product` must contain at minimum a `nutriments`
        dict; `ingredients_text`, `product_name`, `barcode`/`code`, and
        `category`/`categories` are used when present.
        """
        self._validate_product(product)

        # 1. Nutrition Engine
        nutrition_report = self._nutrition_engine.analyze(product)

        # 2. Disease Engine
        disease_report = self._disease_engine.analyze(product)

        # 3. Additive Engine
        additive_report = self._additive_engine.analyze(product)

        # 4. NOVA Engine
        nova_report = self._nova_engine.analyze(product)

        # 5. Blend nutrition + additive + nova into the true overall score
        additive_points = self._additive_engine.compute_additive_score_points(
            additive_report, self._weights["additives"]
        )
        nova_points = self._nova_engine.compute_nova_score_points(
            nova_report, self._weights["nova_group"]
        )
        overall_score = self._nutrition_engine.apply_external_score_component(
            nutrition_report, additive_points, nova_points
        )
        overall_band = self._band_for_score(overall_score)

        # 6. Recommendation Engine (needs the final score + NOVA group)
        recommendation_report = self._recommendation_engine.generate(
            product, overall_score, nova_report.nova_group
        )

        # 7. Aggregate warnings/positives across engines that produce them
        warnings = list(nutrition_report.warnings)
        positives = list(nutrition_report.positives)
        warnings.extend(self._additive_warnings(additive_report))
        warnings.extend(self._disease_warnings(disease_report))

        return AnalysisResult(
            product=product,
            nutrition=nutrition_report,
            disease=disease_report,
            additives=additive_report,
            nova=nova_report,
            recommendations=recommendation_report,
            overall_score=overall_score,
            overall_band=overall_band,
            warnings=warnings,
            positives=positives,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_product(product: Dict[str, Any]) -> None:
        if not isinstance(product, dict):
            raise ValueError("product must be a dict/JSON object.")
        if "nutriments" not in product:
            raise ValueError("product JSON must include a 'nutriments' object.")

    def _band_for_score(self, score: float) -> str:
        for band in self._bands.values():
            if band["min"] <= score <= band["max"]:
                return band["label"]
        return "Unknown"

    @staticmethod
    def _additive_warnings(additive_report: AdditiveReport) -> List[Warning]:
        from backend.schemas.common import Severity

        warnings: List[Warning] = []
        for additive in additive_report.detected_additives:
            if additive.risk_level.value in ("high", "very_high"):
                warnings.append(
                    Warning(
                        code=f"additive_{additive.scientific_name.lower().replace(' ', '_')}",
                        title=f"Contains {additive.detected_name}",
                        reason=(
                            f"{additive.detected_name} ({additive.scientific_name}) is classified as "
                            f"{additive.risk_level.value.replace('_', ' ')} risk. Purpose: {additive.purpose}."
                        ),
                        severity=Severity.HIGH if additive.risk_level.value == "very_high" else Severity.MEDIUM,
                        category="additive",
                        recommendation=(
                            f"Consider an alternative such as: {additive.alternative}."
                            if additive.alternative
                            else "Consult product labelling for more information."
                        ),
                    )
                )
        return warnings

    @staticmethod
    def _disease_warnings(disease_report: DiseaseReport) -> List[Warning]:
        from backend.schemas.common import Severity

        warnings: List[Warning] = []
        for assessment in disease_report.assessments:
            if assessment.risk.value in ("high", "very_high"):
                warnings.append(
                    Warning(
                        code=f"disease_{assessment.condition.lower().replace(' ', '_')}",
                        title=f"Risk for {assessment.condition}",
                        reason=assessment.reason,
                        severity=Severity.HIGH,
                        category=f"disease:{assessment.condition.lower().replace(' ', '_')}",
                        recommendation=assessment.recommendation,
                    )
                )
        return warnings
