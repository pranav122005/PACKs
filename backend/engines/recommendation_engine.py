"""
backend/engines/recommendation_engine.py

Recommendation Engine — generates actionable next steps for the user:

    - Healthier alternatives (higher overall health score, same category)
    - Lower sugar alternatives
    - Higher protein alternatives
    - Natural-ingredient alternatives (fewer/no additives, lower NOVA group)
    - Gym / fitness-goal specific recommendation
    - Daily intake recommendation (how this product fits into a daily budget)

This engine depends on an injected repository (ProductRepositoryProtocol)
following the Dependency Inversion Principle: it does not know or care
whether alternatives come from SQLite, an API, or a cache. If no
repository is supplied, it still returns fully-reasoned gym/daily-intake
guidance, just without concrete product alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from backend.utils.config_loader import HealthRulesConfig, get_health_rules_config


class ProductRepositoryProtocol(Protocol):
    """
    Minimal contract the Recommendation Engine needs from any product
    repository implementation (e.g. backend/repositories/product_repository.py).
    """

    def find_alternatives_by_category(
        self,
        category: str,
        exclude_barcode: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Return candidate alternative products (as raw product dicts, same
        shape consumed by the engines) belonging to `category`, excluding
        the current product's barcode, capped at `limit` results.
        """
        ...


@dataclass
class AlternativeProduct:
    """A single suggested alternative product with the reason it was chosen."""

    barcode: Optional[str]
    product_name: str
    brand: Optional[str]
    reason: str
    health_score: Optional[float] = None
    sugar_100g: Optional[float] = None
    protein_100g: Optional[float] = None
    nova_group: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "barcode": self.barcode,
            "product_name": self.product_name,
            "brand": self.brand,
            "reason": self.reason,
            "health_score": self.health_score,
            "sugar_100g": self.sugar_100g,
            "protein_100g": self.protein_100g,
            "nova_group": self.nova_group,
        }


@dataclass
class GymRecommendation:
    """Fitness-context guidance for this specific product."""

    suitable_for_pre_workout: bool
    suitable_for_post_workout: bool
    protein_per_serving_note: str
    guidance: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suitable_for_pre_workout": self.suitable_for_pre_workout,
            "suitable_for_post_workout": self.suitable_for_post_workout,
            "protein_per_serving_note": self.protein_per_serving_note,
            "guidance": self.guidance,
        }


@dataclass
class DailyIntakeRecommendation:
    """How much of this product fits within standard daily reference intakes."""

    calories_pct_of_daily: Optional[float]
    sugar_pct_of_daily: Optional[float]
    salt_pct_of_daily: Optional[float]
    saturated_fat_pct_of_daily: Optional[float]
    protein_pct_of_daily: Optional[float]
    fiber_pct_of_daily: Optional[float]
    guidance: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calories_pct_of_daily": self.calories_pct_of_daily,
            "sugar_pct_of_daily": self.sugar_pct_of_daily,
            "salt_pct_of_daily": self.salt_pct_of_daily,
            "saturated_fat_pct_of_daily": self.saturated_fat_pct_of_daily,
            "protein_pct_of_daily": self.protein_pct_of_daily,
            "fiber_pct_of_daily": self.fiber_pct_of_daily,
            "guidance": self.guidance,
        }


@dataclass
class RecommendationReport:
    """Complete output of the Recommendation Engine for a single product."""

    healthier_alternatives: List[AlternativeProduct] = field(default_factory=list)
    lower_sugar_alternatives: List[AlternativeProduct] = field(default_factory=list)
    higher_protein_alternatives: List[AlternativeProduct] = field(default_factory=list)
    natural_alternatives: List[AlternativeProduct] = field(default_factory=list)
    gym_recommendation: Optional[GymRecommendation] = None
    daily_intake_recommendation: Optional[DailyIntakeRecommendation] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "healthier_alternatives": [a.to_dict() for a in self.healthier_alternatives],
            "lower_sugar_alternatives": [a.to_dict() for a in self.lower_sugar_alternatives],
            "higher_protein_alternatives": [a.to_dict() for a in self.higher_protein_alternatives],
            "natural_alternatives": [a.to_dict() for a in self.natural_alternatives],
            "gym_recommendation": self.gym_recommendation.to_dict() if self.gym_recommendation else None,
            "daily_intake_recommendation": (
                self.daily_intake_recommendation.to_dict() if self.daily_intake_recommendation else None
            ),
        }


class RecommendationEngine:
    """Produces alternative-product suggestions and contextual guidance."""

    def __init__(
        self,
        config: Optional[HealthRulesConfig] = None,
        product_repository: Optional[ProductRepositoryProtocol] = None,
    ) -> None:
        self._config = config or get_health_rules_config()
        self._settings = self._config.section("recommendation_settings")
        self._repository = product_repository

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        product: Dict[str, Any],
        overall_score: float,
        nova_group: int,
    ) -> RecommendationReport:
        """Build the full RecommendationReport for a product given its computed score/NOVA group."""
        report = RecommendationReport()
        category = product.get("category") or product.get("categories", "")
        barcode = product.get("barcode") or product.get("code")
        nutriments = product.get("nutriments", {}) or {}

        candidates: List[Dict[str, Any]] = []
        if self._repository is not None and category:
            limit = self._settings.get("max_alternatives_returned", 5) * 4
            candidates = self._repository.find_alternatives_by_category(category, barcode, limit)

        report.healthier_alternatives = self._rank_healthier(candidates, overall_score)
        report.lower_sugar_alternatives = self._rank_lower_sugar(candidates, nutriments)
        report.higher_protein_alternatives = self._rank_higher_protein(candidates, nutriments)
        report.natural_alternatives = self._rank_natural(candidates, nova_group)

        report.gym_recommendation = self._build_gym_recommendation(nutriments)
        report.daily_intake_recommendation = self._build_daily_intake_recommendation(nutriments)

        return report

    # ------------------------------------------------------------------
    # Alternative ranking
    # ------------------------------------------------------------------

    def _to_alternative(self, candidate: Dict[str, Any], reason: str) -> AlternativeProduct:
        nutriments = candidate.get("nutriments", {}) or {}
        return AlternativeProduct(
            barcode=candidate.get("barcode") or candidate.get("code"),
            product_name=candidate.get("product_name", "Unknown product"),
            brand=candidate.get("brand") or candidate.get("brands"),
            reason=reason,
            health_score=candidate.get("health_score"),
            sugar_100g=nutriments.get("sugars_100g"),
            protein_100g=nutriments.get("proteins_100g"),
            nova_group=candidate.get("nova_group"),
        )

    def _rank_healthier(
        self, candidates: List[Dict[str, Any]], overall_score: float
    ) -> List[AlternativeProduct]:
        min_improvement = self._settings.get("min_score_improvement", 10)
        limit = self._settings.get("max_alternatives_returned", 5)
        eligible = [
            c for c in candidates
            if isinstance(c.get("health_score"), (int, float))
            and c["health_score"] >= overall_score + min_improvement
        ]
        eligible.sort(key=lambda c: c["health_score"], reverse=True)
        return [
            self._to_alternative(
                c,
                f"Scores {round(c['health_score'] - overall_score, 1)} points higher on overall health score.",
            )
            for c in eligible[:limit]
        ]

    def _rank_lower_sugar(
        self, candidates: List[Dict[str, Any]], current_nutriments: Dict[str, Any]
    ) -> List[AlternativeProduct]:
        limit = self._settings.get("max_alternatives_returned", 5)
        current_sugar = current_nutriments.get("sugars_100g")
        if current_sugar is None:
            return []
        eligible = []
        for c in candidates:
            candidate_sugar = (c.get("nutriments", {}) or {}).get("sugars_100g")
            if isinstance(candidate_sugar, (int, float)) and candidate_sugar < current_sugar:
                eligible.append((c, candidate_sugar))
        eligible.sort(key=lambda pair: pair[1])
        return [
            self._to_alternative(
                c, f"Contains {round(current_sugar - sugar, 1)}g less sugar per 100g."
            )
            for c, sugar in eligible[:limit]
        ]

    def _rank_higher_protein(
        self, candidates: List[Dict[str, Any]], current_nutriments: Dict[str, Any]
    ) -> List[AlternativeProduct]:
        limit = self._settings.get("max_alternatives_returned", 5)
        current_protein = current_nutriments.get("proteins_100g") or 0.0
        eligible = []
        for c in candidates:
            candidate_protein = (c.get("nutriments", {}) or {}).get("proteins_100g")
            if isinstance(candidate_protein, (int, float)) and candidate_protein > current_protein:
                eligible.append((c, candidate_protein))
        eligible.sort(key=lambda pair: pair[1], reverse=True)
        return [
            self._to_alternative(
                c, f"Provides {round(protein - current_protein, 1)}g more protein per 100g."
            )
            for c, protein in eligible[:limit]
        ]

    def _rank_natural(
        self, candidates: List[Dict[str, Any]], current_nova_group: int
    ) -> List[AlternativeProduct]:
        limit = self._settings.get("max_alternatives_returned", 5)
        eligible = []
        for c in candidates:
            candidate_nova = c.get("nova_group")
            if isinstance(candidate_nova, int) and candidate_nova < current_nova_group:
                eligible.append(c)
        eligible.sort(key=lambda c: c.get("nova_group", 4))
        return [
            self._to_alternative(
                c, f"Less processed (NOVA {c.get('nova_group')}) with fewer artificial ingredients."
            )
            for c in eligible[:limit]
        ]

    # ------------------------------------------------------------------
    # Contextual guidance
    # ------------------------------------------------------------------

    def _build_gym_recommendation(self, nutriments: Dict[str, Any]) -> GymRecommendation:
        protein = nutriments.get("proteins_100g") or 0.0
        sugar = nutriments.get("sugars_100g") or 0.0
        target = self._settings.get("gym_protein_target_g_per_meal", 20)

        suitable_post = protein >= target * 0.5
        suitable_pre = sugar >= 5 and protein < target * 0.5

        if suitable_post:
            guidance = (
                f"Good post-workout choice: provides {protein}g of protein per 100g, supporting "
                "muscle recovery. Pair with a fast-digesting carbohydrate source."
            )
        elif suitable_pre:
            guidance = (
                f"Better suited pre-workout: {sugar}g sugar per 100g offers quick energy, but low "
                f"protein ({protein}g/100g) means it won't aid recovery on its own."
            )
        else:
            guidance = (
                f"Not particularly optimized for gym use — protein ({protein}g/100g) is below the "
                f"typical per-meal target of {target}g. Consider a dedicated protein source around workouts."
            )

        return GymRecommendation(
            suitable_for_pre_workout=suitable_pre,
            suitable_for_post_workout=suitable_post,
            protein_per_serving_note=f"{protein}g protein per 100g",
            guidance=guidance,
        )

    def _build_daily_intake_recommendation(self, nutriments: Dict[str, Any]) -> DailyIntakeRecommendation:
        def pct(value: Optional[float], reference: float) -> Optional[float]:
            if value is None or not reference:
                return None
            return round((value / reference) * 100, 1)

        calories = nutriments.get("energy_kcal_100g")
        sugar = nutriments.get("sugars_100g")
        salt = nutriments.get("salt_100g")
        saturated_fat = nutriments.get("saturated_fat_100g")
        protein = nutriments.get("proteins_100g")
        fiber = nutriments.get("fiber_100g")

        calories_pct = pct(calories, self._settings.get("daily_calorie_reference_kcal", 2000))
        sugar_pct = pct(sugar, self._settings.get("daily_sugar_reference_g", 25))
        salt_pct = pct(salt, self._settings.get("daily_salt_reference_g", 5))
        sat_fat_pct = pct(saturated_fat, self._settings.get("daily_saturated_fat_reference_g", 20))
        protein_pct = pct(protein, self._settings.get("daily_protein_reference_g", 50))
        fiber_pct = pct(fiber, self._settings.get("daily_fiber_reference_g", 30))

        notable = []
        if sugar_pct and sugar_pct >= 30:
            notable.append(f"{sugar_pct}% of your daily sugar reference in just 100g")
        if salt_pct and salt_pct >= 30:
            notable.append(f"{salt_pct}% of your daily salt reference in just 100g")
        if sat_fat_pct and sat_fat_pct >= 30:
            notable.append(f"{sat_fat_pct}% of your daily saturated fat reference in just 100g")

        if notable:
            guidance = "Eating just 100g of this product uses up " + "; ".join(notable) + ". Portion control is important."
        else:
            guidance = "This product fits comfortably within standard daily nutrient references at a 100g serving."

        return DailyIntakeRecommendation(
            calories_pct_of_daily=calories_pct,
            sugar_pct_of_daily=sugar_pct,
            salt_pct_of_daily=salt_pct,
            saturated_fat_pct_of_daily=sat_fat_pct,
            protein_pct_of_daily=protein_pct,
            fiber_pct_of_daily=fiber_pct,
            guidance=guidance,
        )
