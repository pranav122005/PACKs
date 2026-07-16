"""
backend/engines/nutrition_engine.py

Nutrition Engine — analyses the raw nutrient facts of a product (per 100g/100ml)
and produces a fully explained NutritionReport: per-nutrient verdicts, a
weighted health score (0-100), and human-readable warnings/positives.

Design goals
------------
- No hardcoded thresholds: everything is read from health_rules.json via
  HealthRulesConfig.
- Every point gained or lost is explained (ScoreBreakdownItem).
- Pure, side-effect free: given the same product JSON + config, always
  produces the same report. Safe to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.schemas.common import Positive, ScoreBreakdownItem, Severity, Warning
from backend.utils.config_loader import HealthRulesConfig, get_health_rules_config


@dataclass
class NutrientVerdict:
    """Verdict for a single nutrient (e.g. sugar) at its measured value."""

    nutrient: str
    value: Optional[float]
    unit: str
    level: str  # "low" | "moderate" | "high" | "very_high" | "unknown"
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nutrient": self.nutrient,
            "value": self.value,
            "unit": self.unit,
            "level": self.level,
            "explanation": self.explanation,
        }


@dataclass
class NutritionReport:
    """Complete output of the Nutrition Engine for a single product."""

    per_100g_basis: bool
    verdicts: List[NutrientVerdict] = field(default_factory=list)
    score_breakdown: List[ScoreBreakdownItem] = field(default_factory=list)
    health_score: float = 0.0
    health_band: str = "unknown"
    warnings: List[Warning] = field(default_factory=list)
    positives: List[Positive] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_100g_basis": self.per_100g_basis,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "score_breakdown": [s.to_dict() for s in self.score_breakdown],
            "health_score": round(self.health_score, 2),
            "health_band": self.health_band,
            "warnings": [w.to_dict() for w in self.warnings],
            "positives": [p.to_dict() for p in self.positives],
        }


class NutritionEngine:
    """
    Computes a NutritionReport from a product's nutrient facts.

    Expected product JSON shape (extra keys are ignored, missing keys are
    treated as unknown/None rather than raising):

    {
        "product_name": str,
        "nutriments": {
            "energy_kcal_100g": float,
            "sugars_100g": float,
            "salt_100g": float,            # grams of salt per 100g
            "sodium_100g": float,          # optional, mg per 100g (fallback)
            "proteins_100g": float,
            "fiber_100g": float,
            "fat_100g": float,
            "saturated_fat_100g": float,
            "trans_fat_100g": float
        }
    }
    """

    def __init__(self, config: Optional[HealthRulesConfig] = None) -> None:
        self._config = config or get_health_rules_config()
        self._thresholds = self._config.section("nutrition_thresholds")
        self._weights = self._config.section("health_score_weights")
        self._bands = self._config.section("health_score_bands")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, product: Dict[str, Any]) -> NutritionReport:
        """Run the full nutrition analysis pipeline for a single product."""
        nutriments = product.get("nutriments", {}) or {}

        report = NutritionReport(per_100g_basis=True)

        calories = self._to_float(nutriments.get("energy_kcal_100g"))
        sugar = self._to_float(nutriments.get("sugars_100g"))
        salt = self._resolve_salt(nutriments)
        protein = self._to_float(nutriments.get("proteins_100g"))
        fiber = self._to_float(nutriments.get("fiber_100g"))
        total_fat = self._to_float(nutriments.get("fat_100g"))
        saturated_fat = self._to_float(nutriments.get("saturated_fat_100g"))
        trans_fat = self._to_float(nutriments.get("trans_fat_100g"))

        calorie_points = self._analyze_calories(calories, report)
        sugar_points = self._analyze_sugar(sugar, report)
        salt_points = self._analyze_salt(salt, report)
        sat_fat_points = self._analyze_saturated_fat(saturated_fat, report)
        trans_fat_points = self._analyze_trans_fat(trans_fat, report)
        protein_points = self._analyze_protein(protein, report)
        fiber_points = self._analyze_fiber(fiber, report)
        self._analyze_total_fat(total_fat, report)

        # additive_engine / nova_engine scores are injected later by the
        # orchestrator via `apply_external_score_component`, since this
        # engine only owns nutrient-derived nutrition facts.
        weighted_points = (
            calorie_points
            + sugar_points
            + salt_points
            + sat_fat_points
            + trans_fat_points
            + protein_points
            + fiber_points
        )
        max_nutrition_points = sum(
            self._weights[key]
            for key in (
                "calories",
                "sugar",
                "salt",
                "saturated_fat",
                "trans_fat",
                "protein",
                "fiber",
            )
        )

        # Scale nutrition-only points onto a 0-100 basis for this report.
        # (The orchestrator later blends this with additive/NOVA scores
        # using the full weight table for the true overall_score.)
        report.health_score = (
            (weighted_points / max_nutrition_points) * 100 if max_nutrition_points else 0.0
        )
        report.health_band = self._band_for_score(report.health_score)

        return report

    def apply_external_score_component(
        self,
        report: NutritionReport,
        additive_points: float,
        nova_points: float,
    ) -> float:
        """
        Blend additive-engine and nova-engine point contributions into a
        final, fully-weighted 0-100 score. Returns the blended score without
        mutating the nutrition-only score already stored on the report.
        """
        nutrition_weight_sum = sum(
            self._weights[key]
            for key in (
                "calories",
                "sugar",
                "salt",
                "saturated_fat",
                "trans_fat",
                "protein",
                "fiber",
            )
        )
        total_weight_sum = sum(self._weights.values())
        nutrition_component = (report.health_score / 100) * nutrition_weight_sum
        blended = nutrition_component + additive_points + nova_points
        return (blended / total_weight_sum) * 100 if total_weight_sum else 0.0

    # ------------------------------------------------------------------
    # Per-nutrient analysis
    # ------------------------------------------------------------------

    def _analyze_calories(self, value: Optional[float], report: NutritionReport) -> float:
        weight = self._weights["calories"]
        t = self._thresholds["calories"]
        if value is None:
            report.verdicts.append(
                NutrientVerdict("calories", None, t["unit"], "unknown", "Calorie data not available.")
            )
            report.score_breakdown.append(
                ScoreBreakdownItem("calories", weight, weight * 0.5, "No calorie data provided; assumed neutral.")
            )
            return weight * 0.5

        if value <= t["low"]:
            level, ratio = "low", 1.0
            explanation = f"Low calorie density ({value} kcal/100g)."
            report.positives.append(
                Positive("low_calorie", "Low calorie", explanation, "calories", value, "kcal")
            )
        elif value <= t["moderate"]:
            level, ratio = "moderate", 0.8
            explanation = f"Moderate calorie density ({value} kcal/100g)."
        elif value <= t["high"]:
            level, ratio = "high", 0.4
            explanation = f"High calorie density ({value} kcal/100g). Consider portion control."
            report.warnings.append(
                Warning(
                    "high_calorie", "High calorie", explanation, Severity.MEDIUM, "calories", value, "kcal",
                    recommendation="Limit portion size or pair with an active lifestyle.",
                )
            )
        else:
            level, ratio = "very_high", 0.1
            explanation = f"Very high calorie density ({value} kcal/100g). This is calorie-dense food."
            report.warnings.append(
                Warning(
                    "very_high_calorie", "Very high calorie", explanation, Severity.HIGH, "calories", value, "kcal",
                    recommendation="Consume in small portions and infrequently.",
                )
            )

        report.verdicts.append(NutrientVerdict("calories", value, t["unit"], level, explanation))
        awarded = weight * ratio
        report.score_breakdown.append(ScoreBreakdownItem("calories", weight, awarded, explanation))
        return awarded

    def _analyze_sugar(self, value: Optional[float], report: NutritionReport) -> float:
        weight = self._weights["sugar"]
        t = self._thresholds["sugar"]
        if value is None:
            report.verdicts.append(
                NutrientVerdict("sugar", None, t["unit"], "unknown", "Sugar data not available.")
            )
            report.score_breakdown.append(
                ScoreBreakdownItem("sugar", weight, weight * 0.5, "No sugar data provided; assumed neutral.")
            )
            return weight * 0.5

        daily_limit = t.get("who_daily_limit_g", 25)
        pct_of_daily = round((value / daily_limit) * 100, 1) if daily_limit else None

        if value <= t["low"]:
            level, ratio = "low", 1.0
            explanation = f"Low sugar content ({value}g/100g)."
            report.positives.append(Positive("low_sugar", "Low sugar", explanation, "sugar", value, "g"))
        elif value <= t["moderate"]:
            level, ratio = "moderate", 0.75
            explanation = f"Moderate sugar content ({value}g/100g)."
        elif value <= t["high"]:
            level, ratio = "high", 0.35
            explanation = (
                f"High sugar content ({value}g/100g), roughly {pct_of_daily}% of the WHO recommended "
                f"daily free-sugar limit ({daily_limit}g) in just 100g."
            )
            report.warnings.append(
                Warning(
                    "high_sugar", "High sugar", explanation, Severity.HIGH, "sugar", value, "g",
                    recommendation="Choose a lower-sugar alternative or limit serving size.",
                )
            )
        else:
            level, ratio = "very_high", 0.05
            explanation = (
                f"Very high sugar content ({value}g/100g), roughly {pct_of_daily}% of the WHO recommended "
                f"daily free-sugar limit ({daily_limit}g) in just 100g."
            )
            report.warnings.append(
                Warning(
                    "very_high_sugar", "Very high sugar", explanation, Severity.CRITICAL, "sugar", value, "g",
                    recommendation="Avoid frequent consumption; strongly consider a lower-sugar alternative.",
                )
            )

        report.verdicts.append(NutrientVerdict("sugar", value, t["unit"], level, explanation))
        awarded = weight * ratio
        report.score_breakdown.append(ScoreBreakdownItem("sugar", weight, awarded, explanation))
        return awarded

    def _analyze_salt(self, value: Optional[float], report: NutritionReport) -> float:
        weight = self._weights["salt"]
        t = self._thresholds["salt"]
        if value is None:
            report.verdicts.append(
                NutrientVerdict("salt", None, t["unit"], "unknown", "Salt/sodium data not available.")
            )
            report.score_breakdown.append(
                ScoreBreakdownItem("salt", weight, weight * 0.5, "No salt data provided; assumed neutral.")
            )
            return weight * 0.5

        daily_limit = t.get("who_daily_limit_g", 5)
        pct_of_daily = round((value / daily_limit) * 100, 1) if daily_limit else None

        if value <= t["low"]:
            level, ratio = "low", 1.0
            explanation = f"Low salt content ({value}g/100g)."
            report.positives.append(Positive("low_salt", "Low salt", explanation, "salt", value, "g"))
        elif value <= t["moderate"]:
            level, ratio = "moderate", 0.75
            explanation = f"Moderate salt content ({value}g/100g)."
        elif value <= t["high"]:
            level, ratio = "high", 0.35
            explanation = (
                f"High salt content ({value}g/100g), roughly {pct_of_daily}% of the WHO recommended "
                f"daily limit ({daily_limit}g) in just 100g."
            )
            report.warnings.append(
                Warning(
                    "high_salt", "High salt", explanation, Severity.HIGH, "salt", value, "g",
                    recommendation="Watch total daily sodium intake; avoid pairing with other salty foods.",
                )
            )
        else:
            level, ratio = "very_high", 0.05
            explanation = (
                f"Very high salt content ({value}g/100g), roughly {pct_of_daily}% of the WHO recommended "
                f"daily limit ({daily_limit}g) in just 100g."
            )
            report.warnings.append(
                Warning(
                    "very_high_salt", "Very high salt", explanation, Severity.CRITICAL, "salt", value, "g",
                    recommendation="Avoid if managing blood pressure; strongly consider a lower-sodium alternative.",
                )
            )

        report.verdicts.append(NutrientVerdict("salt", value, t["unit"], level, explanation))
        awarded = weight * ratio
        report.score_breakdown.append(ScoreBreakdownItem("salt", weight, awarded, explanation))
        return awarded

    def _analyze_saturated_fat(self, value: Optional[float], report: NutritionReport) -> float:
        weight = self._weights["saturated_fat"]
        t = self._thresholds["saturated_fat"]
        if value is None:
            report.verdicts.append(
                NutrientVerdict("saturated_fat", None, t["unit"], "unknown", "Saturated fat data not available.")
            )
            report.score_breakdown.append(
                ScoreBreakdownItem(
                    "saturated_fat", weight, weight * 0.5, "No saturated fat data provided; assumed neutral."
                )
            )
            return weight * 0.5

        if value <= t["low"]:
            level, ratio = "low", 1.0
            explanation = f"Low saturated fat ({value}g/100g)."
            report.positives.append(
                Positive("low_saturated_fat", "Low saturated fat", explanation, "saturated_fat", value, "g")
            )
        elif value <= t["moderate"]:
            level, ratio = "moderate", 0.7
            explanation = f"Moderate saturated fat ({value}g/100g)."
        elif value <= t["high"]:
            level, ratio = "high", 0.3
            explanation = f"High saturated fat ({value}g/100g)."
            report.warnings.append(
                Warning(
                    "high_saturated_fat", "High saturated fat", explanation, Severity.HIGH,
                    "saturated_fat", value, "g",
                    recommendation="Limit intake; excess saturated fat raises LDL cholesterol.",
                )
            )
        else:
            level, ratio = "very_high", 0.05
            explanation = f"Very high saturated fat ({value}g/100g)."
            report.warnings.append(
                Warning(
                    "very_high_saturated_fat", "Very high saturated fat", explanation, Severity.CRITICAL,
                    "saturated_fat", value, "g",
                    recommendation="Avoid frequent consumption, especially with existing heart conditions.",
                )
            )

        report.verdicts.append(NutrientVerdict("saturated_fat", value, t["unit"], level, explanation))
        awarded = weight * ratio
        report.score_breakdown.append(ScoreBreakdownItem("saturated_fat", weight, awarded, explanation))
        return awarded

    def _analyze_trans_fat(self, value: Optional[float], report: NutritionReport) -> float:
        weight = self._weights["trans_fat"]
        t = self._thresholds["trans_fat"]
        if value is None:
            report.verdicts.append(
                NutrientVerdict("trans_fat", None, t["unit"], "unknown", "Trans fat data not available.")
            )
            report.score_breakdown.append(
                ScoreBreakdownItem("trans_fat", weight, weight * 0.6, "No trans fat data provided; assumed neutral.")
            )
            return weight * 0.6

        if value <= t["low"]:
            level, ratio = "low", 1.0
            explanation = "No detectable trans fat."
            report.positives.append(Positive("no_trans_fat", "No trans fat", explanation, "trans_fat", value, "g"))
        elif value <= t["moderate"]:
            level, ratio = "moderate", 0.5
            explanation = f"Trace trans fat present ({value}g/100g)."
            report.warnings.append(
                Warning(
                    "trace_trans_fat", "Trace trans fat", explanation, Severity.MEDIUM, "trans_fat", value, "g",
                    recommendation="Trans fat has no known safe level; minimize intake.",
                )
            )
        elif value <= t["high"]:
            level, ratio = "high", 0.15
            explanation = f"High trans fat content ({value}g/100g)."
            report.warnings.append(
                Warning(
                    "high_trans_fat", "High trans fat", explanation, Severity.HIGH, "trans_fat", value, "g",
                    recommendation="Avoid regular consumption; trans fat significantly raises heart disease risk.",
                )
            )
        else:
            level, ratio = "very_high", 0.0
            explanation = f"Very high trans fat content ({value}g/100g)."
            report.warnings.append(
                Warning(
                    "very_high_trans_fat", "Very high trans fat", explanation, Severity.CRITICAL,
                    "trans_fat", value, "g",
                    recommendation="Avoid entirely; no safe consumption level for trans fat.",
                )
            )

        report.verdicts.append(NutrientVerdict("trans_fat", value, t["unit"], level, explanation))
        awarded = weight * ratio
        report.score_breakdown.append(ScoreBreakdownItem("trans_fat", weight, awarded, explanation))
        return awarded

    def _analyze_protein(self, value: Optional[float], report: NutritionReport) -> float:
        weight = self._weights["protein"]
        t = self._thresholds["protein"]
        if value is None:
            report.verdicts.append(
                NutrientVerdict("protein", None, t["unit"], "unknown", "Protein data not available.")
            )
            report.score_breakdown.append(
                ScoreBreakdownItem("protein", weight, weight * 0.5, "No protein data provided; assumed neutral.")
            )
            return weight * 0.5

        if value >= t["excellent"]:
            level, ratio = "excellent", 1.0
            explanation = f"Excellent protein density ({value}g/100g)."
            report.positives.append(
                Positive("excellent_protein", "Excellent protein source", explanation, "protein", value, "g")
            )
        elif value >= t["high"]:
            level, ratio = "high", 0.85
            explanation = f"Good protein density ({value}g/100g)."
            report.positives.append(Positive("good_protein", "Good protein source", explanation, "protein", value, "g"))
        elif value >= t["moderate"]:
            level, ratio = "moderate", 0.55
            explanation = f"Moderate protein density ({value}g/100g)."
        elif value >= t["low"]:
            level, ratio = "low", 0.3
            explanation = f"Low protein density ({value}g/100g)."
        else:
            level, ratio = "very_low", 0.15
            explanation = f"Very low protein density ({value}g/100g)."

        report.verdicts.append(NutrientVerdict("protein", value, t["unit"], level, explanation))
        awarded = weight * ratio
        report.score_breakdown.append(ScoreBreakdownItem("protein", weight, awarded, explanation))
        return awarded

    def _analyze_fiber(self, value: Optional[float], report: NutritionReport) -> float:
        weight = self._weights["fiber"]
        t = self._thresholds["fiber"]
        if value is None:
            report.verdicts.append(
                NutrientVerdict("fiber", None, t["unit"], "unknown", "Fiber data not available.")
            )
            report.score_breakdown.append(
                ScoreBreakdownItem("fiber", weight, weight * 0.5, "No fiber data provided; assumed neutral.")
            )
            return weight * 0.5

        if value >= t["excellent"]:
            level, ratio = "excellent", 1.0
            explanation = f"Excellent source of fiber ({value}g/100g)."
            report.positives.append(Positive("excellent_fiber", "Excellent fiber source", explanation, "fiber", value, "g"))
        elif value >= t["high"]:
            level, ratio = "high", 0.85
            explanation = f"Good source of fiber ({value}g/100g)."
            report.positives.append(Positive("good_fiber", "Good fiber source", explanation, "fiber", value, "g"))
        elif value >= t["moderate"]:
            level, ratio = "moderate", 0.55
            explanation = f"Moderate fiber content ({value}g/100g)."
        elif value >= t["low"]:
            level, ratio = "low", 0.3
            explanation = f"Low fiber content ({value}g/100g)."
        else:
            level, ratio = "very_low", 0.15
            explanation = f"Little to no fiber ({value}g/100g)."

        report.verdicts.append(NutrientVerdict("fiber", value, t["unit"], level, explanation))
        awarded = weight * ratio
        report.score_breakdown.append(ScoreBreakdownItem("fiber", weight, awarded, explanation))
        return awarded

    def _analyze_total_fat(self, value: Optional[float], report: NutritionReport) -> None:
        """Total fat is informational only (saturated/trans fat already weighted)."""
        t = self._thresholds["total_fat"]
        if value is None:
            report.verdicts.append(
                NutrientVerdict("total_fat", None, t["unit"], "unknown", "Total fat data not available.")
            )
            return
        if value <= t["low"]:
            level = "low"
        elif value <= t["moderate"]:
            level = "moderate"
        elif value <= t["high"]:
            level = "high"
        else:
            level = "very_high"
        explanation = f"Total fat content is {level} ({value}g/100g)."
        report.verdicts.append(NutrientVerdict("total_fat", value, t["unit"], level, explanation))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_salt(self, nutriments: Dict[str, Any]) -> Optional[float]:
        """Prefer explicit salt_100g; fall back to sodium_100g * 2.5 / 1000."""
        salt = self._to_float(nutriments.get("salt_100g"))
        if salt is not None:
            return salt
        sodium_mg = self._to_float(nutriments.get("sodium_100g"))
        if sodium_mg is not None:
            return round((sodium_mg * 2.5) / 1000, 3)
        return None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _band_for_score(self, score: float) -> str:
        for band_name, band in self._bands.items():
            if band["min"] <= score <= band["max"]:
                return band["label"]
        return "Unknown"
