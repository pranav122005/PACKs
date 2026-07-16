"""
backend/services/health_score_service.py

PACKS - AI Powered Ingredient Intelligence Platform
====================================================
Aggregates outputs from the Nutrition Engine, Disease Engine, Additive
Engine, NOVA classification, and Macro Calculator into a single
overall health score with supporting reasoning.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from backend.services.macro_calculator import MacroProfile

logger = logging.getLogger("packs.health_score_service")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Relative weighting of each sub-engine's contribution to the final score.
_WEIGHT_NUTRITION = 0.30
_WEIGHT_DISEASE = 0.25
_WEIGHT_ADDITIVE = 0.20
_WEIGHT_NOVA = 0.10
_WEIGHT_MACRO = 0.15


@dataclass
class HealthScoreResult:
    """Final aggregated health score output."""

    overall_score: float
    grade: str
    positive_points: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    component_scores: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return the health score result as a plain dictionary."""
        return asdict(self)


class HealthScoreService:
    """
    Combines multiple engine outputs into a single, explainable overall
    health score for a product.

    Each contributing engine is expected to expose a 0-100 score plus
    optional textual insights (positives/warnings). This service is
    agnostic to how each engine computes its score (Dependency Inversion) —
    it only consumes their already-produced outputs.
    """

    def calculate(
        self,
        nutrition_score: Optional[float],
        disease_score: Optional[float],
        additive_score: Optional[float],
        nova_score: Optional[float],
        macro_profile: Optional[MacroProfile],
        nutrition_insights: Optional[Dict[str, List[str]]] = None,
        disease_insights: Optional[Dict[str, List[str]]] = None,
        additive_insights: Optional[Dict[str, List[str]]] = None,
        nova_insights: Optional[Dict[str, List[str]]] = None,
    ) -> HealthScoreResult:
        """
        Compute the overall health score from all engine outputs.

        Args:
            nutrition_score: 0-100 score from the Nutrition Engine.
            disease_score: 0-100 score from the Disease Engine.
            additive_score: 0-100 score from the Additive Engine.
            nova_score: 0-100 score derived from NOVA classification
                (higher = less ultra-processed).
            macro_profile: Computed MacroProfile from MacroCalculator.
            nutrition_insights: Optional dict with "positives"/"warnings" lists.
            disease_insights: Optional dict with "positives"/"warnings" lists.
            additive_insights: Optional dict with "positives"/"warnings" lists.
            nova_insights: Optional dict with "positives"/"warnings" lists.

        Returns:
            HealthScoreResult containing the overall score, grade,
            positive points, warnings, and reasoning trail.
        """
        component_scores: Dict[str, Optional[float]] = {
            "nutrition": nutrition_score,
            "disease": disease_score,
            "additive": additive_score,
            "nova": nova_score,
            "macro": macro_profile.macro_balance_score if macro_profile else None,
        }

        overall_score = self._weighted_average(component_scores)
        grade = self._score_to_grade(overall_score)

        positive_points = self._collect_items(
            [nutrition_insights, disease_insights, additive_insights, nova_insights],
            key="positives",
        )
        warnings = self._collect_items(
            [nutrition_insights, disease_insights, additive_insights, nova_insights],
            key="warnings",
        )

        if macro_profile:
            positive_points.extend(self._macro_positive_points(macro_profile))
            warnings.extend(self._macro_warnings(macro_profile))

        reasoning = self._build_reasoning(component_scores, overall_score, grade)

        result = HealthScoreResult(
            overall_score=overall_score,
            grade=grade,
            positive_points=positive_points,
            warnings=warnings,
            reasoning=reasoning,
            component_scores=component_scores,
        )

        logger.info(
            "Computed overall health score=%.2f grade=%s components=%s",
            overall_score,
            grade,
            component_scores,
        )
        return result

    def _weighted_average(self, component_scores: Dict[str, Optional[float]]) -> float:
        """
        Compute a weighted average across available component scores,
        re-normalizing weights when some components are missing.

        Args:
            component_scores: Dict of component name -> score (or None).

        Returns:
            Overall score in the range 0-100.
        """
        weights = {
            "nutrition": _WEIGHT_NUTRITION,
            "disease": _WEIGHT_DISEASE,
            "additive": _WEIGHT_ADDITIVE,
            "nova": _WEIGHT_NOVA,
            "macro": _WEIGHT_MACRO,
        }

        available = {
            name: score
            for name, score in component_scores.items()
            if score is not None
        }

        if not available:
            logger.warning("No component scores available; defaulting overall score to 0.0")
            return 0.0

        total_weight = sum(weights[name] for name in available)
        if total_weight <= 0:
            return 0.0

        weighted_sum = sum(available[name] * weights[name] for name in available)
        overall = weighted_sum / total_weight
        return round(max(0.0, min(100.0, overall)), 2)

    def _score_to_grade(self, score: float) -> str:
        """
        Map a numeric score to a letter grade.

        Args:
            score: Overall 0-100 score.

        Returns:
            A letter grade string ("A" through "E").
        """
        if score >= 85:
            return "A"
        if score >= 70:
            return "B"
        if score >= 50:
            return "C"
        if score >= 30:
            return "D"
        return "E"

    def _collect_items(
        self,
        insight_dicts: List[Optional[Dict[str, List[str]]]],
        key: str,
    ) -> List[str]:
        """
        Merge a specific list (e.g. 'positives' or 'warnings') across
        multiple engine insight dicts, preserving order and removing
        duplicates.

        Args:
            insight_dicts: List of optional insight dicts from each engine.
            key: The key to extract ("positives" or "warnings").

        Returns:
            A merged, deduplicated list of strings.
        """
        merged: List[str] = []
        seen = set()
        for insights in insight_dicts:
            if not insights:
                continue
            for item in insights.get(key, []) or []:
                if item not in seen:
                    seen.add(item)
                    merged.append(item)
        return merged

    def _macro_positive_points(self, macro_profile: MacroProfile) -> List[str]:
        """
        Derive positive callouts directly from the macro profile.

        Args:
            macro_profile: Computed macro profile.

        Returns:
            List of positive observation strings.
        """
        points: List[str] = []
        if macro_profile.protein_percent is not None and macro_profile.protein_percent >= 15:
            points.append(
                f"Good protein contribution: {macro_profile.protein_percent}% of energy"
            )
        if macro_profile.sugar_percent is not None and macro_profile.sugar_percent <= 5:
            points.append(
                f"Low sugar contribution: {macro_profile.sugar_percent}% of energy"
            )
        return points

    def _macro_warnings(self, macro_profile: MacroProfile) -> List[str]:
        """
        Derive warning callouts directly from the macro profile.

        Args:
            macro_profile: Computed macro profile.

        Returns:
            List of warning strings.
        """
        warnings: List[str] = []
        if macro_profile.sugar_percent is not None and macro_profile.sugar_percent > 20:
            warnings.append(
                f"High sugar contribution: {macro_profile.sugar_percent}% of energy"
            )
        if macro_profile.fat_percent is not None and macro_profile.fat_percent > 40:
            warnings.append(
                f"High fat contribution: {macro_profile.fat_percent}% of energy"
            )
        if macro_profile.energy_density_kcal_per_g is not None and macro_profile.energy_density_kcal_per_g > 4.0:
            warnings.append(
                f"High energy density: {macro_profile.energy_density_kcal_per_g} kcal/g"
            )
        return warnings

    def _build_reasoning(
        self,
        component_scores: Dict[str, Optional[float]],
        overall_score: float,
        grade: str,
    ) -> List[str]:
        """
        Build a human-readable reasoning trail explaining how the
        overall score was derived.

        Args:
            component_scores: Dict of component name -> score.
            overall_score: Final computed overall score.
            grade: Final letter grade.

        Returns:
            List of reasoning strings.
        """
        reasoning: List[str] = []
        for name, score in component_scores.items():
            if score is None:
                reasoning.append(f"{name.capitalize()} score unavailable and excluded from weighting")
            else:
                reasoning.append(f"{name.capitalize()} contributed a score of {score}")
        reasoning.append(f"Final weighted overall score: {overall_score} (Grade {grade})")
        return reasoning