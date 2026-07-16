"""
backend/services/nutrition_analysis_service.py

PACKS - AI Powered Ingredient Intelligence Platform
====================================================
Analyzes a nutrition dictionary using the macro calculator and a
nutrition scoring engine to produce a combined macro/nutrition report.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from backend.services.macro_calculator import MacroCalculator, MacroProfile
from backend.services.nutrition_label_parser import NutritionFacts

logger = logging.getLogger("packs.nutrition_analysis_service")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# Thresholds used for warning generation (per reference serving/100g).
_HIGH_SUGAR_G = 15.0
_HIGH_SODIUM_MG = 600.0
_HIGH_SATURATED_FAT_G = 5.0
_TRANS_FAT_PRESENT_G = 0.0
_LOW_FIBER_G = 3.0
_HIGH_ENERGY_DENSITY_KCAL_PER_G = 4.0


@dataclass
class NutritionAnalysisResult:
    """Structured result combining macro and nutrition analysis."""

    macro_report: Dict[str, Any] = field(default_factory=dict)
    nutrition_report: Dict[str, Any] = field(default_factory=dict)
    energy_density: Optional[float] = None
    protein_density: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    nutrition_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the nutrition analysis result as a plain dictionary."""
        return asdict(self)


class NutritionAnalysisService:
    """
    Combines the MacroCalculator with rule-based nutrition scoring to
    produce a comprehensive nutrition analysis report.
    """

    def __init__(self, macro_calculator: Optional[MacroCalculator] = None) -> None:
        """
        Args:
            macro_calculator: Optional injected MacroCalculator instance.
                Defaults to a new MacroCalculator().
        """
        self._macro_calculator = macro_calculator or MacroCalculator()

    def analyze(self, nutrition_facts: NutritionFacts) -> NutritionAnalysisResult:
        """
        Run full nutrition analysis on parsed nutrition facts.

        Args:
            nutrition_facts: Structured nutrition facts (e.g. from
                NutritionLabelParser).

        Returns:
            NutritionAnalysisResult containing the macro report,
            nutrition report, densities, warnings, and an overall
            nutrition score.
        """
        macro_profile = self._macro_calculator.calculate(nutrition_facts)
        nutrition_report = self._build_nutrition_report(nutrition_facts)
        warnings = self._generate_warnings(nutrition_facts, macro_profile)
        nutrition_score = self._compute_nutrition_score(nutrition_facts, macro_profile, warnings)

        result = NutritionAnalysisResult(
            macro_report=macro_profile.to_dict(),
            nutrition_report=nutrition_report,
            energy_density=macro_profile.energy_density_kcal_per_g,
            protein_density=macro_profile.protein_density_g_per_100kcal,
            warnings=warnings,
            nutrition_score=nutrition_score,
        )

        logger.info(
            "Nutrition analysis complete: score=%s, warnings=%d",
            nutrition_score,
            len(warnings),
        )
        return result

    def _build_nutrition_report(self, nutrition_facts: NutritionFacts) -> Dict[str, Any]:
        """
        Build a plain nutrition report dictionary directly from the
        parsed nutrition facts.

        Args:
            nutrition_facts: Structured nutrition facts.

        Returns:
            Dict representation of the nutrition facts.
        """
        return nutrition_facts.to_dict()

    def _generate_warnings(
        self, nutrition_facts: NutritionFacts, macro_profile: MacroProfile
    ) -> List[str]:
        """
        Generate human-readable warnings based on nutrient thresholds.

        Args:
            nutrition_facts: Structured nutrition facts.
            macro_profile: Computed macro profile.

        Returns:
            List of warning strings.
        """
        warnings: List[str] = []

        if nutrition_facts.sugar_g is not None and nutrition_facts.sugar_g >= _HIGH_SUGAR_G:
            warnings.append(f"High sugar content: {nutrition_facts.sugar_g}g")

        if nutrition_facts.sodium_mg is not None and nutrition_facts.sodium_mg >= _HIGH_SODIUM_MG:
            warnings.append(f"High sodium content: {nutrition_facts.sodium_mg}mg")

        if (
            nutrition_facts.saturated_fat_g is not None
            and nutrition_facts.saturated_fat_g >= _HIGH_SATURATED_FAT_G
        ):
            warnings.append(f"High saturated fat content: {nutrition_facts.saturated_fat_g}g")

        if nutrition_facts.trans_fat_g is not None and nutrition_facts.trans_fat_g > _TRANS_FAT_PRESENT_G:
            warnings.append(f"Contains trans fat: {nutrition_facts.trans_fat_g}g")

        if nutrition_facts.fiber_g is not None and nutrition_facts.fiber_g < _LOW_FIBER_G:
            warnings.append(f"Low dietary fiber content: {nutrition_facts.fiber_g}g")

        if (
            macro_profile.energy_density_kcal_per_g is not None
            and macro_profile.energy_density_kcal_per_g >= _HIGH_ENERGY_DENSITY_KCAL_PER_G
        ):
            warnings.append(
                f"High energy density: {macro_profile.energy_density_kcal_per_g} kcal/g"
            )

        if macro_profile.sugar_percent is not None and macro_profile.sugar_percent > 20:
            warnings.append(f"Sugar contributes {macro_profile.sugar_percent}% of total energy")

        return warnings

    def _compute_nutrition_score(
        self,
        nutrition_facts: NutritionFacts,
        macro_profile: MacroProfile,
        warnings: List[str],
    ) -> float:
        """
        Compute an overall 0-100 nutrition score using macro balance as
        a baseline, penalized by the number/severity of warnings.

        Args:
            nutrition_facts: Structured nutrition facts.
            macro_profile: Computed macro profile.
            warnings: List of generated warnings.

        Returns:
            A 0-100 nutrition score.
        """
        base_score = (
            macro_profile.macro_balance_score
            if macro_profile.macro_balance_score is not None
            else 50.0
        )

        penalty = min(len(warnings) * 8.0, 60.0)
        score = base_score - penalty

        if nutrition_facts.fiber_g is not None and nutrition_facts.fiber_g >= _LOW_FIBER_G:
            score += 5.0

        return round(max(0.0, min(100.0, score)), 2)