"""
backend/services/macro_calculator.py

PACKS - AI Powered Ingredient Intelligence Platform
====================================================
Calculates macro-nutrient ratios and densities from parsed nutrition
facts, used as input to downstream health scoring.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from backend.services.nutrition_label_parser import NutritionFacts

logger = logging.getLogger("packs.macro_calculator")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Standard macronutrient energy factors (kcal per gram).
_KCAL_PER_G_PROTEIN = 4.0
_KCAL_PER_G_CARB = 4.0
_KCAL_PER_G_FAT = 9.0

# Reference serving mass (grams) used for density calculations when the
# parsed serving size cannot be resolved to a numeric gram value.
_DEFAULT_REFERENCE_GRAMS = 100.0


@dataclass
class MacroProfile:
    """Structured macro-nutrient ratios and densities."""

    protein_percent: Optional[float] = None
    sugar_percent: Optional[float] = None
    fat_percent: Optional[float] = None
    carbs_percent: Optional[float] = None
    energy_density_kcal_per_g: Optional[float] = None
    protein_density_g_per_100kcal: Optional[float] = None
    sugar_density_g_per_100kcal: Optional[float] = None
    macro_balance_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Return the macro profile as a plain dictionary."""
        return asdict(self)


class MacroCalculator:
    """
    Computes macro-nutrient percentages, densities, and an overall
    macro balance score from a NutritionFacts record.
    """

    def calculate(
        self,
        nutrition_facts: NutritionFacts,
        reference_grams: Optional[float] = None,
    ) -> MacroProfile:
        """
        Compute the macro profile for a given set of nutrition facts.

        Args:
            nutrition_facts: Parsed nutrition facts for the product.
            reference_grams: Serving mass in grams used for density
                calculations. If not supplied, attempts to parse from
                `nutrition_facts.serving_size`, falling back to 100g.

        Returns:
            A populated MacroProfile. Fields default to None when the
            underlying data is insufficient to compute them.
        """
        grams = reference_grams or self._resolve_reference_grams(nutrition_facts)

        calories = nutrition_facts.calories
        protein_g = nutrition_facts.protein_g
        fat_g = nutrition_facts.fat_g
        carbs_g = nutrition_facts.carbohydrates_g
        sugar_g = nutrition_facts.sugar_g

        total_energy_kcal = calories if calories is not None else self._estimate_calories(
            protein_g, fat_g, carbs_g
        )

        profile = MacroProfile(
            protein_percent=self._percent_of_energy(protein_g, _KCAL_PER_G_PROTEIN, total_energy_kcal),
            fat_percent=self._percent_of_energy(fat_g, _KCAL_PER_G_FAT, total_energy_kcal),
            carbs_percent=self._percent_of_energy(carbs_g, _KCAL_PER_G_CARB, total_energy_kcal),
            sugar_percent=self._percent_of_energy(sugar_g, _KCAL_PER_G_CARB, total_energy_kcal),
            energy_density_kcal_per_g=self._energy_density(total_energy_kcal, grams),
            protein_density_g_per_100kcal=self._density_per_100kcal(protein_g, total_energy_kcal),
            sugar_density_g_per_100kcal=self._density_per_100kcal(sugar_g, total_energy_kcal),
        )

        profile.macro_balance_score = self._macro_balance_score(profile)

        logger.info("Computed macro profile: %s", profile.to_dict())
        return profile

    def _resolve_reference_grams(self, nutrition_facts: NutritionFacts) -> float:
        """
        Attempt to extract a numeric gram value from the serving size
        string, falling back to a standard 100g reference.

        Args:
            nutrition_facts: Parsed nutrition facts.

        Returns:
            A positive float representing reference grams.
        """
        serving_size = nutrition_facts.serving_size
        if not serving_size:
            return _DEFAULT_REFERENCE_GRAMS

        digits = "".join(c for c in serving_size if c.isdigit() or c == ".")
        try:
            value = float(digits) if digits else _DEFAULT_REFERENCE_GRAMS
            return value if value > 0 else _DEFAULT_REFERENCE_GRAMS
        except ValueError:
            return _DEFAULT_REFERENCE_GRAMS

    def _estimate_calories(
        self,
        protein_g: Optional[float],
        fat_g: Optional[float],
        carbs_g: Optional[float],
    ) -> Optional[float]:
        """
        Estimate total calories from macronutrient grams using the
        Atwater factors, when direct calorie data is unavailable.

        Args:
            protein_g: Grams of protein.
            fat_g: Grams of fat.
            carbs_g: Grams of carbohydrates.

        Returns:
            Estimated total kcal, or None if no macros are available.
        """
        if protein_g is None and fat_g is None and carbs_g is None:
            return None

        total = 0.0
        total += (protein_g or 0.0) * _KCAL_PER_G_PROTEIN
        total += (fat_g or 0.0) * _KCAL_PER_G_FAT
        total += (carbs_g or 0.0) * _KCAL_PER_G_CARB
        return total if total > 0 else None

    def _percent_of_energy(
        self,
        grams: Optional[float],
        kcal_per_gram: float,
        total_energy_kcal: Optional[float],
    ) -> Optional[float]:
        """
        Compute what percentage of total energy a given macronutrient
        contributes.

        Args:
            grams: Grams of the macronutrient.
            kcal_per_gram: Energy factor for that macronutrient.
            total_energy_kcal: Total energy of the product/serving.

        Returns:
            Percentage (0-100+) or None if inputs are insufficient.
        """
        if grams is None or not total_energy_kcal or total_energy_kcal <= 0:
            return None
        return round((grams * kcal_per_gram / total_energy_kcal) * 100, 2)

    def _energy_density(
        self, total_energy_kcal: Optional[float], grams: float
    ) -> Optional[float]:
        """
        Compute energy density (kcal per gram of product).

        Args:
            total_energy_kcal: Total energy for the reference serving.
            grams: Reference serving mass in grams.

        Returns:
            kcal per gram, or None if energy data is unavailable.
        """
        if total_energy_kcal is None or grams <= 0:
            return None
        return round(total_energy_kcal / grams, 3)

    def _density_per_100kcal(
        self, grams: Optional[float], total_energy_kcal: Optional[float]
    ) -> Optional[float]:
        """
        Compute grams of a nutrient per 100 kcal of the product, useful
        for comparing nutrient quality independent of portion size.

        Args:
            grams: Grams of the nutrient.
            total_energy_kcal: Total energy for the reference serving.

        Returns:
            Grams per 100 kcal, or None if inputs are insufficient.
        """
        if grams is None or not total_energy_kcal or total_energy_kcal <= 0:
            return None
        return round((grams / total_energy_kcal) * 100, 3)

    def _macro_balance_score(self, profile: MacroProfile) -> Optional[float]:
        """
        Compute a 0-100 macro balance score. Higher is better balanced,
        penalizing excessive sugar/fat share and rewarding adequate
        protein share.

        Args:
            profile: Partially populated MacroProfile.

        Returns:
            A 0-100 balance score, or None if insufficient data exists.
        """
        if profile.protein_percent is None and profile.fat_percent is None and profile.sugar_percent is None:
            return None

        score = 100.0

        protein_pct = profile.protein_percent or 0.0
        fat_pct = profile.fat_percent or 0.0
        sugar_pct = profile.sugar_percent or 0.0

        # Reward protein contribution up to a healthy ceiling of ~25%.
        protein_bonus = min(protein_pct, 25.0) * 0.4
        score = score - 40 + protein_bonus

        # Penalize excessive sugar share of total energy.
        if sugar_pct > 10.0:
            score -= min((sugar_pct - 10.0) * 1.5, 40.0)

        # Penalize excessive fat share of total energy.
        if fat_pct > 35.0:
            score -= min((fat_pct - 35.0) * 1.0, 30.0)

        score = max(0.0, min(100.0, score))
        return round(score, 2)