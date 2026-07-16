"""
backend/services/ingredient_analysis_service.py

PACKS - AI Powered Ingredient Intelligence Platform
====================================================
Analyzes a normalized ingredient list to detect additives, artificial
ingredients, preservatives, sweeteners, colours, flavours, and produce
an overall risk summary.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("packs.ingredient_analysis_service")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# Category classification sets (lowercase canonical/alias names).
_PRESERVATIVES = {
    "sodium benzoate", "potassium sorbate", "calcium propionate",
    "sodium nitrite", "sodium nitrate", "bha", "bht", "sulphur dioxide",
    "sodium metabisulphite",
}

_SWEETENERS = {
    "sugar", "hfcs", "aspartame", "sucralose", "saccharin", "stevia",
    "sorbitol", "xylitol", "maltitol", "acesulfame potassium", "monk fruit sweetener",
}

_COLOURS = {
    "tartrazine", "sunset yellow", "carmoisine", "allura red", "brilliant blue",
    "caramel colour", "beta-carotene", "annatto",
}

_FLAVOURS = {
    "artificial flavour", "natural flavour", "vanillin", "artificial vanilla flavour",
    "nature identical flavouring substances",
}

_ARTIFICIAL_MARKERS = {
    "msg", "hfcs", "tartrazine", "sunset yellow", "carmoisine", "allura red",
    "aspartame", "sucralose", "saccharin", "acesulfame potassium",
    "artificial flavour", "artificial vanilla flavour", "sodium benzoate",
    "bha", "bht",
}

_HIGH_RISK_SET = {"tartrazine", "sunset yellow", "carmoisine", "bha", "bht", "sodium nitrite"}
_MODERATE_RISK_SET = {"msg", "hfcs", "sodium benzoate", "palm oil", "salt", "sugar", "aspartame"}


@dataclass
class IngredientAnalysisResult:
    """Structured result of ingredient-level additive/category analysis."""

    total_ingredients: int = 0
    additives: List[str] = field(default_factory=list)
    artificial_ingredients: List[str] = field(default_factory=list)
    preservatives: List[str] = field(default_factory=list)
    sweeteners: List[str] = field(default_factory=list)
    colours: List[str] = field(default_factory=list)
    flavours: List[str] = field(default_factory=list)
    risk_summary: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Return the analysis result as a plain dictionary."""
        return asdict(self)


class IngredientAnalysisService:
    """
    Categorizes normalized ingredients into functional groups (additive,
    preservative, sweetener, colour, flavour) and produces a risk summary,
    optionally enriched with knowledge-base risk ratings.
    """

    def analyze(
        self,
        ingredients: List[str],
        knowledge: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> IngredientAnalysisResult:
        """
        Analyze a normalized ingredient list.

        Args:
            ingredients: List of normalized ingredient names (e.g. from
                IngredientNormalizer).
            knowledge: Optional dict of ingredient name -> knowledge
                record (from KnowledgeService.lookup_many), used to
                refine risk classification when available.

        Returns:
            IngredientAnalysisResult with categorized ingredient lists
            and an overall risk summary.
        """
        if not ingredients:
            logger.info("No ingredients supplied to ingredient analysis service")
            return IngredientAnalysisResult(total_ingredients=0)

        knowledge = knowledge or {}

        preservatives: List[str] = []
        sweeteners: List[str] = []
        colours: List[str] = []
        flavours: List[str] = []
        artificial: List[str] = []
        additives: List[str] = []

        for ingredient in ingredients:
            lowered = ingredient.lower().strip()

            is_additive = False

            if lowered in _PRESERVATIVES:
                preservatives.append(ingredient)
                is_additive = True
            if lowered in _SWEETENERS:
                sweeteners.append(ingredient)
                is_additive = True
            if lowered in _COLOURS:
                colours.append(ingredient)
                is_additive = True
            if lowered in _FLAVOURS:
                flavours.append(ingredient)
                is_additive = True
            if lowered in _ARTIFICIAL_MARKERS:
                artificial.append(ingredient)
                is_additive = True

            record = knowledge.get(ingredient, {})
            category = str(record.get("category", "")).lower() if record else ""
            if category:
                if "preservative" in category and ingredient not in preservatives:
                    preservatives.append(ingredient)
                    is_additive = True
                if "sweetener" in category and ingredient not in sweeteners:
                    sweeteners.append(ingredient)
                    is_additive = True
                if "colour" in category or "color" in category:
                    if ingredient not in colours:
                        colours.append(ingredient)
                    is_additive = True
                if "flavour" in category or "flavor" in category:
                    if ingredient not in flavours:
                        flavours.append(ingredient)
                    is_additive = True

            if is_additive and ingredient not in additives:
                additives.append(ingredient)

        risk_summary = self._build_risk_summary(ingredients, additives, knowledge)

        result = IngredientAnalysisResult(
            total_ingredients=len(ingredients),
            additives=additives,
            artificial_ingredients=artificial,
            preservatives=preservatives,
            sweeteners=sweeteners,
            colours=colours,
            flavours=flavours,
            risk_summary=risk_summary,
        )

        logger.info(
            "Ingredient analysis complete: %d additives detected out of %d ingredients",
            len(additives),
            len(ingredients),
        )
        return result

    def _build_risk_summary(
        self,
        ingredients: List[str],
        additives: List[str],
        knowledge: Dict[str, Dict[str, object]],
    ) -> Dict[str, object]:
        """
        Build an aggregate risk summary from the detected additives,
        using knowledge-base risk ratings when available and falling
        back to built-in risk sets.

        Args:
            ingredients: Full normalized ingredient list.
            additives: Subset of ingredients classified as additives.
            knowledge: Knowledge records keyed by ingredient name.

        Returns:
            Dict summarizing high/moderate/low risk counts and flagged
            ingredient names.
        """
        high_risk: List[str] = []
        moderate_risk: List[str] = []
        low_risk: List[str] = []

        for ingredient in additives:
            lowered = ingredient.lower().strip()
            record = knowledge.get(ingredient, {})
            risk_level = str(record.get("risk", "")).lower() if record else ""

            if risk_level == "high" or lowered in _HIGH_RISK_SET:
                high_risk.append(ingredient)
            elif risk_level == "moderate" or lowered in _MODERATE_RISK_SET:
                moderate_risk.append(ingredient)
            else:
                low_risk.append(ingredient)

        if high_risk:
            overall_level = "High"
        elif moderate_risk:
            overall_level = "Moderate"
        elif additives:
            overall_level = "Low"
        else:
            overall_level = "Minimal"

        return {
            "overall_risk_level": overall_level,
            "high_risk_ingredients": high_risk,
            "moderate_risk_ingredients": moderate_risk,
            "low_risk_ingredients": low_risk,
            "total_additives_detected": len(additives),
            "total_ingredients_analyzed": len(ingredients),
        }