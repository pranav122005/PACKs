"""
backend/engines/nova_engine.py

NOVA Engine — classifies a product into the NOVA food-processing groups:

    NOVA 1: Unprocessed or minimally processed foods
    NOVA 2: Processed culinary ingredients
    NOVA 3: Processed foods
    NOVA 4: Ultra-processed food and drink products

Classification is driven by ingredient count plus the presence of
"markers of ultra-processing": artificial colours, artificial flavours,
non-nutritive sweeteners, emulsifiers/stabilizers, and preservatives.
Thresholds live in health_rules.json under "nova_classification".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from backend.schemas.common import NovaGroup
from backend.utils.config_loader import HealthRulesConfig, get_health_rules_config

# Emulsifiers/stabilizers are not modeled in the additive engine (which
# focuses on sweeteners/colours/preservatives/flavour enhancers), so the
# NOVA engine keeps its own lightweight keyword set for this marker.
_EMULSIFIER_KEYWORDS = (
    "lecithin", "mono- and diglycerides", "monoglycerides", "diglycerides",
    "polysorbate", "carrageenan", "carboxymethyl cellulose", "xanthan gum",
    "guar gum", "sodium stearoyl lactylate", "dats", "csl", "e471", "e472",
    "e433", "e466",
)

_ARTIFICIAL_FLAVOUR_KEYWORDS = (
    "artificial flavour", "artificial flavor", "nature identical flavouring",
    "nature-identical flavouring", "synthetic flavouring", "flavouring substance",
)

_ARTIFICIAL_COLOUR_KEYWORDS = (
    "tartrazine", "sunset yellow", "carmoisine", "ponceau", "allura red",
    "brilliant blue", "quinoline yellow", "artificial colour", "artificial color",
)

_SWEETENER_KEYWORDS = (
    "aspartame", "sucralose", "acesulfame", "saccharin", "cyclamate", "neotame",
)

_PRESERVATIVE_KEYWORDS = (
    "sodium benzoate", "potassium sorbate", "sodium nitrite", "sodium nitrate",
    "sulphur dioxide", "sulfur dioxide", "bha", "bht", "calcium propionate",
)

_CULINARY_INGREDIENT_KEYWORDS = (
    "salt", "sugar", "oil", "butter", "honey", "vinegar", "starch",
)


@dataclass
class NovaMarker:
    """A single ultra-processing marker found in the ingredient list."""

    marker: str
    matched_terms: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"marker": self.marker, "matched_terms": self.matched_terms}


@dataclass
class NovaReport:
    """Complete output of the NOVA Engine for a single product."""

    nova_group: int
    label: str
    description: str
    ingredient_count: int
    markers: List[NovaMarker] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nova_group": self.nova_group,
            "label": self.label,
            "description": self.description,
            "ingredient_count": self.ingredient_count,
            "markers": [m.to_dict() for m in self.markers],
            "reasoning": self.reasoning,
        }


class NovaEngine:
    """Classifies a product's degree of processing using the NOVA system."""

    def __init__(self, config: Optional[HealthRulesConfig] = None) -> None:
        self._config = config or get_health_rules_config()
        self._rules = self._config.section("nova_classification")

    def analyze(self, product: Dict[str, Any]) -> NovaReport:
        """Classify a product into NOVA 1-4 with full reasoning."""
        ingredients_text = (product.get("ingredients_text") or "").lower().strip()
        ingredients = self._split_ingredients(ingredients_text)
        ingredient_count = len(ingredients)

        markers = self._detect_markers(ingredients_text)
        marker_names = {m.marker for m in markers}
        cosmetic_markers = marker_names & {"artificial_colour", "artificial_flavour", "sweetener", "emulsifier"}

        reasoning: List[str] = [f"Detected {ingredient_count} top-level ingredient(s)."]
        if markers:
            reasoning.append(
                "Ultra-processing markers found: " + ", ".join(sorted(marker_names)) + "."
            )
        else:
            reasoning.append("No ultra-processing markers (artificial colours/flavours/sweeteners/emulsifiers/preservatives) detected.")

        if ingredient_count == 0:
            nova_group = NovaGroup.PROCESSED_FOODS
            reasoning.append("Ingredient list unavailable; defaulted to NOVA 3 pending manual review.")
        elif ingredient_count <= self._rules["nova_1"]["max_ingredient_count"] and not markers:
            nova_group = NovaGroup.UNPROCESSED_OR_MINIMALLY_PROCESSED
            reasoning.append("Single natural ingredient with no additives qualifies as NOVA 1.")
        elif (
            ingredient_count <= self._rules["nova_2"]["max_ingredient_count"]
            and not markers
            and self._is_pure_culinary_ingredient(ingredients)
        ):
            nova_group = NovaGroup.PROCESSED_CULINARY_INGREDIENTS
            reasoning.append("Composed only of basic culinary ingredients (oil/sugar/salt/etc.) qualifies as NOVA 2.")
        elif cosmetic_markers or len(markers) >= self._rules["nova_4"]["min_additive_count"]:
            nova_group = NovaGroup.ULTRA_PROCESSED
            if cosmetic_markers:
                reasoning.append(
                    "Presence of cosmetic additive marker(s) ("
                    + ", ".join(sorted(cosmetic_markers))
                    + ") not typically used in home cooking qualifies this as NOVA 4 (ultra-processed)."
                )
            else:
                reasoning.append(
                    f"{len(markers)} distinct additive markers detected, meeting the NOVA 4 threshold "
                    f"(>= {self._rules['nova_4']['min_additive_count']})."
                )
        elif len(markers) <= self._rules["nova_3"]["max_additive_count"]:
            nova_group = NovaGroup.PROCESSED_FOODS
            reasoning.append(
                f"{len(markers)} additive marker(s) detected, within the NOVA 3 threshold "
                f"(<= {self._rules['nova_3']['max_additive_count']}) for processed foods with added salt/sugar/oil/preservatives."
            )
        else:
            nova_group = NovaGroup.ULTRA_PROCESSED
            reasoning.append("Additive profile exceeds processed-food thresholds; classified as NOVA 4.")

        group_key = f"nova_{nova_group.value}"
        group_meta = self._rules.get(group_key, {})

        return NovaReport(
            nova_group=nova_group.value,
            label=group_meta.get("label", nova_group.name),
            description=group_meta.get("description", ""),
            ingredient_count=ingredient_count,
            markers=markers,
            reasoning=reasoning,
        )

    def compute_nova_score_points(self, report: NovaReport, max_points: float) -> float:
        """Convert the NOVA classification into a points contribution for the overall score."""
        ratio_by_group = {1: 1.0, 2: 0.9, 3: 0.55, 4: 0.15}
        return max_points * ratio_by_group.get(report.nova_group, 0.5)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_ingredients(text: str) -> List[str]:
        """
        Split an ingredient string into top-level items, ignoring commas
        that occur inside parentheses (sub-ingredient breakdowns).
        """
        if not text:
            return []
        items: List[str] = []
        depth = 0
        current = []
        for char in text:
            if char == "(":
                depth += 1
                current.append(char)
            elif char == ")":
                depth = max(0, depth - 1)
                current.append(char)
            elif char == "," and depth == 0:
                items.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            items.append("".join(current).strip())
        return [i for i in items if i]

    @staticmethod
    def _is_pure_culinary_ingredient(ingredients: List[str]) -> bool:
        for ingredient in ingredients:
            if not any(keyword in ingredient for keyword in _CULINARY_INGREDIENT_KEYWORDS):
                return False
        return True

    @staticmethod
    def _detect_markers(text: str) -> List[NovaMarker]:
        markers: List[NovaMarker] = []

        def find_matches(keywords) -> List[str]:
            return [kw for kw in keywords if kw in text]

        colour_matches = find_matches(_ARTIFICIAL_COLOUR_KEYWORDS)
        if colour_matches:
            markers.append(NovaMarker("artificial_colour", colour_matches))

        flavour_matches = find_matches(_ARTIFICIAL_FLAVOUR_KEYWORDS)
        if flavour_matches:
            markers.append(NovaMarker("artificial_flavour", flavour_matches))

        sweetener_matches = find_matches(_SWEETENER_KEYWORDS)
        if sweetener_matches:
            markers.append(NovaMarker("sweetener", sweetener_matches))

        emulsifier_matches = find_matches(_EMULSIFIER_KEYWORDS)
        if emulsifier_matches:
            markers.append(NovaMarker("emulsifier", emulsifier_matches))

        preservative_matches = find_matches(_PRESERVATIVE_KEYWORDS)
        if preservative_matches:
            markers.append(NovaMarker("preservative", preservative_matches))

        # Raw E-number / INS-number tokens also count as one generic marker
        # if nothing else caught them, since bare codes still imply
        # industrial formulation.
        if re.search(r"\bE\d{3}[a-zA-Z]?\b", text, re.IGNORECASE) and not (
            colour_matches or flavour_matches or sweetener_matches or emulsifier_matches or preservative_matches
        ):
            markers.append(NovaMarker("coded_additive", re.findall(r"\bE\d{3}[a-zA-Z]?\b", text, re.IGNORECASE)))

        return markers
