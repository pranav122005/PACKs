"""
backend/services/knowledge_service.py

PACKS - AI Powered Ingredient Intelligence Platform
====================================================
Queries the ingredient knowledge database, resolving both aliases and
scientific names, and returns structured enrichment records.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("packs.knowledge_service")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


@dataclass
class IngredientKnowledge:
    """Structured knowledge record for a single ingredient."""

    name: str
    scientific_name: Optional[str] = None
    purpose: Optional[str] = None
    category: Optional[str] = None
    risk: Optional[str] = None
    evidence: Optional[str] = None
    daily_limit: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)
    health_notes: Optional[str] = None
    found: bool = False

    def to_dict(self) -> Dict[str, object]:
        """Return the knowledge record as a plain dictionary."""
        return asdict(self)


# Built-in seed knowledge base. Keys are normalized (lowercase) canonical
# or alias/scientific names mapping to a shared record key.
_SEED_KNOWLEDGE: Dict[str, Dict[str, object]] = {
    "msg": {
        "name": "MSG",
        "scientific_name": "Monosodium Glutamate",
        "purpose": "Flavor enhancer",
        "category": "Flavour Enhancer",
        "risk": "Moderate",
        "evidence": "Generally recognized as safe in moderate amounts; some individuals report sensitivity",
        "daily_limit": "No formal ADI established; moderate consumption advised",
        "alternatives": ["Yeast Extract", "Sea Salt"],
        "health_notes": "May cause headache or flushing in sensitive individuals (MSG symptom complex)",
    },
    "hfcs": {
        "name": "HFCS",
        "scientific_name": "High Fructose Corn Syrup",
        "purpose": "Sweetener",
        "category": "Sweetener",
        "risk": "High",
        "evidence": "Associated with increased risk of obesity and metabolic syndrome in excess intake",
        "daily_limit": "Limit added sugars to under 25g/day (WHO guidance)",
        "alternatives": ["Cane Sugar", "Honey", "Stevia"],
        "health_notes": "Frequently linked to fatty liver disease and insulin resistance with chronic overconsumption",
    },
    "palm oil": {
        "name": "Palm Oil",
        "scientific_name": "Elaeis guineensis oil",
        "purpose": "Fat / texturizer",
        "category": "Fat",
        "risk": "Moderate",
        "evidence": "High in saturated fat; linked to cardiovascular risk with high intake",
        "daily_limit": "Saturated fat intake below 10% of total daily calories recommended",
        "alternatives": ["Olive Oil", "Sunflower Oil"],
        "health_notes": "Also associated with environmental and deforestation concerns",
    },
    "sodium benzoate": {
        "name": "Sodium Benzoate",
        "scientific_name": "Sodium Benzoate (E211)",
        "purpose": "Preservative",
        "category": "Preservative",
        "risk": "Moderate",
        "evidence": "Can form benzene when combined with ascorbic acid under heat/light exposure",
        "daily_limit": "ADI of 5 mg/kg body weight (JECFA)",
        "alternatives": ["Potassium Sorbate", "Natural Fermentation"],
        "health_notes": "Some studies link to hyperactivity in children when combined with certain colourants",
    },
    "tartrazine": {
        "name": "Tartrazine",
        "scientific_name": "Tartrazine (E102)",
        "purpose": "Colouring agent",
        "category": "Colour",
        "risk": "High",
        "evidence": "Linked to hyperactivity in children and allergic reactions in sensitive individuals",
        "daily_limit": "ADI of 7.5 mg/kg body weight (JECFA)",
        "alternatives": ["Beta-Carotene", "Turmeric Extract"],
        "health_notes": "Requires warning labels in several jurisdictions due to behavioral effects in children",
    },
    "soy lecithin": {
        "name": "Soy Lecithin",
        "scientific_name": "Soy Lecithin (E322)",
        "purpose": "Emulsifier",
        "category": "Emulsifier",
        "risk": "Low",
        "evidence": "Generally recognized as safe; rare allergic reactions in soy-sensitive individuals",
        "daily_limit": "No specific limit established",
        "alternatives": ["Sunflower Lecithin"],
        "health_notes": "Derived from soy; a concern only for individuals with soy allergies",
    },
    "citric acid": {
        "name": "Citric Acid",
        "scientific_name": "Citric Acid (E330)",
        "purpose": "Acidity regulator / preservative",
        "category": "Acidity Regulator",
        "risk": "Low",
        "evidence": "Naturally occurring in citrus fruits; considered safe at typical food levels",
        "daily_limit": "No specific limit established",
        "alternatives": ["Lemon Juice", "Ascorbic Acid"],
        "health_notes": "Rarely associated with tooth enamel erosion at high frequency of intake",
    },
    "salt": {
        "name": "Salt",
        "scientific_name": "Sodium Chloride",
        "purpose": "Flavoring / preservative",
        "category": "Seasoning",
        "risk": "Moderate",
        "evidence": "Excess sodium intake linked to hypertension and cardiovascular disease",
        "daily_limit": "Under 5g/day (WHO guidance)",
        "alternatives": ["Herbs", "Potassium Chloride Blends"],
        "health_notes": "Monitor cumulative sodium intake across all meals",
    },
    "sugar": {
        "name": "Sugar",
        "scientific_name": "Sucrose",
        "purpose": "Sweetener",
        "category": "Sweetener",
        "risk": "Moderate",
        "evidence": "Excess added sugar intake linked to obesity, diabetes, and dental caries",
        "daily_limit": "Under 25-50g/day added sugar (WHO guidance)",
        "alternatives": ["Stevia", "Monk Fruit Sweetener"],
        "health_notes": "Consider total daily added sugar across all consumed products",
    },
}

# Alias map from raw alternate spellings/names to the seed knowledge key.
_ALIAS_TO_KEY: Dict[str, str] = {
    "monosodium glutamate": "msg",
    "ins621": "msg",
    "e621": "msg",
    "high fructose corn syrup": "hfcs",
    "palmolein": "palm oil",
    "palm olein": "palm oil",
    "palm kernel oil": "palm oil",
    "ins211": "sodium benzoate",
    "e211": "sodium benzoate",
    "ins102": "tartrazine",
    "e102": "tartrazine",
    "ins330": "citric acid",
    "e330": "citric acid",
    "ins322": "soy lecithin",
    "e322": "soy lecithin",
    "lecithin": "soy lecithin",
    "sodium chloride": "salt",
    "common salt": "salt",
    "table salt": "salt",
    "sucrose": "sugar",
    "cane sugar": "sugar",
    "sugar syrup": "sugar",
}


class KnowledgeService:
    """
    Provides lookups against the ingredient knowledge database, resolving
    both common aliases and scientific names to a canonical knowledge
    record.
    """

    def __init__(self, knowledge_base: Optional[Dict[str, Dict[str, object]]] = None) -> None:
        """
        Args:
            knowledge_base: Optional custom knowledge base keyed by
                normalized ingredient name. Defaults to the built-in
                PACKS seed knowledge base.
        """
        self._knowledge_base = knowledge_base or _SEED_KNOWLEDGE
        self._alias_to_key = _ALIAS_TO_KEY
        self._scientific_name_index = self._build_scientific_name_index()

    def lookup(self, ingredient_name: str) -> IngredientKnowledge:
        """
        Look up a single ingredient by name, alias, or scientific name.

        Args:
            ingredient_name: Raw or normalized ingredient name.

        Returns:
            IngredientKnowledge record. If no match is found, returns a
            record with `found=False` and the original name preserved.
        """
        if not ingredient_name or not ingredient_name.strip():
            return IngredientKnowledge(name=ingredient_name, found=False)

        key = self._normalize(ingredient_name)

        record = self._knowledge_base.get(key)
        if record is None:
            resolved_key = self._alias_to_key.get(key)
            if resolved_key:
                record = self._knowledge_base.get(resolved_key)

        if record is None:
            resolved_key = self._scientific_name_index.get(key)
            if resolved_key:
                record = self._knowledge_base.get(resolved_key)

        if record is None:
            logger.info("No knowledge record found for ingredient: %s", ingredient_name)
            return IngredientKnowledge(name=ingredient_name, found=False)

        return IngredientKnowledge(
            name=str(record.get("name", ingredient_name)),
            scientific_name=record.get("scientific_name"),
            purpose=record.get("purpose"),
            category=record.get("category"),
            risk=record.get("risk"),
            evidence=record.get("evidence"),
            daily_limit=record.get("daily_limit"),
            alternatives=list(record.get("alternatives", [])),
            health_notes=record.get("health_notes"),
            found=True,
        )

    def lookup_many(self, ingredient_names: List[str]) -> Dict[str, Dict[str, object]]:
        """
        Look up multiple ingredients in a single call.

        Args:
            ingredient_names: List of raw or normalized ingredient names.

        Returns:
            Dict mapping the original ingredient name to its knowledge
            record dictionary.
        """
        results: Dict[str, Dict[str, object]] = {}
        for name in ingredient_names:
            record = self.lookup(name)
            results[name] = record.to_dict()

        found_count = sum(1 for r in results.values() if r.get("found"))
        logger.info(
            "Knowledge lookup completed: %d/%d ingredients matched",
            found_count,
            len(ingredient_names),
        )
        return results

    def search(self, query: str) -> List[IngredientKnowledge]:
        """
        Search the knowledge base for entries whose name, alias, or
        scientific name contains the given query substring.

        Args:
            query: Free-text search string.

        Returns:
            List of matching IngredientKnowledge records.
        """
        if not query or not query.strip():
            return []

        normalized_query = self._normalize(query)
        matches: List[IngredientKnowledge] = []
        seen_keys = set()

        for key, record in self._knowledge_base.items():
            searchable = " ".join(
                str(v) for v in [
                    key,
                    record.get("name", ""),
                    record.get("scientific_name", ""),
                ]
            ).lower()

            if normalized_query in searchable and key not in seen_keys:
                seen_keys.add(key)
                matches.append(
                    IngredientKnowledge(
                        name=str(record.get("name", key)),
                        scientific_name=record.get("scientific_name"),
                        purpose=record.get("purpose"),
                        category=record.get("category"),
                        risk=record.get("risk"),
                        evidence=record.get("evidence"),
                        daily_limit=record.get("daily_limit"),
                        alternatives=list(record.get("alternatives", [])),
                        health_notes=record.get("health_notes"),
                        found=True,
                    )
                )

        logger.info("Knowledge search for %r returned %d results", query, len(matches))
        return matches

    def _normalize(self, text: str) -> str:
        """Normalize text into a consistent lookup key."""
        text = text.strip().lower()
        text = re.sub(r"[().]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _build_scientific_name_index(self) -> Dict[str, str]:
        """
        Build an index mapping normalized scientific names to their
        knowledge base key, for reverse lookup by scientific name.

        Returns:
            Dict mapping normalized scientific name -> knowledge base key.
        """
        index: Dict[str, str] = {}
        for key, record in self._knowledge_base.items():
            scientific_name = record.get("scientific_name")
            if scientific_name:
                index[self._normalize(str(scientific_name))] = key
        return index