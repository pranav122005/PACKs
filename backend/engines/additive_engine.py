"""
backend/engines/additive_engine.py

Additive Engine — scans a product's ingredient text for additives of
concern: sweeteners, colours, preservatives, flavour enhancers, syrups,
and raw E-number / INS-number codes.

Matching strategy
------------------
1. Exact / substring match against known aliases (fast path).
2. RapidFuzz fuzzy match (token_sort_ratio) to catch OCR noise and
   spelling variants (e.g. "Tartrazin", "Sodiun Benzoate").
3. Regex extraction of raw E-numbers (E102, E-102) and INS numbers
   (INS 102, 102) that fuzzy matching might miss.

Every detected additive is enriched with scientific name, purpose,
risk level, daily limit, a suggested alternative and an evidence level,
all sourced from health_rules.json plus a small in-code knowledge base
for descriptive fields (purpose/alternative/evidence) that don't belong
in the numeric config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from backend.schemas.common import DetectedAdditive, EvidenceLevel, RiskLevel
from backend.utils.config_loader import HealthRulesConfig, get_health_rules_config

_FUZZY_MATCH_THRESHOLD = 88.0

_E_NUMBER_PATTERN = re.compile(r"\bE[\s\-]?(\d{3}[a-zA-Z]?)\b", re.IGNORECASE)
_INS_NUMBER_PATTERN = re.compile(r"\bINS[\s\-]?(\d{3}[a-zA-Z]?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class _AdditiveKnowledge:
    """Descriptive metadata not stored in health_rules.json (purpose/alternative/evidence)."""

    purpose: str
    alternative: str
    evidence_level: EvidenceLevel
    aliases: Tuple[str, ...]


# Supplementary knowledge base: descriptive fields keyed by the same slug
# used in health_rules.json sections (e.g. "aspartame", "sodium_benzoate").
_KNOWLEDGE_BASE: Dict[str, _AdditiveKnowledge] = {
    "aspartame": _AdditiveKnowledge(
        "Artificial sweetener, ~200x sweeter than sugar",
        "Stevia or monk fruit extract",
        EvidenceLevel.MODERATE,
        ("aspartame",),
    ),
    "sucralose": _AdditiveKnowledge(
        "Artificial sweetener, ~600x sweeter than sugar",
        "Stevia or monk fruit extract",
        EvidenceLevel.LIMITED,
        ("sucralose", "splenda"),
    ),
    "acesulfame_k": _AdditiveKnowledge(
        "Artificial sweetener, often blended with other sweeteners",
        "Stevia or monk fruit extract",
        EvidenceLevel.MODERATE,
        ("acesulfame k", "acesulfame potassium", "ace-k", "acesulfame-k"),
    ),
    "saccharin": _AdditiveKnowledge(
        "Artificial sweetener, one of the oldest synthetic sweeteners",
        "Stevia or monk fruit extract",
        EvidenceLevel.MODERATE,
        ("saccharin",),
    ),
    "cyclamate": _AdditiveKnowledge(
        "Artificial sweetener, often blended with saccharin",
        "Stevia or monk fruit extract",
        EvidenceLevel.LIMITED,
        ("cyclamate",),
    ),
    "neotame": _AdditiveKnowledge(
        "Ultra-potent artificial sweetener",
        "Stevia or monk fruit extract",
        EvidenceLevel.LIMITED,
        ("neotame",),
    ),
    "stevia": _AdditiveKnowledge(
        "Natural non-caloric sweetener from the stevia plant",
        "Already a natural alternative",
        EvidenceLevel.STRONG,
        ("stevia", "steviol glycosides"),
    ),
    "tartrazine": _AdditiveKnowledge(
        "Synthetic yellow food colour",
        "Turmeric or beta-carotene based natural colour",
        EvidenceLevel.STRONG,
        ("tartrazine",),
    ),
    "sunset_yellow": _AdditiveKnowledge(
        "Synthetic orange-yellow food colour",
        "Beta-carotene or paprika extract",
        EvidenceLevel.STRONG,
        ("sunset yellow", "sunset yellow fcf"),
    ),
    "carmoisine": _AdditiveKnowledge(
        "Synthetic red-brown food colour",
        "Beetroot red (E162) or anthocyanins",
        EvidenceLevel.MODERATE,
        ("carmoisine", "azorubine"),
    ),
    "ponceau_4r": _AdditiveKnowledge(
        "Synthetic red food colour",
        "Beetroot red or anthocyanins",
        EvidenceLevel.MODERATE,
        ("ponceau 4r", "ponceau4r", "cochineal red a"),
    ),
    "allura_red": _AdditiveKnowledge(
        "Synthetic red food colour",
        "Beetroot red or anthocyanins",
        EvidenceLevel.MODERATE,
        ("allura red", "allura red ac"),
    ),
    "brilliant_blue": _AdditiveKnowledge(
        "Synthetic blue food colour",
        "Spirulina extract",
        EvidenceLevel.LIMITED,
        ("brilliant blue", "brilliant blue fcf"),
    ),
    "quinoline_yellow": _AdditiveKnowledge(
        "Synthetic yellow food colour",
        "Turmeric extract",
        EvidenceLevel.STRONG,
        ("quinoline yellow",),
    ),
    "sodium_benzoate": _AdditiveKnowledge(
        "Antimicrobial preservative",
        "Rosemary extract or refrigeration-based preservation",
        EvidenceLevel.MODERATE,
        ("sodium benzoate",),
    ),
    "potassium_sorbate": _AdditiveKnowledge(
        "Antimicrobial preservative, mold/yeast inhibitor",
        "Rosemary extract or refrigeration-based preservation",
        EvidenceLevel.LIMITED,
        ("potassium sorbate",),
    ),
    "sodium_nitrite": _AdditiveKnowledge(
        "Curing agent/preservative used in processed meats",
        "Celery powder based natural curing or uncured products",
        EvidenceLevel.STRONG,
        ("sodium nitrite",),
    ),
    "sodium_nitrate": _AdditiveKnowledge(
        "Curing agent/preservative used in processed meats",
        "Celery powder based natural curing or uncured products",
        EvidenceLevel.STRONG,
        ("sodium nitrate",),
    ),
    "sulphur_dioxide": _AdditiveKnowledge(
        "Preservative/antioxidant, common in dried fruit and wine",
        "Sulphite-free dried fruit",
        EvidenceLevel.MODERATE,
        ("sulphur dioxide", "sulfur dioxide", "sulphites", "sulfites"),
    ),
    "bha": _AdditiveKnowledge(
        "Synthetic antioxidant preservative",
        "Vitamin E (tocopherols) based preservation",
        EvidenceLevel.STRONG,
        ("bha", "butylated hydroxyanisole"),
    ),
    "bht": _AdditiveKnowledge(
        "Synthetic antioxidant preservative",
        "Vitamin E (tocopherols) based preservation",
        EvidenceLevel.STRONG,
        ("bht", "butylated hydroxytoluene"),
    ),
    "msg": _AdditiveKnowledge(
        "Flavour enhancer (umami)",
        "Naturally occurring glutamate sources like parmesan or mushrooms",
        EvidenceLevel.LIMITED,
        ("msg", "monosodium glutamate", "flavour enhancer 621"),
    ),
    "disodium_guanylate": _AdditiveKnowledge(
        "Flavour enhancer, often paired with MSG",
        "Mushroom or seaweed extract",
        EvidenceLevel.LIMITED,
        ("disodium guanylate",),
    ),
    "disodium_inosinate": _AdditiveKnowledge(
        "Flavour enhancer, often paired with MSG",
        "Mushroom or seaweed extract",
        EvidenceLevel.LIMITED,
        ("disodium inosinate",),
    ),
    "hfcs": _AdditiveKnowledge(
        "High-fructose sweetening syrup",
        "Whole fruit, honey, or reduced overall sugar",
        EvidenceLevel.STRONG,
        ("high fructose corn syrup", "high-fructose corn syrup", "corn syrup", "hfcs"),
    ),
    "invert_sugar_syrup": _AdditiveKnowledge(
        "Sweetening syrup used to prevent crystallization",
        "Honey or reduced overall sugar",
        EvidenceLevel.LIMITED,
        ("invert sugar", "invert syrup"),
    ),
}

_CATEGORY_BY_SLUG: Dict[str, str] = {
    "aspartame": "sweetener", "sucralose": "sweetener", "acesulfame_k": "sweetener",
    "saccharin": "sweetener", "cyclamate": "sweetener", "neotame": "sweetener", "stevia": "sweetener",
    "tartrazine": "colour", "sunset_yellow": "colour", "carmoisine": "colour",
    "ponceau_4r": "colour", "allura_red": "colour", "brilliant_blue": "colour", "quinoline_yellow": "colour",
    "sodium_benzoate": "preservative", "potassium_sorbate": "preservative",
    "sodium_nitrite": "preservative", "sodium_nitrate": "preservative",
    "sulphur_dioxide": "preservative", "bha": "preservative", "bht": "preservative",
    "msg": "flavour_enhancer", "disodium_guanylate": "flavour_enhancer", "disodium_inosinate": "flavour_enhancer",
    "hfcs": "sweetening_syrup", "invert_sugar_syrup": "sweetening_syrup",
}

_CONFIG_SECTIONS_BY_SLUG_PREFIX = (
    "artificial_sweeteners",
    "artificial_colours",
    "artificial_preservatives",
    "flavour_enhancers",
    "high_risk_sweetener_syrups",
)


@dataclass
class AdditiveReport:
    """Aggregate output of the Additive Engine for a single product."""

    detected_additives: List[DetectedAdditive] = field(default_factory=list)
    total_count: int = 0
    high_risk_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_additives": [a.to_dict() for a in self.detected_additives],
            "total_count": self.total_count,
            "high_risk_count": self.high_risk_count,
        }


class AdditiveEngine:
    """Detects and profiles food additives from ingredient text."""

    def __init__(self, config: Optional[HealthRulesConfig] = None) -> None:
        self._config = config or get_health_rules_config()
        self._config_lookup = self._build_config_lookup()

    def _build_config_lookup(self) -> Dict[str, Dict[str, Any]]:
        """Flatten all additive config sections into slug -> {e_number, ins_number, risk_level, ...}."""
        merged: Dict[str, Dict[str, Any]] = {}
        for section_name in _CONFIG_SECTIONS_BY_SLUG_PREFIX:
            section = self._config.data.get(section_name, {})
            for slug, details in section.items():
                merged[slug] = details
        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, product: Dict[str, Any]) -> AdditiveReport:
        """Detect and profile all additives found in a product's ingredient text."""
        ingredients_text = (product.get("ingredients_text") or "").lower()
        report = AdditiveReport()

        matched_slugs = set()
        matched_slugs.update(self._match_by_alias(ingredients_text))
        matched_slugs.update(self._match_by_fuzzy(ingredients_text))

        detected: List[DetectedAdditive] = []
        for slug in sorted(matched_slugs):
            additive = self._build_detected_additive(slug, ingredients_text)
            if additive:
                detected.append(additive)

        detected.extend(self._match_raw_codes(ingredients_text, matched_slugs))

        report.detected_additives = detected
        report.total_count = len(detected)
        report.high_risk_count = sum(
            1 for a in detected if a.risk_level in (RiskLevel.HIGH, RiskLevel.VERY_HIGH)
        )
        return report

    def compute_additive_score_points(self, report: AdditiveReport, max_points: float) -> float:
        """
        Convert additive findings into a points contribution (0..max_points)
        for the overall weighted health score. Every high-risk additive
        costs more than a low-risk one; more additives cost more overall.
        """
        if report.total_count == 0:
            return max_points

        risk_penalty_weights = {
            RiskLevel.VERY_LOW: 0.02,
            RiskLevel.LOW: 0.05,
            RiskLevel.MODERATE: 0.15,
            RiskLevel.HIGH: 0.30,
            RiskLevel.VERY_HIGH: 0.40,
            RiskLevel.UNKNOWN: 0.10,
        }
        total_penalty_ratio = sum(
            risk_penalty_weights.get(a.risk_level, 0.1) for a in report.detected_additives
        )
        remaining_ratio = max(0.0, 1.0 - total_penalty_ratio)
        return max_points * remaining_ratio

    # ------------------------------------------------------------------
    # Matching strategies
    # ------------------------------------------------------------------

    def _match_by_alias(self, text: str) -> List[str]:
        matches = []
        for slug, knowledge in _KNOWLEDGE_BASE.items():
            if any(alias in text for alias in knowledge.aliases):
                matches.append(slug)
        return matches

    def _match_by_fuzzy(self, text: str) -> List[str]:
        """Catch spelling variants / OCR noise using RapidFuzz token_sort_ratio."""
        matches = []
        tokens = self._extract_candidate_phrases(text)
        for slug, knowledge in _KNOWLEDGE_BASE.items():
            for alias in knowledge.aliases:
                for token in tokens:
                    score = fuzz.token_sort_ratio(alias, token)
                    if score >= _FUZZY_MATCH_THRESHOLD:
                        matches.append(slug)
                        break
                else:
                    continue
                break
        return matches

    @staticmethod
    def _extract_candidate_phrases(text: str) -> List[str]:
        """
        Split ingredient text into comma-separated phrases and generate
        sliding 1-3 word windows so fuzzy matching can compare against
        multi-word additive names.
        """
        raw_phrases = [p.strip() for p in re.split(r"[,;()\[\]]", text) if p.strip()]
        phrases: List[str] = []
        for phrase in raw_phrases:
            words = phrase.split()
            phrases.append(phrase)
            for window in (1, 2, 3):
                for i in range(len(words) - window + 1):
                    phrases.append(" ".join(words[i : i + window]))
        return phrases

    def _match_raw_codes(self, text: str, already_matched: set) -> List[DetectedAdditive]:
        """Catch bare E-numbers / INS-numbers not resolved to a known slug alias."""
        found: List[DetectedAdditive] = []
        seen_codes = set()

        for pattern, prefix in ((_E_NUMBER_PATTERN, "E"), (_INS_NUMBER_PATTERN, "INS")):
            for match in pattern.finditer(text):
                code = match.group(1).upper()
                normalized_e = f"E{code}"
                # Skip if this code already resolved to a known slug via alias/fuzzy match
                slug_for_code = self._slug_for_code(normalized_e)
                if slug_for_code and slug_for_code in already_matched:
                    continue
                if normalized_e in seen_codes:
                    continue
                seen_codes.add(normalized_e)

                if slug_for_code:
                    additive = self._build_detected_additive(slug_for_code, text)
                    if additive:
                        found.append(additive)
                else:
                    found.append(
                        DetectedAdditive(
                            detected_name=f"{prefix} {code}",
                            scientific_name=f"Additive coded {normalized_e}",
                            category="unclassified",
                            purpose="Purpose not in local knowledge base; verify with regulatory database.",
                            risk_level=RiskLevel.UNKNOWN,
                            daily_limit=None,
                            alternative=None,
                            evidence_level=EvidenceLevel.INCONCLUSIVE,
                            e_number=normalized_e,
                            ins_number=code,
                        )
                    )
        return found

    def _slug_for_code(self, e_number: str) -> Optional[str]:
        for slug, details in self._config_lookup.items():
            if details.get("e_number", "").upper() == e_number.upper():
                return slug
        return None

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def _build_detected_additive(self, slug: str, source_text: str) -> Optional[DetectedAdditive]:
        knowledge = _KNOWLEDGE_BASE.get(slug)
        if not knowledge:
            return None
        config_details = self._config_lookup.get(slug, {})

        risk_level_str = config_details.get("risk_level", "unknown")
        try:
            risk_level = RiskLevel(risk_level_str)
        except ValueError:
            risk_level = RiskLevel.UNKNOWN

        daily_limit = self._format_daily_limit(config_details)
        matched_alias = next((a for a in knowledge.aliases if a in source_text), knowledge.aliases[0])

        return DetectedAdditive(
            detected_name=matched_alias.title(),
            scientific_name=slug.replace("_", " ").title(),
            category=_CATEGORY_BY_SLUG.get(slug, "other"),
            purpose=knowledge.purpose,
            risk_level=risk_level,
            daily_limit=daily_limit,
            alternative=knowledge.alternative,
            evidence_level=knowledge.evidence_level,
            e_number=config_details.get("e_number"),
            ins_number=config_details.get("ins_number"),
        )

    @staticmethod
    def _format_daily_limit(config_details: Dict[str, Any]) -> Optional[str]:
        limit = config_details.get("daily_limit_mg_per_kg_bw")
        if limit is None:
            return None
        return f"{limit} mg/kg body weight/day (ADI)"
