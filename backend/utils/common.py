"""
backend/schemas/common.py

Shared enums and dataclasses used by every engine and service in the
PACKS analysis pipeline. Keeping these in one place avoids duplicated,
divergent definitions of "what a warning looks like" across engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RiskLevel(str, Enum):
    """Normalized risk vocabulary used across nutrition, additive and disease engines."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Severity of a single warning/positive note surfaced to the user."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceLevel(str, Enum):
    """How strong the scientific backing is for a given additive claim."""

    STRONG = "strong"          # Multiple peer-reviewed human studies / regulatory consensus
    MODERATE = "moderate"      # Some human studies, mixed or limited results
    LIMITED = "limited"        # Mostly animal studies or preliminary data
    INCONCLUSIVE = "inconclusive"


class NovaGroup(int, Enum):
    """NOVA food processing classification groups (1-4)."""

    UNPROCESSED_OR_MINIMALLY_PROCESSED = 1
    PROCESSED_CULINARY_INGREDIENTS = 2
    PROCESSED_FOODS = 3
    ULTRA_PROCESSED = 4


@dataclass
class Warning:
    """A single negative finding surfaced to the user, always with a reason."""

    code: str
    title: str
    reason: str
    severity: Severity
    category: str  # e.g. "sugar", "additive", "disease:diabetes"
    value: Optional[float] = None
    unit: Optional[str] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "reason": self.reason,
            "severity": self.severity.value,
            "category": self.category,
            "value": self.value,
            "unit": self.unit,
            "recommendation": self.recommendation,
        }


@dataclass
class Positive:
    """A single positive finding surfaced to the user, always with a reason."""

    code: str
    title: str
    reason: str
    category: str
    value: Optional[float] = None
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "reason": self.reason,
            "category": self.category,
            "value": self.value,
            "unit": self.unit,
        }


@dataclass
class ScoreBreakdownItem:
    """Explains exactly how many points a single nutrient contributed/cost."""

    factor: str
    max_points: float
    awarded_points: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.factor,
            "max_points": round(self.max_points, 2),
            "awarded_points": round(self.awarded_points, 2),
            "points_lost": round(self.max_points - self.awarded_points, 2),
            "explanation": self.explanation,
        }


@dataclass
class DiseaseAssessment:
    """Result of analysing a product against a single health condition."""

    condition: str
    risk: RiskLevel
    reason: str
    recommendation: str
    contributing_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "risk": self.risk.value,
            "reason": self.reason,
            "recommendation": self.recommendation,
            "contributing_factors": self.contributing_factors,
        }


@dataclass
class DetectedAdditive:
    """A single additive/ingredient of concern identified in a product."""

    detected_name: str
    scientific_name: str
    category: str  # sweetener / colour / preservative / flavour_enhancer / other
    purpose: str
    risk_level: RiskLevel
    daily_limit: Optional[str]
    alternative: Optional[str]
    evidence_level: EvidenceLevel
    e_number: Optional[str] = None
    ins_number: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_name": self.detected_name,
            "scientific_name": self.scientific_name,
            "category": self.category,
            "purpose": self.purpose,
            "risk_level": self.risk_level.value,
            "daily_limit": self.daily_limit,
            "alternative": self.alternative,
            "evidence_level": self.evidence_level.value,
            "e_number": self.e_number,
            "ins_number": self.ins_number,
        }
