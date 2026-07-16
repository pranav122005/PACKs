"""
backend/services/report_engine.py

Report Engine — takes the raw AnalysisResult produced by
ProductAnalysisService and assembles the final, API-ready JSON report:

{
    "product": {...},
    "nutrition": {...},
    "disease": {...},
    "nova": {...},
    "additives": {...},
    "recommendations": {...},
    "overall_score": float,
    "overall_band": str,
    "warnings": [...],
    "positives": [...],
    "summary": str
}

This is intentionally a thin presentation layer: it does not compute any
new health facts, it only formats and narrates what the engines already
determined. Keeping this separate from ProductAnalysisService follows
the Single Responsibility Principle — orchestration vs. presentation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

from backend.schemas.common import Severity
from backend.services.product_analysis_service import AnalysisResult


@dataclass
class ReportEngine:
    """Builds the final, client-facing analysis report from an AnalysisResult."""

    include_generated_at: bool = True

    def generate(self, result: AnalysisResult) -> Dict[str, Any]:
        """Build the complete final report dictionary for a single product."""
        product_summary = self._build_product_summary(result.product)

        report: Dict[str, Any] = {
            "product": product_summary,
            "nutrition": result.nutrition.to_dict(),
            "disease": result.disease.to_dict(),
            "nova": result.nova.to_dict(),
            "additives": result.additives.to_dict(),
            "recommendations": result.recommendations.to_dict(),
            "overall_score": round(result.overall_score, 2),
            "overall_band": result.overall_band,
            "warnings": [w.to_dict() for w in self._sorted_warnings(result.warnings)],
            "positives": [p.to_dict() for p in result.positives],
            "summary": self._build_summary(result),
        }

        if self.include_generated_at:
            report["generated_at"] = datetime.now(timezone.utc).isoformat()

        return report

    def generate_json(self, result: AnalysisResult, indent: int = 2) -> str:
        """Convenience wrapper returning the final report as a JSON string."""
        return json.dumps(self.generate(result), indent=indent, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_product_summary(product: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "barcode": product.get("barcode") or product.get("code"),
            "product_name": product.get("product_name", "Unknown product"),
            "brand": product.get("brand") or product.get("brands"),
            "category": product.get("category") or product.get("categories"),
            "quantity": product.get("quantity"),
            "image_url": product.get("image_url"),
        }

    @staticmethod
    def _sorted_warnings(warnings: List[Any]) -> List[Any]:
        severity_rank = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        return sorted(warnings, key=lambda w: severity_rank.get(w.severity, 5))

    def _build_summary(self, result: AnalysisResult) -> str:
        """Compose a short, human-readable narrative summary of the full analysis."""
        product_name = result.product.get("product_name", "This product")
        sentences: List[str] = []

        sentences.append(
            f"{product_name} scores {round(result.overall_score, 1)}/100 "
            f"({result.overall_band}) on the PACKS health score."
        )

        critical_or_high = [
            w for w in result.warnings if w.severity in (Severity.CRITICAL, Severity.HIGH)
        ]
        if critical_or_high:
            top_issues = ", ".join(w.title for w in critical_or_high[:3])
            sentences.append(f"Key concerns: {top_issues}.")
        elif result.warnings:
            sentences.append("No major concerns detected, though minor notes are listed below.")
        else:
            sentences.append("No significant nutritional or additive concerns were detected.")

        if result.positives:
            top_positives = ", ".join(p.title for p in result.positives[:3])
            sentences.append(f"Notable positives: {top_positives}.")

        sentences.append(
            f"Classified as NOVA {result.nova.nova_group} ({result.nova.label})."
        )

        high_risk_conditions = result.disease.high_risk_conditions()
        if high_risk_conditions:
            sentences.append(
                "Elevated risk flagged for: " + ", ".join(high_risk_conditions) + "."
            )

        if result.additives.total_count > 0:
            sentences.append(
                f"{result.additives.total_count} additive(s) of interest detected, "
                f"{result.additives.high_risk_count} of which are high/very-high risk."
            )

        return " ".join(sentences)
