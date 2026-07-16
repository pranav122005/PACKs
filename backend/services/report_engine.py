from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("packs.report_engine")


class ReportEngine:
    """Produces the final client-facing report."""

    def generate(self, product: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "product": {
                "barcode": product.get("barcode") or product.get("code"),
                "product_name": product.get("product_name", "Unknown"),
                "brand": product.get("brand") or product.get("brands"),
            },
            "nutrition": analysis.get("nutrition", {}),
            "disease": analysis.get("disease", {}),
            "nova": analysis.get("nova", {}),
            "additives": analysis.get("additives", {}),
            "recommendations": analysis.get("recommendations", {}),
            "overall_score": round(analysis.get("overall_score", 0), 2),
            "overall_band": analysis.get("overall_band", "unknown"),
            "warnings": [],
            "positives": [],
            "summary": "",
        }
