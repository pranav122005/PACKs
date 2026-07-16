"""
backend/ai/recommendation_ai.py

Recommendation AI — turns the rule-based, deterministic output of
backend/engines/recommendation_engine.py (and the full product report)
into a short, natural-language, personalized narrative for the user via
Ollama/llama3.2.

This service NEVER invents alternatives, scores, or nutrient facts — it
is explicitly prompted (via PromptBuilder) to narrate only what the
RecommendationEngine and ReportEngine already computed. This keeps the
LLM in a strictly "explain and personalize" role rather than a
"decide what's healthy" role, which stays with the deterministic engines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.ai.ollama_client import OllamaClient, OllamaClientError
from backend.ai.prompt_builder import PromptBuilder
from backend.schemas.user_profile import UserProfile


@dataclass
class RecommendationNarrative:
    """Structured result of narrating a product's recommendations for a user."""

    narrative: str
    structured_recommendations: Dict[str, Any] = field(default_factory=dict)
    retrieved_context: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "narrative": self.narrative,
            "structured_recommendations": self.structured_recommendations,
            "retrieved_context": self.retrieved_context,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "success": self.success,
            "error": self.error,
        }


class RecommendationAIService:
    """
    Narrates a product's final report + structured recommendation data
    into personalized, human-friendly guidance:

        product_report + recommendation_report + user_profile
          -> PromptBuilder (RAG + strictly-grounded prompt)
          -> OllamaClient (llama3.2 generation)
          -> RecommendationNarrative
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> None:
        self._client = ollama_client or OllamaClient()
        self._prompt_builder = prompt_builder or PromptBuilder()

    def generate_narrative(
        self,
        product_report: Dict[str, Any],
        recommendation_report: Dict[str, Any],
        user_profile: Optional[UserProfile] = None,
    ) -> RecommendationNarrative:
        """
        Generate a short personalized narrative. `product_report` is the
        final dict from ReportEngine.generate(); `recommendation_report`
        is typically `product_report["recommendations"]` but can be passed
        separately if generated independently.
        """
        start = time.perf_counter()

        if not product_report:
            return RecommendationNarrative(
                narrative="",
                success=False,
                error="product_report is required to generate a recommendation narrative.",
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

        prompt_bundle = self._prompt_builder.build_recommendation_prompt(
            product_report=product_report,
            recommendation_data=recommendation_report or {},
            user_profile=user_profile,
        )

        try:
            response = self._client.generate(prompt=prompt_bundle["prompt"], temperature=0.5)
        except OllamaClientError as exc:
            return RecommendationNarrative(
                narrative="",
                structured_recommendations=recommendation_report or {},
                retrieved_context=prompt_bundle["retrieved_context"],
                success=False,
                error=str(exc),
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

        return RecommendationNarrative(
            narrative=response.text,
            structured_recommendations=recommendation_report or {},
            retrieved_context=prompt_bundle["retrieved_context"],
            processing_time_ms=(time.perf_counter() - start) * 1000,
            success=True,
        )

    def generate_gym_narrative(
        self,
        product_report: Dict[str, Any],
        user_profile: Optional[UserProfile] = None,
    ) -> RecommendationNarrative:
        """Convenience method focusing the narrative specifically on gym/fitness guidance."""
        recommendations = product_report.get("recommendations", {})
        gym_only = {"gym_recommendation": recommendations.get("gym_recommendation")}
        return self.generate_narrative(product_report, gym_only, user_profile)

    def generate_daily_intake_narrative(
        self,
        product_report: Dict[str, Any],
        user_profile: Optional[UserProfile] = None,
    ) -> RecommendationNarrative:
        """Convenience method focusing the narrative specifically on daily-intake guidance."""
        recommendations = product_report.get("recommendations", {})
        intake_only = {"daily_intake_recommendation": recommendations.get("daily_intake_recommendation")}
        return self.generate_narrative(product_report, intake_only, user_profile)
