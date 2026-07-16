"""
backend/ai/ingredient_explainer.py

Ingredient Explainer — generates a plain-language explanation of a
single ingredient/additive, grounded first in PACKS' own structured
Additive Engine data (if the ingredient is recognized) and supplemented
with RAG context, to minimize hallucination risk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.ai.ollama_client import OllamaClient, OllamaClientError
from backend.ai.prompt_builder import PromptBuilder
from backend.engines.additive_engine import AdditiveEngine
from backend.schemas.user_profile import UserProfile


@dataclass
class IngredientExplanation:
    """Structured result of explaining a single ingredient."""

    ingredient_name: str
    explanation: str
    matched_structured_data: Optional[Dict[str, Any]] = None
    retrieved_context: List[Dict[str, Any]] = field(default_factory=list)
    is_grounded: bool = False  # True if matched_structured_data was found in the additive knowledge base
    processing_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ingredient_name": self.ingredient_name,
            "explanation": self.explanation,
            "matched_structured_data": self.matched_structured_data,
            "retrieved_context": self.retrieved_context,
            "is_grounded": self.is_grounded,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "success": self.success,
            "error": self.error,
        }


class IngredientExplainerService:
    """
    Explains a single ingredient/additive by name:

        ingredient_name
          -> AdditiveEngine (lookup known structured data, if any)
          -> PromptBuilder (RAG + grounded prompt)
          -> OllamaClient (llama3.2 generation)
          -> IngredientExplanation
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        additive_engine: Optional[AdditiveEngine] = None,
    ) -> None:
        self._client = ollama_client or OllamaClient()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._additive_engine = additive_engine or AdditiveEngine()

    def explain(
        self,
        ingredient_name: str,
        user_profile: Optional[UserProfile] = None,
    ) -> IngredientExplanation:
        """Generate a grounded, plain-language explanation for a single ingredient."""
        start = time.perf_counter()

        if not ingredient_name or not ingredient_name.strip():
            return IngredientExplanation(
                ingredient_name=ingredient_name,
                explanation="",
                success=False,
                error="Ingredient name cannot be empty.",
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

        structured_data = self._lookup_structured_data(ingredient_name)
        prompt_bundle = self._prompt_builder.build_ingredient_explainer_prompt(
            ingredient_name=ingredient_name,
            additive_details=structured_data,
            user_profile=user_profile,
        )

        try:
            response = self._client.generate(prompt=prompt_bundle["prompt"])
        except OllamaClientError as exc:
            return IngredientExplanation(
                ingredient_name=ingredient_name,
                explanation="",
                matched_structured_data=structured_data,
                retrieved_context=prompt_bundle["retrieved_context"],
                is_grounded=structured_data is not None,
                success=False,
                error=str(exc),
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

        return IngredientExplanation(
            ingredient_name=ingredient_name,
            explanation=response.text,
            matched_structured_data=structured_data,
            retrieved_context=prompt_bundle["retrieved_context"],
            is_grounded=structured_data is not None,
            processing_time_ms=(time.perf_counter() - start) * 1000,
            success=True,
        )

    def explain_batch(
        self,
        ingredient_names: List[str],
        user_profile: Optional[UserProfile] = None,
    ) -> List[IngredientExplanation]:
        """Explain multiple ingredients in sequence (e.g. all additives found in a product)."""
        return [self.explain(name, user_profile) for name in ingredient_names]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _lookup_structured_data(self, ingredient_name: str) -> Optional[Dict[str, Any]]:
        """
        Reuse the Additive Engine's own detection logic to check whether
        this ingredient is a recognized additive, by running it through
        `analyze()` against a synthetic single-ingredient product. This
        avoids duplicating the additive knowledge base in this file.
        """
        synthetic_product = {"ingredients_text": ingredient_name}
        additive_report = self._additive_engine.analyze(synthetic_product)
        if not additive_report.detected_additives:
            return None
        # Prefer the highest-confidence (first) match; the additive engine
        # already sorts/dedupes by slug during detection.
        return additive_report.detected_additives[0].to_dict()
