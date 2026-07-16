"""
backend/ai/prompt_builder.py

Prompt Builder — the RAG (Retrieval-Augmented Generation) and prompt
engineering layer shared by every AI service in backend/ai/.

Responsibilities:
    - Retrieve relevant knowledge snippets ("RAG context") for a query,
      either from an injected knowledge repository (e.g. the Ingredient
      Knowledge Base / product database) or a small built-in fallback
      corpus, ranked with RapidFuzz so PACKS' existing fuzzy-matching
      stack doubles as its retrieval scorer (no separate vector DB
      dependency required).
    - Assemble strictly-grounded, role-based prompts that combine:
        system instructions -> user profile -> product report ->
        retrieved RAG context -> the user's actual question
    - Enforce prompt-engineering guardrails: explicit output format,
      instructions not to hallucinate beyond the provided context, and
      a consistent "consult a professional" disclaimer for medical topics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from rapidfuzz import fuzz

from backend.schemas.user_profile import UserProfile

# ----------------------------------------------------------------------
# Built-in fallback knowledge corpus (used when no repository is injected,
# or as a supplement to it). Keeps the RAG layer functional standalone.
# ----------------------------------------------------------------------
_FALLBACK_KNOWLEDGE_CORPUS: List[Dict[str, str]] = [
    {
        "topic": "sugar",
        "text": "The WHO recommends limiting free sugar intake to under 25g/day for an average adult. "
                "Excess added sugar is linked to weight gain, type 2 diabetes risk, and dental decay.",
    },
    {
        "topic": "sodium",
        "text": "The WHO recommends under 5g of salt (about 2000mg sodium) per day. Excess sodium intake "
                "is a major driver of high blood pressure and cardiovascular disease risk.",
    },
    {
        "topic": "trans fat",
        "text": "Trans fat has no known safe consumption level. It raises LDL ('bad') cholesterol and "
                "lowers HDL ('good') cholesterol, increasing heart disease risk even in small amounts.",
    },
    {
        "topic": "nova classification",
        "text": "NOVA groups classify foods by degree of processing: NOVA 1 unprocessed/minimally "
                "processed, NOVA 2 processed culinary ingredients, NOVA 3 processed foods, NOVA 4 "
                "ultra-processed foods with industrial additives rarely used in home cooking.",
    },
    {
        "topic": "artificial sweeteners",
        "text": "Artificial sweeteners like aspartame and sucralose are non-caloric alternatives to "
                "sugar, regulated with an Acceptable Daily Intake (ADI). Evidence on long-term metabolic "
                "effects is still debated among researchers.",
    },
    {
        "topic": "artificial colours",
        "text": "Some synthetic food colours (e.g. tartrazine, sunset yellow) have been associated in "
                "some studies with hyperactivity in children, prompting mandatory warning labels in the EU.",
    },
    {
        "topic": "msg",
        "text": "Monosodium glutamate (MSG) is a flavour enhancer providing umami taste. Major food safety "
                "bodies (FDA, EFSA) consider it safe at typical consumption levels; a subset of individuals "
                "report sensitivity symptoms.",
    },
    {
        "topic": "diabetes diet",
        "text": "For blood sugar management, prioritizing low-glycemic-index foods, fiber, and limiting "
                "free sugars helps reduce post-meal glucose spikes.",
    },
    {
        "topic": "protein needs",
        "text": "General recommendations suggest 0.8-1.6g of protein per kg of bodyweight daily depending "
                "on activity level, with higher intakes benefiting muscle repair and growth for active individuals.",
    },
    {
        "topic": "fiber benefits",
        "text": "Dietary fiber slows digestion, promotes satiety, supports gut health, and can help "
                "moderate blood sugar response when eaten alongside sugars or refined carbohydrates.",
    },
]


class KnowledgeRepositoryProtocol(Protocol):
    """
    Minimal contract for an external knowledge source (e.g. the Ingredient
    Knowledge Base repository) that can supply RAG context chunks.
    """

    def search_knowledge(self, query: str, limit: int) -> List[Dict[str, str]]:
        """Return candidate knowledge chunks, each a dict with at least a 'text' key."""
        ...


@dataclass
class RetrievedChunk:
    """A single piece of retrieved context with its relevance score."""

    topic: str
    text: str
    relevance_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"topic": self.topic, "text": self.text, "relevance_score": round(self.relevance_score, 2)}


class KnowledgeRetriever:
    """
    RAG retriever: ranks knowledge chunks (from an injected repository
    and/or the built-in fallback corpus) against a query using RapidFuzz,
    returning the top-k most relevant chunks to ground an LLM prompt.
    """

    def __init__(
        self,
        repository: Optional[KnowledgeRepositoryProtocol] = None,
        use_fallback_corpus: bool = True,
        min_relevance_score: float = 30.0,
    ) -> None:
        self._repository = repository
        self._use_fallback_corpus = use_fallback_corpus
        self._min_relevance_score = min_relevance_score

    def retrieve(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        """Retrieve the top-k most relevant knowledge chunks for `query`."""
        if not query or not query.strip():
            return []

        candidates: List[Dict[str, str]] = []
        if self._repository is not None:
            try:
                candidates.extend(self._repository.search_knowledge(query, limit=20))
            except Exception:
                # RAG retrieval failures should never break the AI pipeline;
                # fall through to the built-in corpus instead.
                pass
        if self._use_fallback_corpus:
            candidates.extend(_FALLBACK_KNOWLEDGE_CORPUS)

        scored: List[RetrievedChunk] = []
        for chunk in candidates:
            text = chunk.get("text", "")
            topic = chunk.get("topic", "")
            if not text:
                continue
            score = max(
                fuzz.partial_ratio(query.lower(), text.lower()),
                fuzz.partial_ratio(query.lower(), topic.lower()) if topic else 0,
                fuzz.token_set_ratio(query.lower(), text.lower()),
            )
            if score >= self._min_relevance_score:
                scored.append(RetrievedChunk(topic=topic or "general", text=text, relevance_score=score))

        # Deduplicate identical text chunks, keep highest score, then rank.
        deduped: Dict[str, RetrievedChunk] = {}
        for chunk in scored:
            existing = deduped.get(chunk.text)
            if existing is None or chunk.relevance_score > existing.relevance_score:
                deduped[chunk.text] = chunk

        ranked = sorted(deduped.values(), key=lambda c: c.relevance_score, reverse=True)
        return ranked[:top_k]


class PromptBuilder:
    """
    Assembles grounded, role-based prompts for each AI service. Every
    prompt follows the same structure for consistency and to make
    prompt-injection via product/ingredient text harder to exploit:

        [SYSTEM INSTRUCTIONS]
        [USER PROFILE]
        [PRODUCT REPORT]  (if applicable)
        [RETRIEVED KNOWLEDGE CONTEXT]
        [USER QUESTION / TASK]
        [OUTPUT FORMAT INSTRUCTIONS]
    """

    _BASE_SYSTEM_INSTRUCTIONS = (
        "You are the PACKS Nutrition Assistant, an AI that explains packaged-food health analysis "
        "to everyday consumers in clear, plain language. Follow these rules strictly:\n"
        "1. Only use facts given to you in the PRODUCT REPORT and KNOWLEDGE CONTEXT sections below. "
        "Do not invent nutrient values, additive names, or health claims not present in that data.\n"
        "2. If the provided data is insufficient to answer, say so honestly instead of guessing.\n"
        "3. You are not a doctor. For any medical question, give general nutrition-education "
        "information and clearly recommend consulting a qualified healthcare professional for "
        "personal medical decisions.\n"
        "4. Be concise, warm, and non-judgmental about the user's food choices.\n"
        "5. Never fabricate citations, studies, or statistics beyond what is provided."
    )

    def __init__(self, retriever: Optional[KnowledgeRetriever] = None) -> None:
        self._retriever = retriever or KnowledgeRetriever()

    # ------------------------------------------------------------------
    # Nutrition Chat
    # ------------------------------------------------------------------

    def build_nutrition_chat_prompt(
        self,
        user_question: str,
        product_report: Optional[Dict[str, Any]],
        user_profile: Optional[UserProfile],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Build the full message list for a `/api/chat`-style conversation
        about a specific product report.
        """
        retrieved = self._retriever.retrieve(user_question, top_k=4)

        system_message = self._BASE_SYSTEM_INSTRUCTIONS
        context_sections: List[str] = []

        if user_profile:
            context_sections.append(f"USER PROFILE:\n{user_profile.to_prompt_summary()}")

        if product_report:
            context_sections.append(f"PRODUCT REPORT (JSON):\n{self._summarize_product_report(product_report)}")

        if retrieved:
            knowledge_text = "\n".join(f"- ({c.topic}) {c.text}" for c in retrieved)
            context_sections.append(f"KNOWLEDGE CONTEXT:\n{knowledge_text}")

        context_sections.append(
            "OUTPUT FORMAT: Answer in 2-5 short sentences of plain language, no markdown headers."
        )

        full_system_message = system_message + "\n\n" + "\n\n".join(context_sections)

        messages: List[Dict[str, str]] = [{"role": "system", "content": full_system_message}]
        for turn in (conversation_history or []):
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_question})

        return {"messages": messages, "retrieved_context": [c.to_dict() for c in retrieved]}

    # ------------------------------------------------------------------
    # Ingredient Explainer
    # ------------------------------------------------------------------

    def build_ingredient_explainer_prompt(
        self,
        ingredient_name: str,
        additive_details: Optional[Dict[str, Any]],
        user_profile: Optional[UserProfile] = None,
    ) -> Dict[str, Any]:
        """Build a grounded prompt asking the LLM to explain a single ingredient/additive."""
        retrieved = self._retriever.retrieve(ingredient_name, top_k=3)

        sections = [self._BASE_SYSTEM_INSTRUCTIONS]

        if additive_details:
            sections.append(
                "STRUCTURED ADDITIVE DATA (JSON, treat as ground truth):\n"
                f"{self._format_dict_as_bullets(additive_details)}"
            )
        else:
            sections.append(
                f"No structured data is available for '{ingredient_name}' in the PACKS database. "
                "Clearly state that this ingredient is not in the local knowledge base and give only "
                "very general, cautious context if you are highly confident about the classification "
                "(e.g. 'commonly used as a preservative'); do not invent a risk level or daily limit."
            )

        if retrieved:
            knowledge_text = "\n".join(f"- ({c.topic}) {c.text}" for c in retrieved)
            sections.append(f"RELATED KNOWLEDGE CONTEXT:\n{knowledge_text}")

        if user_profile:
            sections.append(f"USER PROFILE:\n{user_profile.to_prompt_summary()}")

        sections.append(
            "TASK: Explain what this ingredient is, why it's used, and its risk level in plain, "
            "friendly language for a non-expert. Keep it under 100 words. If a daily limit or "
            "alternative is provided in the structured data, mention it briefly."
        )

        prompt = "\n\n".join(sections)
        return {"prompt": prompt, "retrieved_context": [c.to_dict() for c in retrieved]}

    # ------------------------------------------------------------------
    # Recommendation AI
    # ------------------------------------------------------------------

    def build_recommendation_prompt(
        self,
        product_report: Dict[str, Any],
        recommendation_data: Dict[str, Any],
        user_profile: Optional[UserProfile] = None,
    ) -> Dict[str, Any]:
        """Build a grounded prompt asking the LLM to narrate personalized recommendations."""
        query = self._build_recommendation_query(product_report, user_profile)
        retrieved = self._retriever.retrieve(query, top_k=4)

        sections = [self._BASE_SYSTEM_INSTRUCTIONS]
        sections.append(f"PRODUCT REPORT SUMMARY:\n{self._summarize_product_report(product_report)}")
        sections.append(
            "STRUCTURED RECOMMENDATION DATA (JSON, treat as ground truth — do not contradict this):\n"
            f"{self._format_dict_as_bullets(recommendation_data)}"
        )

        if user_profile:
            sections.append(f"USER PROFILE:\n{user_profile.to_prompt_summary()}")

        if retrieved:
            knowledge_text = "\n".join(f"- ({c.topic}) {c.text}" for c in retrieved)
            sections.append(f"KNOWLEDGE CONTEXT:\n{knowledge_text}")

        sections.append(
            "TASK: Write a short, encouraging, personalized recommendation (3-5 sentences) explaining "
            "whether this product fits the user's profile and goals, referencing the specific "
            "alternatives, gym guidance, or daily-intake data above where relevant. Do not introduce "
            "any alternative product, nutrient value, or claim not present in the structured data above."
        )

        prompt = "\n\n".join(sections)
        return {"prompt": prompt, "retrieved_context": [c.to_dict() for c in retrieved]}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_recommendation_query(product_report: Dict[str, Any], user_profile: Optional[UserProfile]) -> str:
        parts = [product_report.get("summary", "")]
        if user_profile:
            parts.append(" ".join(user_profile.health_conditions))
            if user_profile.fitness_goal:
                parts.append(user_profile.fitness_goal)
        return " ".join(p for p in parts if p)

    @staticmethod
    def _summarize_product_report(product_report: Dict[str, Any]) -> str:
        """
        Compact, LLM-friendly textual summary of the (potentially large)
        final report JSON, avoiding dumping the entire raw structure into
        the prompt (keeps token usage bounded and predictable).
        """
        product = product_report.get("product", {})
        nutrition = product_report.get("nutrition", {})
        nova = product_report.get("nova", {})
        additives = product_report.get("additives", {})

        lines = [
            f"Product: {product.get('product_name', 'Unknown')} ({product.get('brand', 'Unknown brand')})",
            f"Overall Score: {product_report.get('overall_score')} ({product_report.get('overall_band')})",
            f"NOVA Group: {nova.get('nova_group')} - {nova.get('label')}",
            f"Additives detected: {additives.get('total_count', 0)} "
            f"({additives.get('high_risk_count', 0)} high-risk)",
        ]
        for verdict in nutrition.get("verdicts", [])[:8]:
            lines.append(
                f"- {verdict.get('nutrient')}: {verdict.get('value')}{verdict.get('unit', '')} "
                f"({verdict.get('level')})"
            )
        warnings = product_report.get("warnings", [])
        if warnings:
            lines.append("Top warnings: " + "; ".join(w.get("title", "") for w in warnings[:3]))
        return "\n".join(lines)

    @staticmethod
    def _format_dict_as_bullets(data: Dict[str, Any], max_depth: int = 2) -> str:
        """Render a (possibly nested) dict as a compact bullet list for prompt injection safety and brevity."""

        def render(value: Any, depth: int) -> List[str]:
            lines: List[str] = []
            indent = "  " * depth
            if isinstance(value, dict):
                for key, val in value.items():
                    if isinstance(val, (dict, list)) and depth < max_depth:
                        lines.append(f"{indent}- {key}:")
                        lines.extend(render(val, depth + 1))
                    else:
                        lines.append(f"{indent}- {key}: {val}")
            elif isinstance(value, list):
                for item in value[:5]:
                    if isinstance(item, (dict, list)) and depth < max_depth:
                        lines.extend(render(item, depth + 1))
                    else:
                        lines.append(f"{indent}- {item}")
            else:
                lines.append(f"{indent}{value}")
            return lines

        return "\n".join(render(data, 0))
