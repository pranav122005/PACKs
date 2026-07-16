"""
backend/ai/nutrition_chat.py

Nutrition Chat — a conversational AI service that lets users ask
free-form questions about a specific product's PACKS analysis report,
grounded via RAG context and personalized to their UserProfile.

Maintains lightweight in-memory conversation sessions so follow-up
questions ("what about for my kids?") retain context. In a horizontally
scaled deployment, swap `_sessions` for a Redis-backed store behind the
same interface.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional

from backend.ai.ollama_client import OllamaClient, OllamaClientError
from backend.ai.prompt_builder import PromptBuilder
from backend.schemas.user_profile import UserProfile

_MAX_HISTORY_TURNS = 12  # user+assistant messages kept per session (excludes system message)


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class NutritionChatResponse:
    """Structured result of a single chat turn."""

    session_id: str
    answer: str
    retrieved_context: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    processing_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "answer": self.answer,
            "retrieved_context": self.retrieved_context,
            "model": self.model,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "success": self.success,
            "error": self.error,
        }


class NutritionChatService:
    """
    Orchestrates conversational Q&A about a product report:

        question + history + product_report + user_profile
          -> PromptBuilder (RAG + prompt assembly)
          -> OllamaClient (llama3.2 chat completion)
          -> NutritionChatResponse
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        max_history_turns: int = _MAX_HISTORY_TURNS,
    ) -> None:
        self._client = ollama_client or OllamaClient()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._max_history_turns = max_history_turns
        self._sessions: Dict[str, List[ChatTurn]] = {}
        self._sessions_lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_session(self) -> str:
        """Create a new empty conversation session and return its id."""
        session_id = str(uuid.uuid4())
        with self._sessions_lock:
            self._sessions[session_id] = []
        return session_id

    def ask(
        self,
        question: str,
        product_report: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None,
        session_id: Optional[str] = None,
    ) -> NutritionChatResponse:
        """
        Ask a question about `product_report`, optionally continuing an
        existing `session_id`'s conversation history. If no session_id is
        given, a new one is created and returned in the response.
        """
        start = time.perf_counter()
        session_id = session_id or self.start_session()

        if not question or not question.strip():
            return NutritionChatResponse(
                session_id=session_id,
                answer="",
                success=False,
                error="Question cannot be empty.",
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

        history = self._get_history(session_id)
        prompt_bundle = self._prompt_builder.build_nutrition_chat_prompt(
            user_question=question,
            product_report=product_report,
            user_profile=user_profile,
            conversation_history=[turn.to_dict() for turn in history],
        )

        try:
            response = self._client.chat(messages=prompt_bundle["messages"])
        except OllamaClientError as exc:
            return NutritionChatResponse(
                session_id=session_id,
                answer="",
                retrieved_context=prompt_bundle["retrieved_context"],
                success=False,
                error=str(exc),
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

        self._append_turn(session_id, ChatTurn("user", question))
        self._append_turn(session_id, ChatTurn("assistant", response.text))

        return NutritionChatResponse(
            session_id=session_id,
            answer=response.text,
            retrieved_context=prompt_bundle["retrieved_context"],
            model=response.model,
            processing_time_ms=(time.perf_counter() - start) * 1000,
            success=True,
        )

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return the conversation history for a session as plain dicts."""
        return [turn.to_dict() for turn in self._get_history(session_id)]

    def clear_session(self, session_id: str) -> None:
        """Discard a conversation session's history."""
        with self._sessions_lock:
            self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_history(self, session_id: str) -> List[ChatTurn]:
        with self._sessions_lock:
            return list(self._sessions.get(session_id, []))

    def _append_turn(self, session_id: str, turn: ChatTurn) -> None:
        with self._sessions_lock:
            history = self._sessions.setdefault(session_id, [])
            history.append(turn)
            # Trim to the most recent N turns to bound prompt size/token cost.
            if len(history) > self._max_history_turns:
                del history[: len(history) - self._max_history_turns]
