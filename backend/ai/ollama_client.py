"""
backend/ai/ollama_client.py

Ollama Client — production-hardened HTTP wrapper around a local/remote
Ollama server, defaulting to the `llama3.2` model.

Responsibilities:
    - Talk to Ollama's `/api/generate` and `/api/chat` endpoints.
    - Handle timeouts, connection errors and retries with backoff.
    - Support both streaming and non-streaming responses.
    - Isolate all Ollama-specific HTTP/JSON details behind typed
      dataclasses so the rest of PACKS never touches raw HTTP.

This is the only file in PACKS that imports `requests` for LLM calls.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

import requests


class OllamaClientError(RuntimeError):
    """Raised for unrecoverable Ollama communication failures."""


class OllamaTimeoutError(OllamaClientError):
    """Raised when a request to Ollama exceeds the configured timeout."""


class OllamaConnectionError(OllamaClientError):
    """Raised when the Ollama server cannot be reached (not running, wrong host, etc.)."""


@dataclass
class OllamaResponse:
    """Normalized result of a single (non-streaming) Ollama call."""

    text: str
    model: str
    done: bool
    total_duration_ms: Optional[float] = None
    eval_count: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "done": self.done,
            "total_duration_ms": self.total_duration_ms,
            "eval_count": self.eval_count,
            "prompt_eval_count": self.prompt_eval_count,
        }


class OllamaClient:
    """
    Thin, dependency-injectable client for a running Ollama instance.

    Usage:
        client = OllamaClient(model="llama3.2")
        response = client.generate(prompt="Explain sodium benzoate simply.")
        print(response.text)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3:8b",
        request_timeout_seconds: float = 300.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
        default_temperature: float = 0.3,
        default_num_predict: int = 512,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = request_timeout_seconds
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff_seconds
        self._default_temperature = default_temperature
        self._default_num_predict = default_num_predict
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        model: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> OllamaResponse:
        """Single-turn generation via `/api/generate` (non-streaming)."""
        payload: Dict[str, Any] = {
            "model": model or self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self._default_temperature,
                "num_predict": num_predict or self._default_num_predict,
            },
        }
        if system:
            payload["system"] = system
        if stop:
            payload["options"]["stop"] = stop

        raw = self._post_with_retries("/api/generate", payload)
        return OllamaResponse(
            text=raw.get("response", "").strip(),
            model=raw.get("model", payload["model"]),
            done=raw.get("done", True),
            total_duration_ms=self._ns_to_ms(raw.get("total_duration")),
            eval_count=raw.get("eval_count"),
            prompt_eval_count=raw.get("prompt_eval_count"),
            raw=raw,
        )

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Streaming generation via `/api/generate`; yields text chunks as they arrive."""
        payload: Dict[str, Any] = {
            "model": model or self._model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else self._default_temperature,
                "num_predict": num_predict or self._default_num_predict,
            },
        }
        if system:
            payload["system"] = system

        url = f"{self._base_url}/api/generate"
        try:
            with self._session.post(url, json=payload, timeout=self._timeout, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    if chunk.get("response"):
                        yield chunk["response"]
                    if chunk.get("done"):
                        break
        except requests.exceptions.Timeout as exc:
            raise OllamaTimeoutError(f"Ollama streaming request timed out after {self._timeout}s") from exc
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self._base_url}. Is `ollama serve` running?"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise OllamaClientError(f"Ollama returned an HTTP error: {exc}") from exc

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        model: Optional[str] = None,
    ) -> OllamaResponse:
        """
        Multi-turn conversation via `/api/generate`.
        Converts message list into a formatted prompt string,
        extracting the system message as a separate field.
        """
        system = None
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt_parts.append("Assistant: ")
        prompt = "\n".join(prompt_parts)

        payload: Dict[str, Any] = {
            "model": model or self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self._default_temperature,
                "num_predict": num_predict or self._default_num_predict,
            },
        }
        if system:
            payload["system"] = system

        raw = self._post_with_retries("/api/generate", payload)
        return OllamaResponse(
            text=raw.get("response", "").strip(),
            model=raw.get("model", payload["model"]),
            done=raw.get("done", True),
            total_duration_ms=self._ns_to_ms(raw.get("total_duration")),
            eval_count=raw.get("eval_count"),
            prompt_eval_count=raw.get("prompt_eval_count"),
            raw=raw,
        )

    def health_check(self) -> bool:
        """Return True if the Ollama server is reachable and responsive."""
        try:
            response = self._session.get(f"{self._base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def is_model_available(self, model: Optional[str] = None) -> bool:
        """Check whether the target model has been pulled locally."""
        target_model = model or self._model
        try:
            response = self._session.get(f"{self._base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
            available = [m.get("name", "") for m in response.json().get("models", [])]
            return any(target_model in name for name in available)
        except requests.exceptions.RequestException:
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _post_with_retries(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._session.post(url, json=payload, timeout=self._timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout as exc:
                last_error = OllamaTimeoutError(
                    f"Ollama request to {path} timed out after {self._timeout}s"
                )
            except requests.exceptions.ConnectionError as exc:
                last_error = OllamaConnectionError(
                    f"Could not connect to Ollama at {self._base_url}. Is `ollama serve` running?"
                )
            except requests.exceptions.HTTPError as exc:
                last_error = OllamaClientError(f"Ollama returned an HTTP error: {exc}")
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = OllamaClientError(f"Ollama returned an unparseable response: {exc}")

            if attempt < self._max_retries:
                time.sleep(self._retry_backoff * (attempt + 1))

        raise last_error or OllamaClientError("Unknown error communicating with Ollama.")

    @staticmethod
    def _ns_to_ms(nanoseconds: Optional[int]) -> Optional[float]:
        if nanoseconds is None:
            return None
        return round(nanoseconds / 1_000_000, 2)
