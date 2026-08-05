
NO_CONTEXT_ANSWER = (
    "I do not have enough information in the synced documents to answer that. "
    "Try rephrasing your question, or sync additional fitness/nutrition documents "
    "via Google Drive first."
)

SYSTEM_PROMPT = """You are the PACKS Fitness Assistant, a factual question-answering system for a \
fitness and macro tracking app. You answer user questions ONLY using the numbered CONTEXT \
passages provided in the user message. You have no other source of truth.

Rules you MUST follow:
1. Use ONLY the information in the CONTEXT passages to answer. Do not use outside knowledge, \
even if you believe it to be true, and do not make assumptions beyond what the context states.
2. Every factual claim in your answer must be followed by a citation marker like [1] or [2] \
referencing the CONTEXT passage(s) it came from. If a sentence draws on multiple passages, cite \
all of them, e.g. [1][3].
3. If the CONTEXT passages do not contain enough information to answer the question -- even \
partially -- respond with EXACTLY this sentence and nothing else: \
"I do not have enough information in the synced documents to answer that."
4. Never fabricate a citation number that wasn't provided in the CONTEXT.
5. Do not give medical advice, diagnoses, or safety-critical dosing/medication guidance even if \
present in the context; if the question calls for that, recommend the user consult a qualified \
professional instead.
6. Keep answers concise and directly responsive to the question. Do not pad with disclaimers \
beyond what these rules already require.

Respond in plain text (not JSON, not markdown headers) with inline [n] citation markers."""


def format_context_block(retrieved_chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Formats retrieved FAISS results into a numbered CONTEXT block for the
    prompt, and builds the parallel citations list to return to the client.

    Args:
        retrieved_chunks: output of indexer.search(), i.e. a list of
            {"id": int, "score": float, "metadata": {"content", "filename",
             "source_id", "location", "chunk_index", ...}}

    Returns:
        (context_block_text, citations)
        citations: [{"marker": 1, "filename": ..., "location": ...,
                     "source_id": ..., "score": ...}, ...]
    """
    context_lines = []
    citations = []

    for i, result in enumerate(retrieved_chunks, start=1):
        meta = result.get("metadata", {})
        content = meta.get("content", "")
        filename = meta.get("filename", "unknown source")
        location = meta.get("location")
        source_id = meta.get("source_id")

        location_str = f", {location}" if location else ""
        context_lines.append(f"[{i}] (Source: {filename}{location_str})\n{content}")

        citations.append(
            {
                "marker": i,
                "filename": filename,
                "location": location,
                "source_id": source_id,
                "score": round(result.get("score", 0.0), 4),
            }
        )

    context_block = "\n\n".join(context_lines)
    return context_block, citations


def build_user_message(question: str, context_block: str) -> str:
    """Builds the user-turn content combining the numbered context and the question."""
    return (
        f"CONTEXT:\n\n{context_block}\n\n"
        f"---\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the CONTEXT above, with [n] citation markers."
    )


def build_messages(question: str, retrieved_chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Builds the full Groq chat `messages` list for a grounded RAG answer,
    plus the citations list to return alongside the model's answer.

    Args:
        question: the user's question.
        retrieved_chunks: output of indexer.search().

    Returns:
        (messages, citations)
        messages: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        citations: see format_context_block()
    """
    context_block, citations = format_context_block(retrieved_chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, context_block)},
    ]

    return messages, citations
