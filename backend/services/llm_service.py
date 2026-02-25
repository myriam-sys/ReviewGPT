"""
LLM service — Groq API chat completion for the RAG pipeline.

Two public functions:

build_rag_prompt(question, retrieved_reviews, session_stats)
    Assembles the ``messages`` list expected by the Groq chat completion
    API.  The system message provides dataset context and strict grounding
    instructions; the user message contains the formatted reviews and the
    question.

generate_response(question, retrieved_reviews, session_stats)
    Async function that calls the Groq API (``AsyncGroq``) and returns a
    structured dict with the answer, model name, token usage, and source
    metadata.

The ``AsyncGroq`` client is a module-level lazy singleton so it is
created once and reused across all requests.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from groq import AsyncGroq

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ── Singleton ─────────────────────────────────────────────────────────────────

_groq_client: Optional[AsyncGroq] = None


def _get_groq_client() -> AsyncGroq:
    """Return (or lazily create) the shared AsyncGroq client."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    if not settings.groq_api_key:
        raise RuntimeError(
            "Groq API key is not configured.  "
            "Set GROQ_API_KEY in your .env file."
        )
    _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    logger.info("Groq async client initialised — model=%s", settings.groq_model)
    return _groq_client


# ── Public API ────────────────────────────────────────────────────────────────


def build_rag_prompt(
    question: str,
    retrieved_reviews: list[dict[str, Any]],
    session_stats: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Build the ``messages`` array for the Groq chat completion API.

    System message
    --------------
    Describes the assistant's role, embeds dataset statistics so the LLM
    can reference them ("based on your 142 reviews…"), and gives strict
    grounding instructions:

    - Answer ONLY from the provided reviews.
    - Cite reviewer names and ratings when relevant.
    - Respond in the same language as the user's question.
    - Admit when no retrieved review is relevant.

    User message
    ------------
    Lists the retrieved reviews in a consistent structured format, then
    appends the user's question.

    Parameters
    ----------
    question:
        The user's natural-language question.
    retrieved_reviews:
        Dicts returned by ``retrieve_similar_reviews``, each with
        ``author``, ``rating``, ``date``, ``text``, ``language``,
        ``similarity`` keys.
    session_stats:
        Dict from ``get_session_stats`` with ``total_reviews``,
        ``avg_rating``, ``language_distribution``, ``date_range``.

    Returns
    -------
    list[dict[str, str]]
        A two-element list: ``[system_message, user_message]``.
    """
    # ── System message ────────────────────────────────────────────────────────
    total = session_stats.get("total_reviews", 0)
    avg_rating = session_stats.get("avg_rating")
    lang_dist = session_stats.get("language_distribution", {})
    date_range = session_stats.get("date_range")

    avg_str = f"{avg_rating:.2f}/5" if avg_rating is not None else "N/A"
    lang_str = (
        ", ".join(f"{lang} ({cnt})" for lang, cnt in list(lang_dist.items())[:5])
        if lang_dist
        else "unknown"
    )
    date_str = (
        f"{date_range['earliest']} to {date_range['latest']}"
        if date_range
        else "unknown"
    )

    system_content = (
        "You are an expert analyst of Google Maps business reviews. "
        "You help business owners and their teams understand what customers think.\n\n"
        f"Dataset context:\n"
        f"- Total reviews in session: {total}\n"
        f"- Average rating: {avg_str}\n"
        f"- Languages: {lang_str}\n"
        f"- Date range: {date_str}\n\n"
        "Instructions:\n"
        "- Answer ONLY based on the reviews provided below. "
        "Do not invent, extrapolate, or recall information outside the provided text.\n"
        "- When relevant, cite the reviewer's name and rating "
        "(e.g. \"According to Marie (5★)…\").\n"
        "- If no provided review is relevant to the question, say so honestly "
        "and suggest the user rephrase.\n"
        "- Detect the language of the user's question and respond in that same language.\n"
        "- Be concise, structured, and highlight patterns across multiple reviews "
        "where applicable."
    )

    # ── User message — formatted reviews + question ───────────────────────────
    if retrieved_reviews:
        review_lines: list[str] = []
        for i, r in enumerate(retrieved_reviews, start=1):
            author = r.get("author") or "Anonymous"
            rating = r.get("rating")
            rating_str = f"{rating}/5" if rating is not None else "N/A"
            date = r.get("date") or "unknown date"
            language = r.get("language") or "unknown"
            text = r.get("text") or ""
            similarity = r.get("similarity", 0.0)

            review_lines.append(
                f"[Review {i} — relevance score: {similarity:.2f}]\n"
                f"Author: {author}\n"
                f"Rating: {rating_str}\n"
                f"Date: {date}\n"
                f"Language: {language}\n"
                f"Review: \"{text}\""
            )

        reviews_block = "\n\n".join(review_lines)
        user_content = (
            f"Here are the most relevant reviews retrieved for your question:\n\n"
            f"{reviews_block}\n\n"
            f"---\n"
            f"Question: {question}"
        )
    else:
        user_content = (
            "No reviews were found that are sufficiently similar to your question.\n\n"
            f"Question: {question}"
        )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


async def generate_response(
    question: str,
    retrieved_reviews: list[dict[str, Any]],
    session_stats: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate an LLM answer for *question* grounded in *retrieved_reviews*.

    Calls the Groq chat completion API with ``temperature=0.3`` for
    consistent, factual answers.

    Parameters
    ----------
    question:
        The user's question.
    retrieved_reviews:
        Retrieved review dicts (see ``retrieve_similar_reviews``).
    session_stats:
        Session aggregate stats (see ``get_session_stats``).

    Returns
    -------
    dict
        Keys: ``answer`` (str), ``model`` (str), ``tokens_used`` (int),
        ``sources_count`` (int), ``retrieved_reviews`` (list[dict]).

    Raises
    ------
    RuntimeError
        When the Groq API key is not configured.
    groq.APIError
        Propagated from the Groq SDK on API-level failures.
    """
    messages = build_rag_prompt(question, retrieved_reviews, session_stats)
    client = _get_groq_client()

    logger.debug(
        "Calling Groq — model=%s reviews=%d question=%.80r",
        settings.groq_model,
        len(retrieved_reviews),
        question,
    )

    response = await client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=1000,
        temperature=0.3,
    )

    answer = response.choices[0].message.content or ""
    tokens_used = response.usage.total_tokens if response.usage else 0

    logger.info(
        "Groq response — model=%s tokens=%d sources=%d",
        response.model,
        tokens_used,
        len(retrieved_reviews),
    )

    return {
        "answer": answer,
        "model": response.model,
        "tokens_used": tokens_used,
        "sources_count": len(retrieved_reviews),
        "retrieved_reviews": retrieved_reviews,
    }
