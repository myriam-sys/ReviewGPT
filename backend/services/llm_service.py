"""
LLM service — Groq API chat completion for the RAG pipeline.

Two public functions:

build_rag_prompt(question, context)
    Assembles the ``messages`` list expected by the Groq chat completion
    API.  The system message provides the full dataset statistics block
    (exact SQL aggregates) and strict grounding instructions.  The user
    message is formatted differently depending on the question type
    detected by ``retrieve_context``.

generate_response(question, context)
    Async function that calls the Groq API (``AsyncGroq``) and returns a
    structured dict with the answer, model name, token usage, source
    metadata, and the detected question type string.

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


# ── Private helpers ────────────────────────────────────────────────────────────


def _format_review(i: int, r: dict[str, Any], show_similarity: bool = True) -> str:
    """Format a single review dict into a structured text block."""
    author = r.get("author") or "Anonymous"
    rating = r.get("rating")
    rating_str = f"{rating}/5" if rating is not None else "N/A"
    date = r.get("date") or "unknown date"
    language = r.get("language") or "unknown"
    text = r.get("text") or ""
    similarity = r.get("similarity")

    header = f"[Review {i}"
    if show_similarity and similarity is not None:
        header += f" — relevance score: {similarity:.2f}"
    header += "]"

    return (
        f"{header}\n"
        f"Author: {author}\n"
        f"Rating: {rating_str}\n"
        f"Date: {date}\n"
        f"Language: {language}\n"
        f'Review: "{text}"'
    )


# ── Public API ────────────────────────────────────────────────────────────────


def build_rag_prompt(
    question: str,
    context: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Build the ``messages`` array for the Groq chat completion API.

    System message
    --------------
    Contains a ``## Dataset Statistics`` block (exact SQL aggregates) so
    the LLM can answer numerical questions ("what is the average rating?",
    "how many 5-star reviews?") directly from pre-computed counts rather
    than guessing from the sampled reviews.  The ``## Instructions``
    section tells the LLM to use stats for numbers and retrieved reviews
    for qualitative evidence.

    User message
    ------------
    Formats retrieved reviews according to the question type from
    ``context["question_type"]``:

    - ``comparative``: two labelled sections — positive (4-5★) and
      negative (1-2★) reviews.
    - ``temporal``   : reviews ordered by date with a date-ordering note.
    - ``analytical`` : stats-use reminder prepended to the review block.
    - others         : standard numbered list with relevance scores.

    Parameters
    ----------
    question:
        The user's natural-language question.
    context:
        Dict returned by ``retrieve_context``.  Always contains ``stats``
        (dict) and ``question_type`` (dict).  Non-comparative types have
        ``reviews`` (list[dict]); comparative has ``positive_reviews`` /
        ``negative_reviews``.

    Returns
    -------
    list[dict[str, str]]
        A two-element list: ``[system_message, user_message]``.
    """
    stats = context.get("stats", {})
    question_type = context.get("question_type", {})
    qtype = question_type.get("type", "search")

    # ── System message ────────────────────────────────────────────────────────
    total = stats.get("total_reviews", 0)
    reviews_with_text = stats.get("reviews_with_text", 0)
    avg_rating = stats.get("avg_rating")
    lang_dist = stats.get("language_distribution", {})
    rating_dist = stats.get("rating_distribution", {})
    date_range = stats.get("date_range")

    avg_str = f"{avg_rating:.2f}/5" if avg_rating is not None else "N/A"
    rating_dist_str = (
        ", ".join(f"{star}★: {rating_dist.get(star, 0)}" for star in range(1, 6))
        if rating_dist
        else "N/A"
    )
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
        "## Dataset Statistics (exact SQL aggregates — use these for any numerical questions)\n"
        f"- Total reviews: {total} ({reviews_with_text} with text)\n"
        f"- Average rating: {avg_str}\n"
        f"- Rating distribution: {rating_dist_str}\n"
        f"- Languages: {lang_str}\n"
        f"- Date range: {date_str}\n\n"
        "## Instructions\n"
        "- For analytical questions (e.g. 'what is the average rating?', "
        "'how many 5-star reviews?'), answer directly from the Dataset Statistics "
        "above — these are exact counts from SQL.\n"
        "- Use the retrieved reviews below for qualitative context: themes, "
        "specific examples, and representative quotes.\n"
        "- Answer ONLY based on the provided data. Do not invent or extrapolate.\n"
        "- When relevant, cite the reviewer's name and rating "
        "(e.g. \"According to Marie (5★)…\").\n"
        "- If no provided review is relevant to the question, say so honestly "
        "and suggest the user rephrase.\n"
        "- Detect the language of the user's question and respond in that same language.\n"
        "- Be concise, structured, and highlight patterns across multiple reviews "
        "where applicable."
    )

    # ── User message — formatted reviews + question ───────────────────────────
    if qtype == "comparative":
        pos = context.get("positive_reviews", [])
        neg = context.get("negative_reviews", [])
        pos_block = (
            "\n\n".join(_format_review(i, r) for i, r in enumerate(pos, start=1))
            if pos
            else "No positive reviews retrieved."
        )
        neg_block = (
            "\n\n".join(_format_review(i, r) for i, r in enumerate(neg, start=1))
            if neg
            else "No negative reviews retrieved."
        )
        user_content = (
            "## Positive reviews (4-5★)\n\n"
            f"{pos_block}\n\n"
            "## Negative reviews (1-2★)\n\n"
            f"{neg_block}\n\n"
            "---\n"
            f"Question: {question}"
        )

    elif qtype == "temporal":
        reviews = context.get("reviews", [])
        if reviews:
            reviews_block = "\n\n".join(
                _format_review(i, r, show_similarity=False)
                for i, r in enumerate(reviews, start=1)
            )
            user_content = (
                "The following reviews are ordered by date (most recent first):\n\n"
                f"{reviews_block}\n\n"
                "---\n"
                f"Question: {question}"
            )
        else:
            user_content = (
                "No reviews were found for this session.\n\n"
                f"Question: {question}"
            )

    elif qtype == "analytical":
        reviews = context.get("reviews", [])
        if reviews:
            reviews_block = "\n\n".join(
                _format_review(i, r) for i, r in enumerate(reviews, start=1)
            )
            user_content = (
                "Note: For numerical answers, use the Dataset Statistics in the "
                "system prompt (exact SQL counts). The reviews below provide "
                "qualitative context only.\n\n"
                f"{reviews_block}\n\n"
                "---\n"
                f"Question: {question}"
            )
        else:
            user_content = (
                "Note: Use the Dataset Statistics in the system prompt for "
                "numerical answers.\n\n"
                f"Question: {question}"
            )

    else:  # author, search
        reviews = context.get("reviews", [])
        if reviews:
            reviews_block = "\n\n".join(
                _format_review(i, r) for i, r in enumerate(reviews, start=1)
            )
            user_content = (
                "Here are the most relevant reviews retrieved for your question:\n\n"
                f"{reviews_block}\n\n"
                "---\n"
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
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate an LLM answer for *question* grounded in *context*.

    Calls the Groq chat completion API with ``temperature=0.3`` for
    consistent, factual answers.

    Parameters
    ----------
    question:
        The user's question.
    context:
        Full context dict from ``retrieve_context`` (reviews + stats +
        question_type).

    Returns
    -------
    dict
        Keys: ``answer`` (str), ``model`` (str), ``tokens_used`` (int),
        ``sources_count`` (int), ``retrieved_reviews`` (list[dict]),
        ``question_type`` (str).

    Raises
    ------
    RuntimeError
        When the Groq API key is not configured.
    groq.APIError
        Propagated from the Groq SDK on API-level failures.
    """
    # Flatten all retrieved reviews for the response payload.
    if "positive_reviews" in context or "negative_reviews" in context:
        all_reviews: list[dict[str, Any]] = (
            context.get("positive_reviews", []) + context.get("negative_reviews", [])
        )
    else:
        all_reviews = context.get("reviews", [])

    question_type_str = context.get("question_type", {}).get("type", "search")

    messages = build_rag_prompt(question, context)
    client = _get_groq_client()

    logger.debug(
        "Calling Groq — model=%s reviews=%d type=%s question=%.80r",
        settings.groq_model,
        len(all_reviews),
        question_type_str,
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
        "Groq response — model=%s tokens=%d sources=%d type=%s",
        response.model,
        tokens_used,
        len(all_reviews),
        question_type_str,
    )

    return {
        "answer": answer,
        "model": response.model,
        "tokens_used": tokens_used,
        "sources_count": len(all_reviews),
        "retrieved_reviews": all_reviews,
        "question_type": question_type_str,
    }
