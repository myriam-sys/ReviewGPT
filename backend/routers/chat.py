"""
Chat router — RAG-based conversational interface for Google Maps reviews.

Endpoint
--------
POST /chat
    1. Validates the session exists in Supabase and that embeddings are ready.
    2. Runs pgvector cosine similarity search to retrieve the top-k most
       relevant reviews for the user's question.
    3. Fetches session statistics for LLM context.
    4. Calls the Groq LLM to generate a grounded, multilingual answer.
    5. Returns the answer, model metadata, and retrieved sources.

Error responses
---------------
400 Bad Question     — empty question string (enforced by Pydantic schema)
404 Session Not Found — no reviews stored under the given session_id
409 Embeddings Not Ready — reviews exist but none have been embedded yet
    (poll GET /upload/{session_id}/embedding-status)
502 LLM Error       — Groq API call failed
503 DB Unavailable  — asyncpg pool or Supabase credentials missing
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from backend.db.supabase_client import get_asyncpg_pool
from backend.models.schemas import ChatRequest, ChatResponse
from backend.services.llm_service import generate_response
from backend.services.retrieval_service import get_session_stats, retrieve_similar_reviews

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about your Google Maps reviews",
    description=(
        "Runs the full RAG pipeline: embeds the question, retrieves similar "
        "reviews from Supabase pgvector, and generates a grounded answer via "
        "the Groq LLM.  The session must have completed embedding before chat "
        "is available — poll ``GET /upload/{session_id}/embedding-status`` first."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    RAG pipeline for a single conversational turn.

    Parameters
    ----------
    request:
        ``{ session_id, question, top_k }``

    Returns
    -------
    ChatResponse
        LLM answer plus retrieved sources and token usage metadata.

    Raises
    ------
    404 Not Found
        When no reviews are stored for ``session_id``.
    409 Conflict
        When reviews exist but no embeddings are ready yet.
    502 Bad Gateway
        When the Groq API call fails.
    503 Service Unavailable
        When the database connection pool cannot be created.
    """
    session_id = request.session_id
    question = request.question
    top_k = request.top_k

    logger.info(
        "Chat request — session=%s top_k=%d question=%.80r",
        session_id,
        top_k,
        question,
    )

    # ── 1. Acquire pool (503 if credentials missing) ──────────────────────────
    try:
        pool = await get_asyncpg_pool()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is not configured.",
        ) from exc

    # ── 2. Validate session exists ────────────────────────────────────────────
    async with pool.acquire() as conn:
        total_count: int = await conn.fetchval(
            "SELECT COUNT(*)::int FROM reviews WHERE session_id = $1::uuid",
            session_id,
        )
        if total_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No reviews found for session '{session_id}'.",
            )

        # ── 3. Check embeddings are ready ─────────────────────────────────────
        embedded_count: int = await conn.fetchval(
            """
            SELECT COUNT(*)::int FROM reviews
            WHERE session_id = $1::uuid AND embedding IS NOT NULL
            """,
            session_id,
        )
        if embedded_count == 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Embeddings are not ready yet for this session.  "
                    "Poll GET /upload/{session_id}/embedding-status and retry "
                    "once status is 'complete'."
                ),
            )

    # ── 4. Retrieve similar reviews ───────────────────────────────────────────
    try:
        retrieved = await retrieve_similar_reviews(
            query=question,
            session_id=session_id,
            top_k=top_k,
        )
    except Exception as exc:
        logger.exception("Similarity search failed — session=%s", session_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve similar reviews from the database.",
        ) from exc

    logger.info(
        "Retrieved %d relevant reviews — session=%s", len(retrieved), session_id
    )

    # ── 5. Fetch session statistics for LLM context ───────────────────────────
    try:
        stats = await get_session_stats(session_id)
    except Exception:
        # Non-fatal — the LLM can still answer without stats context.
        logger.exception(
            "Failed to fetch session stats — session=%s — continuing without stats",
            session_id,
        )
        stats = {
            "total_reviews": 0,
            "reviews_with_text": 0,
            "avg_rating": None,
            "language_distribution": {},
            "date_range": None,
        }

    # ── 6. Generate LLM response ──────────────────────────────────────────────
    try:
        llm_result = await generate_response(
            question=question,
            retrieved_reviews=retrieved,
            session_stats=stats,
        )
    except RuntimeError as exc:
        # Groq API key not configured
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Groq API call failed — session=%s", session_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The language model call failed.  Please try again.",
        ) from exc

    logger.info(
        "Chat complete — session=%s model=%s tokens=%d sources=%d",
        session_id,
        llm_result["model"],
        llm_result["tokens_used"],
        llm_result["sources_count"],
    )

    return ChatResponse(
        answer=llm_result["answer"],
        model=llm_result["model"],
        tokens_used=llm_result["tokens_used"],
        sources_count=llm_result["sources_count"],
        retrieved_reviews=llm_result["retrieved_reviews"],
        session_id=session_id,
        question=question,
    )
