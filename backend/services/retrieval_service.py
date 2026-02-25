"""
Retrieval service — pgvector cosine similarity search via asyncpg.

Two public functions are exposed:

retrieve_similar_reviews(query, session_id, top_k)
    Embeds the user's question and runs a nearest-neighbour search against
    the ``reviews`` table, returning the top-k results for the session.
    Results with cosine similarity < 0.3 are filtered out as irrelevant.

get_session_stats(session_id)
    Returns aggregate metadata about a session (review count, average rating,
    language distribution, date range) so the LLM has dataset context.

Both functions are async and acquire connections from the shared asyncpg pool
maintained by ``supabase_client.get_asyncpg_pool()``.  The pool is
initialised lazily on first use.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.db.supabase_client import get_asyncpg_pool

logger = logging.getLogger(__name__)

# Minimum cosine similarity to consider a review relevant.
# Cosine similarity ranges from 0 (orthogonal) to 1 (identical direction).
# Reviews below this threshold are excluded from LLM context.
_SIMILARITY_THRESHOLD: float = 0.3


# ── Public API ────────────────────────────────────────────────────────────────


async def retrieve_similar_reviews(
    query: str,
    session_id: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Find the *top_k* reviews most semantically similar to *query*.

    Pipeline
    --------
    1. Embed *query* with ``embed_query()`` (prefixes ``"query: "`` per E5
       spec).
    2. Serialise the vector to pgvector's ``[f1,f2,...]`` text format.
    3. Run a cosine-distance nearest-neighbour search via raw asyncpg SQL
       so that the ``<=>`` pgvector operator is available.
    4. Filter results below ``_SIMILARITY_THRESHOLD`` (irrelevant hits).

    Parameters
    ----------
    query:
        The user's natural-language question.
    session_id:
        The upload session UUID — scopes the search to one tenant's data.
    top_k:
        Maximum number of reviews to return (before threshold filtering).

    Returns
    -------
    list[dict]
        Each dict has keys: ``review_id``, ``author``, ``rating``, ``date``,
        ``text``, ``language``, ``similarity``.  May be empty if no reviews
        pass the threshold.
    """
    # Inline import avoids a module-level circular dependency:
    # retrieval_service → embedding_service → (no further deps in this direction)
    from backend.services.embedding_service import embed_query  # noqa: PLC0415

    query_vector = embed_query(query)

    # asyncpg passes Python lists as PostgreSQL arrays, not as pgvector text
    # literals.  We must serialise manually and cast with ::vector in SQL.
    vector_str = "[" + ",".join(map(str, query_vector)) + "]"

    pool = await get_asyncpg_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                review_id::text,
                author,
                rating,
                date::text,
                text,
                language,
                1 - (embedding <=> $1::vector) AS similarity
            FROM reviews
            WHERE session_id = $2::uuid
              AND has_text    = TRUE
              AND embedding   IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            vector_str,
            session_id,
            top_k,
        )

    results: list[dict[str, Any]] = []
    for row in rows:
        similarity = float(row["similarity"])
        if similarity < _SIMILARITY_THRESHOLD:
            continue
        results.append(
            {
                "review_id": row["review_id"],
                "author": row["author"],
                "rating": row["rating"],
                "date": row["date"],
                "text": row["text"],
                "language": row["language"],
                "similarity": round(similarity, 4),
            }
        )

    logger.debug(
        "Similarity search — session=%s query=%.50r top_k=%d returned=%d",
        session_id,
        query,
        top_k,
        len(results),
    )
    return results


async def get_session_stats(session_id: str) -> dict[str, Any]:
    """
    Return aggregate metadata for a session, used to give the LLM context.

    Runs two queries in the same connection:
    1. Aggregate counts / averages / date range.
    2. Language distribution (GROUP BY language).

    Parameters
    ----------
    session_id:
        The upload session UUID to aggregate.

    Returns
    -------
    dict
        Keys: ``total_reviews`` (int), ``reviews_with_text`` (int),
        ``avg_rating`` (float | None), ``language_distribution``
        (dict[str, int]), ``date_range`` (dict | None with ``earliest``
        and ``latest`` ISO strings).
    """
    pool = await get_asyncpg_pool()
    async with pool.acquire() as conn:
        stats_row = await conn.fetchrow(
            """
            SELECT
                COUNT(*)::int                                   AS total_reviews,
                COUNT(*) FILTER (WHERE has_text)::int           AS reviews_with_text,
                ROUND(AVG(rating)::numeric, 2)::float           AS avg_rating,
                MIN(date)                                        AS earliest_date,
                MAX(date)                                        AS latest_date
            FROM reviews
            WHERE session_id = $1::uuid
            """,
            session_id,
        )

        lang_rows = await conn.fetch(
            """
            SELECT language, COUNT(*)::int AS count
            FROM reviews
            WHERE session_id = $1::uuid AND language IS NOT NULL
            GROUP BY language
            ORDER BY count DESC
            """,
            session_id,
        )

    language_distribution = {row["language"]: row["count"] for row in lang_rows}

    def _iso(val: Any) -> str | None:
        if val is None:
            return None
        return val.isoformat() if hasattr(val, "isoformat") else str(val)

    earliest = _iso(stats_row["earliest_date"])
    latest = _iso(stats_row["latest_date"])
    date_range = {"earliest": earliest, "latest": latest} if earliest and latest else None

    return {
        "total_reviews": stats_row["total_reviews"] or 0,
        "reviews_with_text": stats_row["reviews_with_text"] or 0,
        "avg_rating": float(stats_row["avg_rating"]) if stats_row["avg_rating"] is not None else None,
        "language_distribution": language_distribution,
        "date_range": date_range,
    }
