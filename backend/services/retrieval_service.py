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
import unicodedata
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
        (dict[str, int]), ``rating_distribution`` (dict[int, int] mapping
        star 1–5 to count), ``date_range`` (dict | None with ``earliest``
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

        rating_rows = await conn.fetch(
            """
            SELECT ROUND(rating)::int AS star, COUNT(*)::int AS count
            FROM reviews
            WHERE session_id = $1::uuid
            GROUP BY star
            ORDER BY star
            """,
            session_id,
        )

    language_distribution = {row["language"]: row["count"] for row in lang_rows}
    rating_distribution = {row["star"]: row["count"] for row in rating_rows}

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
        "rating_distribution": rating_distribution,
        "date_range": date_range,
    }


# ── Query classification ───────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """
    Lowercase and strip accent marks from *text* for accent-insensitive matching.

    Uses NFD decomposition to separate base characters from combining diacritical
    marks, then drops the diacritics via ASCII encoding.  This means "récemment"
    and "recemment" both normalize to "recemment", so a single keyword covers
    both spellings.
    """
    return (
        unicodedata.normalize("NFD", text.lower())
        .encode("ascii", "ignore")
        .decode("utf-8")
    )


# All keywords are stored pre-normalized (no accents, lowercase) so that
# _normalize(question) can be compared against them directly.
_ANALYTICAL_KEYWORDS: frozenset[str] = frozenset({
    "moyenne", "average", "combien", "how many", "pourcentage",
    "percent", "total", "distribution", "note", "rating", "score",
    "nombre", "count", "statistique", "stats",
})
_TEMPORAL_KEYWORDS: frozenset[str] = frozenset({
    "evolution", "evolue", "tendance", "trend", "recent", "dernier",
    "latest", "avant", "apres", "after", "annee", "year",
    "mois", "month", "amelior", "degrade", "over time", "changed", "progress",
})
_COMPARATIVE_KEYWORDS: frozenset[str] = frozenset({
    "difference", "compar", "positif", "negatif",
    "positive", "negative", "mieux", "better", "worse", "pire",
    "versus", "vs", "contrast",
})
_AUTHOR_KEYWORDS: frozenset[str] = frozenset({
    "qui", "who", "auteur", "author", "client", "customer",
    "personne", "person", "nom", "name", "mentionn",
})


def classify_question(question: str) -> dict[str, Any]:
    """
    Classify a question into one of five types using case-insensitive keyword
    matching (no LLM call — pure Python, zero latency).

    Types
    -----
    "analytical"  : numerical questions (averages, counts, distributions).
    "temporal"    : questions about trends or evolution over time.
    "comparative" : questions comparing positive vs negative feedback.
    "author"      : questions about specific reviewers or people.
    "search"      : default — qualitative questions about topics / themes.

    Returns
    -------
    dict
        ``type`` (str), ``suggested_top_k`` (int),
        ``fetch_positive_negative_split`` (bool),
        ``use_stats_prominently`` (bool).
    """
    q = _normalize(question)
    if any(kw in q for kw in _ANALYTICAL_KEYWORDS):
        return {
            "type": "analytical",
            "suggested_top_k": 5,
            "fetch_positive_negative_split": False,
            "use_stats_prominently": True,
        }
    if any(kw in q for kw in _TEMPORAL_KEYWORDS):
        return {
            "type": "temporal",
            "suggested_top_k": 10,
            "fetch_positive_negative_split": False,
            "use_stats_prominently": True,
        }
    if any(kw in q for kw in _COMPARATIVE_KEYWORDS):
        return {
            "type": "comparative",
            "suggested_top_k": 10,
            "fetch_positive_negative_split": True,
            "use_stats_prominently": False,
        }
    if any(kw in q for kw in _AUTHOR_KEYWORDS):
        return {
            "type": "author",
            "suggested_top_k": 5,
            "fetch_positive_negative_split": False,
            "use_stats_prominently": False,
        }
    return {
        "type": "search",
        "suggested_top_k": 10,
        "fetch_positive_negative_split": False,
        "use_stats_prominently": False,
    }


async def retrieve_context(
    question: str,
    session_id: str,
    top_k: int,
) -> dict[str, Any]:
    """
    Orchestrate context assembly for the RAG pipeline based on question type.

    1. Classifies the question with ``classify_question`` (keyword matching).
    2. Fetches session statistics (non-fatal — falls back to empty dict on error).
    3. Retrieves reviews in a type-appropriate way:

       analytical  — top-5 similarity search; stats are the primary answer source.
       temporal    — most recent ``top_k`` reviews ordered by date DESC.
       comparative — ``top_k // 2`` low-rated (≤ 2★) reviews +
                     ``top_k // 2`` high-rated (≥ 4★) reviews,
                     both ordered by similarity to the question.
       author / search — standard similarity search with ``top_k`` results.

    Parameters
    ----------
    question:
        The user's natural-language question.
    session_id:
        The upload session UUID.
    top_k:
        Caller-supplied retrieval budget.

    Returns
    -------
    dict
        Always contains ``stats`` (dict) and ``question_type`` (dict).
        Non-comparative types: also ``reviews`` (list[dict]).
        Comparative type: ``positive_reviews`` and ``negative_reviews`` instead.
        Analytical type: also ``note`` = ``"answer from stats"``.
    """
    question_type = classify_question(question)
    qtype = question_type["type"]
    logger.info(
        "Question classified — session=%s type=%s question=%.60r",
        session_id,
        qtype,
        question,
    )

    # Stats are non-fatal; all question types benefit from them in the prompt.
    try:
        stats = await get_session_stats(session_id)
    except Exception:
        logger.exception(
            "Failed to fetch session stats — session=%s — continuing without stats",
            session_id,
        )
        stats = {
            "total_reviews": 0,
            "reviews_with_text": 0,
            "avg_rating": None,
            "language_distribution": {},
            "rating_distribution": {},
            "date_range": None,
        }

    if qtype == "analytical":
        reviews = await retrieve_similar_reviews(question, session_id, top_k=5)
        return {
            "reviews": reviews,
            "stats": stats,
            "question_type": question_type,
            "note": "answer from stats",
        }

    if qtype == "temporal":
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
                    language
                FROM reviews
                WHERE session_id = $1::uuid
                  AND has_text   = TRUE
                ORDER BY date DESC NULLS LAST
                LIMIT $2
                """,
                session_id,
                top_k,
            )
        reviews = [
            {
                "review_id": row["review_id"],
                "author": row["author"],
                "rating": row["rating"],
                "date": row["date"],
                "text": row["text"],
                "language": row["language"],
                "similarity": None,
            }
            for row in rows
        ]
        return {"reviews": reviews, "stats": stats, "question_type": question_type}

    if qtype == "comparative":
        from backend.services.embedding_service import embed_query  # noqa: PLC0415

        query_vector = embed_query(question)
        vector_str = "[" + ",".join(map(str, query_vector)) + "]"
        half_k = max(1, top_k // 2)

        pool = await get_asyncpg_pool()
        async with pool.acquire() as conn:
            neg_rows = await conn.fetch(
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
                  AND rating     <= 2
                  AND has_text   = TRUE
                  AND embedding  IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                vector_str,
                session_id,
                half_k,
            )
            pos_rows = await conn.fetch(
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
                  AND rating     >= 4
                  AND has_text   = TRUE
                  AND embedding  IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                vector_str,
                session_id,
                half_k,
            )

        def _row_to_dict(row: Any) -> dict[str, Any]:
            return {
                "review_id": row["review_id"],
                "author": row["author"],
                "rating": row["rating"],
                "date": row["date"],
                "text": row["text"],
                "language": row["language"],
                "similarity": round(float(row["similarity"]), 4),
            }

        return {
            "positive_reviews": [_row_to_dict(r) for r in pos_rows],
            "negative_reviews": [_row_to_dict(r) for r in neg_rows],
            "stats": stats,
            "question_type": question_type,
        }

    # author or search — standard similarity search
    reviews = await retrieve_similar_reviews(question, session_id, top_k=top_k)
    return {"reviews": reviews, "stats": stats, "question_type": question_type}
