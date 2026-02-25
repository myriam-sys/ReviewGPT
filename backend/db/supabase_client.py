"""
Supabase client — lazy singletons for SDK and asyncpg pool.

Two clients are provided:

get_client()
    A synchronous Supabase SDK client (``supabase.Client``) initialised with
    the project URL and service_role key.  Used by the ingestion service to
    insert rows via the REST/PostgREST API.  The service_role key bypasses
    Row-Level Security so the backend can write without additional policies.

get_asyncpg_pool()
    An asyncpg connection pool pointed at the direct PostgreSQL URL.  Used by
    the embedding and retrieval services (Phase 3) for pgvector similarity
    searches which are not supported through the PostgREST layer.

Both are initialised lazily on first call and cached for the lifetime of the
process.  Neither function opens a connection at import time, so the module is
safe to import even when Supabase credentials are absent (e.g. during tests).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional
import asyncpg
from supabase import Client, create_client

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ── Lazy singletons ───────────────────────────────────────────────────────────

_supabase_client: Optional[Client] = None
_asyncpg_pool: Optional[asyncpg.Pool] = None
_pool_lock = asyncio.Lock()


# ── Public accessors ──────────────────────────────────────────────────────────


def get_client() -> Client:
    """
    Return the shared Supabase SDK client, creating it on first call.

    Uses ``settings.supabase_url`` and ``settings.supabase_key`` (the
    service_role key).  Raises ``RuntimeError`` if either value is missing so
    misconfiguration surfaces at call time with a helpful message rather than
    an obscure library error.

    Returns
    -------
    supabase.Client
        The initialised client.

    Raises
    ------
    RuntimeError
        When ``SUPABASE_URL`` or ``SUPABASE_KEY`` is not set.
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    url = settings.supabase_url
    key = settings.supabase_key

    if not url or not key:
        raise RuntimeError(
            "Supabase credentials are not configured.  "
            "Set SUPABASE_URL and SUPABASE_KEY in your .env file."
        )

    logger.info("Initialising Supabase SDK client (url=%s)", url)
    _supabase_client = create_client(url, key)
    return _supabase_client


async def get_asyncpg_pool() -> asyncpg.Pool:
    """
    Return the shared asyncpg connection pool, creating it on first call.

    Uses ``settings.supabase_db_url`` (the direct PostgreSQL connection
    string).  The pool is created with ``min_size=1, max_size=5`` which is
    appropriate for Phase 3 similarity-search workloads.

    Thread-safe: creation is guarded by an asyncio.Lock so concurrent
    coroutines don't race to create multiple pools.

    Returns
    -------
    asyncpg.Pool
        The initialised pool.

    Raises
    ------
    RuntimeError
        When ``SUPABASE_DB_URL`` is not set.
    """
    global _asyncpg_pool

    if _asyncpg_pool is not None:
        return _asyncpg_pool

    async with _pool_lock:
        # Re-check inside the lock in case another coroutine created it first.
        if _asyncpg_pool is not None:
            return _asyncpg_pool

        db_url = settings.supabase_db_url
        if not db_url:
            raise RuntimeError(
                "Database URL is not configured.  "
                "Set SUPABASE_DB_URL in your .env file."
            )

        logger.info("Creating asyncpg connection pool")
        _asyncpg_pool = await asyncpg.create_pool(
            db_url,
            ssl="require",
            min_size=1,
            max_size=5,
        )

    return _asyncpg_pool
