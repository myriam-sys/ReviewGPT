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
from urllib.parse import quote_plus, urlparse, urlunparse

import asyncpg
from supabase import Client, create_client

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ── Lazy singletons ───────────────────────────────────────────────────────────

_supabase_client: Optional[Client] = None
_asyncpg_pool: Optional[asyncpg.Pool] = None
_pool_lock = asyncio.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_safe_db_url(raw_url: str) -> str:
    """
    URL-encode the password component of a PostgreSQL connection string.

    Supabase auto-generates passwords that may contain special characters
    (``@``, ``#``, ``%``, ``+``, etc.).  asyncpg parses the URL naively, so
    an unencoded special character in the password breaks the netloc split and
    causes an authentication error.

    ``urllib.parse.quote_plus`` percent-encodes every character that is not
    safe in a URL password field, which asyncpg then decodes correctly before
    sending it to PostgreSQL.

    Parameters
    ----------
    raw_url:
        The raw ``SUPABASE_DB_URL`` value from the environment, e.g.
        ``postgresql://postgres:p@$$w0rd!@db.xyz.supabase.co:5432/postgres``.

    Returns
    -------
    str
        A fully URL-safe connection string with the password percent-encoded.
    """
    parsed = urlparse(raw_url)
    safe_password = quote_plus(parsed.password or "")
    netloc = f"{parsed.username}:{safe_password}@{parsed.hostname}:{parsed.port}"
    clean_url = urlunparse(parsed._replace(netloc=netloc))
    # Supabase Session Pooler requires SSL; direct connections do not.
    if "pooler.supabase.com" in raw_url:
        clean_url += "?sslmode=require"
    return clean_url


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
    string).  The pool is created with ``min_size=1, max_size=10`` which is
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

        safe_url = _get_safe_db_url(db_url)
        logger.info("Creating asyncpg connection pool")
        _asyncpg_pool = await asyncpg.create_pool(
            safe_url,
            min_size=1,
            max_size=10,
        )

    return _asyncpg_pool
