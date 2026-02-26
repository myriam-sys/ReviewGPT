"""
Embedding service — dense vector encoding via the Mistral Embed API.

Design notes
------------
* Uses ``mistral-embed`` (1024-dimensional, multilingual) via Mistral's
  REST API instead of running a local model.  This eliminates the ~420 MB
  model download and the 512 MB RAM ceiling that blocked Render free-tier
  deployment.

* The Mistral client is held in a module-level singleton (``_client``) and
  initialised lazily on first use.  ``load_model()`` is kept as a no-op so
  that the FastAPI startup hook requires no changes.

* ``mistral-embed`` outputs L2-normalised 1024-dimensional vectors, which
  are directly compatible with pgvector's ``<=>`` (cosine) operator.

* Batch size is determined by the API (up to 512 texts per request).
  No local batch size tuning is needed.
"""

from __future__ import annotations

import logging

from mistralai import Mistral

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Output dimension of mistral-embed.
# Must match the vector(N) column created by migration 004.
_EMBEDDING_DIMENSION: int = 1024

# ── Singleton ─────────────────────────────────────────────────────────────────

_client: Mistral | None = None


def _get_client() -> Mistral:
    global _client
    if _client is None:
        if not settings.mistral_api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY is not configured. "
                "Set it in your .env file or Render environment variables."
            )
        _client = Mistral(api_key=settings.mistral_api_key)
        logger.info("Mistral client initialised")
    return _client


# ── Public API ────────────────────────────────────────────────────────────────


def load_model() -> None:
    """
    No-op for API-based embeddings — kept for startup hook compatibility.

    The FastAPI ``startup`` event calls this function; with a local model it
    triggered a download.  Here it just logs a confirmation so the startup
    logs remain informative.
    """
    logger.info(
        "Embedding backend: Mistral API (mistral-embed, %d-dim). "
        "No local model to load.",
        _EMBEDDING_DIMENSION,
    )


def embed_passages(texts: list[str]) -> list[list[float]]:
    """
    Encode a list of review texts via the Mistral Embed API.

    Parameters
    ----------
    texts:
        Review bodies to embed.  Empty list returns immediately without an
        API call.

    Returns
    -------
    list[list[float]]
        One 1024-dimensional float vector per input text.
    """
    if not texts:
        return []

    client = _get_client()
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=texts,
    )
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    """
    Encode a single user query string via the Mistral Embed API.

    Parameters
    ----------
    text:
        The user's question or search phrase.

    Returns
    -------
    list[float]
        A 1024-dimensional float vector.
    """
    return embed_passages([text])[0]


def get_embedding_dimension() -> int:
    """
    Return the fixed output dimension of the embedding model.

    Returns
    -------
    int
        ``1024`` for ``mistral-embed``.
    """
    return _EMBEDDING_DIMENSION
