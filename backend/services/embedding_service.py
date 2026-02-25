"""
Embedding service — dense vector encoding via multilingual-e5-large.

Design notes
------------
* The ``SentenceTransformer`` model is held in a module-level singleton
  (``_model``) so it is loaded exactly once per process.  Subsequent calls
  to ``load_model()`` are near-instant cache hits.

* ``intfloat/multilingual-e5-large`` requires instruction prefixes:
    - Passages (documents to store): ``"passage: <text>"``
    - Queries (user questions):      ``"query: <text>"``
  Forgetting these prefixes degrades retrieval quality significantly.

* ``normalize_embeddings=True`` L2-normalises each vector so that
  dot-product similarity equals cosine similarity.  This matches
  the recommendation in the E5 paper and lets pgvector use the
  faster ``<=>`` (cosine) or ``<#>`` (negative inner product) operators.

* ``batch_size=32`` keeps peak RAM under ~2 GB on CPU, which is safe
  for typical Render.com or Railway instances.  Increase on GPU machines.

* Model weights (~1.1 GB) are downloaded from HuggingFace on the first
  run and cached in ``~/.cache/huggingface/``.  Subsequent starts reuse
  the cache and take ~10 seconds to load into memory.
"""

from __future__ import annotations

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Fixed output dimension for intfloat/multilingual-e5-large.
# Used as a sanity check at startup and referenced by the migration SQL.
_EMBEDDING_DIMENSION: int = 1024

# Safe batch size for CPU inference.  Raise to 64-128 on a GPU instance.
_BATCH_SIZE: int = 32

# ── Singleton ─────────────────────────────────────────────────────────────────

_model: Optional[SentenceTransformer] = None


# ── Public API ────────────────────────────────────────────────────────────────


def load_model() -> SentenceTransformer:
    """
    Load ``settings.embedding_model`` and cache it for the process lifetime.

    Intended to be called once at application startup (via the FastAPI
    ``startup`` event) so the first upload request is not penalised by the
    ~10-second load time.  Subsequent calls return the cached instance
    immediately.

    Returns
    -------
    SentenceTransformer
        The loaded model, ready for encoding.
    """
    global _model
    if _model is not None:
        return _model

    logger.info(
        "Loading embedding model %r — first run downloads ~1.1 GB from HuggingFace.",
        settings.embedding_model,
    )
    _model = SentenceTransformer(settings.embedding_model)
    logger.info(
        "Embedding model loaded — dimension=%d device=%s",
        _EMBEDDING_DIMENSION,
        _model.device,
    )
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    """
    Encode a list of review texts (documents to be stored and searched).

    Each text is prefixed with ``"passage: "`` as required by the E5 model
    family.  Omitting the prefix degrades retrieval quality.

    Parameters
    ----------
    texts:
        Review bodies to embed.  Must be non-empty strings.  Empty list
        returns an empty list without touching the model.

    Returns
    -------
    list[list[float]]
        One 1024-dimensional float vector per input text.  Vectors are
        L2-normalised so dot-product == cosine similarity.
    """
    if not texts:
        return []

    model = load_model()
    prefixed = [f"passage: {t}" for t in texts]
    vectors = model.encode(
        prefixed,
        batch_size=_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """
    Encode a single user query string.

    The text is prefixed with ``"query: "`` as required by the E5 model
    family.

    Parameters
    ----------
    text:
        The user's question or search phrase.

    Returns
    -------
    list[float]
        A 1024-dimensional L2-normalised float vector.
    """
    model = load_model()
    vector = model.encode(
        f"query: {text}",
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vector.tolist()


def get_embedding_dimension() -> int:
    """
    Return the fixed output dimension of the configured embedding model.

    Used as a startup sanity check to verify that the migration SQL and
    the live model agree on vector size (both should be 1024).

    Returns
    -------
    int
        ``1024`` for ``intfloat/multilingual-e5-large``.
    """
    return _EMBEDDING_DIMENSION
