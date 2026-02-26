"""
Pydantic schemas — request and response models for all API endpoints.

Separation of concerns:
- ``ReviewRaw``   : mirrors a raw CSV row; all fields are optional so dirty /
                    partially-filled rows can be parsed without raising immediately.
- ``ReviewClean`` : the validated, normalised representation stored (and later
                    embedded).  Strict types are enforced here.
- ``UploadResponse`` : returned to the caller after a CSV upload completes.
- ``RowError``    : describes a single validation failure with its row number.
- ``ChatRequest`` / ``ChatResponse`` : placeholders for Phase 4.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field, field_validator


# ── Raw CSV row ───────────────────────────────────────────────────────────────


class ReviewRaw(BaseModel):
    """
    Lenient representation of a single CSV row before validation.

    Every field is ``str | None`` so that pandas rows can be fed in as-is.
    The ingestion service is responsible for converting this to
    ``ReviewClean`` and collecting any conversion errors.
    """

    author: str | None = None
    rating: str | None = None
    date: str | None = None
    text: str | None = None
    language: str | None = None


# ── Cleaned / validated review ────────────────────────────────────────────────


class ReviewClean(BaseModel):
    """
    Validated, normalised review record ready for embedding and storage.

    Fields
    ------
    review_id :
        UUID generated at validation time; unique per review row.
    session_id :
        Identifies the upload session / tenant.  All reviews in one upload
        share the same session_id.
    author :
        Display name of the reviewer.  May be None if the CSV omits it.
    rating :
        Numeric score in the range [1.0, 5.0].
    date :
        Publication date of the review.  ``None`` when the raw value is
        missing or cannot be parsed by either strptime or dateparser (a
        warning is logged in that case; the row is still ingested).
    text :
        Body of the review after whitespace stripping, or ``None`` when the
        cell was empty or contained only whitespace.  Rows with ``text=None``
        are still ingested; they will be skipped by the embedding service in
        Phase 3.
    has_text :
        Computed convenience flag — ``True`` when ``text`` is non-``None``.
        Phase 3 uses this to filter out un-embeddable reviews without
        re-checking the field directly.
    language :
        ISO 639-1 language code assigned by ``langdetect`` (e.g. ``"fr"``),
        or ``"unknown"`` when detection fails.
    original_language_detected :
        ``True`` when the language was inferred automatically by langdetect;
        ``False`` when it was supplied directly in the CSV.
    """

    review_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: str
    author: str | None = None
    rating: float = Field(..., ge=1.0, le=5.0)
    date: Optional[datetime] = None
    text: str | None = None
    language: str | None = None
    original_language_detected: bool = False
    content_hash: str = Field(
        default="",
        description="MD5 hex digest of 'author|rating|text|date' used for upsert deduplication.",
    )

    @field_validator("text", mode="before")
    @classmethod
    def normalise_text(cls, v: Any) -> str | None:
        """Strip whitespace; treat blank / empty strings as absent (``None``)."""
        if v is None:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None

    @computed_field  # type: ignore[misc]
    @property
    def has_text(self) -> bool:
        """``True`` when the review body is present and non-empty."""
        return self.text is not None

    @field_validator("rating", mode="before")
    @classmethod
    def coerce_rating(cls, v: Any) -> float:
        """Accept ratings supplied as strings (e.g. '4' or '3.5')."""
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid rating value: {v!r}")


# ── Validation error per row ──────────────────────────────────────────────────


class RowError(BaseModel):
    """Describes a single row that failed validation during ingestion."""

    row: int = Field(..., description="1-based row number in the original CSV.")
    reason: str = Field(..., description="Human-readable description of why the row was rejected.")


# ── Upload API response ───────────────────────────────────────────────────────


class UploadResponse(BaseModel):
    """
    Returned by ``POST /upload`` after CSV ingestion completes.

    Fields
    ------
    session_id :
        UUID that identifies this upload session.  Pass it to subsequent
        ``/chat`` and ``/upload/{session_id}/preview`` requests.
    total_rows :
        Total number of data rows found in the file (header excluded).
    valid_rows :
        Number of rows that passed validation and were ingested.
    invalid_rows :
        Number of rows that were rejected.
    reviews_with_text :
        Subset of ``valid_rows`` where the review body is non-empty.  Only
        these rows will be embedded and indexed in Phase 3.
    errors :
        List of per-row errors (empty when all rows are valid).
    """

    session_id: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    reviews_with_text: int
    inserted_rows: int = Field(
        default=0,
        description=(
            "Number of rows successfully written to the database. "
            "0 when the DB write was skipped or failed (reviews are still "
            "available via the preview endpoint for the duration of the session)."
        ),
    )
    skipped_rows: int = Field(
        default=0,
        description=(
            "Number of valid rows silently skipped because an identical review "
            "(same session_id + content_hash) already exists in the database. "
            "Non-zero only on re-uploads of the same CSV."
        ),
    )
    embedding_status: str = Field(
        default="embedding_queued",
        description=(
            "Indicates whether background embedding has been queued. "
            "``embedding_queued`` — background task started, poll "
            "``GET /upload/{session_id}/embedding-status`` for progress. "
            "``nothing_to_embed`` — all valid rows lack review text. "
            "``embedding_skipped`` — DB write failed so there is nothing to embed."
        ),
    )
    errors: list[RowError] = Field(default_factory=list)


# ── Preview response ──────────────────────────────────────────────────────────


class PreviewResponse(BaseModel):
    """Returned by ``GET /upload/{session_id}/preview``."""

    session_id: str
    reviews: list[ReviewClean]


# ── Embedding status response ─────────────────────────────────────────────────


class EmbeddingStatusResponse(BaseModel):
    """
    Returned by ``GET /upload/{session_id}/embedding-status``.

    Fields
    ------
    session_id :
        The upload session being queried.
    total_with_text :
        Number of valid rows whose ``has_text`` flag is ``True`` — i.e. the
        reviews that will be (or have been) embedded.
    embedded :
        Number of those rows whose ``embedding`` column is non-NULL.
    pending :
        ``total_with_text - embedded``.  Decreases toward 0 as the background
        task progresses.
    status :
        ``"complete"`` — all embeddable rows have a vector.
        ``"processing"`` — the background task is still running.
        ``"empty"`` — no text-bearing reviews exist for this session.
    """

    session_id: str
    total_with_text: int
    embedded: int
    pending: int
    status: str  # "complete" | "processing" | "empty"


# ── Chat (Phase 4) ────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Request body for ``POST /chat``."""

    session_id: str = Field(..., description="Session UUID returned by POST /upload.")
    question: str = Field(..., min_length=1, description="The user's natural-language question.")
    top_k: int = Field(
        default=10,
        ge=1,
        le=30,
        description=(
            "Number of similar reviews to retrieve before filtering. "
            "For top_k > 15, consider using mixtral-8x7b-32768 which has "
            "a larger context window to accommodate the extra reviews."
        ),
    )


class ChatResponse(BaseModel):
    """
    Returned by ``POST /chat`` after the full RAG pipeline completes.

    Fields
    ------
    answer :
        The LLM-generated response, grounded in the retrieved reviews.
    model :
        Groq model ID used to generate the response.
    tokens_used :
        Total tokens consumed (prompt + completion).
    sources_count :
        Number of retrieved reviews passed to the LLM as context.
    retrieved_reviews :
        The review dicts used as sources, each with ``author``, ``rating``,
        ``date``, ``text``, ``language``, and ``similarity`` keys.
    session_id :
        The session this response belongs to.
    question :
        Echo of the original question, useful for frontend display.
    """

    answer: str
    model: str
    tokens_used: int
    sources_count: int
    retrieved_reviews: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str
    question: str
    question_type: str = Field(
        default="search",
        description=(
            "Detected question type: ``analytical``, ``temporal``, "
            "``comparative``, ``author``, or ``search``."
        ),
    )
