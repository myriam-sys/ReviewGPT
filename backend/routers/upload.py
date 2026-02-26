"""
Upload router — CSV / XLSX ingestion, preview, and embedding-status endpoints.

Endpoints
---------
POST /upload
    Accept a multipart CSV or Excel file, run the full ingestion pipeline,
    persist reviews to Supabase, queue a background embedding task, and return
    an ``UploadResponse`` describing the outcome.

GET /upload/{session_id}/preview
    Return the first N cleaned reviews for a session so the frontend can show
    the user what was successfully parsed before they proceed to the chat.

GET /upload/{session_id}/embedding-status
    Poll the progress of the background embedding task started by POST /upload.
    Returns counts of embedded vs. pending reviews and a ``status`` string.

Storage note
------------
Validated reviews are stored in the module-level ``_session_store`` dict so
the preview endpoint works even when Supabase is unavailable.  DB persistence
(Phase 2) and vector embedding (Phase 3) are both best-effort — errors are
logged but never surface to the caller as HTTP errors.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from backend.core.config import settings
from backend.db.supabase_client import get_asyncpg_pool
from backend.models.schemas import (
    EmbeddingStatusResponse,
    PreviewResponse,
    ReviewClean,
    UploadResponse,
)
from backend.services.ingestion_service import (
    compute_and_store_embeddings,
    parse_file,
    save_reviews_to_db,
    validate_and_clean,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ── In-memory session store (Phase 1 only) ────────────────────────────────────
# Maps session_id -> list of validated ReviewClean objects.
# Replaced by Supabase persistence in Phase 2.
_session_store: dict[str, list[ReviewClean]] = {}

# Allowed MIME types and file extensions for uploaded files.
# Content-type is browser-reported and unreliable; extension is a secondary
# signal.  The authoritative format check is done via magic bytes inside
# parse_file — these sets only serve as a fast early-rejection gate.
_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({
    # CSV
    "text/csv",
    "application/csv",
    "text/plain",
    # Excel XLSX
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    # Excel XLS (legacy)
    "application/vnd.ms-excel",
    "application/excel",
    # Generic binary — many browsers send this for both CSV and Excel
    "application/octet-stream",
})
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".xlsx", ".xls"})


# ── POST /upload ──────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload a Google Maps review CSV or Excel file",
    description=(
        "Accepts a .csv, .xlsx, or .xls file via multipart/form-data, "
        "validates every row, and returns a summary of what was ingested. "
        "Use the returned ``session_id`` to query the preview or chat endpoints."
    ),
)
async def upload_csv(
    file: Annotated[UploadFile, File(description="CSV or Excel file of Google Maps reviews.")],
    background_tasks: BackgroundTasks,
) -> UploadResponse:
    """
    Ingest a CSV or Excel file of Google Maps reviews.

    The pipeline is:
    1. Reject files whose extension and content-type are both unrecognised.
    2. Enforce the configured maximum file size (``MAX_UPLOAD_BYTES``).
    3. Detect format via magic bytes and parse into a DataFrame.
    4. Validate and clean every row via ``ingestion_service``.
    5. Store the valid reviews in the in-memory session store.
    6. Return ``UploadResponse`` with counts and any per-row errors.

    Parameters
    ----------
    file:
        The uploaded CSV or Excel file.

    Returns
    -------
    UploadResponse
        Session identifier plus ingestion statistics and errors.

    Raises
    ------
    400 Bad Request
        When the file type is not CSV or Excel.
    413 Request Entity Too Large
        When the file exceeds ``MAX_UPLOAD_BYTES``.
    422 Unprocessable Entity
        When the file cannot be parsed (bad encoding, missing required columns).
    500 Internal Server Error
        For unexpected failures.
    """
    _validate_file_type(file)

    try:
        file_bytes = await file.read()
    except Exception as exc:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read the uploaded file.",
        ) from exc

    _validate_file_size(file_bytes, file.filename or "")

    session_id = str(uuid.uuid4())
    logger.info("Starting ingestion — session=%s file=%s", session_id, file.filename)

    try:
        df = parse_file(file_bytes, filename=file.filename or "")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    total_rows = len(df)

    try:
        valid_reviews, errors = validate_and_clean(df, session_id)
    except Exception as exc:
        logger.exception("Unexpected error during validation — session=%s", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while validating the CSV.",
        ) from exc

    # Always persist in the in-memory store so the preview endpoint works
    # regardless of whether the DB write succeeds.
    _session_store[session_id] = valid_reviews

    reviews_with_text = sum(1 for r in valid_reviews if r.has_text)

    # ── Persist to Supabase ───────────────────────────────────────────────────
    # The DB write is best-effort: if credentials are absent or the insert
    # fails for any reason we log the error and continue.  The in-memory store
    # guarantees the preview endpoint still works for the session lifetime.
    inserted_rows = 0
    skipped_rows = 0
    try:
        inserted_rows, skipped_rows = save_reviews_to_db(valid_reviews, session_id)
    except Exception:
        logger.exception(
            "DB write failed — session=%s reviews will only be available "
            "in-memory for this session.",
            session_id,
        )

    # ── Queue background embedding ────────────────────────────────────────────
    # Only useful when rows actually landed in the DB and have text to embed.
    if inserted_rows > 0 and reviews_with_text > 0:
        background_tasks.add_task(compute_and_store_embeddings, session_id)
        embedding_status = "embedding_queued"
    elif reviews_with_text == 0:
        embedding_status = "nothing_to_embed"
    else:
        embedding_status = "embedding_skipped"

    logger.info(
        "Ingestion complete — session=%s total=%d valid=%d invalid=%d "
        "embeddable=%d inserted=%d skipped=%d embedding=%s",
        session_id,
        total_rows,
        len(valid_reviews),
        len(errors),
        reviews_with_text,
        inserted_rows,
        skipped_rows,
        embedding_status,
    )

    return UploadResponse(
        session_id=session_id,
        total_rows=total_rows,
        valid_rows=len(valid_reviews),
        invalid_rows=len(errors),
        reviews_with_text=reviews_with_text,
        inserted_rows=inserted_rows,
        skipped_rows=skipped_rows,
        embedding_status=embedding_status,
        errors=errors,
    )


# ── GET /upload/{session_id}/preview ─────────────────────────────────────────


@router.get(
    "/{session_id}/preview",
    response_model=PreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Preview cleaned reviews for a session",
    description=(
        f"Returns the first {settings.preview_row_limit} validated reviews "
        "for the given session so the frontend can display a data preview."
    ),
)
async def preview_session(session_id: str) -> PreviewResponse:
    """
    Return the first ``PREVIEW_ROW_LIMIT`` cleaned reviews for *session_id*.

    Parameters
    ----------
    session_id:
        The UUID returned by ``POST /upload``.

    Returns
    -------
    PreviewResponse
        Session ID and a list of up to ``PREVIEW_ROW_LIMIT`` clean reviews.

    Raises
    ------
    404 Not Found
        When no session exists for the given ID.
    """
    reviews = _session_store.get(session_id)
    if reviews is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No session found for id '{session_id}'.",
        )

    return PreviewResponse(
        session_id=session_id,
        reviews=reviews[: settings.preview_row_limit],
    )


# ── GET /upload/{session_id}/embedding-status ─────────────────────────────────


@router.get(
    "/{session_id}/embedding-status",
    response_model=EmbeddingStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Poll embedding progress for a session",
    description=(
        "Returns the number of reviews that have been embedded so far "
        "vs. the total that need embedding.  Poll this endpoint after "
        "``POST /upload`` until ``status`` is ``'complete'``."
    ),
)
async def embedding_status_endpoint(session_id: str) -> EmbeddingStatusResponse:
    """
    Query Supabase for embedding progress of *session_id*.

    Counts rows where ``has_text=True`` (total_with_text) and the subset
    where ``embedding IS NOT NULL`` (embedded) to derive progress.

    Parameters
    ----------
    session_id:
        The UUID returned by ``POST /upload``.

    Returns
    -------
    EmbeddingStatusResponse
        Counts and a ``status`` string: ``"complete"``, ``"processing"``,
        or ``"empty"`` when no text-bearing reviews exist.

    Raises
    ------
    503 Service Unavailable
        When the database connection pool cannot be created.
    """
    try:
        pool = await get_asyncpg_pool()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is not configured.",
        ) from exc

    # A single query returns both counts — more efficient than two round-trips.
    # asyncpg is used because the SDK cannot filter on the vector column type.
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE has_text = TRUE)                          AS total_with_text,
                COUNT(*) FILTER (WHERE has_text = TRUE AND embedding IS NOT NULL) AS embedded
            FROM reviews
            WHERE session_id = $1::uuid
            """,
            session_id,
        )

    total_with_text: int = row["total_with_text"] or 0
    embedded: int = row["embedded"] or 0

    if total_with_text == 0:
        return EmbeddingStatusResponse(
            session_id=session_id,
            total_with_text=0,
            embedded=0,
            pending=0,
            status="empty",
        )

    pending = total_with_text - embedded
    embedding_status_str = "complete" if pending == 0 else "processing"

    return EmbeddingStatusResponse(
        session_id=session_id,
        total_with_text=total_with_text,
        embedded=embedded,
        pending=pending,
        status=embedding_status_str,
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _validate_file_type(file: UploadFile) -> None:
    """
    Raise ``HTTP 400`` if *file* is clearly not a supported format.

    This is a fast pre-flight check before the bytes are read.  It rejects
    obviously wrong uploads (e.g. images, PDFs) without consuming memory.
    The authoritative format detection happens later via magic bytes inside
    ``parse_file`` — this gate only checks extension and content-type.

    The file passes if *either* its extension or its content-type is in the
    allowed sets, because browsers report both inconsistently.

    Parameters
    ----------
    file:
        The uploaded file to inspect.

    Raises
    ------
    HTTPException (400)
        When neither the extension nor the content-type is recognised.
    """
    filename = file.filename or ""
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_type = (file.content_type or "").split(";")[0].strip().lower()

    if extension not in _ALLOWED_EXTENSIONS and content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid file type. Expected .csv, .xlsx, or .xls — "
                f"got extension={extension!r} content_type={content_type!r}."
            ),
        )


def _validate_file_size(file_bytes: bytes, filename: str) -> None:
    """
    Raise ``HTTP 413`` if *file_bytes* exceeds ``settings.max_upload_bytes``.

    Parameters
    ----------
    file_bytes:
        The fully-read file content.
    filename:
        Original filename, used only for the error message.

    Raises
    ------
    HTTPException (413)
        When the file is too large.
    """
    size = len(file_bytes)
    limit = settings.max_upload_bytes
    if size > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File '{filename}' is {size:,} bytes, "
                f"which exceeds the {limit:,}-byte limit."
            ),
        )
