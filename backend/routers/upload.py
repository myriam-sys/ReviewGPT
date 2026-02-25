"""
Upload router — CSV / XLSX ingestion and session preview endpoints.

Endpoints
---------
POST /upload
    Accept a multipart CSV or Excel file, run the full ingestion pipeline,
    and return an ``UploadResponse`` describing the outcome.

GET /upload/{session_id}/preview
    Return the first N cleaned reviews for a session so the frontend can show
    the user what was successfully parsed before they proceed to the chat.

Storage note
------------
Validated reviews are stored in the module-level ``_session_store`` dict for
Phase 1.  This is intentionally simple — no DB, no persistence across
restarts.  Phase 2 will replace this with Supabase writes.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.core.config import settings
from backend.models.schemas import PreviewResponse, ReviewClean, UploadResponse
from backend.services.ingestion_service import parse_file, validate_and_clean

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

    # Persist in the in-memory store for the preview endpoint.
    _session_store[session_id] = valid_reviews

    logger.info(
        "Ingestion complete — session=%s total=%d valid=%d invalid=%d",
        session_id,
        total_rows,
        len(valid_reviews),
        len(errors),
    )

    return UploadResponse(
        session_id=session_id,
        total_rows=total_rows,
        valid_rows=len(valid_reviews),
        invalid_rows=len(errors),
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
