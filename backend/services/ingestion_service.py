"""
Ingestion service — pure-Python CSV parsing, column mapping, and row validation.

This module has zero FastAPI dependencies and can be exercised in isolation
(unit tests, CLI scripts, etc.).  The upload router is the only caller in
production.

Pipeline
--------
1. ``parse_csv``         — decode bytes, parse CSV into a DataFrame.
2. ``detect_columns``    — map arbitrary column names to canonical ones.
3. ``validate_and_clean``— row-by-row validation, returns clean records + errors.
4. ``detect_language``   — ISO 639-1 language detection via langdetect.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from typing import Any

import dateparser
import pandas as pd
from langdetect import detect, LangDetectException
from pydantic import ValidationError

from backend.models.schemas import ReviewClean, RowError

logger = logging.getLogger(__name__)

# ── Column alias dictionary ───────────────────────────────────────────────────
# Maps every known alternate column name (lower-cased, stripped) to its
# canonical field name.  Add new aliases here without touching any other code.

COLUMN_ALIASES: dict[str, str] = {
    # author
    "author": "author",
    "auteur": "author",
    "autor": "author",
    "reviewer": "author",
    "user": "author",
    "utilisateur": "author",
    "name": "author",
    "nom": "author",
    # rating
    "rating": "rating",
    "note": "rating",
    "score": "rating",
    "stars": "rating",
    "étoiles": "rating",
    "etoiles": "rating",
    "grade": "rating",
    "calificacion": "rating",
    "nota": "rating",
    # date
    "date": "date",
    "fecha": "date",
    "datum": "date",
    "published": "date",
    "published_at": "date",
    "review_date": "date",
    "created_at": "date",
    "time": "date",
    # text / review body
    "text": "text",
    "review": "text",
    "avis": "text",
    "comentario": "text",
    "comentarios": "text",
    "comment": "text",
    "commentaire": "text",
    "body": "text",
    "content": "text",
    "review_text": "text",
    "feedback": "text",
    "message": "text",
    # language
    "language": "language",
    "langue": "language",
    "idioma": "language",
    "lang": "language",
    "iso_lang": "language",
}

# Canonical columns that must be present after mapping for a row to be valid.
REQUIRED_CANONICAL_COLUMNS: frozenset[str] = frozenset({"rating", "date", "text"})


# ── Public API ────────────────────────────────────────────────────────────────


def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Decode *file_bytes* and parse them into a ``DataFrame``.

    Encoding and separator detection strategy
    -----------------------------------------
    1. Attempt UTF-8 (covers the vast majority of modern exports).
    2. Fall back to Latin-1 / ISO-8859-1, which handles most Western-European
       Google Maps exports that include accented characters.

    For each encoding, ``sep=None, engine='python'`` is passed to
    ``pd.read_csv`` so that Python's built-in CSV sniffer automatically
    detects the delimiter (comma, semicolon, tab, pipe, etc.).  This covers
    the common case of French/European exports that use ``;`` as the
    separator.

    Parameters
    ----------
    file_bytes:
        Raw bytes of the uploaded CSV file.

    Returns
    -------
    pd.DataFrame
        All columns as ``object`` dtype (strings).  The caller is responsible
        for type coercion.

    Raises
    ------
    ValueError
        When the bytes cannot be decoded or parsed as CSV.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            text = file_bytes.decode(encoding)
            df = pd.read_csv(
                io.StringIO(text),
                sep=None,           # let pandas sniff the delimiter
                engine="python",    # required when sep=None
                dtype=str,
                keep_default_na=False,
            )
            logger.debug(
                "CSV parsed — encoding=%s columns=%s shape=%s",
                encoding,
                df.columns.tolist(),
                df.shape,
            )
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError as exc:
            raise ValueError(f"CSV parsing failed: {exc}") from exc

    raise ValueError(
        "Could not decode the file as UTF-8 or Latin-1. "
        "Please re-save the CSV with UTF-8 encoding."
    )


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Build a mapping from *df*'s actual column names to canonical field names.

    The comparison is done case-insensitively and strips leading/trailing
    whitespace, so ``'  Author '``, ``'AUTHOR'``, and ``'author'`` all resolve
    to ``'author'``.

    Parameters
    ----------
    df:
        DataFrame whose columns should be inspected.

    Returns
    -------
    dict[str, str]
        ``{actual_column_name: canonical_name}`` for every column that has a
        known alias.  Columns with no alias are omitted.

    Notes
    -----
    If the same canonical name would be mapped from multiple actual columns
    (e.g. both ``"rating"`` and ``"note"`` are present), the first occurrence
    in ``df.columns`` wins.
    """
    mapping: dict[str, str] = {}
    seen_canonical: set[str] = set()

    for col in df.columns:
        normalised = col.strip().lower()
        canonical = COLUMN_ALIASES.get(normalised)
        if canonical and canonical not in seen_canonical:
            mapping[col] = canonical
            seen_canonical.add(canonical)

    logger.debug("Column mapping resolved: %s", mapping)
    return mapping


def validate_and_clean(
    df: pd.DataFrame,
    session_id: str,
) -> tuple[list[ReviewClean], list[RowError]]:
    """
    Validate and clean every row in *df*, returning clean records and errors.

    Steps per row
    -------------
    1. Rename columns to canonical names using ``detect_columns``.
    2. Check that all required canonical columns are present; if not, abort
       the entire file with a single ``RowError`` at row 0 describing the
       missing columns.
    3. For each data row:
       a. Build a ``ReviewClean`` via Pydantic validation.
       b. If the ``language`` field is absent or blank, run ``detect_language``
          on the review text and set ``original_language_detected = True``.
       c. Collect ``RowError`` for any row that fails validation.

    Parameters
    ----------
    df:
        DataFrame produced by ``parse_csv`` (all columns as strings).
    session_id:
        Session identifier to embed in every ``ReviewClean`` record.

    Returns
    -------
    tuple[list[ReviewClean], list[RowError]]
        A 2-tuple of ``(valid_reviews, errors)``.  Rows that fail appear only
        in *errors*; they are never included in *valid_reviews*.
    """
    col_map = detect_columns(df)
    df = df.rename(columns=col_map)

    # Guard: verify required columns are present after remapping.
    missing = REQUIRED_CANONICAL_COLUMNS - set(df.columns)
    if missing:
        error = RowError(
            row=0,
            reason=(
                f"CSV is missing required column(s): {sorted(missing)}. "
                f"Detected columns after mapping: {sorted(df.columns.tolist())}."
            ),
        )
        return [], [error]

    valid_reviews: list[ReviewClean] = []
    errors: list[RowError] = []

    for idx, row in df.iterrows():
        # idx is 0-based; report 1-based row numbers to the user (+ 1 for header).
        row_number = int(idx) + 2  # type: ignore[arg-type]
        raw: dict[str, Any] = row.to_dict()

        # Extract fields, treating empty strings as None.
        def _get(key: str) -> str | None:
            val = raw.get(key, "")
            return val.strip() if isinstance(val, str) and val.strip() else None

        text_value = _get("text")
        language_from_csv = _get("language")

        # Detect language when not supplied in the CSV.
        language_detected = False
        if not language_from_csv and text_value:
            language_from_csv = detect_language(text_value)
            language_detected = True

        date_value = _parse_date(_get("date"))
        # date_value is None when the field is missing or unparseable;
        # the row is still ingested (ReviewClean.date allows None).

        try:
            review = ReviewClean(
                session_id=session_id,
                author=_get("author"),
                rating=_get("rating"),  # coerced by field_validator
                date=date_value,
                text=text_value or "",  # empty string triggers min_length error
                language=language_from_csv,
                original_language_detected=language_detected,
            )
            valid_reviews.append(review)
        except ValidationError as exc:
            reasons = "; ".join(
                f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            )
            errors.append(RowError(row=row_number, reason=reasons))

    logger.info(
        "Validation complete — session=%s valid=%d invalid=%d",
        session_id,
        len(valid_reviews),
        len(errors),
    )
    return valid_reviews, errors


def detect_language(text: str) -> str:
    """
    Detect the ISO 639-1 language code of *text* using ``langdetect``.

    ``langdetect`` is non-deterministic by default; the result may vary
    slightly between runs on very short texts.  For production use, consider
    calling ``langdetect.DetectorFactory.seed = 0`` once at startup to make
    results reproducible.

    Parameters
    ----------
    text:
        The review body to analyse.  Should be at least a few words long for
        reliable results.

    Returns
    -------
    str
        ISO 639-1 code (e.g. ``"fr"``, ``"en"``, ``"ar"``), or ``"unknown"``
        when detection fails or the text is too short / ambiguous.
    """
    try:
        return detect(text)
    except LangDetectException:
        logger.debug("langdetect could not identify language for text: %.50r", text)
        return "unknown"
    except Exception as exc:  # noqa: BLE001 — broad catch intentional for robustness
        logger.warning("Unexpected error in detect_language: %s", exc)
        return "unknown"


# ── Internal helpers ──────────────────────────────────────────────────────────


def _parse_date(raw: str | None) -> datetime | None:
    """
    Attempt to parse *raw* into a ``datetime``.

    Parsing strategy (in order)
    ---------------------------
    1. Explicit ``strptime`` formats — fast, unambiguous, covers well-formed
       absolute dates in the 10 most common CSV export styles.
    2. ``dateparser.parse`` — handles relative strings in multiple languages,
       e.g. ``"il y a 2 ans"``, ``"Modifié il y a 3 mois"``,
       ``"2 weeks ago"``, ``"X years ago"``.  ``PREFER_DAY_OF_MONTH`` and
       ``RETURN_AS_TIMEZONE_AWARE=False`` keep the output consistent with the
       strptime path.

    Parameters
    ----------
    raw:
        Raw date string from the CSV, or ``None`` if the cell was empty.

    Returns
    -------
    datetime | None
        Parsed datetime object, or ``None`` when *raw* is missing/empty or
        cannot be parsed by either strategy.  A warning is logged in the
        latter case so the issue is visible without rejecting the row.
    """
    if not raw:
        logger.warning("Date field is missing or empty — storing None.")
        return None

    # ── Stage 1: explicit strptime formats ────────────────────────────────────
    # Ordered from most specific to least specific to avoid ambiguous matches
    # (e.g. %d/%m/%Y must come before %m/%d/%Y).
    formats = [
        "%Y-%m-%dT%H:%M:%S",   # ISO 8601 with time
        "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601 UTC
        "%Y-%m-%d %H:%M:%S",   # common DB export format
        "%Y-%m-%d",             # date only
        "%d/%m/%Y %H:%M:%S",   # European with time
        "%d/%m/%Y",             # European date only
        "%m/%d/%Y",             # US date only
        "%d-%m-%Y",             # European dash-separated
        "%B %d, %Y",            # "January 15, 2024"
        "%b %d, %Y",            # "Jan 15, 2024"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue

    # ── Stage 2: dateparser — relative & multilingual strings ─────────────────
    # Handles: "il y a 2 ans", "Modifié il y a 3 mois", "il y a un an",
    #          "il y a 2 semaines", "2 years ago", "3 months ago", etc.
    parsed = dateparser.parse(
        raw,
        settings={
            "PREFER_DAY_OF_MONTH": "first",
            "RETURN_AS_TIMEZONE_AWARE": False,
            # Allow all languages supported by dateparser (FR, EN, AR, ES, PT …)
            "PREFER_LOCALE_DATE_ORDER": True,
        },
    )
    if parsed is not None:
        logger.debug("dateparser resolved %r → %s", raw, parsed)
        return parsed

    logger.warning("Could not parse date %r — storing None.", raw)
    return None
