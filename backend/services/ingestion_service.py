"""
Ingestion service — pure-Python file parsing, column mapping, and row validation.

This module has zero FastAPI dependencies and can be exercised in isolation
(unit tests, CLI scripts, etc.).  The upload router is the only caller in
production.

Pipeline
--------
1. ``parse_file``        — detect format (CSV / XLSX / XLS), parse into a DataFrame.
2. ``detect_columns``    — map arbitrary column names to canonical ones.
3. ``validate_and_clean``— row-by-row validation, returns clean records + errors.
4. ``detect_language``   — ISO 639-1 language detection via langdetect.

Format detection
----------------
``parse_file`` inspects the first 4 bytes of the file (magic bytes) to
determine the format, regardless of browser-reported content-type or file
extension.  Downstream functions (``detect_columns``, ``validate_and_clean``)
always receive a plain ``DataFrame`` and are completely format-agnostic.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import datetime
from typing import Any

import dateparser
import pandas as pd
from dateutil.relativedelta import relativedelta  # transitive dep via dateparser
from langdetect import detect, LangDetectException
from pydantic import ValidationError

from backend.models.schemas import ReviewClean, RowError

logger = logging.getLogger(__name__)

# ── Magic bytes for binary format detection ───────────────────────────────────
# Checked against the first 4 bytes of the uploaded file.  Browser-reported
# content-type is unreliable, so byte signatures are used as ground truth.

_XLSX_MAGIC: bytes = b"\x50\x4B\x03\x04"  # ZIP/PK — XLSX uses the OOXML/ZIP container
_XLS_MAGIC: bytes = b"\xD0\xCF\x11\xE0"   # OLE2 compound document — legacy .xls format

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


def parse_file(file_bytes: bytes, filename: str = "") -> pd.DataFrame:
    """
    Detect the file format and parse *file_bytes* into a ``DataFrame``.

    Format detection strategy (most reliable first)
    ------------------------------------------------
    1. **Magic bytes** — the first 4 bytes identify XLSX (``PK/ZIP``) and
       legacy XLS (``OLE2``) unambiguously, regardless of extension or
       browser-reported content-type.
    2. **Extension fallback** — consulted only when magic bytes produce no
       match, i.e. for text files (CSV, TSV) which have no binary signature.

    Dispatches to
    -------------
    - ``_parse_csv``   : UTF-8 BOM + Latin-1 fallback; ``sep=None`` for
                         automatic delimiter detection.
    - ``_parse_excel`` : ``openpyxl`` engine for XLSX, auto-selected engine
                         for legacy XLS.

    The rest of the pipeline (``detect_columns``, ``validate_and_clean``) is
    format-agnostic — it always receives a plain ``DataFrame``.

    Parameters
    ----------
    file_bytes:
        Raw bytes of the uploaded file.
    filename:
        Original filename, used for the extension-based fallback and log
        messages.  Safe to omit; defaults to an empty string.

    Returns
    -------
    pd.DataFrame
        All columns as ``object`` dtype (strings).  The caller is responsible
        for type coercion.

    Raises
    ------
    ValueError
        When the file cannot be parsed in any supported format.
    """
    fmt = _detect_format(file_bytes, filename)
    logger.debug("Format detected for %r: %s", filename, fmt)

    if fmt == "xlsx":
        return _parse_excel(file_bytes, engine="openpyxl")
    if fmt == "xls":
        # Legacy XLS requires the optional `xlrd` package.  pandas will
        # raise ImportError if it is absent; _parse_excel surfaces that as
        # a user-friendly ValueError.
        return _parse_excel(file_bytes, engine=None)
    return _parse_csv(file_bytes)


# ── Private format helpers ─────────────────────────────────────────────────────


def _detect_format(file_bytes: bytes, filename: str) -> str:
    """
    Return the file format string: ``"xlsx"``, ``"xls"``, or ``"csv"``.

    Magic bytes (first 4 bytes of the file) are the primary signal.  The
    file extension is only consulted when magic bytes do not identify a
    known binary format — i.e. for plain-text files.

    Parameters
    ----------
    file_bytes:
        Raw file bytes (at least 4 bytes expected for reliable detection).
    filename:
        Original filename used for extension-based fallback and warnings.

    Returns
    -------
    str
        One of ``"xlsx"``, ``"xls"``, or ``"csv"``.
    """
    header = file_bytes[:4]

    if header == _XLSX_MAGIC:
        return "xlsx"
    if header == _XLS_MAGIC:
        return "xls"

    # No binary magic matched — file is assumed to be text/CSV.
    # Warn if the extension looks like Excel, as the file may be malformed.
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext in (".xlsx", ".xls"):
        logger.warning(
            "File %r has extension %r but no matching magic bytes — "
            "attempting to parse as CSV.",
            filename,
            ext,
        )
    return "csv"


def _parse_csv(file_bytes: bytes) -> pd.DataFrame:
    """
    Decode *file_bytes* as a delimiter-separated text file and return a DataFrame.

    Encoding strategy
    -----------------
    1. ``utf-8-sig`` — handles both plain UTF-8 and UTF-8-with-BOM (the BOM
       that Excel adds when exporting CSVs is stripped automatically).
    2. ``latin-1`` — covers Western-European exports with accented characters
       that are not valid UTF-8.

    ``sep=None, engine='python'`` delegates delimiter detection to Python's
    built-in ``csv.Sniffer``, covering commas, semicolons, tabs, pipes, and
    other separators automatically.

    Parameters
    ----------
    file_bytes:
        Raw bytes of a CSV or other delimiter-separated text file.

    Returns
    -------
    pd.DataFrame
        All columns as ``object`` dtype (strings).

    Raises
    ------
    ValueError
        When the bytes cannot be decoded or parsed.
    """
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = file_bytes.decode(encoding)
            df = pd.read_csv(
                io.StringIO(text),
                sep=None,        # let Python's csv.Sniffer detect the delimiter
                engine="python", # required when sep=None
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


def _parse_excel(file_bytes: bytes, engine: str | None) -> pd.DataFrame:
    """
    Parse *file_bytes* as an Excel workbook and return the first sheet.

    Parameters
    ----------
    file_bytes:
        Raw bytes of an XLSX or XLS workbook.
    engine:
        ``pandas.read_excel`` engine.  Pass ``"openpyxl"`` for XLSX; pass
        ``None`` to let pandas auto-select for legacy XLS (requires ``xlrd``
        to be installed separately).

    Returns
    -------
    pd.DataFrame
        All columns as ``object`` dtype (strings).  Column names are cast to
        ``str`` to handle workbooks that use numeric headers.

    Raises
    ------
    ValueError
        When the workbook cannot be read, or when the required engine
        (e.g. ``xlrd`` for legacy XLS) is not installed.
    """
    try:
        df = pd.read_excel(
            io.BytesIO(file_bytes),
            dtype=str,
            engine=engine,
            keep_default_na=False,
        )
        # Numeric column headers (e.g. unnamed columns) must be strings for
        # detect_columns to process them correctly.
        df.columns = df.columns.astype(str)
        logger.debug(
            "Excel parsed — engine=%s columns=%s shape=%s",
            engine,
            df.columns.tolist(),
            df.shape,
        )
        return df
    except ImportError as exc:
        raise ValueError(
            "Reading this Excel format requires an additional library. "
            "Install 'xlrd' for legacy .xls files: pip install xlrd"
        ) from exc
    except Exception as exc:
        raise ValueError(f"Excel parsing failed: {exc}") from exc


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
       b. If the ``language`` field is absent or blank AND the review text is
          present, run ``detect_language`` and set
          ``original_language_detected = True``.
       c. Rows with empty/missing text are accepted with ``text=None`` and
          ``has_text=False``; they are not rejected.
       d. Collect ``RowError`` for any row that fails validation (e.g. bad
          rating, missing required column).

    Parameters
    ----------
    df:
        DataFrame produced by ``parse_file`` (all columns as strings).
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
                text=text_value,  # None when blank; has_text derives from this
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


# ── French relative date helpers ──────────────────────────────────────────────

# Matches "il y a <quantity> <unit>" produced by Google Maps FR exports.
# Quantity is either a digit string or the words "un" / "une" (→ 1).
# The prefix "Modifié " is stripped before this regex is applied.
_FR_RELATIVE_RE: re.Pattern[str] = re.compile(
    r"il\s+y\s+a\s+"          # literal "il y a" with any whitespace
    r"(une?|\d+)"              # quantity: "un", "une", or one-or-more digits
    r"\s+"
    r"(ans?|mois|semaines?|jours?|heures?)",  # time unit (singular and plural)
    re.IGNORECASE,
)

# Maps the normalised unit word to the matching relativedelta keyword.
_FR_UNIT_TO_KWARG: dict[str, str] = {
    "an":       "years",
    "ans":      "years",
    "mois":     "months",
    "semaine":  "weeks",
    "semaines": "weeks",
    "jour":     "days",
    "jours":    "days",
    "heure":    "hours",
    "heures":   "hours",
}

# ── Internal helpers ──────────────────────────────────────────────────────────


def _parse_date(raw: str | None) -> datetime | None:
    """
    Parse *raw* into a ``datetime`` using a four-stage strategy.

    Stage 1 — Normalisation
    -----------------------
    - Strip leading/trailing whitespace.
    - Replace ``\\xa0`` (non-breaking space, common in Google Maps exports)
      with a regular space.
    - Remove the ``"Modifié "`` prefix emitted by the French Google Maps UI
      when a review has been edited (e.g. ``"Modifié il y a 2 ans"``).

    Stage 2 — Explicit ``strptime`` formats
    ----------------------------------------
    Ten common absolute date formats are tried in order from most specific
    to least specific to avoid ambiguous matches (e.g. ``%d/%m/%Y`` before
    ``%m/%d/%Y``).

    Stage 3 — French relative patterns (regex)
    -------------------------------------------
    ``_FR_RELATIVE_RE`` matches ``"il y a <qty> <unit>"`` strings, where
    *qty* is a digit string or ``"un"`` / ``"une"`` (→ 1).  The offset is
    applied with ``dateutil.relativedelta`` against ``datetime.now()``.

    Stage 4 — ``dateparser`` fallback
    -----------------------------------
    Catches anything not matched above (other ISO variants, English relative
    strings, etc.).  ``RETURN_AS_TIMEZONE_AWARE=False`` keeps output
    consistent with the strptime path.

    Parameters
    ----------
    raw:
        Raw date string from the file, or ``None`` if the cell was empty.

    Returns
    -------
    datetime | None
        Parsed datetime, or ``None`` when the string is absent or cannot be
        resolved by any stage.  A warning is logged on failure so the issue
        is visible in server logs without rejecting the row.
    """
    if not raw:
        logger.warning("Date field is missing or empty — storing None.")
        return None

    # ── Stage 1: normalise ────────────────────────────────────────────────────
    s = raw.strip().replace("\xa0", " ")
    # Remove "Modifié " prefix (handles both accented é and unaccented e,
    # and any amount of whitespace after the word).
    s = re.sub(r"^[Mm]odifi[eé]\s+", "", s)

    # ── Stage 2: explicit strptime formats ────────────────────────────────────
    # Ordered from most specific to least specific.
    _FORMATS = [
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
    for fmt in _FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    # ── Stage 3: French relative patterns ────────────────────────────────────
    m = _FR_RELATIVE_RE.search(s)
    if m:
        qty_str = m.group(1).lower()
        unit_str = m.group(2).lower()
        amount = 1 if qty_str in ("un", "une") else int(qty_str)
        kwarg = _FR_UNIT_TO_KWARG.get(unit_str)
        if kwarg:
            result = datetime.now() - relativedelta(**{kwarg: amount})
            logger.debug("French relative date %r → %s", raw, result)
            return result

    # ── Stage 4: dateparser fallback ─────────────────────────────────────────
    parsed = dateparser.parse(
        s,
        settings={
            "PREFER_DAY_OF_MONTH": "first",
            "RETURN_AS_TIMEZONE_AWARE": False,
            "PREFER_LOCALE_DATE_ORDER": True,
        },
    )
    if parsed is not None:
        logger.debug("dateparser resolved %r → %s", raw, parsed)
        return parsed

    logger.warning("Could not parse date %r — storing None.", raw)
    return None

    # ── Expected behaviour ────────────────────────────────────────────────────
    # _parse_date(None)                              → None
    # _parse_date("")                                → None
    # _parse_date("2024-01-15")                      → datetime(2024, 1, 15, 0, 0)
    # _parse_date("15/01/2024")                      → datetime(2024, 1, 15, 0, 0)
    # _parse_date("il y a 2 ans")                    → now - relativedelta(years=2)
    # _parse_date("il y a un an")                    → now - relativedelta(years=1)
    # _parse_date("il y a 3 mois")                   → now - relativedelta(months=3)
    # _parse_date("il y a un mois")                  → now - relativedelta(months=1)
    # _parse_date("il y a 2 semaines")               → now - relativedelta(weeks=2)
    # _parse_date("il y a une semaine")              → now - relativedelta(weeks=1)
    # _parse_date("il y a 5 jours")                  → now - relativedelta(days=5)
    # _parse_date("il y a un jour")                  → now - relativedelta(days=1)
    # _parse_date("il y a 4 heures")                 → now - relativedelta(hours=4)
    # _parse_date("il y a une heure")                → now - relativedelta(hours=1)
    # _parse_date("Modifié il y a 2 ans")            → now - relativedelta(years=2)
    # _parse_date("Modifié\xa0il\xa0y\xa0a 3 mois") → now - relativedelta(months=3)
    # _parse_date("unparseable garbage")             → None
