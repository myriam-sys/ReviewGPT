-- ─────────────────────────────────────────────────────────────────────────────
-- Migration 001 — Create the reviews table with pgvector support
--
-- Run this once in your Supabase SQL editor before starting the application.
-- Supabase already ships with the pgvector extension; this migration just
-- ensures it is enabled and creates the table + indexes needed by the app.
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable pgvector (idempotent; safe to run more than once)
CREATE EXTENSION IF NOT EXISTS vector;

-- ── reviews ──────────────────────────────────────────────────────────────────
-- One row per validated review.  Rows without text are stored here too
-- (has_text = FALSE) so the UI can report totals accurately; the embedding
-- service (Phase 3) skips them when populating the `embedding` column.

CREATE TABLE IF NOT EXISTS reviews (
    review_id   UUID        PRIMARY KEY,
    session_id  UUID        NOT NULL,
    author      TEXT,
    rating      FLOAT       NOT NULL CHECK (rating BETWEEN 1.0 AND 5.0),
    date        TIMESTAMPTZ,
    text        TEXT,
    language    VARCHAR(10),
    has_text    BOOLEAN     NOT NULL DEFAULT FALSE,
    -- 1024-dim vector; NULL until Phase 3 embedding service populates it.
    embedding   vector(1024),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Fast lookup of all reviews belonging to a session (used by preview + chat)
CREATE INDEX IF NOT EXISTS idx_reviews_session_id
    ON reviews (session_id);

-- Phase 3 will filter WHERE has_text = TRUE before embedding; this speeds it up
CREATE INDEX IF NOT EXISTS idx_reviews_has_text
    ON reviews (has_text);

-- ── Row-Level Security ────────────────────────────────────────────────────────
-- Enable RLS so that the anon key cannot read any rows.
-- The backend uses the service_role key (via SUPABASE_KEY) which bypasses RLS,
-- so no additional policies are needed for server-side writes/reads.
ALTER TABLE reviews ENABLE ROW LEVEL SECURITY;
