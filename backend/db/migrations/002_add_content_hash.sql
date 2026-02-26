-- Migration 002: add content_hash column and unique index for idempotent upserts.
--
-- content_hash is an MD5 hex digest of "author|rating|text|date" computed at
-- ingestion time.  The unique index on (session_id, content_hash) means that
-- re-uploading the same CSV is a no-op: the upsert with ON CONFLICT DO NOTHING
-- silently skips rows that already exist, so no duplicates are created and
-- existing embeddings are preserved.
--
-- Run in the Supabase SQL Editor (Database → SQL Editor → New query).

ALTER TABLE reviews
ADD COLUMN IF NOT EXISTS content_hash VARCHAR(32);

CREATE UNIQUE INDEX IF NOT EXISTS idx_reviews_content_hash
ON reviews(session_id, content_hash);
