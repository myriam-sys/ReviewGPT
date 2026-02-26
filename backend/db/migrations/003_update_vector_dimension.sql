-- Migration 003: update embedding column from vector(1024) to vector(384).
--
-- Switches from intfloat/multilingual-e5-large (1024-dim) to
-- paraphrase-multilingual-MiniLM-L12-v2 (384-dim) for deployment
-- compatibility on Render.com free-tier (~420 MB vs ~1.1 GB model size).
--
-- WARNING: This drops all existing embeddings.  Re-run the embedding
-- background task after applying this migration to repopulate the column.
--
-- Run in the Supabase SQL Editor (Database → SQL Editor → New query).

-- Step 1: Drop the old column (and any indexes that depend on it).
ALTER TABLE reviews DROP COLUMN IF EXISTS embedding;

-- Step 2: Recreate with the new dimension.
ALTER TABLE reviews ADD COLUMN embedding vector(384);
