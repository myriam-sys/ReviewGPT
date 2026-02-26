-- Migration 004: update embedding column from vector(384) to vector(1024).
--
-- Switches from local sentence-transformers model (384-dim) to the
-- Mistral Embed API (mistral-embed, 1024-dim).  The API approach eliminates
-- the 512 MB RAM constraint on Render free-tier deployment.
--
-- WARNING: This drops all existing embeddings.  Re-upload your CSV files
-- after applying this migration to regenerate embeddings via the new model.
--
-- Run in the Supabase SQL Editor (Database → SQL Editor → New query).

-- Step 1: Drop the old column (and any dependent indexes).
ALTER TABLE reviews DROP COLUMN IF EXISTS embedding;

-- Step 2: Recreate with the new dimension.
ALTER TABLE reviews ADD COLUMN embedding vector(1024);
