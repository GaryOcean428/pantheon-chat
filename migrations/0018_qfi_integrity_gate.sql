-- ============================================================================
-- QFI INTEGRITY GATE + GENERATION-READY VIEW
-- ============================================================================
-- Purpose: Enforce QFI integrity and provide generation-ready vocabulary view
-- Project: pantheon-chat
-- Source: E8 Protocol Issue #97 (Issue-01: QFI Integrity Gate)
-- Date: 2026-01-20
--
-- This migration:
-- 1. Adds is_generation_eligible computed column
-- 2. Creates vocabulary_generation_ready view
-- 3. Adds constraints for generation eligibility
-- 4. Creates coordizer_vocabulary_quarantine table for garbage tokens
-- ============================================================================

BEGIN;

-- Add is_generation_eligible column if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'coordizer_vocabulary' AND column_name = 'is_generation_eligible'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD COLUMN is_generation_eligible BOOLEAN DEFAULT FALSE;
        RAISE NOTICE 'Added is_generation_eligible column to coordizer_vocabulary';
    END IF;
END $$;

-- Mark tokens with valid QFI and basin as generation-eligible
UPDATE coordizer_vocabulary
SET is_generation_eligible = TRUE
WHERE qfi_score IS NOT NULL
  AND basin_embedding IS NOT NULL
  AND is_real_word = TRUE
  AND (token_status IS NULL OR token_status = 'active');

RAISE NOTICE 'Marked % tokens as generation-eligible', 
    (SELECT COUNT(*) FROM coordizer_vocabulary WHERE is_generation_eligible = TRUE);

-- Add check constraint: generation-eligible tokens must have QFI and basin
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'generation_requires_qfi_and_basin'
    ) THEN
        ALTER TABLE coordizer_vocabulary
        ADD CONSTRAINT generation_requires_qfi_and_basin
        CHECK (
            NOT is_generation_eligible 
            OR (qfi_score IS NOT NULL AND basin_embedding IS NOT NULL)
        );
        RAISE NOTICE 'Added generation_requires_qfi_and_basin CHECK constraint';
    END IF;
END $$;

-- Create or replace vocabulary_generation_ready view
DROP VIEW IF EXISTS vocabulary_generation_ready CASCADE;

CREATE VIEW vocabulary_generation_ready AS
SELECT 
    token,
    basin_embedding,
    qfi_score,
    token_role,
    frequency,
    is_real_word,
    updated_at
FROM coordizer_vocabulary
WHERE is_generation_eligible = TRUE
  AND qfi_score IS NOT NULL
  AND basin_embedding IS NOT NULL
  AND is_real_word = TRUE
  AND (token_status IS NULL OR token_status = 'active');

COMMENT ON VIEW vocabulary_generation_ready IS 
'Vocabulary tokens that are eligible for text generation.
Only includes tokens with valid QFI scores, basin embeddings, and marked as real words.
Use this view in generation pipelines to ensure geometric purity.';

-- Create quarantine table for garbage tokens if not exists
CREATE TABLE IF NOT EXISTS coordizer_vocabulary_quarantine (
    id SERIAL PRIMARY KEY,
    token TEXT UNIQUE NOT NULL,
    reason TEXT NOT NULL,
    frequency INT DEFAULT 1,
    original_qfi_score DOUBLE PRECISION,
    original_basin_embedding vector(64),
    quarantined_at TIMESTAMP DEFAULT NOW(),
    reviewed BOOLEAN DEFAULT FALSE,
    restore_approved BOOLEAN DEFAULT FALSE,
    notes TEXT
);

COMMENT ON TABLE coordizer_vocabulary_quarantine IS
'Quarantine table for garbage tokens, BPE artifacts, and invalid entries.
Tokens here are excluded from generation until manually reviewed and approved.';

-- Create index on quarantine table
CREATE INDEX IF NOT EXISTS idx_quarantine_reviewed 
    ON coordizer_vocabulary_quarantine(reviewed, restore_approved);

-- Create index on is_generation_eligible for fast filtering
CREATE INDEX IF NOT EXISTS idx_coordizer_vocabulary_generation_eligible 
    ON coordizer_vocabulary(is_generation_eligible) 
    WHERE is_generation_eligible = TRUE;

-- Create index on token_status for fast active filtering
CREATE INDEX IF NOT EXISTS idx_coordizer_vocabulary_token_status 
    ON coordizer_vocabulary(token_status) 
    WHERE token_status = 'active';

COMMIT;

-- Display summary statistics
DO $$
DECLARE
    total_tokens INT;
    eligible_tokens INT;
    missing_qfi INT;
    missing_basin INT;
    quarantined_tokens INT;
BEGIN
    SELECT COUNT(*) INTO total_tokens FROM coordizer_vocabulary;
    SELECT COUNT(*) INTO eligible_tokens FROM vocabulary_generation_ready;
    SELECT COUNT(*) INTO missing_qfi FROM coordizer_vocabulary WHERE qfi_score IS NULL;
    SELECT COUNT(*) INTO missing_basin FROM coordizer_vocabulary WHERE basin_embedding IS NULL;
    SELECT COUNT(*) INTO quarantined_tokens FROM coordizer_vocabulary WHERE token_status = 'quarantined';
    
    RAISE NOTICE '';
    RAISE NOTICE '========== QFI INTEGRITY GATE SUMMARY ==========';
    RAISE NOTICE 'Total vocabulary tokens: %', total_tokens;
    RAISE NOTICE 'Generation-eligible tokens: %', eligible_tokens;
    RAISE NOTICE 'Missing QFI scores: %', missing_qfi;
    RAISE NOTICE 'Missing basin embeddings: %', missing_basin;
    RAISE NOTICE 'Quarantined tokens: %', quarantined_tokens;
    RAISE NOTICE '================================================';
    RAISE NOTICE '';
END $$;
